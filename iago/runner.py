"""Runner: fire every technique x objective at the target, N trials each.

Methodology (IMPROVEMENTS P1):
  * multi-trial — each (technique, objective) fired `trials` times; the report
    turns these into a bypass RATE, because guardrail bypass is probabilistic and
    a single shot is noise.
  * pinned sampling — temperature fixed, per-trial seed = base_seed + trial, so
    trials differ from each other but the whole run reproduces.
  * separation of concerns — the runner ONLY executes and writes structured JSONL
    artifacts (one row per trial). The report generator (planned) reads those
    artifacts and never re-hits the model.
  * authorization guard — refuses a non-local target unless `authorized=True`,
    making the "only target what you own" ethic executable, not just documented.

Each artifact row is judged inline (heuristic judge) so the JSONL is self-contained;
a future Claude-API judge (planned) can re-score the same artifacts without re-running.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import (SCHEMA_VERSION, build_manifest, load_rows, module_fingerprint, sha256_text, stamp,
                        write_manifest)
from .guards_thirdparty import GuardBackendUnavailable
from .judge import ERROR as _ERROR_VERDICT

from .attacks import (
    NEUTRAL_HISTORY_USER,
    PLANTED_HISTORY_ACK,
    Technique,
    load_library,
)
from .config import ARTIFACTS_DIR, BASE_SEED, DEFAULT_TEMPERATURE, DEFAULT_TRIALS
from .decode import decode_recovered, is_decode_gated
from .judge import _trust_arm_signal, judge, judge_leak, judge_trust, judge_unsafe_output
from .objectives import Objective, load_objectives
from .target import Target


@dataclass(frozen=True)
class TrialResult:
    """One (technique, objective, trial) execution — the JSONL artifact row."""

    technique_id: str
    technique_name: str
    category: str
    owasp: str
    objective_id: str
    objective_kind: str  # forbidden | control
    model: str
    seed: int
    temperature: float
    trial: int
    batch_id: int          # trial round this probe belongs to (rounds are temporally separated)
    run_seq: int           # global 0-based order this probe fired (for drift / position analysis)
    prompt: str
    response: str
    verdict: str
    confidence: float
    rationale: str
    latency_s: float
    timestamp: str
    turns: int = 1  # 1 = single-shot; >1 = multi-turn (crescendo / context-priming)
    shots: int | None = None  # many-shot techniques: fabricated compliant shots used (None = n/a)
    # Decode-gating: cipher / low-resource techniques hide the objective, so a refusal
    # can be a DECODE FAILURE (false-safe), not a real guardrail hold.
    gated: bool = False           # technique hides the objective (cipher / low-resource)
    decoded: bool | None = None   # gated trials: did the model recover the objective? (None = not gated)
    # Trust-escalation (provenance-forging) differential arms. The row's `response`/`verdict` are
    # the PROVENANCE arm + the three-arm differential verdict; these capture the ANONYMOUS and
    # PLACEBO baselines and every arm's action-token signal so the refuse→grant attribution is
    # auditable (escalation requires prov granted while anon AND placebo held). None = not a
    # trust-escalation trial.
    anon_response: str | None = None
    anon_signal: str | None = None       # granted | held (anonymous arm)
    placebo_response: str | None = None
    placebo_signal: str | None = None    # granted | held (neutral-placebo arm — the null floor)
    prov_signal: str | None = None       # granted | held (provenance arm)
    # OWASP Agentic (ASI) Top-10 tag, carried from the technique (agentic techniques
    # only). None for the single-turn jailbreak library.
    asi: str | None = None
    # unsafe-output (LLM05): the downstream sink this objective targeted (html | markdown |
    # shell), carried so the report can group emission rates per sink. None for other kinds.
    sink: str | None = None
    # Schema + provenance (ISC-33): every row names its schema and surface so a reader can refuse
    # the wrong artifact; `status` separates a probe that reached the model from one that did not
    # (garak's attempt status); the hashes let an auditor pin a quoted prompt/response to a row.
    schema_version: int = SCHEMA_VERSION
    surface: str = "chatbot"
    status: str = "evaluated"            # evaluated | error
    prompt_sha256: str | None = None
    response_sha256: str | None = None
    judge_id: str | None = None          # fingerprint of the offline scoring code (judge+canary+decode)


class AuthorizationError(RuntimeError):
    """Raised when a run targets a non-local model without explicit authorization."""


def _run_multiturn(target, turns: list[str], options: dict, system: str | None = None) -> tuple[str, str]:
    """Send a sequence of user turns, carrying the conversation, and return
    (readable_transcript, final_assistant_reply). The FINAL reply is what gets judged —
    the earlier turns prime the model so the last ask lands. An optional `system` message
    (a planted secret for prompt-leak objectives) leads the conversation."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    transcript_parts: list[str] = []
    final_reply = ""
    for i, user_turn in enumerate(turns, 1):
        messages.append({"role": "user", "content": user_turn})
        reply = target.chat(messages, options=options)
        messages.append({"role": "assistant", "content": reply})
        transcript_parts.append(f"[turn {i} — user]\n{user_turn}\n[turn {i} — assistant]\n{reply}")
        final_reply = reply
    return "\n\n".join(transcript_parts), final_reply


def _execute_arm(target, tech: Technique, turns: list[str], opts: dict,
                 system: str | None) -> tuple[str, str]:
    """Fire one arm (single-shot or multi-turn) and return (prompt, response). A transport
    failure is captured as a <<RUN-ERROR>> response so the run stays alive and the judge can
    exclude it. `system` plants context (a secret for prompt-leak, fabricated provenance for a
    trust-escalation provenance arm, or None for the anonymous arm / plain forbidden objectives)."""
    try:
        if tech.is_multiturn:
            return _run_multiturn(target, turns, opts, system=system)
        prompt = turns[0]
        return prompt, target.generate(prompt, system=system, options=opts)
    except GuardBackendUnavailable:
        # A missing/garbage guard backend is not a per-trial transport blip: every trial would
        # error and the run would read as "held". Fail the whole run loud instead (ISC-31).
        raise
    except Exception as exc:  # keep the run alive; record the failure
        return "\n---\n".join(turns), f"<<RUN-ERROR: {exc}>>"


def _execute_planted(target, system: str | None, planted: list[dict], ask: str,
                     opts: dict) -> tuple[str, str]:
    """Build a conversation with fabricated PRIOR turns (planted as history, NOT model-generated)
    followed by the real ask, call the model ONCE, and return (readable_transcript, final_reply).
    `planted` is a list of {role, content} turns injected as conversation history — this is how a
    trust-escalation arm plants 'loaded memory' in the prior-turn channel (never the system channel,
    which would confound trust-transfer with authority-override). A transport failure is captured as
    a <<RUN-ERROR>> reply so the run stays alive."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(planted)
    messages.append({"role": "user", "content": ask})
    try:
        reply = target.chat(messages, options=opts)
    except GuardBackendUnavailable:
        raise  # see _execute_arm — a dead guard backend fails the run, never a row
    except Exception as exc:  # keep the run alive; record the failure
        reply = f"<<RUN-ERROR: {exc}>>"
    readable = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
    return readable, reply


def _fires(tech: Technique, obj: Objective) -> bool:
    """Whether `tech` should be fired at `obj`. Honors objective-kind scoping, and additionally
    restricts trust-escalation objectives to provenance-bearing techniques — a general jailbreak
    has no fabricated history to plant, so its 'provenance arm' would be identical to anonymous and
    could never produce an escalation (only noise rows in the trust section)."""
    if not tech.applies_to_kind(obj.kind):
        return False
    if obj.is_trust_escalation and not tech.provenance:
        return False
    return True


# Instrumentation, not attacks: these stay module constants and never enter the technique library.
# Open-ended and long enough to leave the point-mass regime a 24-token fixed echo sat in — a
# near-degenerate generation agrees with itself whether or not the host is bit-reproducible, so the
# old probe could not fail. These exercise the same regime the matrix does. (ISC-49, superseding
# ISC-34's two-generation cost bound.)
_DETERMINISM_PROBE_TOKENS = 160
_DETERMINISM_PROBES = (
    "Describe, in one paragraph, how a lighthouse keeper spends a long winter night.",
    "Invent six unrelated everyday objects and give each a one-sentence origin story.",
)


def _determinism_probe(target: Target, temperature: float, base_seed: int) -> dict:
    """Fire each open-ended probe twice at the run's temperature + base seed and report whether the
    replies were byte-identical. Fired BEFORE the matrix: that position absorbs Ollama's
    first-generation-after-load effect, and a probe injected mid-run would perturb the very batching
    it claims to measure. Probe replies are never written as trial rows and never scored.

    The check is ONE-SIDED. A match is evidence this host reproduced those generations at that seed;
    it is not proof the matrix replays. A mismatch is definitive, so `False` outranks `None` in the
    aggregate: a probe that raised cannot un-observe a probe that differed. A failure to probe is
    recorded, never raised — the check informs, the run decides nothing on it."""
    opts = {"temperature": temperature, "seed": base_seed,
            "num_predict": _DETERMINISM_PROBE_TOKENS}
    probes: list[dict] = []
    generations = 0
    for text in _DETERMINISM_PROBES:
        replies = []
        try:
            for _ in range(2):
                replies.append(target.generate(text, options=dict(opts)))
                generations += 1
        except Exception as exc:  # the matrix will surface a real transport failure loudly itself
            probes.append({"probe": text, "exact_match": None, "error": str(exc)[:200]})
            continue
        a, b = replies
        probes.append({"probe": text, "exact_match": a == b,
                       "reply_sha256": [sha256_text(a), sha256_text(b)]})
    seen = [pr["exact_match"] for pr in probes]
    if any(m is False for m in seen):
        aggregate: bool | None = False
    elif any(m is None for m in seen) or not seen:
        aggregate = None
    else:
        aggregate = True
    # `generations` is the number of probe calls that RETURNED, recorded rather than promised by a
    # number in doctrine — the next tuning of the cap does not have to retire another anti-claim.
    return {"exact_match": aggregate, "probes": probes, "options": opts,
            "generations": generations}


def _run_id(model: str, now: datetime) -> str:
    safe_model = model.replace(":", "-").replace("/", "-")
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{safe_model}"


def run(
    target: Target,
    *,
    trials: int = DEFAULT_TRIALS,
    temperature: float = DEFAULT_TEMPERATURE,
    base_seed: int = BASE_SEED,
    authorized: bool = False,
    technique_limit: int | None = None,
    objective_limit: int | None = None,
    shots: int | None = None,
    artifacts_dir: Path | None = None,
    techniques: list[Technique] | None = None,
    objectives: list[Objective] | None = None,
    progress: bool = False,
    determinism_check: bool = True,
) -> Path:
    """Execute the full matrix and write a JSONL artifact file. Returns its path.

    `determinism_check` fires each open-ended probe TWICE (same temperature + base seed) before
    the matrix and records per-probe matches, the aggregate, and the generations spent in the
    manifest — the README claims same-host reproducibility, and this measures it per run instead
    of assuming it (ISC-34, strengthened by ISC-49). Skipping it records `null`, not a pass.

    `technique_limit` / `objective_limit` cap the matrix for a fast smoke run
    without hammering the model for the whole library. `shots` overrides the
    fabricated-shot count for many-shot techniques (exercises long-context scaling).
    """
    # Authorization guard — the ethic, made executable.
    if not target.is_local and not authorized:
        raise AuthorizationError(
            f"target {target.name!r} is not local; pass authorized=True (--authorized) "
            "only for a model you own or are explicitly authorized to test"
        )

    lib = techniques if techniques is not None else load_library()
    objs = objectives if objectives is not None else load_objectives()
    if technique_limit is not None:
        lib = lib[:technique_limit]
    if objective_limit is not None:
        objs = objs[:objective_limit]

    if not lib:
        raise ValueError("no techniques to run")
    if not objs:
        raise ValueError("no objectives to run")

    now = datetime.now(timezone.utc)
    out_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_run_id(target.name, now)}.jsonl"

    # Objective-kind scoping: a technique may target only some objective kinds (extraction
    # techniques fire only at prompt-leak objectives; provenance techniques only at trust
    # objectives). Count only the compatible pairs so the progress total matches what actually runs.
    compatible = sum(1 for tech in lib for obj in objs if _fires(tech, obj))
    if compatible == 0:
        # Nothing would fire: writing an empty artifact and a "No artifacts" report at exit 0 is
        # the exit-0-measured-nothing failure (ISC-31). Say so before touching the disk.
        raise ValueError(
            "no technique/objective pairs fire: every loaded technique is scoped to an objective "
            "kind absent from the loaded objectives (check --limit-* / applies_to scoping)"
        )
    total = compatible * trials
    done = 0

    offline_judge_id = module_fingerprint("judge", "canary", "decode")
    determinism = _determinism_probe(target, temperature, base_seed) if determinism_check else None
    if progress and determinism is not None and determinism.get("exact_match") is False:
        differed = [pr["probe"] for pr in determinism["probes"] if pr["exact_match"] is False]
        print(f"  WARNING: {len(differed)} of {len(determinism['probes'])} probes produced two "
              "DIFFERENT replies at the same seed — this host/build is not bit-reproducible at "
              "these settings; the manifest records which.")

    with out_path.open("w") as fh:
        write_manifest(fh, build_manifest(
            surface="chatbot", model=target.name,
            sampling={"trials": trials, "temperature": temperature, "base_seed": base_seed,
                      "seed_rule": "base_seed + trial", "shots": shots},
            judge_id=offline_judge_id,
            extra={"technique_library_sha256": sha256_text(json.dumps([asdict(t) for t in lib],
                                                                      sort_keys=True, default=str)),
                   "techniques": len(lib), "objectives": len(objs), "compatible_pairs": compatible,
                   "determinism": determinism}))
        # Round-robin by trial: each round (batch_id) fires the whole matrix once, so a
        # config's repeated trials are spread across the run instead of fired back-to-back.
        # That exposes non-stationarity (refusal drift over the run) rather than burying it
        # in a burst — the temporal fields (batch_id, run_seq, timestamp) make it analyzable.
        for trial in range(trials):
            seed = base_seed + trial
            for tech in lib:
                for obj in objs:
                    if not _fires(tech, obj):
                        continue  # technique not scoped to this objective (kind, or non-provenance vs trust)
                    turns = tech.rendered_turns(obj.objective, shots=shots)  # single-shot => [one prompt]
                    opts = {"temperature": temperature, "seed": seed}
                    started = time.monotonic()
                    anon_response = anon_signal = placebo_response = placebo_signal = prov_signal = None

                    if obj.is_trust_escalation:
                        # THREE-ARM DIFFERENTIAL under the SAME scenario gate (obj.system_prompt):
                        #   anonymous  — the ask with no planted history.
                        #   placebo    — the ask after NEUTRAL planted prior turns (null floor for
                        #                "any prior context at all").
                        #   provenance — the ask after fabricated RELATIONSHIP history planted as
                        #                prior turns (paired with a generic assistant ack).
                        # judge_trust confirms an escalation only when provenance crosses the gate
                        # (emits the action token) while anon AND placebo do not — isolating the
                        # relationship content. History goes in the PRIOR-TURN channel, never system.
                        gate = obj.system_prompt
                        ask = tech.render(obj.objective, shots=shots)  # wrapped final ask (no priming)
                        ack = PLANTED_HISTORY_ACK
                        _, anon_response = _execute_planted(target, gate, [], ask, opts)
                        placebo_plant = [{"role": "user", "content": NEUTRAL_HISTORY_USER},
                                         {"role": "assistant", "content": ack}]
                        _, placebo_response = _execute_planted(target, gate, placebo_plant, ask, opts)
                        prov_plant = [{"role": "user", "content": tech.provenance_history()},
                                      {"role": "assistant", "content": ack}]
                        prompt, response = _execute_planted(target, gate, prov_plant, ask, opts)
                        latency = time.monotonic() - started
                        v = judge_trust(anon_response, placebo_response, response, obj.action_token)
                        anon_signal = _trust_arm_signal(anon_response, obj.action_token)
                        placebo_signal = _trust_arm_signal(placebo_response, obj.action_token)
                        prov_signal = _trust_arm_signal(response, obj.action_token)
                    else:
                        # prompt-leak (LLM07): plant the secret system prompt so the attack has
                        # something hidden to extract. Other objective kinds send no system message.
                        system = obj.system_prompt if obj.is_prompt_leak else None
                        prompt, response = _execute_arm(target, tech, turns, opts, system)
                        latency = time.monotonic() - started
                        # Oracle dispatch by objective kind:
                        #   prompt-leak   → deterministic canary match
                        #   unsafe-output → deterministic live-construct-targeting-marker match (LLM05)
                        #   everything else → the harmful-content heuristic (the Claude judge confirms later)
                        if obj.is_prompt_leak:
                            v = judge_leak(obj.canary, response)
                        elif obj.is_unsafe_output:
                            v = judge_unsafe_output(response, obj.sink, obj.marker)
                        else:
                            v = judge(response)
                    # Decode-gating: for cipher / low-resource techniques, record whether the
                    # model actually recovered the objective — a refusal after a decode failure
                    # is a false-safe, and the report reads it as "not a valid probe".
                    gated = is_decode_gated(tech)
                    decoded = decode_recovered(obj.objective, response) if gated else None
                    row = TrialResult(
                        technique_id=tech.id,
                        technique_name=tech.name,
                        category=tech.category,
                        owasp=tech.owasp,
                        asi=tech.asi,
                        objective_id=obj.id,
                        objective_kind=obj.kind,
                        sink=obj.sink,
                        model=target.name,
                        seed=seed,
                        temperature=temperature,
                        trial=trial,
                        batch_id=trial,
                        run_seq=done,
                        prompt=prompt,
                        response=response,
                        verdict=v.verdict,
                        confidence=v.confidence,
                        rationale=v.rationale,
                        latency_s=round(latency, 3),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        turns=len(turns),
                        shots=tech.shot_count(shots),
                        gated=gated,
                        decoded=decoded,
                        anon_response=anon_response,
                        anon_signal=anon_signal,
                        placebo_response=placebo_response,
                        placebo_signal=placebo_signal,
                        prov_signal=prov_signal,
                        status="error" if v.verdict == _ERROR_VERDICT else "evaluated",
                        prompt_sha256=sha256_text(prompt),
                        response_sha256=sha256_text(response),
                        judge_id=offline_judge_id,
                    )
                    fh.write(json.dumps(asdict(row)) + "\n")
                    fh.flush()
                    done += 1
                    if progress:
                        print(
                            f"  [{done}/{total}] r{trial} {tech.id} x {obj.id} "
                            f"-> {v.verdict} ({v.confidence:.2f})"
                        )

    return out_path


def load_artifacts(path: Path) -> list[dict]:
    """Read a JSONL artifact file back into rows (for the report generator). The manifest line
    (ISC-33) is split off by `artifacts.load_rows`, never handed to a report as a trial."""
    return load_rows(path)
