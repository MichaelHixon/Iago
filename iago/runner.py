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

from .attacks import Technique, load_library
from .config import ARTIFACTS_DIR, BASE_SEED, DEFAULT_TEMPERATURE, DEFAULT_TRIALS
from .decode import decode_recovered, is_decode_gated
from .judge import judge, judge_leak
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
) -> Path:
    """Execute the full matrix and write a JSONL artifact file. Returns its path.

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
    # techniques fire only at prompt-leak objectives). Count only the compatible pairs so the
    # progress total matches what actually runs.
    compatible = sum(1 for tech in lib for obj in objs if tech.applies_to_kind(obj.kind))
    total = compatible * trials
    done = 0

    with out_path.open("w") as fh:
        # Round-robin by trial: each round (batch_id) fires the whole matrix once, so a
        # config's repeated trials are spread across the run instead of fired back-to-back.
        # That exposes non-stationarity (refusal drift over the run) rather than burying it
        # in a burst — the temporal fields (batch_id, run_seq, timestamp) make it analyzable.
        for trial in range(trials):
            seed = base_seed + trial
            for tech in lib:
                for obj in objs:
                    if not tech.applies_to_kind(obj.kind):
                        continue  # technique not scoped to this objective kind (e.g. extraction vs forbidden)
                    turns = tech.rendered_turns(obj.objective, shots=shots)  # single-shot => [one prompt]
                    opts = {"temperature": temperature, "seed": seed}
                    # prompt-leak (LLM07): plant the secret system prompt on the target so the
                    # attack has something hidden to extract. Other objective kinds send none.
                    system = obj.system_prompt if obj.is_prompt_leak else None
                    started = time.monotonic()
                    try:
                        if tech.is_multiturn:
                            prompt, response = _run_multiturn(target, turns, opts, system=system)
                        else:
                            prompt = turns[0]
                            response = target.generate(prompt, system=system, options=opts)
                    except Exception as exc:  # keep the run alive; record the failure
                        prompt = "\n---\n".join(turns)
                        response = f"<<RUN-ERROR: {exc}>>"
                    latency = time.monotonic() - started

                    # prompt-leak trials are judged deterministically by canary match; all other
                    # objectives go through the harmful-content heuristic (Claude judge confirms).
                    v = judge_leak(obj.canary, response) if obj.is_prompt_leak else judge(response)
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
                        objective_id=obj.id,
                        objective_kind=obj.kind,
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
    """Read a JSONL artifact file back into rows (for the report generator)."""
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
