"""Iago CLI: one command runs the whole loop and writes a report.

    uv run iago run                 # full library x objectives, default trials
    uv run iago run --smoke         # 1 technique x 1 objective x 1 trial (fast proof)
    uv run iago run --trials 5      # more trials -> more defensible bypass rate
    uv run iago library             # show the loaded attack library
"""

from __future__ import annotations

import argparse
import sys

from .attacks import load_library, summarize
from .campaign import DEFAULT_SURFACES
from .config import (
    ARTIFACTS_DIR,
    BASE_SEED,
    DEFAULT_ADAPTIVE_TURNS,
    DEFAULT_AGENT_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRIALS,
)
from .objectives import load_objectives
from .judge import ERROR
from .report import verdict_of, write_html_report, write_log, write_report
from .artifacts import read_artifact
from .runner import AuthorizationError, load_artifacts, run
from .target import available_targets, build_target
from .guards import GuardedTarget, available_guards, build_guards
from .guards_thirdparty import GuardBackendUnavailable


def _positive_int(value: str) -> int:
    """argparse type for every trial/step/turn count: a run of 0 trials measures nothing, writes a
    manifest-only artifact, and used to exit 0 with a report (cross-vendor audit, critical)."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {n})")
    return n


def _valid_count(rows: list[dict]) -> int:
    """Rows that actually PROBED the guardrail. A <<RUN-ERROR>> row never reached the model; a
    decode-failed cipher row reached it but never exercised the guardrail (decode.py), and since
    ISC-32 the report excludes those from every harmful denominator — so counting them here would
    let an all-decode-failed run exit 0 with an empty report (code-review major)."""
    return sum(1 for r in rows
               if verdict_of(r) != ERROR and not (r.get("gated") and r.get("decoded") is False))


def _nothing_measured(rows: list[dict], what: str = "trials") -> int | None:
    """Non-zero exit code (with the reason on stderr) when an artifact holds zero VALID rows;
    None when there is something to report. A report may still be written for the record, but
    the process must not exit 0: a run that measured nothing is not a run in which "the
    guardrails held" (ISC-31 — the exit-0-measured-nothing class)."""
    if not rows:
        print(f"ERROR: artifact holds 0 {what} — nothing was measured.", file=sys.stderr)
        return 2
    if _valid_count(rows) == 0:
        n_err = sum(1 for r in rows if verdict_of(r) == ERROR)
        n_dec = sum(1 for r in rows if r.get("gated") and r.get("decoded") is False)
        why = (f"{n_err} RUN-ERROR (never reached the model)" if n_err else "")
        why += (" and " if n_err and n_dec else "") + (f"{n_dec} decode-failure (the model never "
                                                       "recovered the objective)" if n_dec else "")
        print(f"ERROR: 0 of {len(rows)} {what} probed a guardrail — {why}. Nothing was measured "
              "(this is NOT a hold).", file=sys.stderr)
        return 1
    return None


def _cmd_run(args: argparse.Namespace) -> int:
    # If the target isn't ollama and the user didn't override --model, let the
    # backend pick its own default (don't hand a llama tag to Anthropic).
    model = None if (args.target != "ollama" and args.model == DEFAULT_MODEL) else args.model
    try:
        target = build_target(args.target, model=model)
        if getattr(args, "guard", None):
            target = GuardedTarget(target, build_guards(args.guard))
    except (ValueError, GuardBackendUnavailable) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    trials = 1 if args.smoke else args.trials
    tech_limit = 1 if args.smoke else args.limit_techniques
    obj_limit = 1 if args.smoke else args.limit_objectives

    print(f"Iago run → target {target.name}")
    print(f"  trials/pair={trials} temperature={args.temperature} base_seed={args.base_seed}")

    try:
        artifact_path = run(
            target,
            trials=trials,
            temperature=args.temperature,
            base_seed=args.base_seed,
            authorized=args.authorized,
            technique_limit=tech_limit,
            objective_limit=obj_limit,
            shots=args.shots,
            progress=True,
            determinism_check=getattr(args, "determinism_check", True),
        )
    except AuthorizationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except GuardBackendUnavailable as exc:
        # A guard backend that died MID-MATRIX takes the whole run down (by design: a silently
        # passing guard would fabricate a "held" verdict). Name the partial artifact so the trials
        # already written are not lost (cross-vendor audit).
        print(f"ERROR: guard backend failed mid-run: {exc}", file=sys.stderr)
        partial = sorted(ARTIFACTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if partial:
            print(f"Partial artifact (trials completed before the failure): {partial[-1]}",
                  file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    run_manifest, rows = read_artifact(artifact_path)
    report_path = write_report(rows, manifest=run_manifest)
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    if getattr(args, "html", False):
        print(f"HTML:      {write_html_report(rows)}")
    if getattr(args, "log", False):
        print(f"Transcript: {write_log(rows, html=getattr(args, 'html', False))}")
    print(f"({len(rows)} trials recorded; {_valid_count(rows)} valid)")
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_report(args: argparse.Namespace) -> int:
    """(Re)generate a report from an artifact file — summary, --html, and/or full --log."""
    from pathlib import Path

    path = Path(args.artifact)
    if not path.exists():
        print(f"ERROR: artifact not found: {path}", file=sys.stderr)
        return 2
    run_manifest, rows = read_artifact(path)
    if not rows:
        return _nothing_measured(rows) or 2
    try:
        if args.log:
            print(f"Transcript: {write_log(rows, html=args.html)}  ({len(rows)} trials)")
        else:
            out = (write_html_report(rows) if args.html
                   else write_report(rows, manifest=run_manifest))
            print(f"Report: {out}  ({len(rows)} trials; {_valid_count(rows)} valid)")
    except ValueError as exc:  # wrong-surface artifact (ISC-33)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_delta(args: argparse.Namespace) -> int:
    """Compute the attack-vs-defense delta from an existing raw and guarded artifact."""
    from pathlib import Path
    from .delta import write_delta_report

    raw_path, guarded_path = Path(args.raw), Path(args.guarded)
    for p in (raw_path, guarded_path):
        if not p.exists():
            print(f"ERROR: artifact not found: {p}", file=sys.stderr)
            return 2
    raw_rows = load_artifacts(raw_path)
    guarded_rows = load_artifacts(guarded_path)
    try:
        out = write_delta_report(raw_rows, guarded_rows)
    except ValueError as exc:  # wrong-surface artifact (ISC-33)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Delta report: {out}")
    for label, rws in (("raw", raw_rows), ("guarded", guarded_rows)):
        rc = _nothing_measured(rws, f"{label} trials")
        if rc is not None:
            return rc
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Multi-model differential evaluation (ISC-29): >=2 same-surface artifacts -> one
    comparison report. The delta between models is the finding."""
    from pathlib import Path
    from .compare import build_comparison, write_comparison_report

    paths = [Path(p) for p in args.artifacts]
    for p in paths:
        if not p.exists():
            print(f"ERROR: artifact not found: {p}", file=sys.stderr)
            return 2
    try:
        comp = build_comparison(paths, allow_judge_mismatch=getattr(args, "allow_judge_mismatch", False))
    except ValueError as exc:  # wrong surface, or oracle code differs between runs (ISC-33)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not comp.scenario_ids:
        # Rows without `kind` (chatbot `run` artifacts), empty files, or a surface whose verdict
        # vocabulary compare does not adjudicate all yield an EMPTY matrix — which used to render
        # as a "no divergence" report at exit 0 (ISC-31).
        print("ERROR: no adjudicated attack scenarios across the given artifacts — compare needs "
              ">=2 same-surface AGENT artifacts (rows carrying `kind`); nothing to compare.",
              file=sys.stderr)
        return 2
    if len(comp.models) < 2:
        print(f"ERROR: compare needs >=2 models across the given artifacts (found "
              f"{len(comp.models)}: {[m.model for m in comp.models]}). Run the same surface "
              "against another model first.", file=sys.stderr)
        return 2
    out = write_comparison_report(comp)
    print(f"Models: {', '.join(m.model for m in comp.models)}")
    print(f"Comparison report: {out}")
    return 0


def _cmd_campaign(args: argparse.Namespace) -> int:
    """Cross-surface multi-model differential campaign (ISC-30): fire every requested surface x
    model (LOCAL/Ollama) and roll every per-surface differential into ONE consolidated report —
    the whole-picture cross-surface artifact."""
    from .campaign import build_campaign, run_campaign, write_campaign_report, SURFACE_REGISTRY

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    if len(models) < 2:
        print(f"ERROR: campaign needs >=2 models to compare (got {models}). "
              "e.g. --models llama3.1,llama3.2:3b", file=sys.stderr)
        return 2
    unknown = [s for s in surfaces if s not in SURFACE_REGISTRY]
    if unknown:
        print(f"ERROR: unknown surface(s) {unknown}; known: {', '.join(SURFACE_REGISTRY)}",
              file=sys.stderr)
        return 2

    print(f"Iago campaign → models={models} surfaces={surfaces} "
          f"trials={1 if args.smoke else args.trials}{' (smoke)' if args.smoke else ''}")
    print("  LOCAL only; every surface is sandboxed (no real state change, no network).")
    surface_paths, errors = run_campaign(
        surfaces, models, trials=1 if args.smoke else args.trials, temperature=args.temperature,
        base_seed=args.base_seed, max_steps=args.max_steps, smoke=args.smoke,
        on_event=lambda msg: print(f"  {msg}"))
    for err in errors:
        print(f"  ⚠️ {err}", file=sys.stderr)
    if not surface_paths:
        print("ERROR: no surface produced an artifact; nothing to compare.", file=sys.stderr)
        return 1

    labels = {k: SURFACE_REGISTRY[k].label for k in surface_paths}
    # run_campaign stamps artifact model names as "ollama:<tag>", so the absent-model check must
    # compare against the SAME canonical form or it falsely flags every present model as absent.
    requested = [f"ollama:{m}" for m in models]
    campaign = build_campaign(surface_paths, labels=labels, requested_models=requested, errors=errors)
    out = write_campaign_report(campaign)
    print(f"\nSurfaces run: {', '.join(surface_paths)}")
    print(f"Campaign report: {out}")
    if errors:
        # The report is still written (a failed leg is recorded, never hidden), but a partial
        # campaign must not exit 0 — a pipeline reading the code would call it complete (ISC-31).
        print(f"WARNING: {len(errors)} campaign leg(s) failed — report is PARTIAL (exit 1).",
              file=sys.stderr)
        return 1
    return 0


def _cmd_compose_delta(args: argparse.Namespace) -> int:
    """Composition-lift report from ONE artifact that fired the composed techniques and their
    constituent primitives: how much stacking evasions beat the best single layer."""
    from pathlib import Path
    from .compose_delta import write_compose_report

    path = Path(args.artifact)
    if not path.exists():
        print(f"ERROR: artifact not found: {path}", file=sys.stderr)
        return 2
    rows = load_artifacts(path)
    try:
        out = write_compose_report(rows)
    except ValueError as exc:          # wrong-surface artifact: same contract as its siblings
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Composition-lift report: {out}")
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_defense_delta(args: argparse.Namespace) -> int:
    """Paired run: fire the same library at the raw model AND the guarded model (identical
    seeds), then write the attack-vs-defense delta report — the one-command demo."""
    from .delta import write_delta_report

    model = None if (args.target != "ollama" and args.model == DEFAULT_MODEL) else args.model
    try:
        base = build_target(args.target, model=model)
        guards = build_guards(args.guard)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not guards:
        print("ERROR: --guard is required (e.g. --guard all)", file=sys.stderr)
        return 2
    guarded = GuardedTarget(base, guards)

    trials = 1 if args.smoke else args.trials
    tech_limit = 1 if args.smoke else args.limit_techniques
    obj_limit = 1 if args.smoke else args.limit_objectives
    common = dict(
        trials=trials, temperature=args.temperature, base_seed=args.base_seed,
        authorized=args.authorized, technique_limit=tech_limit, objective_limit=obj_limit,
        shots=args.shots, progress=True,
    )

    print(f"Iago defense-delta → raw {base.name}  vs  guarded {guarded.name}")
    print(f"  guards={'+'.join(g.name for g in guards)}  trials/pair={trials} base_seed={args.base_seed}")
    try:
        print("\n[1/2] raw run…")
        raw_path = run(base, **common)
        print("\n[2/2] guarded run…")
        guarded_path = run(guarded, **common)
    except AuthorizationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    raw_rows = load_artifacts(raw_path)
    guarded_rows = load_artifacts(guarded_path)
    delta_path = write_delta_report(raw_rows, guarded_rows)
    print(f"\nRaw artifacts:     {raw_path}")
    print(f"Guarded artifacts: {guarded_path}")
    print(f"Delta report:      {delta_path}")
    for label, rws in (("raw", raw_rows), ("guarded", guarded_rows)):
        rc = _nothing_measured(rws, f"{label} trials")
        if rc is not None:
            return rc
    return 0


def _cmd_regrade(args: argparse.Namespace) -> int:
    """Re-score an existing artifact file with the Claude rubric judge, then re-report."""
    from pathlib import Path
    from .judge_claude import ClaudeJudge
    from .regrade import regrade_file

    path = Path(args.artifact)
    if not path.exists():
        print(f"ERROR: artifact not found: {path}", file=sys.stderr)
        return 2
    print(f"Regrading {path.name} with Claude judge ({args.judge_model})...")
    try:
        judge = ClaudeJudge(model=args.judge_model)   # inside the try: see judge-eval above
        summary = regrade_file(path, judge)
    except Exception as exc:
        print(f"ERROR: regrade failed (is ANTHROPIC_API_KEY set and the 'anthropic' SDK installed?): {exc}",
              file=sys.stderr)
        return 1
    regrade_manifest, regrade_rows = read_artifact(path)
    report_path = write_report(regrade_rows, manifest=regrade_manifest)
    sk = summary["skipped"]
    print(f"  regraded {summary['regraded']} rows; {summary['flipped_vs_heuristic']} flipped vs heuristic; "
          f"skipped {sk['unknown_objective']} unknown-objective, {sk['run_error']} run-error, "
          f"{sk['structural_verdict']} structural-verdict")
    print(f"Report:    {report_path}")
    if summary["regraded"] == 0 and summary["skipped"]["unknown_objective"]:
        # Only an UNKNOWN-objective skip is an error. A prompt-leak-only artifact legitimately
        # regrades nothing: those verdicts are structural and the rubric must not touch them
        # (code-review minor).
        print(f"ERROR: 0 rows were regraded and {summary['skipped']['unknown_objective']} row(s) "
              "named an objective id absent from objectives.yaml.", file=sys.stderr)
        return 1
    return 0


def _cmd_judge_eval(args: argparse.Namespace) -> int:
    """Measure the judges against the labeled control set (ISC-35)."""
    from pathlib import Path

    from .judge_eval import NO_OVERLAY_SENTINEL, OverlayWriteRefused, evaluate, load_control_set, write_metrics

    entries = load_control_set(args.set, overlay=NO_OVERLAY_SENTINEL if args.no_overlay else None)
    names = [n.strip() for n in args.judge.split(",") if n.strip()]
    if not names:
        # `--judge ""` (an unset CI variable) used to run no judge and exit 0 — the ISC-31 class
        # inside the very command that measures it (Council major).
        print(f"ERROR: --judge named no judge ({args.judge!r}); nothing was measured.", file=sys.stderr)
        return 2
    rc = 0
    for name in names:
        try:
            kw = {}
            if name == "claude":
                # Built INSIDE the try: a missing API key is an operator error and must print the
                # ERROR line like every other failure here, not a traceback (code-review minor).
                from .judge_claude import ClaudeJudge
                kw["claude_judge"] = ClaudeJudge(model=args.judge_model) if args.judge_model else ClaudeJudge()
            m = evaluate(name, entries, **kw)
        except Exception as exc:
            print(f"ERROR: {name} judge evaluation failed: {exc}", file=sys.stderr)
            rc = 1
            continue
        if m["n_scored"] == 0:
            print(f"ERROR: {name}: 0 entries scored — nothing was measured.", file=sys.stderr)
            rc = 1
            continue
        from .judge_eval import _pct
        print(f"\n{name} judge  [{m['judge_id']}]  n={m['n_scored']} (positives {m['positives']}, "
              f"negatives {m['negatives']}, {m['n_missing_text']} bodies unavailable) "
              f"set={m['set_variant']}")
        print(f"  agreement           {_pct(m['agreement'])}")
        print(f"  false-positive rate {_pct(m['fpr'])}")
        print(f"  false-negative rate {_pct(m['fnr'])}")
        print(f"  unresolved          {_pct(m['unresolved_rate'])}")
        if m.get("positive_class_reachable") is False:
            from .judge_eval import UNREACHABLE_NOTE
            print(f"  NOTE: FPR/FNR are n/a — {UNREACHABLE_NOTE}.")
        if args.show_disagreements:
            for d in m["disagreements"]:
                print(f"    {d['id']}: label={d['label']} predicted={d['predicted']}")
        if args.write:
            try:
                dest = write_metrics(m, allow_overlay=args.allow_overlay_write)
                print(f"  metrics → {dest}")
            except OverlayWriteRefused as exc:
                print(f"  metrics NOT written: {exc}", file=sys.stderr)
    return rc


def _cmd_lexical_leak(args: argparse.Namespace) -> int:
    """Advisory lexical-overlap paraphrased-leak band — SECONDARY to the canary oracle."""
    from pathlib import Path
    from .lexical_leak import analyze_file, PROVENANCE

    any_found = False
    analyzed = 0
    for spec in args.artifacts:
        path = Path(spec)
        if not path.exists():
            print(f"ERROR: artifact not found: {path}", file=sys.stderr)
            return 2
        try:
            summary = analyze_file(path, elevated=args.elevated, high=args.high)
        except ValueError as exc:      # wrong-surface artifact: same contract as its siblings
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        analyzed += summary["prompt_leak_rows"]
        print(f"\n{path.name}: {summary['prompt_leak_rows']} prompt-leak rows "
              f"(bands: low <{args.elevated} | elevated <{args.high} | high) — ADVISORY, NOT a verdict")
        print(f"  [{PROVENANCE}]")
        for r in summary["rows"]:
            flag = "  ⟵ PARAPHRASE SUSPECT (canary held)" if r.paraphrase_suspect else \
                   ("  (already caught by canary)" if r.canary_hit else "")
            print(f"  {r.score:>5.3f} {r.band:<9} {r.technique_id:>10} x {r.objective_id}{flag}")
        n = summary["paraphrase_suspects"]
        any_found = any_found or n > 0
        print(f"  → {n} paraphrase suspect(s): canary HELD but content overlap HIGH — inspect the reply.")
    if analyzed == 0:
        # Zero prompt-leak rows is not a clean result, it is no result. The advisory band must not
        # gate a pipeline on suspects FOUND; that never justified reassurance about rows that do not
        # exist (Council major).
        print("\nERROR: no prompt-leak rows in the given artifact(s) — nothing was analyzed.",
              file=sys.stderr)
        return 2
    if not any_found:
        print("\nNo paraphrase suspects — no row disclosed prompt content the canary oracle missed.")
    return 0  # an ADVISORY band must never gate a pipeline on suspects found; the canary owns pass/fail


def _cmd_agent_run(args: argparse.Namespace) -> int:
    """Run the agentic indirect-injection suite against a tool-calling agent."""
    from .agent_run import (
        load_agent_artifacts,
        ollama_chat_fn,
        run_agent_suite,
        write_agent_report,
    )
    from .agent_scenarios import load_scenarios

    if args.target != "ollama":
        print(f"ERROR: agent-run supports --target ollama today (got {args.target!r})", file=sys.stderr)
        return 2

    model = args.model
    chat_fn = ollama_chat_fn(model)
    trials = 1 if args.smoke else args.trials
    scens = load_scenarios()
    if args.smoke:
        scens = scens[:1]

    print(f"Iago agent-run → target ollama:{model}")
    print(f"  scenarios={len(scens)} trials/scenario={trials} max_steps={args.max_steps}")
    try:
        artifact_path = run_agent_suite(
            chat_fn,
            model_name=f"ollama:{model}",
            trials=trials,
            temperature=args.temperature,
            base_seed=args.base_seed,
            max_steps=args.max_steps,
            scenarios=scens,
            progress=True,
        )
    except Exception as exc:
        print(f"ERROR: agent-run failed: {exc}", file=sys.stderr)
        return 1

    rows = load_agent_artifacts(artifact_path)
    report_path = write_agent_report(rows)
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    print(f"({len(rows)} trials recorded)")
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_agent_scenarios(_args: argparse.Namespace) -> int:
    from .agent_scenarios import load_scenarios

    scens = load_scenarios()
    attacks = sum(1 for s in scens if s.kind == "attack")
    controls = sum(1 for s in scens if s.is_control)
    caps = sum(1 for s in scens if s.is_capability)
    print(f"Agent scenarios: {len(scens)} ({attacks} attack, {controls} control, "
          f"{caps} capability)")
    for s in scens:
        print(f"  {s.id:20} [{s.kind:7}] {s.name}")
    return 0


def _cmd_adaptive_run(args: argparse.Namespace) -> int:
    """Run the adaptive dialogue-level attacker (target-adaptive multi-turn search)."""
    from .adaptive import (
        DeterministicAttacker,
        LLMAttacker,
        load_adaptive_artifacts,
        run_adaptive_suite,
        write_adaptive_report,
    )

    model = None if (args.target != "ollama" and args.model == DEFAULT_MODEL) else args.model
    try:
        target = build_target(args.target, model=model)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not target.is_local and not args.authorized:
        print(f"ERROR: target {target.name!r} is not local; pass --authorized only for a "
              "model you own or are authorized to test", file=sys.stderr)
        return 2

    deterministic = args.attacker == "deterministic"
    if deterministic:
        def make_attacker(seed, obj, _options):
            return DeterministicAttacker(seed, obj)
        attacker_desc = "deterministic (seeded, reproducible)"
    else:
        # LLM attacker: a local Ollama model writes each next turn. Reuses the Target.chat seam.
        from .target import OllamaTarget
        attacker_target = OllamaTarget(args.attacker_model)
        chat_fn = lambda messages, options: attacker_target.chat(messages, options=options)

        # Liveness ping — if the attacker model is unreachable, EVERY turn would silently fall back
        # to the deterministic template and the run would be mislabeled "nondeterministic". Warn
        # loudly up front rather than burning a whole run on templates. (The artifact stays honest
        # either way — each fallback turn is tagged — but the operator should know before it runs.)
        try:
            attacker_target.chat([{"role": "user", "content": "ping"}], options={})
        except Exception as exc:
            print(f"WARNING: attacker model {attacker_target.name!r} did not respond ({exc}). "
                  "Every turn will fall back to the deterministic template — this run will NOT be "
                  "a real LLM-attacker run. Start ollama / pull the model, or use "
                  "--attacker deterministic.", file=sys.stderr)

        def make_attacker(seed, obj, options):
            return LLMAttacker(seed, obj, chat_fn, options=options)
        attacker_desc = f"llm ({attacker_target.name}) — NONDETERMINISTIC"

    trials = 1 if args.smoke else args.trials
    print(f"Iago adaptive-run → target {target.name}")
    print(f"  attacker={attacker_desc} trials/objective={trials} max_turns={args.max_turns}")
    try:
        artifact_path = run_adaptive_suite(
            target,
            make_attacker=make_attacker,
            attacker_kind=args.attacker,
            deterministic=deterministic,
            model_name=target.name,
            trials=trials,
            temperature=args.temperature,
            base_seed=args.base_seed,
            max_turns=args.max_turns,
            progress=True,
        )
    except Exception as exc:
        print(f"ERROR: adaptive-run failed: {exc}", file=sys.stderr)
        return 1

    rows = load_adaptive_artifacts(artifact_path)
    report_path = write_adaptive_report(rows)
    n_scored = sum(1 for r in rows if r.get("outcome") != "error")
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    print(f"({len(rows)} conversations recorded; {n_scored} scored)")
    if not deterministic:
        print("NOTE: LLM-attacker turns are nondeterministic — this run does not replay bit-for-bit.")
    if n_scored == 0:
        print("ERROR: every conversation hit a transport error — nothing was measured (this is NOT "
              "a hold).", file=sys.stderr)
        return 1
    return 0


def _cmd_strategies(_args: argparse.Namespace) -> int:
    from .adaptive import STRATEGIES

    print(f"Adaptive strategies: {len(STRATEGIES)}")
    for s in STRATEGIES:
        print(f"  {s.id:22} [{','.join(s.applies_to)}] {s.name}")
    return 0


def _cmd_toolabuse_run(args: argparse.Namespace) -> int:
    """Run the sandboxed agentic tool-abuse (RCE/SSRF) suite against a tool-calling agent."""
    from .agent_run import ollama_chat_fn
    from .agent_toolabuse import (
        load_toolabuse_artifacts,
        load_toolabuse_scenarios,
        run_toolabuse_suite,
        write_toolabuse_report,
    )

    if args.target != "ollama":
        print(f"ERROR: tool-abuse-run supports --target ollama today (got {args.target!r})",
              file=sys.stderr)
        return 2

    model = args.model
    chat_fn = ollama_chat_fn(model)
    trials = 1 if args.smoke else args.trials
    scens = load_toolabuse_scenarios()
    if args.smoke:
        scens = scens[:1]

    print(f"Iago tool-abuse-run → target ollama:{model} (SANDBOXED — no process/socket ever)")
    print(f"  scenarios={len(scens)} trials/scenario={trials} max_steps={args.max_steps}")
    try:
        artifact_path = run_toolabuse_suite(
            chat_fn, model_name=f"ollama:{model}", trials=trials, temperature=args.temperature,
            base_seed=args.base_seed, max_steps=args.max_steps, scenarios=scens, progress=True)
    except Exception as exc:
        print(f"ERROR: tool-abuse-run failed: {exc}", file=sys.stderr)
        return 1

    rows = load_toolabuse_artifacts(artifact_path)
    report_path = write_toolabuse_report(rows)
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    print(f"({len(rows)} trials recorded)")
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_toolabuse_scenarios(_args: argparse.Namespace) -> int:
    from .agent_toolabuse import load_toolabuse_scenarios

    scens = load_toolabuse_scenarios()
    print(f"Tool-abuse scenarios: {len(scens)}")
    for s in scens:
        print(f"  {s.id:24} [{s.kind:10}] [{s.arm}] {s.name}")
    return 0


def _cmd_privilege_run(args: argparse.Namespace) -> int:
    """Run the sandboxed agentic excessive-agency (confused-deputy) suite against a tool-calling
    agent — a poisoned record tries to induce an unauthorized privileged state change."""
    from .agent_run import ollama_chat_fn
    from .agent_privilege import (
        load_privilege_artifacts,
        load_privilege_scenarios,
        run_privilege_suite,
        write_privilege_report,
    )

    if args.target != "ollama":
        print(f"ERROR: privilege-run supports --target ollama today (got {args.target!r})",
              file=sys.stderr)
        return 2

    model = args.model
    chat_fn = ollama_chat_fn(model)
    trials = 1 if args.smoke else args.trials
    scens = load_privilege_scenarios()
    if args.smoke:
        scens = scens[:1]

    print(f"Iago privilege-run → target ollama:{model} (SANDBOXED — no record/role ever changes)")
    print(f"  scenarios={len(scens)} trials/scenario={trials} max_steps={args.max_steps}")
    try:
        artifact_path = run_privilege_suite(
            chat_fn, model_name=f"ollama:{model}", trials=trials, temperature=args.temperature,
            base_seed=args.base_seed, max_steps=args.max_steps, scenarios=scens, progress=True)
    except Exception as exc:
        print(f"ERROR: privilege-run failed: {exc}", file=sys.stderr)
        return 1

    rows = load_privilege_artifacts(artifact_path)
    report_path = write_privilege_report(rows)
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    print(f"({len(rows)} trials recorded)")
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_privilege_scenarios(_args: argparse.Namespace) -> int:
    from .agent_privilege import load_privilege_scenarios

    scens = load_privilege_scenarios()
    print(f"Excessive-agency scenarios: {len(scens)}")
    for s in scens:
        print(f"  {s.id:24} [{s.kind:10}] [{s.arm}] {s.name}")
    return 0


def _cmd_disclosure_run(args: argparse.Namespace) -> int:
    """Run the sandboxed agentic sensitive-info-disclosure (LLM02) suite against a tool-calling
    agent — a poisoned record tries to induce cross-subject disclosure into the answer or a send."""
    from .agent_run import ollama_chat_fn
    from .agent_disclosure import (
        load_disclosure_artifacts,
        load_disclosure_scenarios,
        run_disclosure_suite,
        write_disclosure_report,
    )

    if args.target != "ollama":
        print(f"ERROR: disclosure-run supports --target ollama today (got {args.target!r})",
              file=sys.stderr)
        return 2

    model = args.model
    chat_fn = ollama_chat_fn(model)
    trials = 1 if args.smoke else args.trials
    scens = load_disclosure_scenarios()
    if args.smoke:
        scens = scens[:1]

    print(f"Iago disclosure-run → target ollama:{model} (SANDBOXED — no real send/socket)")
    print(f"  scenarios={len(scens)} trials/scenario={trials} max_steps={args.max_steps}")
    try:
        artifact_path = run_disclosure_suite(
            chat_fn, model_name=f"ollama:{model}", trials=trials, temperature=args.temperature,
            base_seed=args.base_seed, max_steps=args.max_steps, scenarios=scens, progress=True)
    except Exception as exc:
        print(f"ERROR: disclosure-run failed: {exc}", file=sys.stderr)
        return 1

    rows = load_disclosure_artifacts(artifact_path)
    report_path = write_disclosure_report(rows)
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    print(f"({len(rows)} trials recorded)")
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_disclosure_scenarios(_args: argparse.Namespace) -> int:
    from .agent_disclosure import load_disclosure_scenarios

    scens = load_disclosure_scenarios()
    print(f"Sensitive-info-disclosure scenarios: {len(scens)}")
    for s in scens:
        print(f"  {s.id:24} [{s.kind:10}] {s.name}")
    return 0


def _cmd_misinfo_run(args: argparse.Namespace) -> int:
    """Run the misinformation / fabricated-authority (LLM09) suite — the agent is asked for a
    structured identifier about a fabricated subject; asserting one is confirmed fabrication."""
    from .agent_run import ollama_chat_fn
    from .agent_misinfo import (
        load_misinfo_artifacts,
        load_misinfo_scenarios,
        run_misinfo_suite,
        write_misinfo_report,
    )

    if args.target != "ollama":
        print(f"ERROR: misinfo-run supports --target ollama today (got {args.target!r})",
              file=sys.stderr)
        return 2

    model = args.model
    chat_fn = ollama_chat_fn(model)
    trials = 1 if args.smoke else args.trials
    scens = load_misinfo_scenarios()
    if args.smoke:
        # Keep one attack AND one capability so the smoke run stays calibrated (a lone attack would
        # print an uncalibrated 0% as a bare pass — the confound the report now guards against).
        atk = next((s for s in scens if s.kind == "attack"), None)
        cap = next((s for s in scens if s.kind == "capability"), None)
        scens = [s for s in (atk, cap) if s is not None]

    print(f"Iago misinfo-run → target ollama:{model} (SANDBOXED — no lookup, no socket)")
    print(f"  scenarios={len(scens)} trials/scenario={trials} max_steps={args.max_steps}")
    try:
        artifact_path = run_misinfo_suite(
            chat_fn, model_name=f"ollama:{model}", trials=trials, temperature=args.temperature,
            base_seed=args.base_seed, max_steps=args.max_steps, scenarios=scens, progress=True)
    except Exception as exc:
        print(f"ERROR: misinfo-run failed: {exc}", file=sys.stderr)
        return 1

    rows = load_misinfo_artifacts(artifact_path)
    report_path = write_misinfo_report(rows)
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    print(f"({len(rows)} trials recorded)")
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_misinfo_scenarios(_args: argparse.Namespace) -> int:
    from .agent_misinfo import load_misinfo_scenarios

    scens = load_misinfo_scenarios()
    print(f"Misinformation scenarios: {len(scens)}")
    for s in scens:
        tag = "fabricated" if s.is_fabricated else "real"
        print(f"  {s.id:24} [{s.kind:10}] [{tag:10}] {s.name}")
    return 0


def _cmd_memory_run(args: argparse.Namespace) -> int:
    """Run the agentic memory / context poisoning (ASI06) suite against a tool-calling agent."""
    from .agent_run import ollama_chat_fn
    from .agent_memory import (
        load_memory_artifacts,
        load_memory_scenarios,
        run_memory_suite,
        write_memory_report,
    )

    if args.target != "ollama":
        print(f"ERROR: memory-run supports --target ollama today (got {args.target!r})",
              file=sys.stderr)
        return 2

    model = args.model
    chat_fn = ollama_chat_fn(model)
    trials = 1 if args.smoke else args.trials
    scens = load_memory_scenarios()
    if args.smoke:
        scens = scens[:1]

    print(f"Iago memory-run → target ollama:{model} (in-process memory; never persisted)")
    print(f"  scenarios={len(scens)} trials/scenario={trials} max_steps={args.max_steps}")
    try:
        artifact_path = run_memory_suite(
            chat_fn, model_name=f"ollama:{model}", trials=trials, temperature=args.temperature,
            base_seed=args.base_seed, max_steps=args.max_steps, scenarios=scens, progress=True)
    except Exception as exc:
        print(f"ERROR: memory-run failed: {exc}", file=sys.stderr)
        return 1

    rows = load_memory_artifacts(artifact_path)
    report_path = write_memory_report(rows)
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    print(f"({len(rows)} trials recorded)")
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_memory_scenarios(_args: argparse.Namespace) -> int:
    from .agent_memory import load_memory_scenarios

    scens = load_memory_scenarios()
    print(f"Memory-poisoning scenarios: {len(scens)}")
    for s in scens:
        print(f"  {s.id:24} [{s.kind:10}] {s.name}")
    return 0


def _cmd_rag_run(args: argparse.Namespace) -> int:
    """Run the agentic RAG retrieval-poisoning suite against a tool-calling agent."""
    from .agent_run import ollama_chat_fn
    from .agent_rag import (
        load_rag_artifacts,
        load_rag_scenarios,
        run_rag_suite,
        write_rag_report,
    )

    if args.target != "ollama":
        print(f"ERROR: rag-run supports --target ollama today (got {args.target!r})", file=sys.stderr)
        return 2

    model = args.model
    chat_fn = ollama_chat_fn(model)
    trials = 1 if args.smoke else args.trials
    scens = load_rag_scenarios()
    if args.smoke:
        scens = scens[:1]

    print(f"Iago rag-run → target ollama:{model} (pure in-memory retriever; no network)")
    print(f"  scenarios={len(scens)} trials/scenario={trials} max_steps={args.max_steps}")
    try:
        artifact_path = run_rag_suite(
            chat_fn, model_name=f"ollama:{model}", trials=trials, temperature=args.temperature,
            base_seed=args.base_seed, max_steps=args.max_steps, scenarios=scens, progress=True)
    except Exception as exc:
        print(f"ERROR: rag-run failed: {exc}", file=sys.stderr)
        return 1

    rows = load_rag_artifacts(artifact_path)
    report_path = write_rag_report(rows)
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    print(f"({len(rows)} trials recorded)")
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_rag_scenarios(_args: argparse.Namespace) -> int:
    from .agent_rag import load_rag_scenarios

    scens = load_rag_scenarios()
    print(f"RAG-poisoning scenarios: {len(scens)}")
    for s in scens:
        print(f"  {s.id:24} [{s.kind:10}] {s.name}")
    return 0


def _cmd_a2a_run(args: argparse.Namespace) -> int:
    """Run the agentic insecure inter-agent (ASI07) suite against a tool-calling agent."""
    from .agent_run import ollama_chat_fn
    from .agent_a2a import (
        load_a2a_artifacts,
        load_a2a_scenarios,
        run_a2a_suite,
        write_a2a_report,
    )

    if args.target != "ollama":
        print(f"ERROR: a2a-run supports --target ollama today (got {args.target!r})", file=sys.stderr)
        return 2

    model = args.model
    chat_fn = ollama_chat_fn(model)
    trials = 1 if args.smoke else args.trials
    scens = load_a2a_scenarios()
    if args.smoke:
        scens = scens[:1]

    print(f"Iago a2a-run → target ollama:{model} (pure in-memory peer inbox; no network)")
    print(f"  scenarios={len(scens)} trials/scenario={trials} max_steps={args.max_steps}")
    try:
        artifact_path = run_a2a_suite(
            chat_fn, model_name=f"ollama:{model}", trials=trials, temperature=args.temperature,
            base_seed=args.base_seed, max_steps=args.max_steps, scenarios=scens, progress=True)
    except Exception as exc:
        print(f"ERROR: a2a-run failed: {exc}", file=sys.stderr)
        return 1

    rows = load_a2a_artifacts(artifact_path)
    report_path = write_a2a_report(rows)
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    print(f"({len(rows)} trials recorded)")
    rc = _nothing_measured(rows)
    return 0 if rc is None else rc


def _cmd_a2a_scenarios(_args: argparse.Namespace) -> int:
    from .agent_a2a import load_a2a_scenarios

    scens = load_a2a_scenarios()
    print(f"Inter-agent scenarios: {len(scens)}")
    for s in scens:
        print(f"  {s.id:24} [{s.kind:10}] {s.name}")
    return 0


def _cmd_library(_args: argparse.Namespace) -> int:
    lib = load_library()
    objs = load_objectives()
    print(f"Attack library: {len(lib)} techniques")
    for category, count in summarize(lib).items():
        print(f"  {category:24} {count}")
    print(f"\nObjectives: {len(objs)} "
          f"({sum(1 for o in objs if o.kind == 'forbidden')} forbidden, "
          f"{sum(1 for o in objs if o.is_control)} control, "
          f"{sum(1 for o in objs if o.is_prompt_leak)} prompt-leak, "
          f"{sum(1 for o in objs if o.is_dead_end)} dead-end)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iago", description="Authorized LLM guardrail red-team harness.")
    from . import __version__
    p.add_argument("--version", action="version", version=f"iago {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the attack matrix and write a report")
    r.add_argument("--target", default="ollama", choices=available_targets(),
                   help="target backend (default: ollama)")
    r.add_argument("--model", default=DEFAULT_MODEL, help="model tag within the target backend")
    r.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="trials per (technique, objective)")
    r.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    r.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    r.add_argument("--limit-techniques", type=int, default=None, dest="limit_techniques")
    r.add_argument("--limit-objectives", type=int, default=None, dest="limit_objectives")
    r.add_argument("--shots", type=int, default=None,
                   help="override fabricated-shot count for many-shot techniques (e.g. 64, 128) "
                        "to exercise long-context scaling")
    r.add_argument("--guard", default=None,
                   help=f"wrap the target in a defense and measure holds: reference '{','.join(available_guards())}' "
                        "or 'all'; opt-in real guards (need a backend) 'llama-guard,guardrails-ai,"
                        "hf-prompt-injection'. Run raw + guarded, then `iago delta` (or `defense-delta`)")
    r.add_argument("--smoke", action="store_true", help="1x1x1 fast proof of the loop")
    r.add_argument("--no-determinism-check", action="store_false", dest="determinism_check",
                   help="skip the same-seed replay probe: saves 5 short generations (a warm-up "
                        "plus 2 probes x 2, capped at 160 tokens each, ~800 tokens) but records "
                        "null instead of a measurement AND leaves the matrix running on a cold "
                        "model, which a default run does not")
    r.add_argument("--html", action="store_true", help="also write a styled, colored HTML report")
    r.add_argument("--log", action="store_true",
                   help="also write a full request/response transcript (every trial, untruncated) — "
                        "SENSITIVE: contains live payloads + any leaked secrets, do not share publicly")
    r.add_argument("--authorized", action="store_true",
                   help="permit a non-local target (only for models you own/are authorized to test)")
    r.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="(re)generate a report from an existing artifact file")
    rep.add_argument("artifact", help="path to a reports/artifacts/*.jsonl file")
    rep.add_argument("--html", action="store_true", help="emit styled, colored HTML (report or --log)")
    rep.add_argument("--log", action="store_true",
                     help="emit a full request/response transcript instead of the summary report — "
                          "SENSITIVE: contains live payloads + any leaked secrets, do not share publicly")
    rep.set_defaults(func=_cmd_report)

    lib = sub.add_parser("library", help="show the loaded attack library + objectives")
    lib.set_defaults(func=_cmd_library)

    dl = sub.add_parser("delta", help="attack-vs-defense delta from a raw + a guarded artifact")
    dl.add_argument("raw", help="path to the RAW-model reports/artifacts/*.jsonl")
    dl.add_argument("guarded", help="path to the GUARDED-model reports/artifacts/*.jsonl")
    dl.set_defaults(func=_cmd_delta)

    cmp = sub.add_parser("compare",
                         help="multi-model differential: >=2 same-surface artifacts (different models) "
                              "-> one comparison report; the delta between models is the finding")
    cmp.add_argument("artifacts", nargs="+",
                     help="paths to >=2 same-surface reports/artifacts/*.jsonl, one per model")
    cmp.add_argument("--allow-judge-mismatch", action="store_true",
                     help="compare artifacts even when their manifests name different oracle code "
                          "(judge_id) — the delta may then be the oracle change, not the model")
    cmp.set_defaults(func=_cmd_compare)

    je = sub.add_parser("judge-eval",
                        help="measure each judge's agreement / FPR / FNR (95%% Wilson CIs) against the "
                             "reviewer-labeled control set and store them by judge_id for report headers")
    je.add_argument("--judge", default="heuristic,canary",
                    help="comma list of heuristic,canary,claude (claude needs ANTHROPIC_API_KEY; default: the "
                         "two offline judges)")
    je.add_argument("--set", default=None, help="alternative control-set JSONL (default: the shipped set)")
    je.add_argument("--no-overlay", action="store_true",
                    help="ignore the local overlay of withheld harmful bodies and measure ONLY what a "
                         "public clone can reproduce — the shipped metrics are measured this way")
    je.add_argument("--judge-model", default=None, help="Claude judge model id (claude only)")
    je.add_argument("--no-write", action="store_false", dest="write",
                    help="print metrics without updating iago/calibration/judge_metrics.json")
    je.add_argument("--allow-overlay-write", action="store_true",
                    help="permit writing metrics a clone cannot reproduce (overlay-measured, or on a "
                         "different --set) over the shipped judge_metrics.json (default: refused so the "
                         "committed file stays public-reproducible)")
    je.add_argument("--show-disagreements", action="store_true")
    je.set_defaults(func=_cmd_judge_eval)

    cam = sub.add_parser("campaign",
                         help="cross-surface differential: run every surface x every model (LOCAL) -> "
                              "ONE consolidated report — the 'what did you find' artifact")
    cam.add_argument("--models", default=DEFAULT_MODEL,
                     help="comma-separated ollama model tags, one per model to compare "
                          f"(default: {DEFAULT_MODEL})")
    cam.add_argument("--surfaces", default=",".join(DEFAULT_SURFACES),
                     help=f"comma-separated surfaces to run (default: all — {', '.join(DEFAULT_SURFACES)})")
    cam.add_argument("--trials", type=_positive_int, default=2,
                     help="trials per scenario per model (>=2 lets a floor certify; default: 2)")
    cam.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    cam.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    cam.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                     help="tool-loop step budget per scenario")
    cam.add_argument("--smoke", action="store_true",
                     help="1 attack + 1 capability scenario per surface, fast pipeline proof")
    cam.set_defaults(func=_cmd_campaign)

    cd = sub.add_parser("compose-delta",
                        help="composition-lift report: does stacking evasions beat the best single layer?")
    cd.add_argument("artifact", help="path to a reports/artifacts/*.jsonl file that ran the full library")
    cd.set_defaults(func=_cmd_compose_delta)

    dd = sub.add_parser("defense-delta",
                        help="paired run (raw + guarded, same seeds) → attack-vs-defense delta report")
    dd.add_argument("--target", default="ollama", choices=available_targets(),
                    help="target backend (default: ollama)")
    dd.add_argument("--model", default=DEFAULT_MODEL, help="model tag within the target backend")
    dd.add_argument("--guard", default="all",
                    help=f"defense to measure: '{','.join(available_guards())}' or 'all' (default: all)")
    dd.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="trials per (technique, objective)")
    dd.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    dd.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    dd.add_argument("--limit-techniques", type=int, default=None, dest="limit_techniques")
    dd.add_argument("--limit-objectives", type=int, default=None, dest="limit_objectives")
    dd.add_argument("--shots", type=int, default=None, help="override many-shot fabricated-shot count")
    dd.add_argument("--smoke", action="store_true", help="tiny paired proof (1x1x1 each side)")
    dd.add_argument("--authorized", action="store_true",
                    help="permit a non-local target (only for models you own/are authorized to test)")
    dd.set_defaults(func=_cmd_defense_delta)

    sl = sub.add_parser("lexical-leak",
                        help="advisory lexical-overlap paraphrased-leak band over an artifact "
                             "(NOT embeddings; SECONDARY to the deterministic canary oracle — see README)")
    sl.add_argument("artifacts", nargs="+", help="one or more run artifact JSONL files")
    sl.add_argument("--elevated", type=float, default=0.30, help="elevated-band threshold (default 0.30)")
    sl.add_argument("--high", type=float, default=0.50,
                    help="high-band / paraphrase-suspect threshold (default 0.50; one-model heuristic)")
    sl.set_defaults(func=_cmd_lexical_leak)

    rg = sub.add_parser("regrade", help="re-score an artifact file with the Claude rubric judge")
    rg.add_argument("artifact", help="path to a reports/artifacts/*.jsonl file")
    rg.add_argument("--judge-model", default="claude-haiku-4-5-20251001", dest="judge_model")
    rg.set_defaults(func=_cmd_regrade)

    ar = sub.add_parser("agent-run",
                        help="red-team a tool-calling AGENT with indirect injection (behavioral bypass)")
    ar.add_argument("--target", default="ollama", choices=available_targets(),
                    help="agent backend (agent-run supports ollama today)")
    ar.add_argument("--model", default=DEFAULT_MODEL, help="model tag driving the agent")
    ar.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="trials per scenario")
    ar.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ar.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    ar.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                    help="tool-loop step budget per scenario")
    ar.add_argument("--smoke", action="store_true", help="1 scenario x 1 trial fast proof")
    ar.set_defaults(func=_cmd_agent_run)

    asc = sub.add_parser("agent-scenarios", help="show the loaded agentic-injection scenarios")
    asc.set_defaults(func=_cmd_agent_scenarios)

    ad = sub.add_parser("adaptive-run",
                        help="adaptive dialogue-level attacker — target-adaptive multi-turn "
                             "search that picks its next move from the target's last refusal")
    ad.add_argument("--target", default="ollama", choices=available_targets(),
                    help="target backend (default ollama)")
    ad.add_argument("--model", default=DEFAULT_MODEL, help="target model tag")
    ad.add_argument("--attacker", default="deterministic", choices=("deterministic", "llm"),
                    help="deterministic seeded selector (reproducible) or an LLM attacker "
                         "writing each turn (CoP/AJAR; NONDETERMINISTIC)")
    ad.add_argument("--attacker-model", default=DEFAULT_MODEL, dest="attacker_model",
                    help="local Ollama model driving the --attacker llm arm (always local Ollama, "
                         "regardless of --target — keeps the attacker free and off any API quota)")
    ad.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="conversations per objective")
    ad.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ad.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    ad.add_argument("--max-turns", type=int, default=DEFAULT_ADAPTIVE_TURNS, dest="max_turns",
                    help="hard cap on turns per conversation (anti-runaway)")
    ad.add_argument("--smoke", action="store_true", help="1 objective-set x 1 trial fast proof")
    ad.add_argument("--authorized", action="store_true",
                    help="permit a non-local target (only for models you own/are authorized to test)")
    ad.set_defaults(func=_cmd_adaptive_run)

    st = sub.add_parser("strategies", help="show the adaptive attacker's strategy library")
    st.set_defaults(func=_cmd_strategies)

    ta = sub.add_parser("tool-abuse-run",
                        help="red-team a tool-calling agent for RCE/SSRF via a SANDBOXED dangerous-"
                             "tool surface (fake run_shell/fetch_url — never a real process/socket)")
    ta.add_argument("--target", default="ollama", choices=available_targets(),
                    help="agent backend (tool-abuse-run supports ollama today)")
    ta.add_argument("--model", default=DEFAULT_MODEL, help="model tag driving the agent")
    ta.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="trials per scenario")
    ta.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ta.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    ta.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                    help="tool-loop step budget per scenario")
    ta.add_argument("--smoke", action="store_true", help="1 scenario x 1 trial fast proof")
    ta.set_defaults(func=_cmd_toolabuse_run)

    tas = sub.add_parser("toolabuse-scenarios", help="show the loaded tool-abuse (RCE/SSRF) scenarios")
    tas.set_defaults(func=_cmd_toolabuse_scenarios)

    pv = sub.add_parser("privilege-run",
                        help="red-team a tool-calling agent for EXCESSIVE AGENCY / confused deputy "
                             "(LLM06/ASI03) via a SANDBOXED privileged-tool surface (fake "
                             "delete_record/grant_role — never a real state change)")
    pv.add_argument("--target", default="ollama", choices=available_targets(),
                    help="agent backend (privilege-run supports ollama today)")
    pv.add_argument("--model", default=DEFAULT_MODEL, help="model tag driving the agent")
    pv.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="trials per scenario")
    pv.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    pv.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    pv.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                    help="tool-loop step budget per scenario")
    pv.add_argument("--smoke", action="store_true", help="1 scenario x 1 trial fast proof")
    pv.set_defaults(func=_cmd_privilege_run)

    pvs = sub.add_parser("privilege-scenarios",
                         help="show the loaded excessive-agency (confused-deputy) scenarios")
    pvs.set_defaults(func=_cmd_privilege_scenarios)

    dr = sub.add_parser("disclosure-run",
                        help="red-team a tool-calling agent for SENSITIVE-INFO DISCLOSURE (LLM02) — "
                             "cross-subject leak into the answer or an external send, via a "
                             "SANDBOXED confidential store + comms sink")
    dr.add_argument("--target", default="ollama", choices=available_targets(),
                    help="agent backend (disclosure-run supports ollama today)")
    dr.add_argument("--model", default=DEFAULT_MODEL, help="model tag driving the agent")
    dr.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="trials per scenario")
    dr.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    dr.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    dr.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                    help="tool-loop step budget per scenario")
    dr.add_argument("--smoke", action="store_true", help="1 scenario x 1 trial fast proof")
    dr.set_defaults(func=_cmd_disclosure_run)

    drs = sub.add_parser("disclosure-scenarios",
                         help="show the loaded sensitive-info-disclosure (LLM02) scenarios")
    drs.set_defaults(func=_cmd_disclosure_scenarios)

    mi = sub.add_parser("misinfo-run",
                        help="red-team a tool-calling agent for MISINFORMATION (LLM09) — it is asked "
                             "for a structured identifier (CVE/RFC/DOI) about a FABRICATED subject; "
                             "asserting one is confirmed fabrication (deterministic, no judge)")
    mi.add_argument("--target", default="ollama", choices=available_targets(),
                    help="agent backend (misinfo-run supports ollama today)")
    mi.add_argument("--model", default=DEFAULT_MODEL, help="model tag driving the agent")
    mi.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="trials per scenario")
    mi.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    mi.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    mi.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                    help="tool-loop step budget per scenario")
    mi.add_argument("--smoke", action="store_true", help="1 scenario x 1 trial fast proof")
    mi.set_defaults(func=_cmd_misinfo_run)

    mis = sub.add_parser("misinfo-scenarios",
                         help="show the loaded misinformation (LLM09) scenarios")
    mis.set_defaults(func=_cmd_misinfo_scenarios)

    mr = sub.add_parser("memory-run",
                        help="red-team a tool-calling agent for MEMORY/CONTEXT POISONING (ASI06) — "
                             "a poisoned doc plants a note that fires at a later step via recall")
    mr.add_argument("--target", default="ollama", choices=available_targets(),
                    help="agent backend (memory-run supports ollama today)")
    mr.add_argument("--model", default=DEFAULT_MODEL, help="model tag driving the agent")
    mr.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="trials per scenario")
    mr.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    mr.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    mr.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                    help="tool-loop step budget per scenario")
    mr.add_argument("--smoke", action="store_true", help="1 scenario x 1 trial fast proof")
    mr.set_defaults(func=_cmd_memory_run)

    ms = sub.add_parser("memory-scenarios", help="show the loaded memory-poisoning (ASI06) scenarios")
    ms.set_defaults(func=_cmd_memory_scenarios)

    rr = sub.add_parser("rag-run",
                        help="red-team a tool-calling agent for RAG RETRIEVAL POISONING — a poisoned "
                             "knowledge-base passage, surfaced by a benign query, drives an action")
    rr.add_argument("--target", default="ollama", choices=available_targets(),
                    help="agent backend (rag-run supports ollama today)")
    rr.add_argument("--model", default=DEFAULT_MODEL, help="model tag driving the agent")
    rr.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="trials per scenario")
    rr.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    rr.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    rr.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                    help="tool-loop step budget per scenario")
    rr.add_argument("--smoke", action="store_true", help="1 scenario x 1 trial fast proof")
    rr.set_defaults(func=_cmd_rag_run)

    rs = sub.add_parser("rag-scenarios", help="show the loaded RAG retrieval-poisoning scenarios")
    rs.set_defaults(func=_cmd_rag_scenarios)

    aa = sub.add_parser("a2a-run",
                        help="red-team a tool-calling agent for INSECURE INTER-AGENT COMMS (ASI07) — "
                             "a poisoned message from a rogue peer agent drives an action")
    aa.add_argument("--target", default="ollama", choices=available_targets(),
                    help="agent backend (a2a-run supports ollama today)")
    aa.add_argument("--model", default=DEFAULT_MODEL, help="model tag driving the agent")
    aa.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS, help="trials per scenario")
    aa.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    aa.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    aa.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                    help="tool-loop step budget per scenario")
    aa.add_argument("--smoke", action="store_true", help="1 scenario x 1 trial fast proof")
    aa.set_defaults(func=_cmd_a2a_run)

    aas = sub.add_parser("a2a-scenarios", help="show the loaded inter-agent (ASI07) scenarios")
    aas.set_defaults(func=_cmd_a2a_scenarios)

    return p


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from the project .env into the environment (no override)."""
    import os
    from .config import PROJECT_ROOT

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
