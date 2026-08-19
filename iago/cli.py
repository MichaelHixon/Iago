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
from .config import (
    BASE_SEED,
    DEFAULT_ADAPTIVE_TURNS,
    DEFAULT_AGENT_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRIALS,
)
from .objectives import load_objectives
from .report import write_html_report, write_log, write_report
from .runner import AuthorizationError, load_artifacts, run
from .target import available_targets, build_target
from .guards import GuardedTarget, available_guards, build_guards


def _cmd_run(args: argparse.Namespace) -> int:
    # If the target isn't ollama and the user didn't override --model, let the
    # backend pick its own default (don't hand a llama tag to Anthropic).
    model = None if (args.target != "ollama" and args.model == DEFAULT_MODEL) else args.model
    try:
        target = build_target(args.target, model=model)
        if getattr(args, "guard", None):
            target = GuardedTarget(target, build_guards(args.guard))
    except ValueError as exc:
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
        )
    except AuthorizationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    rows = load_artifacts(artifact_path)
    report_path = write_report(rows)
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    if getattr(args, "html", False):
        print(f"HTML:      {write_html_report(rows)}")
    if getattr(args, "log", False):
        print(f"Transcript: {write_log(rows, html=getattr(args, 'html', False))}")
    print(f"({len(rows)} trials recorded)")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """(Re)generate a report from an artifact file — summary, --html, and/or full --log."""
    from pathlib import Path

    path = Path(args.artifact)
    if not path.exists():
        print(f"ERROR: artifact not found: {path}", file=sys.stderr)
        return 2
    rows = load_artifacts(path)
    if args.log:
        print(f"Transcript: {write_log(rows, html=args.html)}  ({len(rows)} trials)")
    else:
        out = write_html_report(rows) if args.html else write_report(rows)
        print(f"Report: {out}  ({len(rows)} trials)")
    return 0


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
    out = write_delta_report(raw_rows, guarded_rows)
    print(f"Delta report: {out}")
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
    out = write_compose_report(rows)
    print(f"Composition-lift report: {out}")
    return 0


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
    judge = ClaudeJudge(model=args.judge_model)
    print(f"Regrading {path.name} with Claude judge ({args.judge_model})...")
    try:
        summary = regrade_file(path, judge)
    except Exception as exc:
        print(f"ERROR: regrade failed (is ANTHROPIC_API_KEY set and the 'anthropic' SDK installed?): {exc}",
              file=sys.stderr)
        return 1
    report_path = write_report(load_artifacts(path))
    print(f"  regraded {summary['regraded']} rows; {summary['flipped_vs_heuristic']} flipped vs heuristic")
    print(f"Report:    {report_path}")
    return 0


def _cmd_lexical_leak(args: argparse.Namespace) -> int:
    """Advisory lexical-overlap paraphrased-leak band — SECONDARY to the canary oracle."""
    from pathlib import Path
    from .lexical_leak import analyze_file, PROVENANCE

    any_found = False
    for spec in args.artifacts:
        path = Path(spec)
        if not path.exists():
            print(f"ERROR: artifact not found: {path}", file=sys.stderr)
            return 2
        summary = analyze_file(path, elevated=args.elevated, high=args.high)
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
    if not any_found:
        print("\nNo paraphrase suspects — no row disclosed prompt content the canary oracle missed.")
    return 0  # always 0: an ADVISORY band must never gate a pipeline; the canary oracle owns pass/fail


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
    return 0


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
    print(f"\nArtifacts: {artifact_path}")
    print(f"Report:    {report_path}")
    print(f"({len(rows)} conversations recorded)")
    if not deterministic:
        print("NOTE: LLM-attacker turns are nondeterministic — this run does not replay bit-for-bit.")
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
    return 0


def _cmd_toolabuse_scenarios(_args: argparse.Namespace) -> int:
    from .agent_toolabuse import load_toolabuse_scenarios

    scens = load_toolabuse_scenarios()
    print(f"Tool-abuse scenarios: {len(scens)}")
    for s in scens:
        print(f"  {s.id:24} [{s.kind:10}] [{s.arm}] {s.name}")
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
    return 0


def _cmd_memory_scenarios(_args: argparse.Namespace) -> int:
    from .agent_memory import load_memory_scenarios

    scens = load_memory_scenarios()
    print(f"Memory-poisoning scenarios: {len(scens)}")
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
          f"{sum(1 for o in objs if o.is_prompt_leak)} prompt-leak)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iago", description="Authorized LLM guardrail red-team harness.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the attack matrix and write a report")
    r.add_argument("--target", default="ollama", choices=available_targets(),
                   help="target backend (default: ollama)")
    r.add_argument("--model", default=DEFAULT_MODEL, help="model tag within the target backend")
    r.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per (technique, objective)")
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
    dd.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per (technique, objective)")
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
    ar.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per scenario")
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
    ad.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="conversations per objective")
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
    ta.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per scenario")
    ta.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ta.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    ta.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                    help="tool-loop step budget per scenario")
    ta.add_argument("--smoke", action="store_true", help="1 scenario x 1 trial fast proof")
    ta.set_defaults(func=_cmd_toolabuse_run)

    tas = sub.add_parser("toolabuse-scenarios", help="show the loaded tool-abuse (RCE/SSRF) scenarios")
    tas.set_defaults(func=_cmd_toolabuse_scenarios)

    mr = sub.add_parser("memory-run",
                        help="red-team a tool-calling agent for MEMORY/CONTEXT POISONING (ASI06) — "
                             "a poisoned doc plants a note that fires at a later step via recall")
    mr.add_argument("--target", default="ollama", choices=available_targets(),
                    help="agent backend (memory-run supports ollama today)")
    mr.add_argument("--model", default=DEFAULT_MODEL, help="model tag driving the agent")
    mr.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per scenario")
    mr.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    mr.add_argument("--base-seed", type=int, default=BASE_SEED, dest="base_seed")
    mr.add_argument("--max-steps", type=int, default=DEFAULT_AGENT_STEPS, dest="max_steps",
                    help="tool-loop step budget per scenario")
    mr.add_argument("--smoke", action="store_true", help="1 scenario x 1 trial fast proof")
    mr.set_defaults(func=_cmd_memory_run)

    ms = sub.add_parser("memory-scenarios", help="show the loaded memory-poisoning (ASI06) scenarios")
    ms.set_defaults(func=_cmd_memory_scenarios)

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
