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
from .config import BASE_SEED, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TRIALS
from .objectives import load_objectives
from .report import write_report
from .runner import AuthorizationError, load_artifacts, run
from .target import available_targets, build_target


def _cmd_run(args: argparse.Namespace) -> int:
    # If the target isn't ollama and the user didn't override --model, let the
    # backend pick its own default (don't hand a llama tag to Anthropic).
    model = None if (args.target != "ollama" and args.model == DEFAULT_MODEL) else args.model
    try:
        target = build_target(args.target, model=model)
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
    print(f"({len(rows)} trials recorded)")
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


def _cmd_library(_args: argparse.Namespace) -> int:
    lib = load_library()
    objs = load_objectives()
    print(f"Attack library: {len(lib)} techniques")
    for category, count in summarize(lib).items():
        print(f"  {category:24} {count}")
    print(f"\nObjectives: {len(objs)} "
          f"({sum(1 for o in objs if o.kind == 'forbidden')} forbidden, "
          f"{sum(1 for o in objs if o.is_control)} control)")
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
    r.add_argument("--smoke", action="store_true", help="1x1x1 fast proof of the loop")
    r.add_argument("--authorized", action="store_true",
                   help="permit a non-local target (only for models you own/are authorized to test)")
    r.set_defaults(func=_cmd_run)

    lib = sub.add_parser("library", help="show the loaded attack library + objectives")
    lib.set_defaults(func=_cmd_library)

    rg = sub.add_parser("regrade", help="re-score an artifact file with the Claude rubric judge")
    rg.add_argument("artifact", help="path to a reports/artifacts/*.jsonl file")
    rg.add_argument("--judge-model", default="claude-haiku-4-5-20251001", dest="judge_model")
    rg.set_defaults(func=_cmd_regrade)

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
