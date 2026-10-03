"""Pin the exact `--help` text of `iago` and every subcommand (ISC-82).

Captured from the code BEFORE the agentic `*-run` / `*-scenarios` handlers were collapsed onto one
surface registry: the registry builds those subparsers in a loop, so a help string, a flag, a
default, or the ORDER subcommands are listed in that drifts in the rewrite fails here.

Regenerate only for an intended CLI change:
    PYTHONDONTWRITEBYTECODE=1 uv run python -m tests.test_cli_help_pin
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from iago.cli import build_parser

GOLDEN = Path(__file__).parent / "golden" / "cli_help.json"
_ENV = {"COLUMNS": "100", "LINES": "50", "NO_COLOR": "1", "PYTHON_COLORS": "0"}


def help_texts() -> dict[str, str]:
    """`{"iago": top-level help, "<subcommand>": its help, ...}` in subcommand registration order."""
    saved = {k: os.environ.get(k) for k in _ENV}
    os.environ.update(_ENV)
    try:
        parser = build_parser()
        out = {"iago": parser.format_help()}
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        for name, sp in sub.choices.items():
            out[name] = sp.format_help()
        return out
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def changed_commands(got: dict[str, str], golden: dict[str, str]) -> list[str]:
    """Every command whose help differs from the golden, plus any added or removed, in one list:
    a bisect over a refactor that broke several commands needs all of them, not the first."""
    names = list(golden) + [n for n in got if n not in golden]
    return [f"`{'iago' if n == 'iago' else 'iago ' + n} --help` "
            + ("added" if n not in golden else "removed" if n not in got else "changed")
            for n in names if got.get(n) != golden.get(n)]


def test_help_text_of_every_command_is_unchanged():
    golden = json.loads(GOLDEN.read_text())
    got = help_texts()
    assert changed_commands(got, golden) == []
    assert list(got) == list(golden), "subcommand registration order changed"


def test_the_pin_names_every_changed_command_at_once():
    golden = {"iago": "top", "run": "r", "gate": "g", "old": "o"}
    got = {"iago": "TOP", "run": "R", "gate": "G", "new": "n"}
    assert changed_commands(got, golden) == [
        "`iago --help` changed", "`iago run --help` changed", "`iago gate --help` changed", "`iago old --help` removed",
        "`iago new --help` added"]


if __name__ == "__main__":  # pragma: no cover — regeneration entry point
    GOLDEN.write_text(json.dumps(help_texts(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {GOLDEN}")
