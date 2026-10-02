"""Pin what every agentic `*-run` / `*-scenarios` subcommand prints and returns (ISC-82).

Written against the eight per-surface handlers BEFORE they collapsed onto one registry-driven
handler, so the same assertions hold on both sides of the refactor. Never touches Ollama: each
surface module's loader, suite runner, artifact loader, report writer and `ollama_chat_fn` are
monkeypatched at their MODULE attribute, which is where both the old handlers and the registry's
lazy loaders resolve them at call time.

Expected console text lives in `golden/cli_surface_stdout.json`, captured from the pre-refactor
handlers. Regenerate only for an intended output change:
    PYTHONDONTWRITEBYTECODE=1 uv run python -m tests.test_cli_surface_stdout
"""

from __future__ import annotations

import importlib
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from iago import cli

GOLDEN = Path(__file__).parent / "golden" / "cli_surface_stdout.json"

# surface -> (run command, scenarios command, scenarios module, loader attr,
#             suite module, run_suite attr, load_artifacts attr, write_report attr)
SURFACES = {
    "agent": ("agent-run", "agent-scenarios", "iago.agent_scenarios", "load_scenarios",
              "iago.agent_run", "run_agent_suite", "load_agent_artifacts", "write_agent_report"),
    "toolabuse": ("tool-abuse-run", "toolabuse-scenarios", "iago.agent_toolabuse",
                  "load_toolabuse_scenarios", "iago.agent_toolabuse", "run_toolabuse_suite",
                  "load_toolabuse_artifacts", "write_toolabuse_report"),
    "privilege": ("privilege-run", "privilege-scenarios", "iago.agent_privilege",
                  "load_privilege_scenarios", "iago.agent_privilege", "run_privilege_suite",
                  "load_privilege_artifacts", "write_privilege_report"),
    "disclosure": ("disclosure-run", "disclosure-scenarios", "iago.agent_disclosure",
                   "load_disclosure_scenarios", "iago.agent_disclosure", "run_disclosure_suite",
                   "load_disclosure_artifacts", "write_disclosure_report"),
    "misinfo": ("misinfo-run", "misinfo-scenarios", "iago.agent_misinfo", "load_misinfo_scenarios",
                "iago.agent_misinfo", "run_misinfo_suite", "load_misinfo_artifacts",
                "write_misinfo_report"),
    "memory": ("memory-run", "memory-scenarios", "iago.agent_memory", "load_memory_scenarios",
               "iago.agent_memory", "run_memory_suite", "load_memory_artifacts", "write_memory_report"),
    "rag": ("rag-run", "rag-scenarios", "iago.agent_rag", "load_rag_scenarios",
            "iago.agent_rag", "run_rag_suite", "load_rag_artifacts", "write_rag_report"),
    "a2a": ("a2a-run", "a2a-scenarios", "iago.agent_a2a", "load_a2a_scenarios",
            "iago.agent_a2a", "run_a2a_suite", "load_a2a_artifacts", "write_a2a_report"),
}

ARTIFACT = Path("/fake/artifacts/run.jsonl")
REPORT = Path("/fake/reports/run.md")
VALID_ROW = {"verdict": "RESISTED", "objective_kind": "attack"}


def _scen(sid: str, kind: str, *, arm: str = "direct", fabricated: bool = True) -> SimpleNamespace:
    return SimpleNamespace(id=sid, kind=kind, name=f"{sid} name", arm=arm, is_fabricated=fabricated,
                           is_control=kind == "control", is_capability=kind == "capability")


def fake_scenarios() -> list[SimpleNamespace]:
    """First scenario is an attack (as in every shipped library), so a bare `[:1]` smoke slice
    would drop the capability floor."""
    return [_scen("atk-one", "attack"), _scen("atk-two", "attack", arm="channel"),
            _scen("cap-one", "capability", fabricated=False), _scen("ctl-one", "control")]


class Harness:
    """Monkeypatches one surface's module entry points and records what the CLI handed them."""

    def __init__(self, monkeypatch, surface: str, *, rows=None, raise_in_run: Exception | None = None):
        _, _, scen_mod, scen_attr, suite_mod, run_attr, load_attr, write_attr = SURFACES[surface]
        self.calls: list[dict] = []
        self.scenarios = fake_scenarios()
        rows = [VALID_ROW] if rows is None else rows

        def run_suite(chat_fn, **kw):
            self.calls.append({"chat_fn": chat_fn, **kw})
            if raise_in_run is not None:
                raise raise_in_run
            return ARTIFACT

        monkeypatch.setattr(importlib.import_module(scen_mod), scen_attr, lambda: list(self.scenarios))
        sm = importlib.import_module(suite_mod)
        monkeypatch.setattr(sm, run_attr, run_suite)
        monkeypatch.setattr(sm, load_attr, lambda path: list(rows))
        monkeypatch.setattr(sm, write_attr, lambda r: REPORT)
        monkeypatch.setattr(importlib.import_module("iago.agent_run"), "ollama_chat_fn",
                            lambda model: f"chat:{model}")


def invoke(argv: list[str]) -> tuple[int, str, str]:
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        args = cli.build_parser().parse_args(argv)
        rc = args.func(args)
    return rc, out.getvalue(), err.getvalue()


RUN_ARGV = ["--model", "m1", "--trials", "3", "--max-steps", "5", "--temperature", "0.25",
            "--base-seed", "7"]


def capture(monkeypatch) -> dict:
    """Every pinned case, keyed `<surface>/<case>` -> {rc, stdout, stderr}."""
    out: dict = {}
    for surface, (run_cmd, scen_cmd, *_rest) in SURFACES.items():
        Harness(monkeypatch, surface)
        rc, so, se = invoke([run_cmd, *RUN_ARGV])
        out[f"{surface}/run"] = {"rc": rc, "stdout": so, "stderr": se}
        rc, so, se = invoke([run_cmd, "--target", "anthropic"])
        out[f"{surface}/wrong-target"] = {"rc": rc, "stdout": so, "stderr": se}
        Harness(monkeypatch, surface, raise_in_run=RuntimeError("boom"))
        rc, so, se = invoke([run_cmd, *RUN_ARGV])
        out[f"{surface}/suite-raises"] = {"rc": rc, "stdout": so, "stderr": se}
        Harness(monkeypatch, surface, rows=[])
        rc, so, se = invoke([run_cmd, *RUN_ARGV])
        out[f"{surface}/nothing-measured"] = {"rc": rc, "stdout": so, "stderr": se}
        Harness(monkeypatch, surface)
        rc, so, se = invoke([scen_cmd])
        out[f"{surface}/scenarios"] = {"rc": rc, "stdout": so, "stderr": se}
    return out


def test_every_surface_prints_exactly_what_it_did_before(monkeypatch):
    golden = json.loads(GOLDEN.read_text())
    got = capture(monkeypatch)
    assert sorted(got) == sorted(golden)
    for case in golden:
        assert got[case] == golden[case], f"{case} console output or exit code changed"


@pytest.mark.parametrize("surface", list(SURFACES))
def test_run_hands_the_suite_every_flag(monkeypatch, surface):
    h = Harness(monkeypatch, surface)
    rc, _, _ = invoke([SURFACES[surface][0], *RUN_ARGV])
    assert rc == 0
    assert h.calls == [{"chat_fn": "chat:m1", "model_name": "ollama:m1", "trials": 3,
                        "temperature": 0.25, "base_seed": 7, "max_steps": 5,
                        "scenarios": h.scenarios, "progress": True}]


@pytest.mark.parametrize("surface", list(SURFACES))
def test_smoke_forces_one_trial(monkeypatch, surface):
    h = Harness(monkeypatch, surface)
    rc, _, _ = invoke([SURFACES[surface][0], *RUN_ARGV, "--smoke"])
    assert rc == 0 and h.calls[0]["trials"] == 1


if __name__ == "__main__":  # pragma: no cover — regeneration entry point
    mp = pytest.MonkeyPatch()
    try:
        data = capture(mp)
    finally:
        mp.undo()
    GOLDEN.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {GOLDEN}", file=sys.stderr)
