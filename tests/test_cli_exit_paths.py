"""CLI exit-code coverage for the paths a mutation run found unguarded.

Deleting the nothing-measured check from `_cmd_run` — the command ISC-31 is named for — left the
suite green, as did removing both `require_surface` calls from the delta report and the new
regrade / adaptive / judge-eval branches. These drive the handlers directly.
"""

import json
import types

import pytest

from iago import cli
from iago.attacks import Technique
from iago.judge import REFUSED
from iago.objectives import Objective
from iago.target import Target


class _T(Target):
    name, is_local = "fake:model", True

    def __init__(self, exc=None):
        self.exc = exc

    def generate(self, prompt, system=None, options=None):
        if self.exc:
            raise self.exc
        return "A" * 300

    def chat(self, messages, options=None):
        return self.generate(None)


def _run_args(**kw):
    base = dict(target="ollama", model="llama3.1", trials=1, temperature=0.8, base_seed=1,
                limit_techniques=None, limit_objectives=None, category=None, shots=None, guard=None, smoke=True,
                html=False, log=False, authorized=False, determinism_check=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_cmd_run_exits_nonzero_when_every_trial_errored(tmp_path, monkeypatch, capsys):
    """The command ISC-31 is named for: deleting its gate previously left the suite green."""
    techs = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
    objs = [Objective("o1", "forbidden thing", "forbidden")]

    def fake_run(target, **kwargs):
        from iago.runner import run as real_run
        return real_run(_T(exc=RuntimeError("daemon down")), techniques=techs, objectives=objs,
                        trials=1, artifacts_dir=tmp_path, determinism_check=False)

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "build_target", lambda *a, **k: _T())
    monkeypatch.setattr(cli, "write_report", lambda rows, **kw: tmp_path / "r.md")
    rc = cli._cmd_run(_run_args())
    assert rc == 1
    assert "NOT a hold" in capsys.readouterr().err


def test_cmd_run_returns_zero_on_a_measured_run(tmp_path, monkeypatch):
    techs = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
    objs = [Objective("o1", "forbidden thing", "forbidden")]

    def fake_run(target, **kwargs):
        from iago.runner import run as real_run
        return real_run(_T(), techniques=techs, objectives=objs, trials=1,
                        artifacts_dir=tmp_path, determinism_check=False)

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "build_target", lambda *a, **k: _T())
    monkeypatch.setattr(cli, "write_report", lambda rows, **kw: tmp_path / "r.md")
    assert cli._cmd_run(_run_args()) == 0


def test_trials_below_one_is_refused_by_the_parser():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--trials", "0"])
    assert parser.parse_args(["run", "--trials", "1"]).trials == 1


def test_cmd_delta_refuses_an_agent_artifact(tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    row = {"kind": "attack", "scenario_id": "s", "verdict": "resisted", "model": "m"}
    for p in (a, b):
        p.write_text(json.dumps(row) + "\n")
    rc = cli._cmd_delta(types.SimpleNamespace(raw=str(a), guarded=str(b)))
    assert rc == 2
    assert "reads chatbot artifacts" in capsys.readouterr().err


def _chat_row(**kw):
    base = dict(technique_id="t1", technique_name="Direct", category="direct-ask", objective_id="o1",
                objective_kind="forbidden", model="m", seed=1, temperature=0.8, trial=0, prompt="p",
                response="I can't help.", verdict=REFUSED, confidence=0.9, rationale="r",
                latency_s=0.1, timestamp="t")
    base.update(kw)
    return base


def test_cmd_regrade_exits_nonzero_when_nothing_was_regraded(tmp_path, monkeypatch, capsys):
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps(_chat_row(objective_id="does-not-exist")) + "\n")

    class _Judge:
        judge_id = "stub"

        def judge(self, *a, **k):
            raise AssertionError("must not be called for an unknown objective")

    monkeypatch.setattr("iago.judge_claude.ClaudeJudge", lambda **k: _Judge())
    monkeypatch.setattr(cli, "write_report", lambda rows, **kw: tmp_path / "r.md")
    rc = cli._cmd_regrade(types.SimpleNamespace(artifact=str(p), judge_model="m"))
    assert rc == 1
    assert "0 rows were regraded" in capsys.readouterr().err


def test_cmd_adaptive_run_exits_nonzero_when_every_conversation_errored(tmp_path, monkeypatch, capsys):
    rows = [{"objective_id": "o1", "objective_kind": "forbidden", "outcome": "error", "trial": 0,
             "turns_used": 1, "trace": [], "model": "m", "surface": "adaptive"}]
    art = tmp_path / "a.jsonl"
    art.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(cli, "build_target", lambda *a, **k: _T())
    monkeypatch.setattr("iago.adaptive.run_adaptive_suite", lambda *a, **k: art)
    monkeypatch.setattr("iago.adaptive.load_adaptive_artifacts", lambda p: rows)
    monkeypatch.setattr("iago.adaptive.write_adaptive_report", lambda r: tmp_path / "r.md")
    args = types.SimpleNamespace(target="ollama", model="llama3.1", trials=1, temperature=0.8,
                                 base_seed=1, max_turns=2, smoke=True, attacker="deterministic",
                                 attacker_model=None, authorized=False)
    rc = cli._cmd_adaptive_run(args)
    assert rc == 1
    assert "nothing was measured" in capsys.readouterr().err.lower()


def test_judge_eval_cli_runs_offline_and_reports(tmp_path, capsys):
    args = types.SimpleNamespace(judge="heuristic,canary", set=None, judge_model=None, write=False,
                                 show_disagreements=False, no_overlay=True)
    assert cli._cmd_judge_eval(args) == 0
    out = capsys.readouterr().out
    assert "heuristic judge" in out and "canary judge" in out
    assert "false-positive rate n/a" in out          # the unreachable-positive judge
    assert "set=public" in out


def test_judge_eval_cli_reports_a_failing_judge_without_crashing(capsys):
    args = types.SimpleNamespace(judge="claude", set=None, judge_model=None, write=False,
                                 show_disagreements=False, no_overlay=True)

    import iago.judge_claude as jc

    class _Boom:
        def __init__(self, **k):
            raise RuntimeError("no API key")

    original = jc.ClaudeJudge
    jc.ClaudeJudge = _Boom
    try:
        rc = cli._cmd_judge_eval(args)
    finally:
        jc.ClaudeJudge = original
    assert rc == 1
    assert "claude judge evaluation failed" in capsys.readouterr().err
