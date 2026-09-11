"""ISC-31 — no command exits 0 having measured nothing.

The class: a run whose every trial errored (dead guard backend, daemon down), an empty
artifact, or a matrix in which no technique/objective pair fires used to write a report that
read "the guardrails held" and return 0. These pin the loud path.
"""

import types

import pytest

from iago.attacks import Technique
from iago.guards_thirdparty import GuardBackendUnavailable
from iago.objectives import Objective
from iago.report import build_report, ci_str
from iago.runner import load_artifacts, run
from iago.target import Target


class _Target(Target):
    def __init__(self, exc=None, reply="A" * 300):
        self.name = "fake:model"
        self.is_local = True
        self.exc, self.reply = exc, reply

    def generate(self, prompt, system=None, options=None):
        if self.exc:
            raise self.exc
        return self.reply

    def chat(self, messages, options=None):
        return self.generate(None)


FORBIDDEN = [Objective("o1", "forbidden thing", "forbidden")]


def test_ci_str_with_no_data_is_na():
    assert ci_str(0, 0) == "n/a"


def test_run_with_zero_fireable_pairs_raises_before_writing(tmp_path):
    leak_only = [Technique("pe-x", "Extract", "prompt-extraction", "d", "Leak: {objective}",
                           applies_to=("prompt-leak",))]
    with pytest.raises(ValueError, match="no technique/objective pairs fire"):
        run(_Target(), techniques=leak_only, objectives=FORBIDDEN, trials=1,
            artifacts_dir=tmp_path)
    assert list(tmp_path.glob("*.jsonl")) == []


def test_all_error_run_report_says_nothing_measured_not_held(tmp_path):
    techs = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
    path = run(_Target(exc=RuntimeError("daemon down")), techniques=techs,
               objectives=FORBIDDEN, trials=2, artifacts_dir=tmp_path)
    rows = load_artifacts(path)
    assert rows and all("<<RUN-ERROR" in r["response"] for r in rows)
    text = build_report(rows)
    assert "NOT a hold" in text
    assert "guardrails held across all harmful" not in text
    assert "0%–0%" not in text


def test_guard_backend_unavailable_fails_the_run_not_a_row(tmp_path):
    techs = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
    with pytest.raises(GuardBackendUnavailable):
        run(_Target(exc=GuardBackendUnavailable("pull llama-guard3")), techniques=techs,
            objectives=FORBIDDEN, trials=1, artifacts_dir=tmp_path)


def test_build_thirdparty_guard_probes_backend_at_build(monkeypatch):
    from iago import guards_thirdparty as gt

    class Dead:
        name, stage = "dead", "input"

        def inspect(self, text, *, system=None):
            raise GuardBackendUnavailable("not installed")

    monkeypatch.setitem(gt.THIRDPARTY_GUARD_BUILDERS, "dead", Dead)
    with pytest.raises(GuardBackendUnavailable):
        gt.build_thirdparty_guard("dead")
    assert isinstance(gt.build_thirdparty_guard("dead", probe=False), Dead)


def test_cmd_report_empty_artifact_exits_2(tmp_path, capsys):
    from iago.cli import _cmd_report

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    rc = _cmd_report(types.SimpleNamespace(artifact=str(empty), log=False, html=False))
    assert rc == 2
    assert "nothing was measured" in capsys.readouterr().err


def test_cmd_report_all_error_artifact_exits_1(tmp_path, monkeypatch, capsys):
    from iago import cli
    from iago import report as report_mod

    techs = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
    path = run(_Target(exc=RuntimeError("down")), techniques=techs, objectives=FORBIDDEN,
               trials=1, artifacts_dir=tmp_path)
    monkeypatch.setattr(cli, "write_report", lambda rows: tmp_path / "r.md")
    rc = cli._cmd_report(types.SimpleNamespace(artifact=str(path), log=False, html=False))
    assert rc == 1
    assert "NOT a hold" in capsys.readouterr().err


def test_cmd_compare_with_no_scenarios_exits_2(tmp_path, monkeypatch, capsys):
    from iago import cli, compare as compare_mod

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("{}\n"); b.write_text("{}\n")
    fake = types.SimpleNamespace(models=[types.SimpleNamespace(model="x"),
                                         types.SimpleNamespace(model="y")], scenario_ids=[])
    monkeypatch.setattr(compare_mod, "build_comparison", lambda paths, **kw: fake)
    monkeypatch.setattr(compare_mod, "write_comparison_report",
                        lambda comp: pytest.fail("must not write a report for an empty matrix"))
    rc = cli._cmd_compare(types.SimpleNamespace(artifacts=[str(a), str(b)]))
    assert rc == 2
    assert "nothing to compare" in capsys.readouterr().err


def test_cmd_campaign_with_failed_legs_exits_1(tmp_path, monkeypatch, capsys):
    from iago import campaign as campaign_mod
    from iago.cli import _cmd_campaign

    monkeypatch.setattr(campaign_mod, "run_campaign",
                        lambda s, m, **kw: ({"privilege": [tmp_path / "a.jsonl"]}, ["leg died"]))
    monkeypatch.setattr(campaign_mod, "build_campaign", lambda *a, **kw: object())
    monkeypatch.setattr(campaign_mod, "write_campaign_report", lambda c: tmp_path / "r.md")
    args = types.SimpleNamespace(models="llama3.1,llama3.2", surfaces="privilege", smoke=False,
                                 trials=1, temperature=None, base_seed=None, max_steps=None)
    assert _cmd_campaign(args) == 1
    assert "PARTIAL" in capsys.readouterr().err
