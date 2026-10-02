"""the runner executes the matrix, pins sampling, writes JSONL, guards auth."""

import json

import pytest

from iago.artifacts import read_artifact
from iago.attacks import Technique
from iago.objectives import Objective
from iago.runner import AuthorizationError, load_artifacts, run
from iago.target import Target


class FakeTarget(Target):
    """Records the options it was called with; returns a canned substantive reply."""

    def __init__(self, is_local=True, reply="A" * 300):
        self.name = "fake:model"
        self.is_local = is_local
        self.reply = reply
        self.calls = []

    def generate(self, prompt, system=None, options=None):
        self.calls.append({"prompt": prompt, "options": options})
        return self.reply

    def chat(self, messages, options=None):
        self.calls.append({"messages": list(messages), "options": options})
        return self.reply


TECHS = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
OBJS = [
    Objective("o1", "forbidden thing", "forbidden"),
    Objective("oc", "benign thing", "control"),
]


def test_multiturn_technique_runs_all_turns_and_judges_final(tmp_path):
    mt = Technique("mt1", "Crescendo", "multi-turn", "d",
                   "Final ask: {objective}", turns=("prime one", "prime two"))
    t = FakeTarget(reply="X" * 300)
    out = run(t, trials=1, artifacts_dir=tmp_path, techniques=[mt], objectives=OBJS[:1])
    row = load_artifacts(out)[0]
    # 3 user turns (2 priming + 1 final) => 3 chat calls, each with growing history.
    chat_calls = [c for c in t.calls if "messages" in c]
    assert len(chat_calls) == 3
    assert len(chat_calls[-1]["messages"]) == 5  # u,a,u,a,u
    assert row["turns"] == 3
    assert "[turn 3" in row["prompt"]  # transcript recorded


def test_singleshot_technique_uses_generate_not_chat(tmp_path):
    t = FakeTarget()
    run(t, trials=1, artifacts_dir=tmp_path, techniques=TECHS, objectives=OBJS[:1])
    assert all("prompt" in c for c in t.calls)  # generate() path, no chat()
    assert not any("messages" in c for c in t.calls)


def test_writes_one_row_per_trial(tmp_path):
    t = FakeTarget()
    out = run(t, trials=2, artifacts_dir=tmp_path, techniques=TECHS, objectives=OBJS)
    rows = load_artifacts(out)
    # 1 technique x 2 objectives x 2 trials = 4 rows
    assert len(rows) == 4
    assert all("verdict" in r for r in rows)


def test_seed_is_pinned_per_trial(tmp_path):
    t = FakeTarget()
    run(t, trials=3, base_seed=1000, artifacts_dir=tmp_path, determinism_check=False, techniques=TECHS, objectives=OBJS[:1])
    seeds = [c["options"]["seed"] for c in t.calls]
    assert seeds == [1000, 1001, 1002]  # base + trial index, reproducible


def test_objective_kind_recorded(tmp_path):
    out = run(FakeTarget(), trials=1, artifacts_dir=tmp_path, techniques=TECHS, objectives=OBJS)
    kinds = {r["objective_id"]: r["objective_kind"] for r in load_artifacts(out)}
    assert kinds == {"o1": "forbidden", "oc": "control"}


def test_non_local_target_refused_without_authorization(tmp_path):
    remote = FakeTarget(is_local=False)
    with pytest.raises(AuthorizationError):
        run(remote, artifacts_dir=tmp_path, techniques=TECHS, objectives=OBJS)


def test_non_local_target_allowed_with_authorization(tmp_path):
    remote = FakeTarget(is_local=False)
    out = run(remote, trials=1, authorized=True, artifacts_dir=tmp_path,
              techniques=TECHS, objectives=OBJS[:1])
    assert out.exists()


def test_generate_error_does_not_kill_run(tmp_path):
    class Boom(FakeTarget):
        def generate(self, prompt, system=None, options=None):
            raise RuntimeError("model exploded")

    out = run(Boom(), trials=1, artifacts_dir=tmp_path, techniques=TECHS, objectives=OBJS[:1])
    row = load_artifacts(out)[0]
    assert "RUN-ERROR" in row["response"]


def test_run_error_records_exception_class(tmp_path):
    # A harness defect (KeyError) must read differently from a backend failure (RuntimeError):
    # before the class name was recorded both collapsed to the bare message.
    class Defect(FakeTarget):
        def generate(self, prompt, system=None, options=None):
            raise KeyError("missing_field")

    out = run(Defect(), trials=1, artifacts_dir=tmp_path, techniques=TECHS, objectives=OBJS[:1],
              determinism_check=False)
    row = load_artifacts(out)[0]
    assert row["response"].startswith("<<RUN-ERROR: KeyError: ")
    assert row["verdict"] == "error"


def test_manifest_names_controls_the_backend_drops(tmp_path, capsys):
    class NoSeed(FakeTarget):
        supported_options = frozenset({"temperature", "num_predict"})

    out = run(NoSeed(), trials=1, artifacts_dir=tmp_path, techniques=TECHS, objectives=OBJS[:1],
              determinism_check=False)
    manifest, _ = read_artifact(out)
    assert manifest["sampling"]["unsupported_controls"] == ["seed"]
    assert "ignores seed" in capsys.readouterr().err


def test_manifest_records_no_gap_for_pass_through_backend(tmp_path, capsys):
    out = run(FakeTarget(), trials=1, artifacts_dir=tmp_path, techniques=TECHS,
              objectives=OBJS[:1], determinism_check=False)
    manifest, _ = read_artifact(out)
    assert manifest["sampling"]["unsupported_controls"] == []
    assert "ignores" not in capsys.readouterr().err


def test_guarded_target_keeps_the_inner_backends_gap(tmp_path):
    # A wrapper that defaulted supported_options to pass-through hid the dropped seed.
    from iago.guards import GuardedTarget

    class NoSeed(FakeTarget):
        supported_options = frozenset({"temperature"})

    out = run(GuardedTarget(NoSeed(), []), trials=1, artifacts_dir=tmp_path, techniques=TECHS,
              objectives=OBJS[:1], determinism_check=False)
    manifest, _ = read_artifact(out)
    assert manifest["sampling"]["unsupported_controls"] == ["seed"]


def test_planted_history_run_error_records_exception_class():
    from iago.runner import _execute_planted

    class Defect(FakeTarget):
        def chat(self, messages, options=None):
            raise KeyError("missing_field")

    _, reply = _execute_planted(Defect(), "sys", [], "ask", {})
    assert reply.startswith("<<RUN-ERROR: KeyError: ")


def test_report_discloses_dropped_controls():
    from iago.report import determinism_disclosure

    text = determinism_disclosure({"sampling": {"unsupported_controls": ["seed"]},
                                   "determinism": {"mismatch_detected": False, "probes": []}})
    assert "ignored `seed`" in text
    clean = determinism_disclosure({"sampling": {"unsupported_controls": []},
                                    "determinism": {"mismatch_detected": False, "probes": []}})
    assert "ignored" not in clean
