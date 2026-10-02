"""ISC-75/76/80: `delta` checks the arms' fingerprints, gate/compare/delta share one fingerprint
policy, and the YAML loaders refuse an empty, null or wrong-shape file by name."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from iago import cli
from iago.agent_a2a import load_a2a_scenarios
from iago.agent_disclosure import load_disclosure_scenarios
from iago.agent_memory import load_memory_scenarios
from iago.agent_misinfo import load_misinfo_scenarios
from iago.agent_privilege import load_privilege_scenarios
from iago.agent_rag import load_rag_scenarios
from iago.agent_scenarios import load_scenarios
from iago.agent_toolabuse import load_toolabuse_scenarios
from iago.artifacts import MANIFEST_RECORD, fingerprint_status
from iago.attacks import load_library
from iago.gate import evaluate
from iago.judge import BYPASSED, REFUSED
from iago.objectives import load_objectives


# --- ISC-76: the shared policy ----------------------------------------------------------------
def test_fingerprint_status_treats_a_missing_value_as_unknown_never_as_a_match():
    assert fingerprint_status({"a": "x", "b": "x"}) == ({"x"}, [])
    assert fingerprint_status({"a": "x", "b": "y"}) == ({"x", "y"}, [])
    assert fingerprint_status({"a": "x", "b": None}) == ({"x"}, ["b"])
    assert fingerprint_status({"a": None, "b": ""}) == (set(), ["a", "b"])


def _gate_artifact(path, judge_id, rubric="claude:m:r"):
    manifest = {"record": MANIFEST_RECORD}
    if judge_id is not None:
        manifest["judge_id"] = judge_id
    rows = [{"surface": "chatbot", "technique_id": "t", "technique_name": "T", "category": "c",
             "objective_kind": "forbidden", "verdict": REFUSED, "claude_verdict": REFUSED,
             "claude_judge_name": "claude", "claude_judge_id": rubric} for _ in range(5)]
    path.write_text("\n".join(json.dumps(x) for x in [manifest, *rows]) + "\n")
    return path


@pytest.mark.parametrize("cur_id,base_id", [("j1", None), (None, "j1"), (None, None)])
def test_gate_notes_an_unknown_judge_id_instead_of_refusing_or_passing_it(tmp_path, cur_id, base_id):
    """Before ISC-76 the gate refused a legacy (None) side against a stamped one and silently
    matched two legacy sides; `compare` noted both. Now all three follow one policy.
    Revert check: restoring the tuple comparison `mine != theirs` reds the first two cases."""
    notes: list[str] = []
    evaluate(_gate_artifact(tmp_path / "c.jsonl", cur_id), _gate_artifact(tmp_path / "b.jsonl", base_id),
             notes=notes)
    unknown = [n for n in notes if "no judge_id" in n]
    assert len(unknown) == 1 and "UNKNOWN" in unknown[0], notes
    sides = " and ".join(s for s, j in (("current", cur_id), ("baseline", base_id)) if j is None)
    assert f"no judge_id in the {sides} manifest" in unknown[0], unknown[0]


# --- ISC-75: delta refuses mismatched arms ----------------------------------------------------
def _delta_artifact(path, **manifest):
    rows = [
        {"technique_id": "t1", "technique_name": "n", "category": "role-play", "owasp": "LLM01",
         "objective_id": "o1", "objective_kind": "forbidden", "model": "ollama:test", "trial": 0,
         "verdict": BYPASSED, "response": "..."},
    ]
    lines = ([{"record": MANIFEST_RECORD, **manifest}] if manifest else []) + rows
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return str(path)


def _delta(tmp_path, monkeypatch, raw, guarded, *flags):
    monkeypatch.setattr("iago.delta.REPORTS_DIR", tmp_path / "reports")
    args = cli.build_parser().parse_args(["delta", raw, guarded, *flags])
    return args.func(args)


SAME = {"judge_id": "j1", "technique_library_sha256": "lib1"}


@pytest.mark.parametrize("key,flag", [("judge_id", "--allow-judge-mismatch"),
                                      ("technique_library_sha256", "--allow-library-mismatch")])
def test_delta_refuses_arms_with_different_fingerprints_unless_overridden(tmp_path, monkeypatch, capsys,
                                                                         key, flag):
    """Revert check: deleting the `raise` in `fingerprint_notes` turns the refusal into rc 0."""
    raw = _delta_artifact(tmp_path / "raw.jsonl", **SAME)
    guarded = _delta_artifact(tmp_path / "g.jsonl", **{**SAME, key: "other"})
    assert _delta(tmp_path, monkeypatch, raw, guarded) == 2
    assert key in capsys.readouterr().err

    assert _delta(tmp_path, monkeypatch, raw, guarded, flag) == 0
    report = next((tmp_path / "reports").glob("delta_*.md")).read_text()
    assert key in report and flag in report


def test_delta_names_an_unknown_fingerprint_and_is_silent_on_a_match(tmp_path, monkeypatch):
    raw = _delta_artifact(tmp_path / "raw.jsonl", **SAME)
    assert _delta(tmp_path, monkeypatch, raw, _delta_artifact(tmp_path / "g.jsonl", **SAME)) == 0
    clean = next((tmp_path / "reports").glob("delta_*.md"))
    assert "UNKNOWN" not in clean.read_text()
    clean.unlink()

    assert _delta(tmp_path, monkeypatch, raw, _delta_artifact(tmp_path / "legacy.jsonl")) == 0
    report = next((tmp_path / "reports").glob("delta_*.md")).read_text()
    for key in SAME:
        assert f"No `{key}` in the guarded arm's manifest" in report
        assert "UNKNOWN" in report


# --- ISC-80: loud YAML loaders ----------------------------------------------------------------
LOADERS = {
    "objectives": load_objectives, "agent": load_scenarios, "privilege": load_privilege_scenarios,
    "memory": load_memory_scenarios, "misinfo": load_misinfo_scenarios,
    "toolabuse": load_toolabuse_scenarios, "rag": load_rag_scenarios, "a2a": load_a2a_scenarios,
    "disclosure": load_disclosure_scenarios,
}
BAD_DOCUMENTS = {"empty": "", "null": "null\n", "empty-list": "[]\n", "mapping": "id: x\n",
                 "scalar-record": "- just a string\n"}


@pytest.mark.parametrize("doc", BAD_DOCUMENTS.values(), ids=BAD_DOCUMENTS.keys())
@pytest.mark.parametrize("loader", LOADERS.values(), ids=LOADERS.keys())
def test_every_record_loader_refuses_a_file_with_no_usable_records(tmp_path, loader, doc):
    """Every loader names the file when it holds no usable records. For these single-file loaders
    this pins the contract rather than catching a revert: restoring `or []` in one was measured
    still red-free, because each already failed later (no attack scenario, a missing field). The
    library test below is the one where `or []` was silent."""
    path = tmp_path / "bad.yaml"
    path.write_text(doc)
    with pytest.raises(ValueError, match=r"^bad\.yaml: "):
        loader(path)


@pytest.mark.parametrize("doc", BAD_DOCUMENTS.values(), ids=BAD_DOCUMENTS.keys())
def test_the_technique_library_refuses_an_empty_file_beside_good_ones(tmp_path, doc):
    """The silent case the old `or []` really had: the library spans many files, so one emptied
    file just shrank the run while the others loaded. (A lone bad file in a single-file loader
    already failed later, on "no attack scenarios" or a missing field.)
    Revert check: restoring `or []` in `load_library` makes the empty/null/empty-list rows pass."""
    shipped = Path(__file__).resolve().parent.parent / "attacks" / "direct_ask.yaml"
    (tmp_path / "direct_ask.yaml").write_text(shipped.read_text())
    assert load_library(tmp_path), "the good file alone must load"
    (tmp_path / "zz_bad.yaml").write_text(doc)
    with pytest.raises(ValueError, match=r"^zz_bad\.yaml: "):
        load_library(tmp_path)


@pytest.mark.parametrize("name", ["wrappers.local.yaml", "provenance.local.yaml"])
@pytest.mark.parametrize("doc", ["", "null\n", "{}\n", "- a\n"], ids=["empty", "null", "empty-map", "list"])
def test_an_operator_frame_override_file_with_nothing_in_it_is_refused(tmp_path, monkeypatch, name, doc):
    """An empty override file used to fall back to the inert defaults without a word, so the
    operator believed the operative frames were loaded."""
    from iago import attacks

    monkeypatch.setattr(attacks, "ATTACKS_DIR", tmp_path)
    (tmp_path / name).write_text(doc)
    frames = attacks._wrapper_frames if name.startswith("wrappers") else attacks._provenance_frames
    with pytest.raises(ValueError, match=rf"^{name.replace('.', r'[.]')}: "):
        frames()


def test_gate_records_an_overridden_judge_mismatch_and_still_names_an_unknown_side(tmp_path):
    """With `--allow-judge-mismatch` the gate used to pass silently, so a forced comparison read
    exactly like a clean one; and the unknown-judge note must survive a rubric mismatch too.
    Revert check: dropping the overridden-mismatch note, or emitting the unknown note only when
    the rubric stamps agree, reds this."""
    notes: list[str] = []
    evaluate(_gate_artifact(tmp_path / "c.jsonl", None, rubric="claude:m:r2"),
             _gate_artifact(tmp_path / "b.jsonl", "j1", rubric="claude:m:r1"),
             allow_judge_mismatch=True, notes=notes)
    assert any("scored differently" in n and "--allow-judge-mismatch given" in n for n in notes), notes
    assert any("no judge_id in the current manifest" in n for n in notes), notes


def test_compare_records_an_overridden_judge_mismatch(tmp_path):
    from iago.compare import build_comparison, write_comparison_report

    def art(name, model, judge):
        rows = [{"record": MANIFEST_RECORD, "judge_id": judge},
                {"surface": "agent", "kind": "attack", "scenario_id": "s", "scenario_name": "s",
                 "model": model, "trial": 0, "verdict": "resisted"}]
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return p

    comp = build_comparison([art("a.jsonl", "A", "j1"), art("b.jsonl", "B", "j2")],
                            allow_judge_mismatch=True)
    text = write_comparison_report(comp, reports_dir=tmp_path).read_text()
    assert "Different oracle code" in text and "--allow-judge-mismatch" in text


# --- delta refuses arms the rubric judge scored differently ------------------------------------
def _graded_artifact(path, *, graded, stamp="claude:m:r1", n=4):
    rows = [{"technique_id": f"t{i}", "technique_name": "n", "category": "role-play",
             "owasp": "LLM01", "objective_id": "o1", "objective_kind": "forbidden",
             "model": "ollama:test", "trial": 0, "verdict": REFUSED, "response": "..."}
            for i in range(n)]
    for i, r in enumerate(rows):
        if graded is True or (graded == "partial" and i == 0):
            r.update(claude_verdict=BYPASSED, claude_judge_name="claude", claude_judge_id=stamp)
    path.write_text("\n".join(json.dumps(x) for x in [{"record": MANIFEST_RECORD, **SAME}, *rows]) + "\n")
    return str(path)


@pytest.mark.parametrize("raw,guarded", [
    (dict(graded=True), dict(graded=False)),                   # only raw regraded: the 100-point "win"
    (dict(graded=False), dict(graded=True)),
    (dict(graded="partial"), dict(graded="partial")),
    (dict(graded=True), dict(graded=True, stamp="claude:m:r2")),
], ids=["raw-only", "guarded-only", "partial", "different-rubric"])
def test_delta_refuses_arms_the_rubric_judge_scored_differently(tmp_path, monkeypatch, capsys, raw, guarded):
    """Revert check: deleting the `raise` in `adjudication_notes` reds every case."""
    r = _graded_artifact(tmp_path / "raw.jsonl", **raw)
    g = _graded_artifact(tmp_path / "g.jsonl", **guarded)
    assert _delta(tmp_path, monkeypatch, r, g) == 2
    assert "regrade" in capsys.readouterr().err
    assert _delta(tmp_path, monkeypatch, r, g, "--allow-judge-mismatch") == 0
    report = next((tmp_path / "reports").glob("delta_*.md")).read_text()
    assert "may measure the judging, not the guard" in report


@pytest.mark.parametrize("graded", [True, False], ids=["both-regraded", "neither"])
def test_delta_accepts_arms_judged_alike(tmp_path, monkeypatch, graded):
    r = _graded_artifact(tmp_path / "raw.jsonl", graded=graded)
    g = _graded_artifact(tmp_path / "g.jsonl", graded=graded)
    assert _delta(tmp_path, monkeypatch, r, g) == 0


def test_load_records_checks_every_record_not_just_the_first(tmp_path):
    from iago.yaml_records import load_records

    path = tmp_path / "mixed.yaml"
    path.write_text("- id: ok\n- just a string\n")
    with pytest.raises(ValueError, match=r"^mixed\.yaml: record 1 is a str"):
        load_records(path)


def test_a_broken_frame_override_fails_the_library_but_not_the_other_commands(tmp_path):
    """An empty override used to crash every `iago` command at import, `gate` included, with the
    exit code gate uses for a regression. Frames now load on first use; the library load (start of
    every run) still refuses, naming the path and the remedy. A fresh interpreter, because this
    one imported `iago.attacks` long ago. Revert check: restoring the import-time
    `_WRAPPER_FRAMES = _wrapper_frames()` makes the import itself fail."""
    import subprocess
    import sys

    (tmp_path / "wrappers.local.yaml").write_text("")
    script = (
        "import pathlib, sys\n"
        "import iago.config as c\n"
        f"c.ATTACKS_DIR = pathlib.Path({str(tmp_path)!r})\n"
        "import iago.attacks as a, iago.cli\n"
        "print('IMPORTED')\n"
        "try:\n"
        "    a.load_library(c.ATTACKS_DIR)\n"
        "except ValueError as e:\n"
        "    print('REFUSED', e)\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert "IMPORTED" in out.stdout, out.stderr
    assert re.search(r"REFUSED wrappers\.local\.yaml: .*delete it to use the shipped frames", out.stdout), out.stdout
