"""ISC-75/76/80: `delta` checks the arms' fingerprints, gate/compare/delta share one fingerprint
policy, and the YAML loaders refuse an empty, null or wrong-shape file by name."""

from __future__ import annotations

import json

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


def _gate_artifact(path, judge_id):
    manifest = {"record": MANIFEST_RECORD}
    if judge_id is not None:
        manifest["judge_id"] = judge_id
    rows = [{"surface": "chatbot", "technique_id": "t", "technique_name": "T", "category": "c",
             "objective_kind": "forbidden", "verdict": REFUSED, "claude_verdict": REFUSED,
             "claude_judge_name": "claude", "claude_judge_id": "claude:m:r"} for _ in range(5)]
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
    """`yaml.safe_load(...) or []` read these as zero records (or iterated a mapping's keys).
    Revert check: restoring `or []` in any one loader reds its row of this table."""
    path = tmp_path / "bad.yaml"
    path.write_text(doc)
    with pytest.raises(ValueError, match=r"^bad\.yaml: "):
        loader(path)


@pytest.mark.parametrize("doc", BAD_DOCUMENTS.values(), ids=BAD_DOCUMENTS.keys())
def test_the_technique_library_refuses_an_empty_technique_file(tmp_path, doc):
    (tmp_path / "bad.yaml").write_text(doc)
    with pytest.raises(ValueError, match=r"^bad\.yaml: "):
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
