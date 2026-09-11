"""ISC-33 — provenance manifest + schema on every artifact; readers refuse the wrong surface."""

import json
import os

import pytest

from iago.artifacts import (SCHEMA_VERSION, build_manifest, load_rows, module_fingerprint, ollama_info,
                            read_artifact, require_surface, surface_of)
from iago.attacks import Technique
from iago.compare import build_comparison
from iago.judge import BYPASSED, REFUSED
from iago.objectives import Objective
from iago.regrade import regrade_file
from iago.report import build_report
from iago.runner import load_artifacts, run
from iago.target import Target


class _T(Target):
    name, is_local = "ollama:fake-model", True

    def generate(self, prompt, system=None, options=None):
        return "A" * 300

    def chat(self, messages, options=None):
        return "A" * 300


TECHS = [Technique("t1", "Direct", "direct-ask", "d", "Do: {objective}")]
OBJS = [Objective("o1", "forbidden thing", "forbidden")]


def test_run_writes_manifest_first_and_stamps_rows(tmp_path):
    path = run(_T(), techniques=TECHS, objectives=OBJS, trials=1, artifacts_dir=tmp_path)
    manifest, rows = read_artifact(path)
    assert manifest and manifest["record"] == "manifest" and manifest["schema_version"] == SCHEMA_VERSION
    for key in ("iago_version", "git", "model", "sampling", "judge_id", "ollama", "ollama_env", "host",
                "technique_library_sha256", "compatible_pairs", "created"):
        assert key in manifest, key
    assert manifest["surface"] == "chatbot" and manifest["sampling"]["trials"] == 1
    assert manifest["judge_id"].startswith("judge-")
    assert manifest["ollama_env"]["OLLAMA_HOST"] in (None, "") or "@" not in manifest["ollama_env"]["OLLAMA_HOST"]
    r = rows[0]
    assert r["schema_version"] == SCHEMA_VERSION and r["surface"] == "chatbot" and r["status"] == "evaluated"
    assert len(r["prompt_sha256"]) == 64 and len(r["response_sha256"]) == 64
    # the manifest is never handed to a report as a trial
    assert load_artifacts(path) == rows and all(not x.get("record") for x in rows)
    assert "No artifacts" not in build_report(rows)


def test_legacy_artifact_without_manifest_still_loads(tmp_path):
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps({"technique_id": "t", "objective_kind": "forbidden"}) + "\n")
    manifest, rows = read_artifact(p)
    assert manifest is None and len(rows) == 1 and surface_of(rows[0]) == "chatbot"


def test_report_refuses_agent_artifact_loudly(tmp_path):
    """Asserted on write_report / write_html_report / write_log, not just build_report: the guard
    originally sat only on the build_* seam, which no CLI path reaches first, so the headline
    KeyError this claim is named for still fired (Council blocker)."""
    from iago.report import write_html_report, write_log, write_report

    agent_rows = [{"kind": "attack", "scenario_id": "s", "verdict": "resisted", "model": "m"}]
    with pytest.raises(ValueError, match="iago report reads chatbot artifacts"):
        build_report(agent_rows)
    for writer in (write_report, write_html_report, write_log):
        with pytest.raises(ValueError, match="iago report reads chatbot artifacts"):
            writer(agent_rows, reports_dir=tmp_path)


def test_compare_refuses_chatbot_artifact_loudly(tmp_path):
    path = run(_T(), techniques=TECHS, objectives=OBJS, trials=1, artifacts_dir=tmp_path)
    with pytest.raises(ValueError, match="iago compare reads agent artifacts"):
        build_comparison([path])


def _agent_artifact(tmp_path, name, judge_id):
    p = tmp_path / name
    lines = [build_manifest(surface="privilege", model="ollama:m", sampling={}, judge_id=judge_id),
             {"model": name, "kind": "capability", "scenario_id": "c", "scenario_name": "c",
              "verdict": "hijacked", "floor_fired": True, "surface": "privilege"},
             {"model": name, "kind": "attack", "scenario_id": "s", "scenario_name": "s",
              "verdict": "resisted", "surface": "privilege"}]
    p.write_text("\n".join(json.dumps(x) for x in lines))
    return p


def test_compare_refuses_differing_judge_ids_unless_allowed(tmp_path):
    a = _agent_artifact(tmp_path, "a.jsonl", "agent_oracle-aaaaaaaaaaaa")
    b = _agent_artifact(tmp_path, "b.jsonl", "agent_oracle-bbbbbbbbbbbb")
    with pytest.raises(ValueError, match="judge_id differs"):
        build_comparison([a, b])
    comp = build_comparison([a, b], allow_judge_mismatch=True)
    assert len(comp.models) == 2 and len(comp.judge_ids) == 2


def test_regrade_preserves_manifest_and_stamps_judge_id(tmp_path):
    path = run(_T(), techniques=TECHS, objectives=OBJS, trials=1, artifacts_dir=tmp_path)

    class J:
        judge_id = "claude-test-000000000000"

        def judge(self, objective, response, kind="forbidden"):
            from iago.judge import Verdict
            return Verdict(REFUSED, 0.9, "stub")

    regrade_file(path, J(), objectives={"o1": "x"})
    manifest, rows = read_artifact(path)
    assert manifest is not None and rows[0]["claude_judge_id"] == "claude-test-000000000000"
    assert json.loads(path.read_text().splitlines()[0])["record"] == "manifest"


def test_module_fingerprint_changes_when_a_scored_module_changes(tmp_path, monkeypatch):
    """`compare` refuses to mix runs whose oracle code differs, and that guard is only as good as
    this fingerprint moving when the bytes move. Calling a pure function twice proved nothing."""
    import iago.artifacts as art

    before = module_fingerprint("judge", "canary")
    assert before.startswith("judge-")
    assert module_fingerprint("judge") != module_fingerprint("canary")

    fake = tmp_path / "pkg"
    fake.mkdir()
    (fake / "judge.py").write_text((art._PKG_DIR / "judge.py").read_text() + "\n# edited\n")
    (fake / "canary.py").write_text((art._PKG_DIR / "canary.py").read_text())
    monkeypatch.setattr(art, "_PKG_DIR", fake)
    assert module_fingerprint("judge", "canary") != before


def test_ollama_info_never_raises_when_daemon_is_absent(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    info = ollama_info("ollama:nonexistent-model:latest")
    assert info["model"] == "nonexistent-model:latest"
    assert info["server_version"] is None and info["digest"] is None


def test_require_surface_accepts_matching_and_empty():
    require_surface([], "chatbot", reader="x")
    require_surface([{"surface": "chatbot"}], "chatbot", reader="x")
    require_surface([{"kind": "attack", "scenario_id": "s"}], "agent", reader="x")
