"""`--strict-fingerprints` turns an UNKNOWN fingerprint into a refusal in gate, compare and delta.

Without the flag a manifest missing `judge_id` or a library hash is a note (`fingerprint_status`:
unknown is never a match, never a refusal). CI that must fail closed passes the flag, and then the
unverifiable side exits 2 by name. A verified match still passes under the flag."""

from __future__ import annotations

import json

import pytest

from iago import cli
from iago.artifacts import MANIFEST_RECORD
from iago.judge import BYPASSED, REFUSED

# command -> its library fingerprint key
LIBRARY_KEY = {"gate": "technique_library_sha256", "delta": "technique_library_sha256",
               "compare": "scenario_library_sha256"}


def _write(path, manifest, rows):
    path.write_text("\n".join(json.dumps(x) for x in [{"record": MANIFEST_RECORD, **manifest}, *rows])
                    + "\n")
    return str(path)


def _chatbot_rows(verdict, **extra):
    return [{"surface": "chatbot", "technique_id": "t", "technique_name": "T", "category": "c",
             "owasp": "LLM01", "objective_id": "o1", "objective_kind": "forbidden",
             "model": "ollama:test", "trial": i, "verdict": verdict, "response": "...", **extra}
            for i in range(5)]


GRADED = {"claude_verdict": REFUSED, "claude_judge_name": "claude", "claude_judge_id": "claude:m:r"}


def _agent_rows(model):
    return [{"surface": "agent", "kind": "attack", "scenario_id": "s", "scenario_name": "s",
             "model": model, "trial": 0, "verdict": "resisted"}]


def _argv(command, tmp_path, known, legacy):
    """argv for `command` over two artifacts: one carrying `known`, one carrying `legacy`."""
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    if command == "compare":
        return ["compare", _write(a, known, _agent_rows("A")), _write(b, legacy, _agent_rows("B"))]
    if command == "delta":
        return ["delta", _write(a, known, _chatbot_rows(BYPASSED)), _write(b, legacy, _chatbot_rows(REFUSED))]
    return ["gate", _write(a, known, _chatbot_rows(REFUSED, **GRADED)), "--baseline",
            _write(b, legacy, _chatbot_rows(REFUSED, **GRADED))]


def _run(argv, tmp_path, monkeypatch):
    monkeypatch.setattr("iago.delta.REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr("iago.compare.REPORTS_DIR", tmp_path / "reports")
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def _full(command):
    return {"judge_id": "j1", LIBRARY_KEY[command]: "lib1"}


@pytest.mark.parametrize("command", ["gate", "compare", "delta"])
@pytest.mark.parametrize("missing", ["judge_id", "library"])
def test_an_unknown_fingerprint_is_a_note_by_default_and_a_refusal_under_the_flag(
        tmp_path, monkeypatch, capsys, command, missing):
    """Revert check: making `refuse_unknown_fingerprint` a no-op turns every flag-on case to rc 0."""
    key = "judge_id" if missing == "judge_id" else LIBRARY_KEY[command]
    known = _full(command)
    legacy = {k: v for k, v in known.items() if k != key}
    argv = _argv(command, tmp_path, known, legacy)

    assert _run(argv, tmp_path, monkeypatch) == 0
    capsys.readouterr()

    assert _run([*argv, "--strict-fingerprints"], tmp_path, monkeypatch) == 2
    err = capsys.readouterr().err
    assert f"no {key} in the" in err and "--strict-fingerprints" in err, err


@pytest.mark.parametrize("command", ["gate", "compare", "delta"])
def test_the_flag_passes_a_pair_whose_fingerprints_are_known_and_equal(tmp_path, monkeypatch, command):
    full = _full(command)
    assert _run([*_argv(command, tmp_path, full, full), "--strict-fingerprints"], tmp_path, monkeypatch) == 0


def test_the_refusal_names_the_unverified_side(tmp_path, monkeypatch, capsys):
    full = _full("delta")
    argv = _argv("delta", tmp_path, full, {"technique_library_sha256": "lib1"})
    assert _run([*argv, "--strict-fingerprints"], tmp_path, monkeypatch) == 2
    assert "no judge_id in the guarded manifest" in capsys.readouterr().err
