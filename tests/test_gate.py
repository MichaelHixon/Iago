"""`iago gate`: a CI pass/fail on harmful-content bypass rates, decided on Wilson intervals."""

import json

import pytest

from iago.artifacts import MANIFEST_RECORD
from iago.cli import main
from iago.gate import OVERALL, GateError, category_rates, evaluate
from iago.judge import BYPASSED, ERROR, NEEDS_REVIEW, REFUSED
from iago.report import harmful_valid


def _row(category, verdict, *, regraded=True, rubric="claude:m1:r1", **extra):
    row = {"surface": "chatbot", "technique_id": f"t-{category}", "technique_name": "T",
           "category": category, "objective_kind": "forbidden", "verdict": REFUSED}
    if regraded:
        row["claude_verdict"] = verdict
        row["claude_judge_name"] = "claude"
        row["claude_judge_id"] = rubric
    else:
        row["verdict"] = verdict
    row.update(extra)
    return row


def _artifact(path, rows, judge_id="j1"):
    lines = [{"record": MANIFEST_RECORD, "judge_id": judge_id}, *rows]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def _rows(category, hits, total):
    return [_row(category, BYPASSED if i < hits else REFUSED) for i in range(total)]


def test_category_rates_share_the_reports_denominator():
    rows = [*_rows("role-play", 2, 4),
            _row("role-play", ERROR),                                    # not a probe
            _row("cipher", REFUSED, gated=True, decoded=False),          # never decoded
            _row("x", BYPASSED, objective_kind="control")]               # not harmful-content
    rates = category_rates(harmful_valid(rows))
    assert rates["role-play"] == (2, 4)
    assert rates[OVERALL] == (2, 4)
    assert "cipher" not in rates and "x" not in rates


def test_noise_does_not_trip_a_regression(tmp_path):
    # 1/10 -> 3/10: the point rate tripled, but the intervals overlap.
    base = _artifact(tmp_path / "b.jsonl", _rows("role-play", 1, 10))
    cur = _artifact(tmp_path / "c.jsonl", _rows("role-play", 3, 10))
    assert not any(c.failed for c in evaluate(cur, base))


def test_a_measurable_rise_is_a_regression(tmp_path):
    base = _artifact(tmp_path / "b.jsonl", _rows("role-play", 0, 30))
    cur = _artifact(tmp_path / "c.jsonl", _rows("role-play", 15, 30))
    checks = {c.category: c for c in evaluate(cur, base)}
    assert checks["role-play"].regression and checks[OVERALL].regression


def test_max_rate_breach_uses_the_lower_bound(tmp_path):
    cur = _artifact(tmp_path / "c.jsonl", _rows("role-play", 15, 30))  # CI ~33%-67%
    assert evaluate(cur, max_rate=0.25)[0].breach
    assert not evaluate(cur, max_rate=0.40)[0].breach


def test_unresolved_trials_cannot_certify_a_pass(tmp_path):
    # The heuristic never assigns BYPASSED, so an unregraded run would always "pass".
    cur = _artifact(tmp_path / "c.jsonl", [_row("role-play", NEEDS_REVIEW, regraded=False)])
    with pytest.raises(GateError, match="regrade"):
        evaluate(cur, max_rate=0.5)


def test_judge_mismatch_is_refused_unless_allowed(tmp_path):
    base = _artifact(tmp_path / "b.jsonl", _rows("role-play", 0, 10), judge_id="old")
    cur = _artifact(tmp_path / "c.jsonl", _rows("role-play", 0, 10), judge_id="new")
    with pytest.raises(GateError, match="scored differently"):
        evaluate(cur, base)
    assert evaluate(cur, base, allow_judge_mismatch=True)


def test_nothing_measured_cannot_pass(tmp_path):
    cur = _artifact(tmp_path / "c.jsonl", [_row("role-play", ERROR)])
    with pytest.raises(GateError, match="nothing was measured"):
        evaluate(cur, max_rate=0.5)


def test_cli_exit_codes(tmp_path, capsys):
    base = _artifact(tmp_path / "b.jsonl", _rows("role-play", 0, 30))
    same = _artifact(tmp_path / "s.jsonl", _rows("role-play", 0, 30))
    worse = _artifact(tmp_path / "w.jsonl", _rows("role-play", 15, 30))
    assert main(["gate", str(same), "--baseline", str(base)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert main(["gate", str(worse), "--baseline", str(base)]) == 1
    assert "FAIL: (overall), role-play" in capsys.readouterr().out
    assert main(["gate", str(worse)]) == 2  # nothing to gate against


def test_a_different_rubric_fingerprint_is_a_mismatch(tmp_path):
    # Same backend name, different model or rubric version: still a different judge.
    base = _artifact(tmp_path / "b.jsonl", [_row("role-play", REFUSED, rubric="claude:m1:r1")])
    cur = _artifact(tmp_path / "c.jsonl", [_row("role-play", REFUSED, rubric="claude:m2:r1")])
    with pytest.raises(GateError, match="scored differently"):
        evaluate(cur, base)
