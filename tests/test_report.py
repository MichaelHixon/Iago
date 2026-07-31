"""the report computes bypass rates and reads only artifacts."""

from iago.report import build_report


def _row(**kw):
    base = dict(
        technique_id="t1", technique_name="Direct", category="direct-ask",
        objective_id="o1", objective_kind="forbidden", model="fake:model",
        seed=1337, temperature=0.8, trial=0, prompt="Do: X", response="A" * 300,
        verdict="bypassed", confidence=0.8, rationale="substantive",
        latency_s=0.1, timestamp="2026-07-24T00:00:00Z",
    )
    base.update(kw)
    return base


def test_bypass_rate_math():
    rows = [
        _row(trial=0, verdict="bypassed"),
        _row(trial=1, verdict="refused"),
        _row(trial=2, verdict="bypassed"),
        _row(trial=3, verdict="bypassed"),
    ]
    md = build_report(rows)
    assert "3 / 4 forbidden-objective trials were confirmed bypasses" in md
    assert "75%" in md


def test_control_calibration_note_present():
    rows = [
        _row(objective_id="o1", objective_kind="forbidden", verdict="refused"),
        _row(objective_id="oc", objective_kind="control", verdict="bypassed"),
    ]
    md = build_report(rows)
    assert "control" in md.lower()
    assert "1/1" in md or "1 / 1" in md.replace("/1", "/1")


def test_empty_rows_safe():
    assert "No artifacts" in build_report([])


def test_hardening_section_present():
    md = build_report([_row()])
    assert "Hardening Recommendations" in md
