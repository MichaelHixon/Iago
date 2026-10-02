"""ISC-71 — k-of-n reliability beside the any-trial rate in the main report, with the headline
named. trials=3, one technique bypassing 1/3 and one bypassing 3/3."""

from iago.judge import BYPASSED, REFUSED
from iago.report import build_html_report, build_report, kofn_stats


def _row(tid, trial, verdict, oid="o1"):
    return dict(technique_id=tid, technique_name=f"n-{tid}", category="role-play", objective_id=oid,
                objective_kind="forbidden", model="fake:model", seed=1, temperature=0.8, trial=trial,
                prompt="p", response="A" * 300, verdict=verdict, confidence=0.8, rationale="r",
                latency_s=0.1, timestamp="2026-07-24T00:00:00Z")


def _rows():
    rows = [_row("t-flaky", t, BYPASSED if t == 0 else REFUSED) for t in range(3)]
    rows += [_row("t-solid", t, BYPASSED) for t in range(3)]
    return rows


def test_kofn_stats_counts_configs_by_any_and_every_trial():
    stats = {s["id"]: s for s in kofn_stats(_rows())}
    assert stats["t-flaky"]["configs"] == 1 and stats["t-flaky"]["any"] == 1 and stats["t-flaky"]["every"] == 0
    assert stats["t-solid"]["configs"] == 1 and stats["t-solid"]["any"] == 1 and stats["t-solid"]["every"] == 1
    assert stats["t-flaky"]["n"] == 3 and stats["t-solid"]["n"] == 3


def test_kofn_handles_uneven_trial_counts_per_config():
    # A config whose only valid trial bypassed is "every-trial" at n=1 — the n is shown, never hidden.
    rows = _rows() + [_row("t-thin", 0, BYPASSED, oid="o2")]
    stats = {s["id"]: s for s in kofn_stats(rows)}
    assert stats["t-thin"]["every"] == 1 and stats["t-thin"]["n"] == 1


def test_report_shows_any_trial_and_every_trial_beside_the_per_trial_rate():
    md = build_report(_rows())
    sec = md.split("## Reliability — any-trial vs every-trial bypass")[1].split("\n## ")[0]
    assert "| n-t-flaky (`t-flaky`) | 1 | 3 | 1/3 (33%) | 1/1 | 0/1 |" in sec
    assert "| n-t-solid (`t-solid`) | 1 | 3 | 3/3 (100%) | 1/1 | 1/1 |" in sec
    assert "The **per-trial bypass rate** (with its 95% Wilson CI) is the headline" in sec
    assert "any-trial" in sec and "every-trial" in sec


def test_report_kofn_section_absent_without_forbidden_trials():
    md = build_report([_row("t1", 0, REFUSED) | {"objective_kind": "control"}])
    assert "## Reliability — any-trial vs every-trial bypass" not in md


def test_html_report_carries_the_same_reliability_section():
    html = build_html_report(_rows())
    assert "any-trial vs every-trial" in html
    assert "t-solid" in html and "1/1" in html
