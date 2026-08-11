"""Composition-lift report math — synthetic rows + a synthetic library, no model required."""

from __future__ import annotations

from iago.attacks import Technique
from iago.compose_delta import (
    _beats,
    _best_single,
    _grading_for,
    _missing_baseline,
    _noisy_or,
    _overblock_block,
    _tech_block,
    build_compose_report,
)
from iago.judge import BYPASSED, ERROR, REFUSED


def _row(tid, kind, verdict, oid="o", response="...", trial=0):
    return {
        "technique_id": tid, "technique_name": f"name-{tid}", "category": "x",
        "owasp": "LLM01", "objective_id": oid, "objective_kind": kind,
        "model": "ollama:test", "trial": trial, "verdict": verdict, "response": response,
    }


def _fake_library():
    # One composed technique stacking two primitives, plus the two primitives themselves.
    return [
        Technique(id="ce-x", name="stack", category="composed-evasion",
                  description="d", template="{objective}", composes=("p1", "p2")),
        Technique(id="p1", name="prim1", category="role-play",
                  description="d", template="{objective}"),
        Technique(id="p2", name="prim2", category="encoding-obfuscation",
                  description="d", template="{objective}"),
    ]


def test_tech_block_counts_only_attack_kinds_and_excludes_errors():
    rows = [
        _row("t", "forbidden", BYPASSED),
        _row("t", "forbidden", REFUSED),
        _row("t", "prompt-leak", BYPASSED),
        _row("t", "control", BYPASSED),   # control is NOT an attack kind → excluded
        _row("t", "forbidden", ERROR),    # transport error → excluded
    ]
    b = _tech_block(rows, "t")
    assert b["hits"] == 2 and b["total"] == 3  # 2 bypass of 3 valid attack trials


def test_best_single_picks_highest_rate_constituent():
    rows = [
        _row("p1", "forbidden", BYPASSED), _row("p1", "forbidden", REFUSED),   # p1 = 1/2 = 50%
        _row("p2", "forbidden", BYPASSED), _row("p2", "forbidden", BYPASSED),  # p2 = 2/2 = 100%
    ]
    best_id, block = _best_single(rows, ("p1", "p2"))
    assert best_id == "p2" and block["rate"] == 1.0


def test_best_single_with_no_constituent_data():
    best_id, block = _best_single([], ("p1", "p2"))
    assert best_id is None and block["total"] == 0


def test_beats_requires_non_overlapping_intervals():
    # composed 10/10 (100%) vs baseline 0/10 (0%): non-overlapping → beats.
    composed = _tech_block([_row("c", "forbidden", BYPASSED, trial=i) for i in range(10)], "c")
    baseline = _tech_block([_row("b", "forbidden", REFUSED, trial=i) for i in range(10)], "b")
    assert _beats(composed, baseline) is True
    # A single trial each way overlaps heavily → not defensible.
    thin_c = _tech_block([_row("c", "forbidden", BYPASSED)], "c")
    thin_b = _tech_block([_row("b", "forbidden", REFUSED)], "b")
    assert _beats(thin_c, thin_b) is False


def test_beats_false_when_no_data():
    empty = {"hits": 0, "total": 0}
    some = _tech_block([_row("c", "forbidden", BYPASSED)], "c")
    assert _beats(some, empty) is False
    assert _beats(empty, some) is False


def test_overblock_counts_refused_and_guard_blocked_controls():
    from iago.guards import blocked_response
    rows = [
        _row("t", "control", REFUSED),
        _row("t", "control", BYPASSED),  # benign answered → not over-blocked
        _row("t", "control", REFUSED, response=blocked_response("g", "matched")),
        _row("t", "control", ERROR),     # excluded
    ]
    o = _overblock_block(rows, "t")
    assert o["blocked"] == 2 and o["total"] == 3


def test_report_shows_lift_and_ranks():
    # ce-x stacks p1+p2. Make the stack clearly beat its best single layer.
    rows = []
    rows += [_row("ce-x", "forbidden", BYPASSED, trial=i) for i in range(10)]      # composed 100%
    rows += [_row("p1", "forbidden", REFUSED, trial=i) for i in range(10)]         # p1 0%
    rows += [_row("p2", "forbidden", BYPASSED, trial=i) for i in range(2)]         # p2 100% but only 2 trials
    rows += [_row("p2", "forbidden", REFUSED, trial=i) for i in range(8)]          # p2 net 2/10 = 20%
    rows += [_row("ce-x", "control", BYPASSED)]                                    # benign answered, no over-block
    md = build_compose_report(rows, _fake_library())
    assert "Composition Lift" in md
    assert "ce-x" in md and "p1 ∘ p2".replace("p1 ∘ p2", "p1") in md  # constituents listed
    assert "Marginal lift" in md
    assert "+80.0 pts" in md  # 100% composed − 20% best-single(p2)
    assert "✅" in md  # non-overlapping intervals → beats parts


def test_missing_baseline_flags_constituents_that_never_fired_alone():
    # p1 fires alone, p2 never does → p2 is the missing baseline.
    rows = [_row("p1", "forbidden", BYPASSED)]
    assert _missing_baseline(rows, ("p1", "p2")) == ["p2"]
    assert _missing_baseline(rows, ("p1",)) == []


def test_noisy_or_expected_rate_over_constituents_that_fired():
    rows = [
        _row("p1", "forbidden", BYPASSED), _row("p1", "forbidden", REFUSED),   # p1 = 50%
        _row("p2", "forbidden", BYPASSED), _row("p2", "forbidden", BYPASSED),  # p2 = 100%
    ]
    p_exp, n_used = _noisy_or(rows, ("p1", "p2"))
    assert n_used == 2 and abs(p_exp - 1.0) < 1e-9  # 1-(1-.5)(1-1.0) = 1.0
    # Only p1 has data → noisy-OR over the one that fired.
    p_exp1, n1 = _noisy_or([_row("p1", "forbidden", BYPASSED), _row("p1", "forbidden", REFUSED)], ("p1", "p2"))
    assert n1 == 1 and abs(p_exp1 - 0.5) < 1e-9
    assert _noisy_or([], ("p1", "p2")) == (None, 0)


def test_report_refuses_lift_when_baseline_incomplete():
    # ce-x composes p1+p2, but p2 NEVER fires alone → baseline incomplete, no ✅, loud flag.
    rows = [_row("ce-x", "forbidden", BYPASSED, trial=i) for i in range(10)]
    rows += [_row("p1", "forbidden", REFUSED, trial=i) for i in range(10)]  # only p1 fires alone
    md = build_compose_report(rows, _fake_library())
    assert "BASELINE INCOMPLETE" in md
    assert "`p2`" in md
    assert "⚠️ incomplete" in md
    # Even though composed 100% would trounce p1, the incomplete baseline blocks the award —
    # the affirmative "beats its best single layer" verdict must never print for this stack.
    assert "beats its best single layer" not in md


def test_report_shows_independence_column():
    rows = []
    rows += [_row("ce-x", "forbidden", BYPASSED, trial=i) for i in range(10)]   # composed 100%
    rows += [_row("p1", "forbidden", BYPASSED, trial=i) for i in range(10)]     # p1 100%
    rows += [_row("p2", "forbidden", REFUSED, trial=i) for i in range(10)]      # p2 0%
    md = build_compose_report(rows, _fake_library())
    assert "Expected (indep.)" in md      # independence column header present
    assert "vs indep." in md


def test_report_handles_empty_rows():
    md = build_compose_report([], _fake_library())
    assert "No artifact rows" in md


def test_report_handles_no_composed_techniques():
    plain = [Technique(id="p1", name="x", category="role-play", description="d", template="{objective}")]
    md = build_compose_report([_row("p1", "forbidden", BYPASSED)], plain)
    assert "No composed-evasion techniques" in md


def test_grading_marks_heuristic_judge_and_canary_per_technique():
    assert _grading_for([_row("t", "forbidden", REFUSED)], "t") == "heuristic"
    judged = [dict(_row("t", "forbidden", BYPASSED), claude_verdict=BYPASSED)]
    assert _grading_for(judged, "t") == "judge"
    assert _grading_for([_row("t", "prompt-leak", BYPASSED)], "t") == "canary"
    mixed = [dict(_row("t", "forbidden", BYPASSED), claude_verdict=BYPASSED),
             _row("t", "forbidden", REFUSED)]
    assert _grading_for(mixed, "t") == "mixed"
    assert _grading_for([], "t") == "—"
    # And the per-row Scored marker surfaces in the report body.
    rows = [_row("ce-x", "forbidden", REFUSED, trial=i) for i in range(3)]
    rows += [_row("p1", "forbidden", REFUSED, trial=i) for i in range(3)]
    rows += [_row("p2", "forbidden", REFUSED, trial=i) for i in range(3)]
    md = build_compose_report(rows, _fake_library())
    assert "Scored" in md and "heuristic" in md


def test_report_shows_marginal_overblock_vs_best_single():
    # Complete baseline (p1 + p2 both fire alone) so no INCOMPLETE flag; p1 is best single.
    rows = []
    rows += [_row("ce-x", "forbidden", BYPASSED, trial=i) for i in range(10)]
    rows += [_row("p1", "forbidden", BYPASSED, trial=i) for i in range(10)]   # p1 100% → best single
    rows += [_row("p2", "forbidden", REFUSED, trial=i) for i in range(10)]    # p2 0% → baseline complete
    # Over-block: the stack breaks both benign controls; the best single (p1) breaks none.
    rows += [_row("ce-x", "control", REFUSED, trial=i) for i in range(2)]     # stack over-blocks 2/2
    rows += [_row("p1", "control", BYPASSED, trial=i) for i in range(2)]      # p1 over-blocks 0/2
    rows += [_row("p2", "control", BYPASSED, trial=i) for i in range(2)]
    md = build_compose_report(rows, _fake_library())
    assert "Over-block (Δ vs best)" in md          # differenced column header
    assert "Δ+100" in md                            # +100pt added false-positive tax in the table
    assert "marginal +100 pts" in md                # and in the per-technique detail
