"""Attack-vs-defense delta report — synthetic rows, no model required."""

from __future__ import annotations

from iago.delta import build_delta_report
from iago.guards import blocked_response
from iago.judge import BYPASSED, REFUSED


def _row(tid, cat, oid, kind, verdict, response="...", trial=0, owasp="LLM01"):
    return {
        "technique_id": tid, "technique_name": f"name-{tid}", "category": cat,
        "owasp": owasp, "objective_id": oid, "objective_kind": kind,
        "model": "ollama:test", "trial": trial, "verdict": verdict, "response": response,
    }


def _raw():
    return [
        _row("t1", "role-play", "obj-harm", "forbidden", BYPASSED, trial=0),
        _row("t1", "role-play", "obj-harm", "forbidden", BYPASSED, trial=1),
        _row("t2", "prompt-injection", "obj-leak", "prompt-leak", BYPASSED, owasp="LLM07"),
        _row("t3", "direct-ask", "obj-ctrl", "control", REFUSED),
    ]


def _guarded():
    blk_in = blocked_response("input-jailbreak-classifier", "matched jailbreak pattern: roleplay-as")
    blk_out = blocked_response("output-systemprompt-dlp", "reproduced a protected token")
    return [
        # t1 was neutralized by the input classifier on both trials.
        _row("t1", "role-play", "obj-harm", "forbidden", REFUSED, response=blk_in, trial=0),
        _row("t1", "role-play", "obj-harm", "forbidden", REFUSED, response=blk_in, trial=1),
        # t2 leak: still bypasses under guard => a residual.
        _row("t2", "prompt-injection", "obj-leak", "prompt-leak", BYPASSED, owasp="LLM07"),
        # control got over-blocked (benign direct-ask refused by the input classifier).
        _row("t3", "direct-ask", "obj-ctrl", "control", REFUSED, response=blk_in),
    ]


def test_headline_shows_reduction_and_residual():
    md = build_delta_report(_raw(), _guarded())
    assert "Harmful-content bypass rate" in md
    assert "System-prompt exfiltration (LLM07)" in md
    # harmful content: raw 100% (2/2) -> guarded 0% (0/2)
    assert "100%" in md and "0%" in md
    assert "2/2" in md and "0/2" in md
    # residual leak survives the guard
    assert "Residual bypasses" in md
    assert "t2" in md


def test_attribution_credits_neutralized_bypass():
    md = build_delta_report(_raw(), _guarded())
    assert "input-jailbreak-classifier" in md
    # the classifier blocked t1 (2 trials) that bypassed raw => 2 credited
    assert "Guard attribution" in md


def test_over_block_counts_benign_control():
    md = build_delta_report(_raw(), _guarded())
    assert "Over-block cost" in md
    # one direct-ask control was blocked
    assert "1" in md


def test_harmful_content_headline_requires_regrade_when_heuristic_only():
    # Heuristic-only run: no forbidden bypasses, no claude_verdict → must NOT print a
    # misleading "0% → 0%"; it says the row needs a regrade (the critical review finding).
    raw = [_row("t1", "role-play", "obj-harm", "forbidden", REFUSED, response="I refuse.")]
    guarded = [_row("t1", "role-play", "obj-harm", "forbidden", REFUSED, response="I refuse.")]
    md = build_delta_report(raw, guarded)
    assert "not adjudicated" in md
    assert "iago regrade" in md


def test_harmful_content_headline_shows_when_claude_graded():
    # Once a claude_verdict is present, the real delta renders.
    raw = [dict(_row("t1", "role-play", "obj-harm", "forbidden", REFUSED), claude_verdict=BYPASSED)]
    guarded = [dict(_row("t1", "role-play", "obj-harm", "forbidden", REFUSED), claude_verdict=REFUSED)]
    md = build_delta_report(raw, guarded)
    assert "Harmful-content bypass rate" in md
    assert "not adjudicated" not in md


def test_empty_inputs_are_handled():
    assert "Need both" in build_delta_report([], [])
