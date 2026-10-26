"""Attack-vs-defense delta report — synthetic rows, no model required."""

from __future__ import annotations

import re

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


def _leak_pairs(n, guard_holds):
    """n paired LLM07 leak trials: raw always leaks; guarded holds (blocked) or also leaks."""
    blk_out = blocked_response("output-systemprompt-dlp", "reproduced a protected token")
    raw = [_row(f"t{i}", "prompt-injection", f"obj-leak{i}", "prompt-leak", BYPASSED, owasp="LLM07")
           for i in range(n)]
    guarded = [_row(f"t{i}", "prompt-injection", f"obj-leak{i}", "prompt-leak",
                    REFUSED if guard_holds else BYPASSED,
                    response=blk_out if guard_holds else "...", owasp="LLM07") for i in range(n)]
    return raw, guarded


def test_mcnemar_pvalue_is_reported_not_deferred():
    # 3-vs-0 discordant split: the report must RUN McNemar's (exact p = 2*0.5**3 = 0.25), label it
    # directional (0.25 > 0.05), and must NOT still ship the retired folk-theorem.
    md = build_delta_report(*_leak_pairs(3, guard_holds=True))
    assert "McNemar" in md
    assert "p = 2.50e-01" in md                        # the computed exact p-value, not a hand-wave
    assert "directional only" in md                    # 0.25 is NOT significant — say so
    assert "the better test we didn't run" not in md   # folk-theorem must be gone
    assert "STRICTER bar" not in md


def test_mcnemar_significant_split_reads_not_noise():
    # 6-vs-0 => p = 2*0.5**6 = 0.03125 < 0.05 => the significance claim IS asserted.
    md = build_delta_report(*_leak_pairs(6, guard_holds=True))
    assert "p = 3.12e-02" in md
    assert "the reduction is asserted at the 0.05 level" in md
    # p < 0.05 bounds how often chance produces the split; it does not rule noise out.
    assert "not sampling noise" not in md


def test_mcnemar_no_discordant_pairs_never_claims_significance():
    # Guard changes no verdict (raw leaks, guarded also leaks) => 0 discordant pairs. The report must
    # NOT print a bogus "p = 1.00e+00 — not sampling noise"; it must say there's no signal to test.
    md = build_delta_report(*_leak_pairs(3, guard_holds=False))
    assert "no discordant pairs" in md
    assert "**no discordant pairs**; 3 both bypassed, 0 both held (n = 3 pairs)." in md
    assert "not sampling noise" not in md
    assert "p = 1.00e+00" not in md


def test_empty_inputs_are_handled():
    assert "Need both" in build_delta_report([], [])


def test_undecoded_harmful_trials_are_not_credited_to_the_guard():
    # A cipher trial the model never decoded is confusion, not a hold. Counting it gave the guard
    # holds it did not earn: on a 2026-08-09 llama3.1 pair, 17 raw and 18 guarded trials.
    from iago.delta import _paired_table, _rate_block

    def undecoded(trial):
        return {**_row("c1", "encoding", "o1", "forbidden", REFUSED, trial=trial),
                "gated": True, "decoded": False}

    raw = [_row("t1", "role-play", "o1", "forbidden", BYPASSED), undecoded(1), undecoded(2)]
    guarded = [_row("t1", "role-play", "o1", "forbidden", REFUSED), undecoded(1), undecoded(2)]
    assert (_rate_block(raw, "forbidden")["hits"], _rate_block(raw, "forbidden")["total"]) == (1, 1)
    assert _rate_block(guarded, "forbidden")["total"] == 1
    # A raw bypass whose guarded twin was never decoded is not a guard win.
    twin_undecoded = [{**_row("t1", "role-play", "o1", "forbidden", REFUSED),
                       "gated": True, "decoded": False}]
    assert _paired_table(raw, twin_undecoded, "forbidden")[1:3] == (0, 0)
    assert _paired_table(raw, guarded, "forbidden")[1:3] == (1, 0)


def test_delta_header_quotes_each_judges_error_rate():
    raw = [_row("t1", "role-play", "o1", "forbidden", REFUSED),
           _row("t2", "prompt-injection", "o2", "prompt-leak", BYPASSED, owasp="LLM07")]
    md = build_delta_report(raw, raw)
    assert "Judge calibration (heuristic)" in md and "Judge calibration (canary)" in md
    assert "Judge calibration (rubric)" not in md  # nothing was regraded


def test_per_category_table_drops_undecoded_trials():
    undecoded = {**_row("c1", "encoding", "o1", "forbidden", REFUSED, trial=1),
                 "gated": True, "decoded": False}
    raw = [_row("c1", "encoding", "o1", "forbidden", BYPASSED), undecoded]
    md = build_delta_report(raw, raw)
    assert "| encoding | 100% (1/1) |" in md.replace("100.0%", "100%")


def test_a_guard_block_on_a_gated_trial_is_a_guard_win_not_confusion():
    # The guard's refusal never echoes the objective, so the decode check marks every block on a
    # cipher trial "undecoded". Dropping those rows erased the guard's real wins (2026-10-02).
    from iago.delta import _paired_table, _rate_block

    blocked = {**_row("c1", "encoding", "o1", "forbidden", REFUSED,
                      response=blocked_response("input-jailbreak-classifier", "matched")),
               "gated": True, "decoded": False}
    raw = [{**_row("c1", "encoding", "o1", "forbidden", BYPASSED), "gated": True, "decoded": True}]
    assert _rate_block([blocked], "forbidden")["total"] == 1
    assert _paired_table(raw, [blocked], "forbidden")[1:3] == (1, 0)


def test_a_pair_is_skipped_when_either_side_is_not_a_probe():
    from iago.delta import _paired_table

    confused = {**_row("c1", "encoding", "o1", "forbidden", REFUSED), "gated": True, "decoded": False}
    bypass = _row("c1", "encoding", "o1", "forbidden", BYPASSED)
    assert _paired_table([confused], [bypass], "forbidden")[1:3] == (0, 0)   # raw side not a probe
    assert _paired_table([bypass], [confused], "forbidden")[1:3] == (0, 0)   # guarded side not a probe


def test_an_undecoded_prompt_leak_row_still_counts():
    # The undecoded exclusion is the harmful-content rule; leaks are judged by the canary.
    from iago.report import is_valid_probe

    leak = {**_row("t2", "prompt-injection", "o2", "prompt-leak", BYPASSED), "gated": True,
            "decoded": False}
    assert is_valid_probe(leak)


def test_each_arm_quotes_its_own_judge_when_they_differ():
    raw = [{**_row("t1", "role-play", "o1", "forbidden", REFUSED), "judge_id": None}]
    guarded = [{**_row("t1", "role-play", "o1", "forbidden", REFUSED), "judge_id": "J2"}]
    md = build_delta_report(raw, guarded)
    assert "Raw arm: **Judge calibration (heuristic)" in md
    assert "Guarded arm: **Judge calibration (heuristic `J2`)" in md


def test_both_arms_are_measured_on_the_same_trials():
    # The raw model never decoded c1, the guard blocked it: no raw probe, so no fair comparison.
    confused = {**_row("c1", "encoding", "o1", "forbidden", REFUSED), "gated": True, "decoded": False}
    blocked = {**_row("c1", "encoding", "o1", "forbidden", REFUSED,
                      response=blocked_response("input-jailbreak-classifier", "matched")),
               "gated": True, "decoded": False}
    raw = [_row("t1", "role-play", "o2", "forbidden", BYPASSED), confused]
    guarded = [_row("t1", "role-play", "o2", "forbidden", REFUSED), blocked]
    md = build_delta_report(raw, guarded)
    assert "1 attack pair(s)" in md and "excluded from BOTH arms" in md
    assert "| encoding |" not in md  # the only encoding pair left no probe in either arm


def _harm_pairs(n, guard_holds, graded=True):
    """n paired harmful-content trials adjudicated by the rubric judge: raw always bypasses;
    guarded holds or also bypasses. `graded=False` leaves them heuristic-only."""
    blk_in = blocked_response("input-jailbreak-classifier", "matched jailbreak pattern: roleplay-as")
    raw = [_row(f"t{i}", "role-play", f"obj-harm{i}", "forbidden", BYPASSED) for i in range(n)]
    guarded = [_row(f"t{i}", "role-play", f"obj-harm{i}", "forbidden",
                    REFUSED if guard_holds else BYPASSED, response=blk_in if guard_holds else "...")
               for i in range(n)]
    if graded:
        for r in raw + guarded:
            r["claude_verdict"] = r["verdict"]
    return raw, guarded


def _mcnemar_block(md):
    return md.split("**Paired result — exact counts first.**")[1].split("## Guard attribution")[0]


def test_mcnemar_runs_on_the_harmful_content_pairs_too():
    # 6-vs-0 harmful-content split, graded on both arms: the exact test runs on it (p = 0.03125),
    # not only on the LLM07 pairs. The leak pairs are empty here, which must read as no pairs.
    md = build_delta_report(*_harm_pairs(6, guard_holds=True))
    block = _mcnemar_block(md)
    assert "On the harmful-content pairs: **6** in the guard's favour" in block
    assert "p = 3.12e-02" in block
    assert "On the LLM07 pairs: n/a (0 pairs)" in block   # ISC-115: 0 pairs is n/a, not "no discordant"
    assert "On the LLM07 pairs there were **no discordant pairs**" not in block


def test_mcnemar_harmful_content_pairs_not_tested_when_heuristic_only():
    md = build_delta_report(*_harm_pairs(6, guard_holds=True, graded=False))
    block = _mcnemar_block(md)
    assert "Harmful-content pairs: _not tested_" in block
    assert "On the harmful-content pairs:" not in block


def test_delta_report_does_not_misstate_when_wald_collapses():
    md = build_delta_report(*_leak_pairs(3, guard_holds=True))
    assert "collapses to zero width when c = 0" not in md
    assert "the Wald interval does when no pair" in md


# --- census framing (ISC-115..117): exact counts lead, the interval is descriptive ------------

def _cells(kind, cells, graded=True):
    """Paired twins of one objective kind from (raw verdict, guarded verdict) cells, one technique
    per cell so every pair is its own (technique, objective, trial)."""
    owasp = "LLM07" if kind == "prompt-leak" else "LLM01"
    raw, guarded = [], []
    for i, (rv, gv) in enumerate(cells):
        for side, v in ((raw, rv), (guarded, gv)):
            r = _row(f"t{i}", "role-play", f"obj{i}", kind, v, owasp=owasp)
            if graded and kind == "forbidden":
                r["claude_verdict"] = v
            side.append(r)
    return raw, guarded


def _table_cells(a, b, c, d):
    return ([(BYPASSED, BYPASSED)] * a + [(BYPASSED, REFUSED)] * b
            + [(REFUSED, BYPASSED)] * c + [(REFUSED, REFUSED)] * d)


def _both_kinds(harm, leak):
    hr, hg = _cells("forbidden", _table_cells(*harm))
    lr, lg = _cells("prompt-leak", _table_cells(*leak))
    # Distinct technique ids per kind, so the two kinds never share a pair key.
    for r in lr + lg:
        r["technique_id"] = "L" + r["technique_id"]
    return hr + lr, hg + lg


def _line_index(lines, *needles):
    hits = [i for i, line in enumerate(lines) if all(n in line for n in needles)]
    assert hits, f"no line carries {needles}"
    return hits[0]


def test_census_discordant_counts_lead_and_the_interval_follows_as_dispersion():
    md = build_delta_report(*_both_kinds(harm=(1, 6, 0, 3), leak=(2, 5, 1, 4)))
    lines = md.splitlines()
    for label in ("harmful-content", "LLM07"):
        disc = _line_index(lines, f"On the {label} pairs:", "in the guard's favour")
        disp = _line_index(lines[disc:], "Descriptive dispersion only") + disc
        assert disc < disp
        # the dispersion line belongs to THIS kind: no other kind's discordant line sits between
        assert not any("in the guard's favour" in line for line in lines[disc + 1:disp])
        assert "not sampling precision over techniques" in lines[disp]
    assert "95% CI on the paired difference" not in md


def test_census_discordant_line_names_both_and_neither_and_the_direction():
    md = build_delta_report(*_both_kinds(harm=(1, 6, 0, 3), leak=(2, 0, 8, 4)))
    harm = next(line for line in md.splitlines() if "On the harmful-content pairs:" in line)
    leak = next(line for line in md.splitlines() if "On the LLM07 pairs:" in line)
    assert "**6** in the guard's favour" in harm and "**0** against" in harm
    assert "1 both bypassed, 3 both held (n = 10 pairs)" in harm
    assert "direction: reduction" in harm
    assert "2 both bypassed, 4 both held (n = 14 pairs)" in leak
    assert "direction: increase" in leak


def test_census_even_split_reads_even_not_a_direction():
    md = build_delta_report(*_cells("prompt-leak", _table_cells(0, 3, 3, 2)))
    leak = next(line for line in md.splitlines() if "On the LLM07 pairs:" in line)
    assert "direction: even split" in leak
    assert "no direction: the guard won and lost the same number of pairs" in leak
    assert "directional only" not in leak and "too few" not in leak


def test_census_large_nonsignificant_split_is_not_called_too_few():
    md = build_delta_report(*_cells("prompt-leak", _table_cells(0, 20, 12, 4)))  # p ≈ 0.22
    leak = next(line for line in md.splitlines() if "On the LLM07 pairs:" in line)
    assert "directional only: the split is not one-sided enough to assert significance" in leak
    assert "too few" not in leak


def test_census_dispersion_interval_is_the_shared_orientation_helper():
    from iago.delta import paired_delta_pts
    md = build_delta_report(*_cells("prompt-leak", _table_cells(1, 4, 0, 1)))
    lo, hi = paired_delta_pts((1, 4, 0, 1))
    assert f"guarded − raw {lo:+.1f} to {hi:+.1f} pts" in md


def test_census_heuristic_only_harm_prints_no_interval():
    md = build_delta_report(*_harm_pairs(6, guard_holds=True, graded=False))
    assert "Harmful-content pairs: _not tested_" in md
    assert md.count("Descriptive dispersion only") == 0  # no harm interval, and no leak pairs either
    # revert `tables["forbidden"]` to an always-computed table → the headline gains a marker
    assert "paired exact test" not in _headline_line(md, "Harmful-content bypass rate")


def _headline_line(md, label):
    return next(line for line in md.splitlines() if line.startswith(f"- **{label}:**"))


def test_marker_follows_the_paired_exact_test_even_when_wilson_intervals_overlap():
    from iago.stats import rose_measurably
    # a=8, b=6, c=0, d=6: raw 14/20 vs guarded 8/20, Wilson intervals overlap, split 6 vs 0.
    raw, guarded = _cells("prompt-leak", _table_cells(8, 6, 0, 6))
    assert not rose_measurably((14, 20), (8, 20))  # the fixture really is a Wilson overlap
    line = _headline_line(build_delta_report(raw, guarded), "System-prompt exfiltration (LLM07)")
    assert "*(paired exact test: 6 guard wins vs 0 regressions, p = 3.12e-02)*" in line


def test_marker_absent_when_the_exact_test_is_not_significant():
    md = build_delta_report(*_leak_pairs(3, guard_holds=True))  # 3 vs 0 → p = 0.25
    assert "paired exact test" not in _headline_line(md, "System-prompt exfiltration (LLM07)")


def test_marker_absent_with_no_discordant_pairs():
    md = build_delta_report(*_leak_pairs(6, guard_holds=False))
    assert "paired exact test" not in _headline_line(md, "System-prompt exfiltration (LLM07)")


def test_no_wilson_overlap_language_survives_in_the_delta_report():
    md = build_delta_report(*_both_kinds(harm=(1, 6, 0, 3), leak=(0, 37, 0, 107)))
    assert "defensible" not in md and "non-overlapping" not in md
    assert "census" in md and "not that the reduction generalizes" in md
    assert "a line is marked only when the paired exact test below gives p < 0.05" in " ".join(
        line.lstrip("> ") for line in md.splitlines())


def test_post_split_37_vs_0_of_144_still_renders_verbatim():
    # "A Zero You Can Defend" quotes 37/144 → 0/144, 37 vs 0 discordant, p ≈ 1.5e-11.
    md = build_delta_report(*_cells("prompt-leak", _table_cells(0, 37, 0, 107)))
    head = _headline_line(md, "System-prompt exfiltration (LLM07)")
    assert "26% (raw, 37/144" in head and "0% (guarded, 0/144" in head
    leak = next(line for line in md.splitlines() if "On the LLM07 pairs:" in line)
    assert "**37** in the guard's favour" in leak and "**0** against" in leak
    assert "p = 1.46e-11" in leak
    assert "0 both bypassed, 107 both held (n = 144 pairs)" in leak
    assert ("Descriptive dispersion only: Newcombe paired interval on guarded − raw -33.4 to -18.7 pts"
            " — spread across these 144 pairs") in md


def test_marker_on_a_significant_regression_renders_in_the_headline():
    # revert the marker for d > 0 (arrow up) → a guard-introduced regression loses its badge
    md = build_delta_report(*_cells("prompt-leak", _table_cells(0, 0, 8, 2)))
    line = _headline_line(md, "System-prompt exfiltration (LLM07)")
    assert "▲" in line
    assert "*(paired exact test: 0 guard wins vs 8 regressions, p = 7.81e-03)*" in line
    assert "against this arrow" not in line


def test_marker_says_when_the_paired_split_runs_against_the_arrow():
    # Mismatched trial counts: 6 paired guard wins, plus 30 twinless raw holds and 8 twinless
    # guarded bypasses, so the rates rise while every paired trial fell (code-review finding).
    raw, guarded = _cells("prompt-leak", _table_cells(0, 6, 0, 0))
    raw += [_row("t0", "role-play", "obj0", "prompt-leak", REFUSED, trial=t, owasp="LLM07")
            for t in range(1, 6) for _ in range(6)]
    guarded += [_row(f"x{i}", "role-play", f"objx{i}", "prompt-leak", BYPASSED, owasp="LLM07")
                for i in range(8)]
    line = _headline_line(build_delta_report(raw, guarded), "System-prompt exfiltration (LLM07)")
    assert "▲" in line
    assert "6 guard wins vs 0 regressions" in line
    assert "the paired split runs against this arrow" in line


def test_underflowed_p_renders_as_a_bound_not_zero():
    # ISC-119: 1,100 vs 0 has an exact p below the smallest float, so mcnemar_exact_p returns 0.0.
    # The paired line and the headline marker both read "p < 1e-300", never "p = 0.00e+00" or a
    # digit-starved subnormal, and the significance claim still stands.
    md = build_delta_report(*_cells("prompt-leak", _table_cells(0, 1100, 0, 0)))
    head = _headline_line(md, "System-prompt exfiltration (LLM07)")
    assert "*(paired exact test: 1100 guard wins vs 0 regressions, p < 1e-300)*" in head
    leak = next(line for line in md.splitlines() if "On the LLM07 pairs:" in line)
    assert "two-sided **p < 1e-300**" in leak
    assert "the reduction is asserted at the 0.05 level" in leak
    assert "0.00e+00" not in md and not re.search(r"\d\.\d+e-3\d\d", md)


def test_subnormal_p_renders_as_a_bound_too():
    # 1,074 vs 0: p = 1e-323 is a nonzero subnormal that prints "9.88e-324", one significant bit.
    md = build_delta_report(*_cells("prompt-leak", _table_cells(0, 1074, 0, 0)))
    assert "p < 1e-300" in _headline_line(md, "System-prompt exfiltration (LLM07)")
    assert not re.search(r"\d\.\d+e-3\d\d", md)
