"""Tests for the Wilson confidence-interval helper and the McNemar exact test."""

from iago.stats import mcnemar_exact_p, wilson_interval


def test_zero_total_is_zero_zero():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_interval_stays_within_unit():
    lo, hi = wilson_interval(1, 3)
    assert 0.0 <= lo <= hi <= 1.0


def test_small_n_all_hits_does_not_collapse_to_point():
    lo, hi = wilson_interval(3, 3)
    assert hi <= 1.0
    assert lo < 1.0  # a 3/3 run is NOT certainty — the interval reflects that


def test_more_trials_tighten_the_interval():
    lo_small, hi_small = wilson_interval(1, 3)   # 33%, n=3
    lo_big, hi_big = wilson_interval(10, 30)     # 33%, n=30 (same point estimate)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_point_estimate_lies_within_the_interval():
    lo, hi = wilson_interval(4, 10)
    assert lo <= 0.4 <= hi


def test_mcnemar_no_discordant_pairs_is_one():
    # No disagreement between the paired runs => nothing to be surprised about.
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_fully_one_sided_matches_hand_calc():
    # 37 wins, 0 regressions: two-sided exact = 2 * 0.5**37. This is the real defense-delta case.
    assert abs(mcnemar_exact_p(37, 0) - 2 * 0.5 ** 37) < 1e-18
    assert mcnemar_exact_p(37, 0) < 1e-10  # overwhelmingly significant


def test_mcnemar_small_one_sided_split():
    # 3 vs 0 => 2 * 0.5**3 = 0.25, a case the chi-square approximation would mishandle.
    assert abs(mcnemar_exact_p(3, 0) - 0.25) < 1e-12


def test_mcnemar_is_symmetric_in_its_arguments():
    # p(b, c) == p(c, b): the test cares about the split, not the direction.
    assert mcnemar_exact_p(5, 2) == mcnemar_exact_p(2, 5)


def test_mcnemar_even_split_is_not_significant():
    # A 4/4 discordant split is exactly what the null predicts => p == 1.0.
    assert mcnemar_exact_p(4, 4) == 1.0
