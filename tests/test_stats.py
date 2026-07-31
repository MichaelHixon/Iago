"""Tests for the Wilson confidence-interval helper."""

from iago.stats import wilson_interval


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
