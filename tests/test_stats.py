"""Tests for the Wilson confidence-interval helper and the McNemar exact test."""

from iago.stats import mcnemar_exact_p, wilson_interval


def test_zero_total_is_uninformative_not_certain():
    # No data is not certainty: an empty/all-error denominator must never render as "0%–0%".
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_out_of_range_inputs_raise_the_guard_not_a_sqrt_domain_error():
    """`match=` is load-bearing: without the guard, wilson_interval(4, 3) reaches math.sqrt with a
    negative variance and raises ValueError anyway, so a bare pytest.raises passes either way and
    cannot tell the guard from the accident (test review)."""
    import pytest
    for hits, total in ((4, 3), (-1, 3), (1, -2)):
        with pytest.raises(ValueError, match="out of range"):
            wilson_interval(hits, total)


def test_exact_values_at_k0_and_kn():
    # Closed form at k=0: low=0, high = z²/(n+z²). k=n mirrors it. Exact, not an inequality.
    z2 = 1.96 ** 2
    lo, hi = wilson_interval(0, 3)
    assert lo == 0.0 and abs(hi - z2 / (3 + z2)) < 1e-12
    lo, hi = wilson_interval(3, 3)
    assert hi == 1.0 and abs(lo - (1 - z2 / (3 + z2))) < 1e-12


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


# --- large discordant totals (ISC-118): the tail sum never becomes an overflowing float --------

def _log_space_mcnemar(b, c):
    """Independent reference: the two-sided exact tail via lgamma + log-sum-exp, never forming
    the huge binomial coefficients the implementation sums."""
    from math import exp, lgamma, log
    n, k = b + c, min(b, c)
    logs = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * log(2) for i in range(k + 1)]
    top = max(logs)
    return min(1.0, 2.0 * exp(top) * sum(exp(x - top) for x in logs))


def test_mcnemar_large_even_split_does_not_overflow():
    # revert to `sum(comb(...)) * (0.5 ** n)` → OverflowError: int too large to convert to float
    assert mcnemar_exact_p(600, 600) == 1.0


def test_mcnemar_large_uneven_split_matches_a_log_space_reference():
    p = mcnemar_exact_p(700, 500)
    ref = _log_space_mcnemar(700, 500)
    assert 0.0 < p < 1e-6               # a 700/500 split is far from a coin flip
    assert abs(p - ref) <= 1e-9 * ref, (p, ref)


def test_mcnemar_k_zero_boundary_is_the_exact_subnormal_or_zero():
    # A boundary check, not a regression test for the overflow: with k = 0 the old code summed the
    # integer 1 and never overflowed either. 2/2^1070 is a nonzero subnormal; 1500/0 underflows.
    assert mcnemar_exact_p(1070, 0) == 2 / 2 ** 1070
    # revert to `2 * (total / 2 ** n)` → 2^-1075 rounds to 0 before the doubling, so this reads 0.0
    assert mcnemar_exact_p(1075, 0) == 5e-324
    p = mcnemar_exact_p(1500, 0)
    assert isinstance(p, float) and p == 0.0


def _exact_mcnemar(b, c):
    """Independent exact reference: a fresh comb() per term and one Fraction, never the
    implementation's term recurrence."""
    from fractions import Fraction
    from math import comb
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, float(Fraction(2 * sum(comb(n, i) for i in range(k + 1)), 2 ** n)))


def test_mcnemar_small_splits_equal_the_exact_reference_bit_for_bit():
    # revert the recurrence to `term // (i + 1) * (n - i)` → (5, 2) reads 0.40625, not 0.453125
    assert mcnemar_exact_p(5, 2) == 0.453125
    bad = [(b, c) for b in range(31) for c in range(31) if mcnemar_exact_p(b, c) != _exact_mcnemar(b, c)]
    assert bad == []


def test_mcnemar_large_uneven_split_in_both_orders_matches_the_references():
    # k = 2400: far past any small cap on the loop; both argument orders take the large path.
    ref = _exact_mcnemar(2600, 2400)
    assert 0.0 < ref < 1.0
    assert mcnemar_exact_p(2600, 2400) == ref == mcnemar_exact_p(2400, 2600)
    assert abs(ref - _log_space_mcnemar(2600, 2400)) <= 1e-9 * ref
    assert mcnemar_exact_p(500, 700) == mcnemar_exact_p(700, 500)
