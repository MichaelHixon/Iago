"""ISC-70 — paired and clustered statistics (`iago/stats.py`).

Reference values are hand-computed from the published formulas, and where a published table
exists it is cited beside the number. Every test here is a falsifier for a specific sentence in
a docstring: the paired CI reduces to Newcombe's unpaired hybrid-score interval when the
correlation term is zero, and it is strictly narrower than that interval when the pairs agree —
which is exactly what an unpaired interval cannot do (the mutation check).
"""

import pytest

from iago.stats import (clustered_interval, newcombe_diff_ci, norm_ppf, paired_difference_ci,
                        paired_sample_size, wilson_interval)


# --- Newcombe hybrid score, unpaired (method 10 of Newcombe 1998a) ---------------------------
# Newcombe RG. "Two-sided confidence intervals for the single proportion: comparison of seven
# methods" is the single-proportion paper; the DIFFERENCE paper is Newcombe RG, "Interval
# estimation for the difference between independent proportions: comparison of eleven methods",
# Stat Med 1998;17:873-890, Table II. Method 10 on 56/70 vs 48/80 gives (0.0524, 0.3339) and on
# 9/10 vs 3/10 gives (0.1705, 0.8090). The same numbers come out of
# R: DescTools::BinomDiffCI(56, 70, 48, 80, method = "score").

def test_newcombe_unpaired_matches_published_table_56_70_vs_48_80():
    lo, hi = newcombe_diff_ci(56, 70, 48, 80)
    assert abs(lo - 0.0524) < 5e-4 and abs(hi - 0.3339) < 5e-4


def test_newcombe_unpaired_matches_published_table_9_10_vs_3_10():
    lo, hi = newcombe_diff_ci(9, 10, 3, 10)
    assert abs(lo - 0.1705) < 5e-4 and abs(hi - 0.8090) < 5e-4


# --- Newcombe paired (method 10 of Newcombe 1998b) --------------------------------------------
# Newcombe RG, "Improved confidence intervals for the difference between binomial proportions
# based on paired data", Stat Med 1998;17:2635-2650. Cells: a = both bypass, b = first-only,
# c = second-only, d = neither. The interval is the unpaired hybrid-score interval with the two
# Wilson half-widths combined through the phi correlation of the 2x2 table.

def test_paired_ci_with_a_zero_margin_reduces_to_the_unpaired_interval():
    # c + d == 0 -> phi is defined as 0 -> identical to the independent interval on the margins.
    # Hand check with the closed-form Wilson at k=n and k=0 (q = z^2/(n+z^2)): with n=3, both
    # half-widths are q, so delta = q*sqrt(2) and the interval is (1 - q*sqrt(2), 1).
    z2 = 1.96 ** 2
    q = z2 / (3 + z2)
    lo, hi = paired_difference_ci(a=0, b=3, c=0, d=0)
    assert abs(lo - (1 - q * 2 ** 0.5)) < 1e-12 and hi == 1.0
    assert (lo, hi) == newcombe_diff_ci(3, 3, 0, 3)


def test_paired_ci_is_strictly_narrower_than_unpaired_when_pairs_agree():
    """THE mutation check: replace the paired interval with the unpaired one and this goes red.
    a=20 concordant bypasses and d=15 concordant holds make phi strongly positive, so the paired
    interval on (b - c)/n must be narrower than the interval that treats the arms as independent."""
    a, b, c, d = 20, 5, 0, 15
    n = a + b + c + d
    plo, phi_ = paired_difference_ci(a=a, b=b, c=c, d=d)
    ulo, uhi = newcombe_diff_ci(a + b, n, a + c, n)
    assert (phi_ - plo) < (uhi - ulo) - 0.01
    assert plo < (b - c) / n < phi_


def test_paired_ci_hand_computed_at_a_small_table():
    # a=2, b=3, c=1, d=4, n=10. p1 = 5/10, p2 = 3/10, diff = 0.2.
    # phi = (ad - bc)/sqrt((a+b)(c+d)(a+c)(b+d)) = (8 - 3)/sqrt(5*5*3*7) = 5/sqrt(525).
    # Wilson 5/10: centre 0.5, half-width 0.2634 -> (0.2366, 0.7634); Wilson 3/10: (0.1078, 0.6032).
    # delta = sqrt((0.5-0.2366)^2 - 2phi(0.5-0.2366)(0.6032-0.3) + (0.6032-0.3)^2)
    # eps   = sqrt((0.7634-0.5)^2 - 2phi(0.7634-0.5)(0.3-0.1078) + (0.3-0.1078)^2)
    phi = 5 / 525 ** 0.5
    l1, u1 = wilson_interval(5, 10)
    l2, u2 = wilson_interval(3, 10)
    delta = ((0.5 - l1) ** 2 - 2 * phi * (0.5 - l1) * (u2 - 0.3) + (u2 - 0.3) ** 2) ** 0.5
    eps = ((u1 - 0.5) ** 2 - 2 * phi * (u1 - 0.5) * (0.3 - l2) + (0.3 - l2) ** 2) ** 0.5
    lo, hi = paired_difference_ci(a=2, b=3, c=1, d=4)
    assert abs(lo - (0.2 - delta)) < 1e-12 and abs(hi - (0.2 + eps)) < 1e-12
    # delta = sqrt(0.069385 - 0.034859 + 0.091942) = 0.35562; eps = sqrt(0.069385 - 0.022096 + 0.036941) = 0.29022
    assert abs(lo - (-0.1556)) < 1e-3 and abs(hi - 0.4902) < 1e-3


def test_paired_ci_n_zero_is_uninformative_and_negative_cells_raise():
    assert paired_difference_ci(a=0, b=0, c=0, d=0) == (-1.0, 1.0)
    with pytest.raises(ValueError, match="negative"):
        paired_difference_ci(a=1, b=-1, c=0, d=0)


def test_paired_ci_is_antisymmetric_in_the_arms():
    lo, hi = paired_difference_ci(a=2, b=3, c=1, d=4)
    lo2, hi2 = paired_difference_ci(a=2, b=1, c=3, d=4)
    assert abs(lo + hi2) < 1e-12 and abs(hi + lo2) < 1e-12


# --- technique-clustered interval -------------------------------------------------------------
# Cluster-robust variance of a pooled proportion with m clusters (y_j hits of n_j trials):
#   Var = m/(m-1) * sum_j (y_j - p n_j)^2 / N^2   (Cameron & Miller 2015, "A Practitioner's
#   Guide to Cluster-Robust Inference", J Hum Resour 50(2), the ratio-estimator form with the
#   m/(m-1) small-sample factor). The design effect deff = Var / (p(1-p)/N) deflates N to
#   N/deff and the Wilson interval is taken on that effective sample (Korn & Graubard 1998).

def test_clustered_interval_hand_computed_design_effect():
    # 3 techniques x 4 trials, hits (4, 0, 0): p = 1/3, N = 12.
    # residuals 8/3, -4/3, -4/3 -> squares sum 96/9; * (3/2) / 144 = 1/9 = Var.
    # binomial var = (1/3)(2/3)/12 = 1/54 -> deff = 6 -> n_eff = 2, hits_eff = 2/3.
    res = clustered_interval([(4, 4), (0, 4), (0, 4)])
    assert abs(res.deff - 6.0) < 1e-9 and abs(res.n_eff - 2.0) < 1e-9
    assert res.interval == pytest.approx(wilson_interval(2 / 3, 2), abs=1e-9)
    plain = wilson_interval(4, 12)
    assert res.interval[0] < plain[0] and res.interval[1] > plain[1]  # wider: the clusters disagree


def test_clustered_interval_never_narrower_than_wilson():
    # Perfectly homogeneous clusters (1/3 each) give deff < 1; it is floored at 1 so the
    # cluster-aware interval can never read tighter than the independent-trials one.
    res = clustered_interval([(1, 3), (1, 3), (1, 3)])
    assert res.deff == 1.0 and res.interval == pytest.approx(wilson_interval(3, 9), abs=1e-12)


def test_clustered_interval_single_cluster_has_no_estimate():
    res = clustered_interval([(2, 5)])
    assert res.interval is None and res.deff is None and "1 cluster" in res.reason


def test_clustered_interval_all_hits_or_none_falls_back_to_wilson():
    assert clustered_interval([(0, 3), (0, 3)]).interval == pytest.approx(wilson_interval(0, 6))
    assert clustered_interval([(3, 3), (3, 3)]).interval == pytest.approx(wilson_interval(6, 6))


# --- normal quantile + paired sample size ------------------------------------------------------

def test_norm_ppf_matches_tabulated_quantiles():
    # Standard normal quantiles to 6 decimals (any statistical table).
    assert abs(norm_ppf(0.975) - 1.959964) < 1e-6
    assert abs(norm_ppf(0.8) - 0.841621) < 1e-6
    assert abs(norm_ppf(0.95) - 1.644854) < 1e-6
    assert abs(norm_ppf(0.5)) < 1e-9
    assert abs(norm_ppf(0.025) + 1.959964) < 1e-6
    with pytest.raises(ValueError):
        norm_ppf(0.0)


def test_paired_sample_size_connor_1987_hand_computed():
    # Connor RJ, "Sample size for testing differences in proportions for the paired-sample
    # design", Biometrics 1987;43:207-211:
    #   n = [z_{a/2} sqrt(psi) + z_b sqrt(psi - delta^2)]^2 / delta^2,
    # psi = discordant proportion, delta = p12 - p21. With delta = 0.10, psi = 0.20:
    #   [1.959964*0.447214 + 0.841621*0.435890]^2 / 0.01 = 1.24340^2 / 0.01 = 154.6 -> 155 pairs.
    assert paired_sample_size(diff=0.10, discordant=0.20) == 155
    # The floor: every discordant pair falls the same way (psi = |delta|).
    #   [1.959964*0.316228 + 0.841621*0.3]^2 / 0.01 = (0.619799 + 0.252486)^2/0.01 = 76.09 -> 77.
    assert paired_sample_size(diff=0.10, discordant=None) == 77


def test_paired_sample_size_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        paired_sample_size(diff=0.0, discordant=0.2)
    with pytest.raises(ValueError):
        paired_sample_size(diff=0.3, discordant=0.2)   # psi < |delta| is impossible
    with pytest.raises(ValueError):
        paired_sample_size(diff=0.1, discordant=0.2, alpha=0.0)
