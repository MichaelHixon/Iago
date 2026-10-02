"""ISC-70 — paired and clustered statistics (`iago/stats.py`).

Reference values are hand-computed from the published formulas, and where a published table
exists it is cited beside the number. Every test here is a falsifier for a specific sentence in
a docstring: the paired CI reproduces Newcombe's own Table III, reduces to his unpaired
hybrid-score interval when the correlation term is zero, and is strictly narrower than that
interval when the pairs agree — which is exactly what an unpaired interval cannot do.
"""

import pytest

from iago.stats import (clustered_interval, newcombe_diff_ci, norm_ppf, paired_difference_ci,
                        paired_phi, paired_sample_size, t_cdf, t_ppf, wilson_interval)


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
# based on paired data", Stat Med 1998;17:2635-2650. Method 10 (§ 5, p. 2639): the hybrid score
# interval with phi = (eh - fg)/sqrt((e+f)(g+h)(e+g)(f+h)), 0 when that denominator is 0, and the
# numerator replaced by max(eh - fg - n/2, 0) when eh > fg. Cells e, f, g, h are our a, b, c, d.
# The anchors below are the method-10 rows of the paper's Table III (pp. 2641-2642), quoted to
# the 4 decimals printed there; the paper uses z = 1.96 as we do. ratesci::pairbinci(method =
# "MOVER_newc") reproduces the same rows, but the paper is the reference, not the package.

_TABLE_III_METHOD_10 = {
    (36, 12, 2, 0): (0.0569, 0.3404),
    (20, 12, 2, 16): (0.0562, 0.3292),
    (18, 12, 2, 18): (0.0562, 0.3290),
    (36, 14, 0, 0): (0.1528, 0.4167),
    (35, 14, 0, 1): (0.1461, 0.4175),
    (18, 14, 0, 18): (0.1441, 0.3963),
    (2, 97, 1, 0): (0.8721, 0.9854),
    (1, 97, 1, 1): (0.8736, 0.9850),
    (0, 29, 1, 0): (0.6666, 0.9882),
    (2, 98, 0, 0): (0.9178, 0.9945),
    (1, 98, 0, 1): (0.9171, 0.9916),
    (0, 30, 0, 0): (0.8395, 1.0),
    (54, 0, 0, 0): (-0.0664, 0.0664),
    (53, 0, 0, 1): (-0.0729, 0.0729),
    (30, 0, 0, 24): (-0.0358, 0.0358),
    (29, 0, 0, 25): (-0.0354, 0.0354),
    (28, 0, 0, 26): (-0.0352, 0.0352),
    (27, 0, 0, 27): (-0.0351, 0.0351),
}


@pytest.mark.parametrize("cells,published", sorted(_TABLE_III_METHOD_10.items()))
def test_paired_ci_matches_newcombe_1998b_table_iii_method_10(cells, published):
    a, b, c, d = cells
    lo, hi = paired_difference_ci(a=a, b=b, c=c, d=d)
    assert abs(lo - published[0]) < 1e-4 and abs(hi - published[1]) < 1e-4, (cells, (lo, hi))


def test_paired_phi_follows_the_published_continuity_correction():
    # (36, 12, 2, 0): eh - fg = 0 - 24 < 0 -> uncorrected, negative. (20, 12, 2, 16): eh - fg = 296,
    # n/2 = 25 -> 271 / sqrt(32*18*22*28). (1, 0, 0, 1): eh - fg = 1 <= n/2 = 1 -> 0, not 1.
    assert paired_phi(36, 12, 2, 0) == pytest.approx(-24 / (48 * 2 * 38 * 12) ** 0.5)
    assert paired_phi(20, 12, 2, 16) == pytest.approx(271 / (32 * 18 * 22 * 28) ** 0.5)
    assert paired_phi(1, 0, 0, 1) == 0.0
    assert paired_phi(0, 3, 0, 0) == 0.0          # a zero margin -> 0 (the paper's rule)


def test_paired_ci_never_has_zero_width_when_n_is_positive():
    """The uncorrected phi is exactly 1 at a = d with b = c = 0, which made (5, 0, 0, 5) and
    (20, 0, 0, 20) collapse to (0.0, 0.0). Newcombe § 6 (iv)(c): methods 8 and 9 produce a
    zero-width interval there; method 10 does not — Table III's (27, 0, 0, 27) row is
    (-0.0351, 0.0351)."""
    for cells in ((5, 0, 0, 5), (20, 0, 0, 20), (1, 0, 0, 1), (1, 0, 0, 0), (0, 0, 0, 1),
                  (3, 0, 0, 1), (0, 2, 0, 0), (0, 0, 2, 0)):
        a, b, c, d = cells
        lo, hi = paired_difference_ci(a=a, b=b, c=c, d=d)
        assert hi - lo > 1e-6, (cells, (lo, hi))
        assert lo <= (b - c) / (a + b + c + d) <= hi


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
    # The quantile is t(m - 1 = 2) at 97.5% = 4.303, not 1.96.
    res = clustered_interval([(4, 4), (0, 4), (0, 4)])
    assert abs(res.deff - 6.0) < 1e-9 and abs(res.n_eff - 2.0) < 1e-9
    assert res.interval == pytest.approx(wilson_interval(2 / 3, 2, z=t_ppf(0.975, 2)), abs=1e-9)
    assert res.interval != pytest.approx(wilson_interval(2 / 3, 2, z=1.96), abs=1e-3)
    plain = wilson_interval(4, 12)
    assert res.interval[0] < plain[0] and res.interval[1] > plain[1]  # wider: the clusters disagree


def test_clustered_interval_uses_t_on_cluster_count_minus_one():
    """Cameron & Miller 2015 § VI.A: with few clusters the normal quantile undercovers; the t(G-1)
    quantile is the standard repair. Ten techniques -> t(9) = 2.262; an explicit z overrides."""
    clusters = [(i % 3, 4) for i in range(10)]
    res = clustered_interval(clusters)
    hits, total = sum(y for y, _ in clusters), 40
    assert res.interval == pytest.approx(
        wilson_interval(hits * res.n_eff / total, res.n_eff, z=t_ppf(0.975, 9)), abs=1e-12)
    forced = clustered_interval(clusters, z=1.96)
    assert forced.interval == pytest.approx(
        wilson_interval(hits * res.n_eff / total, res.n_eff, z=1.96), abs=1e-12)
    assert forced.interval[1] - forced.interval[0] < res.interval[1] - res.interval[0]


def test_clustered_interval_never_narrower_than_wilson():
    # Perfectly homogeneous clusters (1/3 each) give deff < 1; it is floored at 1 so the
    # cluster-aware interval can never read tighter than the independent-trials one (at the same
    # quantile — the t(2) quantile then makes it wider still).
    res = clustered_interval([(1, 3), (1, 3), (1, 3)], z=1.96)
    assert res.deff == 1.0 and res.interval == pytest.approx(wilson_interval(3, 9), abs=1e-12)


def test_clustered_interval_single_cluster_has_no_estimate():
    res = clustered_interval([(2, 5)])
    assert res.interval is None and res.deff is None and "1 cluster" in res.reason


def test_clustered_interval_all_hits_or_none_has_no_estimate_and_says_why():
    """A 0% or 100% category has no between-technique variance, so there is no clustered
    estimate — the result says so instead of quietly handing back the plain Wilson interval
    labelled as if a design effect of 1.0 had been measured."""
    none = clustered_interval([(0, 3), (0, 3)])
    assert none.interval is None and none.deff is None and "0%" in none.reason
    full = clustered_interval([(3, 3), (3, 3)])
    assert full.interval is None and full.deff is None and "100%" in full.reason


# --- Student t quantile -----------------------------------------------------------------------
# Critical values t_{0.975, df} from the standard table (NIST/SEMATECH e-Handbook of Statistical
# Methods § 1.3.6.7.2, "Critical Values of the Student's t Distribution"), to 3 decimals.

@pytest.mark.parametrize("df,critical", [(1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571),
                                         (9, 2.262), (10, 2.228), (20, 2.086), (30, 2.042),
                                         (60, 2.000), (120, 1.980)])
def test_t_ppf_matches_the_standard_table(df, critical):
    assert abs(t_ppf(0.975, df) - critical) < 5e-4


def test_t_ppf_limits_and_symmetry():
    assert abs(t_ppf(0.975, 1e6) - 1.959964) < 1e-5     # -> the normal quantile
    assert abs(t_ppf(0.9995, 1) - 636.619) < 1e-2         # the table's df = 1, 99.95% entry
    assert t_ppf(0.5, 7) == 0.0 and t_ppf(0.05, 5) == pytest.approx(-t_ppf(0.95, 5))
    assert abs(t_cdf(2.228, 10) - 0.975) < 1e-4
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            t_ppf(bad, 3)
    with pytest.raises(ValueError):
        t_ppf(0.9, 0)


# --- normal quantile + paired sample size ------------------------------------------------------

def test_norm_ppf_matches_tabulated_quantiles():
    # Standard normal quantiles to 6 decimals (any statistical table).
    assert abs(norm_ppf(0.975) - 1.959964) < 1e-6
    assert abs(norm_ppf(0.8) - 0.841621) < 1e-6
    assert abs(norm_ppf(0.95) - 1.644854) < 1e-6
    assert abs(norm_ppf(0.5)) < 1e-9
    assert abs(norm_ppf(0.025) + 1.959964) < 1e-6
    # The two tail branches (p < 0.02425 and p > 1 - 0.02425) of Acklam's approximation.
    assert abs(norm_ppf(0.995) - 2.575829) < 1e-6
    assert abs(norm_ppf(0.001) + 3.090232) < 1e-6
    assert abs(norm_ppf(0.9999) - 3.719016) < 1e-6
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
    # alpha = 0.01: z_{0.005} = 2.575829 ->
    #   [2.575829*0.447214 + 0.841621*0.435890]^2 / 0.01 = 1.518801^2 / 0.01 = 230.68 -> 231 pairs.
    assert paired_sample_size(diff=0.10, discordant=0.20, alpha=0.01) == 231


def test_paired_sample_size_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        paired_sample_size(diff=0.0, discordant=0.2)
    with pytest.raises(ValueError):
        paired_sample_size(diff=0.3, discordant=0.2)   # psi < |delta| is impossible
    with pytest.raises(ValueError):
        paired_sample_size(diff=0.1, discordant=0.2, alpha=0.0)
