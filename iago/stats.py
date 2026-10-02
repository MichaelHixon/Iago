"""Statistical helpers for turning trial counts into defensible rates.

A bypass "rate" from a handful of trials is noisy — 1/3 and 30/90 are both 33%
but carry very different confidence. The Wilson score interval gives an honest
95% confidence bound on a binomial proportion, and (unlike the naive normal
approximation) it is small-sample safe: it never runs off the [0, 1] edge and it
does not collapse to a point at 0/N or N/N. That is what turns "we saw it bypass
a third of the time" into a finding you can defend in a report.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, log, sqrt


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact-binomial McNemar p-value on discordant-pair counts (b, c).

    Paired before/after data (each guarded trial has a raw twin) has one correct
    significance test: McNemar's, which throws away the concordant pairs and asks
    how surprising the discordant split (b vs c) is under the null that a discordant
    pair is equally likely to fall either way (p = 0.5). This is the *exact* binomial
    form, not the chi-square approximation, so it stays valid when the discordant
    total is small or entirely one-sided — e.g. 37 vs 0, where the asymptotic form
    has no business being trusted near the boundary.

    Returns a p in [0, 1]; 1.0 when there are no discordant pairs (n == 0).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    lower_tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * lower_tail)


def rose_measurably(new: tuple[int, int], old: tuple[int, int]) -> bool:
    """True when `new` (hits, total) sits measurably above `old`: the 95% Wilson intervals do not
    overlap, upward. A higher point rate inside overlapping intervals is noise, not a change."""
    return wilson_interval(*new)[0] > wilson_interval(*old)[1]


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% (z=1.96) Wilson score interval for a binomial proportion hits/total.

    Returns (low, high), each clamped to [0, 1].

    No data is NOT certainty: total == 0 returns the uninformative (0.0, 1.0) so an empty or
    all-error denominator can never render as a confident "0%–0%" (the exit-0-measured-nothing
    class, ISC-31). Out-of-range inputs raise rather than producing a NaN.
    """
    if total < 0 or hits < 0 or hits > total:
        raise ValueError(f"wilson_interval: hits/total out of range ({hits}/{total})")
    if total == 0:
        return (0.0, 1.0)
    phat = hits / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (phat + z2 / (2 * total)) / denom
    margin = (z / denom) * sqrt(phat * (1 - phat) / total + z2 / (4 * total * total))
    return (max(0.0, center - margin), min(1.0, center + margin))


# --- paired difference ----------------------------------------------------------------------------

def newcombe_diff_ci(x1: int | float, n1: int, x2: int | float, n2: int, *, phi: float = 0.0,
                     z: float = 1.96) -> tuple[float, float]:
    """95% hybrid-score interval on p1 - p2 (Newcombe 1998a, method 10): each proportion's Wilson
    half-widths are combined in quadrature, so the interval inherits Wilson's small-sample
    behaviour (never outside [-1, 1], never a zero-width point at 0/n or n/n). `phi` is the
    correlation between the two arms: 0 for independent samples; `paired_difference_ci` supplies
    the phi of a paired 2x2 table. Reference (Newcombe 1998a, Table II): 56/70 vs 48/80 ->
    (0.0524, 0.3339); 9/10 vs 3/10 -> (0.1705, 0.8090)."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson_interval(x1, n1, z)
    l2, u2 = wilson_interval(x2, n2, z)
    lo_w, hi_w = p1 - l1, u2 - p2         # the half-widths that pull the LOWER limit down
    hi_w1, lo_w2 = u1 - p1, p2 - l2       # the half-widths that push the UPPER limit up
    delta = sqrt(max(0.0, lo_w ** 2 - 2 * phi * lo_w * hi_w + hi_w ** 2))
    eps = sqrt(max(0.0, hi_w1 ** 2 - 2 * phi * hi_w1 * lo_w2 + lo_w2 ** 2))
    d = p1 - p2
    return (max(-1.0, d - delta), min(1.0, d + eps))


def paired_difference_ci(*, a: int, b: int, c: int, d: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI on the paired difference p1 - p2 from a 2x2 table of matched trials: a = both
    arms bypassed, b = arm 1 only, c = arm 2 only, d = neither (Newcombe 1998b, method 10).

    Why this method: the delta / compare pairs are small (tens of trials) and often one-sided
    (c = 0 is the common case for a guard that only ever helps), where the Wald interval on
    (b - c)/n collapses to zero width and the independent-arms interval ignores that the same
    technique/objective/seed fired on both sides. Newcombe's paired method keeps the Wilson
    small-sample behaviour and reduces the width by the arms' correlation phi, which is what the
    pairing buys. It is closed-form (no root-finding) and needs only `math`. Tango's (1998)
    score interval is the usual alternative; it needs an iterative solve for no gain at these
    sizes. phi = (ad - bc)/sqrt((a+b)(c+d)(a+c)(b+d)), taken as 0 when any margin is zero
    (Newcombe's rule; the interval then equals the independent one). The sign of phi is kept: a
    negative correlation widens the interval, which is the conservative direction.

    n == 0 returns the uninformative (-1.0, 1.0); a negative cell raises."""
    if min(a, b, c, d) < 0:
        raise ValueError(f"paired_difference_ci: negative cell count in ({a}, {b}, {c}, {d})")
    n = a + b + c + d
    if n == 0:
        return (-1.0, 1.0)
    margins = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if margins == 0 else (a * d - b * c) / sqrt(margins)
    return newcombe_diff_ci(a + b, n, a + c, n, phi=phi, z=z)


# --- technique-clustered interval -----------------------------------------------------------------

@dataclass(frozen=True)
class ClusteredInterval:
    """Result of `clustered_interval`: `interval` is None when it cannot be estimated (`reason`
    says why); `deff` is the design effect actually applied (floored at 1), `n_eff` the deflated
    sample size the Wilson interval was taken on."""

    interval: tuple[float, float] | None
    deff: float | None
    n_eff: float | None
    reason: str | None


def clustered_interval(clusters: list[tuple[int, int]], z: float = 1.96) -> ClusteredInterval:
    """Cluster-aware 95% interval on a pooled proportion whose trials are grouped — here, every
    trial of one technique is one cluster, because repeated trials of the same technique are not
    independent draws (same prompt, same objective, seeds a step apart).

    Variance of the pooled rate with m clusters of (y_j hits, n_j trials), N = sum n_j, p = sum
    y_j / N: Var = m/(m-1) * sum_j (y_j - p n_j)^2 / N^2 — the cluster-robust ratio-estimator
    variance with the usual m/(m-1) small-sample factor (Cameron & Miller 2015, J Hum Resour
    50(2) § VI). The design effect deff = Var / (p(1-p)/N) deflates N to N/deff and the Wilson
    interval is taken on that effective sample (Korn & Graubard 1998, Survey Methodology 24(2)),
    so the result keeps Wilson's edge behaviour instead of a Wald interval that collapses at 0%.
    deff is floored at 1: an under-dispersed sample is sampling luck, not evidence the trials
    are MORE independent than binomial, so the cluster interval is never narrower than Wilson.
    One cluster gives no estimate (m - 1 = 0) and a rate of exactly 0 or 1 has no between-cluster
    variance to measure, so both fall back to the plain Wilson interval with `reason` set."""
    m = len(clusters)
    hits = sum(y for y, _ in clusters)
    total = sum(n for _, n in clusters)
    if total == 0:
        return ClusteredInterval(None, None, None, "no trials")
    if m < 2:
        return ClusteredInterval(None, None, None, f"{m} cluster: between-technique variance needs >= 2")
    p = hits / total
    binom_var = p * (1 - p) / total
    if binom_var == 0.0:
        return ClusteredInterval(wilson_interval(hits, total, z), 1.0, float(total),
                                 "rate at 0% or 100%: no between-cluster variance to estimate")
    cluster_var = (m / (m - 1)) * sum((y - p * n) ** 2 for y, n in clusters) / (total * total)
    deff = max(1.0, cluster_var / binom_var)
    n_eff = total / deff
    return ClusteredInterval(wilson_interval(hits * n_eff / total, n_eff, z), deff, n_eff, None)


# --- paired sample size ---------------------------------------------------------------------------

def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, |rel err| < 1.15e-9), so the
    power helper needs no scipy. Raises for p outside (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf: p must be in (0, 1), got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425
    if p < plow:
        q = sqrt(-2 * log(p))
        return ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if p > 1 - plow:
        q = sqrt(-2 * log(1 - p))
        return -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                 / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    q = p - 0.5
    r = q * q
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


def paired_sample_size(*, diff: float, discordant: float | None, alpha: float = 0.05,
                       power: float = 0.8) -> int:
    """Pairs needed for McNemar's test to detect a paired difference `diff` = p12 - p21 at
    two-sided `alpha` and `power`, given `discordant` = p12 + p21, the proportion of pairs whose
    verdicts differ (Connor 1987, Biometrics 43:207-211):

        n = [z_{alpha/2} sqrt(psi) + z_{power} sqrt(psi - diff^2)]^2 / diff^2

    `discordant=None` uses the smallest possible psi = |diff| (every discordant pair falls the
    same way), which is a LOWER BOUND on the pairs needed: real runs have some pairs going the
    other way and need more. Rounds up to a whole pair."""
    if diff == 0:
        raise ValueError("paired_sample_size: diff must be non-zero")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("paired_sample_size: alpha and power must be in (0, 1)")
    psi = abs(diff) if discordant is None else discordant
    if psi < abs(diff) or psi > 1:
        raise ValueError(f"paired_sample_size: discordant proportion {psi} must satisfy "
                         f"|diff| <= discordant <= 1 (|diff| = {abs(diff)})")
    z_a = norm_ppf(1 - alpha / 2)
    z_b = norm_ppf(power)
    n = (z_a * sqrt(psi) + z_b * sqrt(psi - diff * diff)) ** 2 / (diff * diff)
    return ceil(n - 1e-9)
