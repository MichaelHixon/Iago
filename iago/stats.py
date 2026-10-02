"""Statistical helpers for turning trial counts into defensible rates.

A bypass "rate" from a handful of trials is noisy — 1/3 and 30/90 are both 33%
but carry very different confidence. The Wilson score interval gives an honest
95% confidence bound on a binomial proportion, and (unlike the naive normal
approximation) it is small-sample safe: it never runs off the [0, 1] edge and it
does not collapse to a point at 0/N or N/N. That is what turns "we saw it bypass
a third of the time" into a finding you can defend in a report.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil, comb, exp, lgamma, log, sqrt


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


def wilson_interval(hits: float, total: float, z: float = 1.96) -> tuple[float, float]:
    """95% (z=1.96) Wilson score interval for a binomial proportion hits/total. Counts may be
    fractional: the clustered and paired callers pass design-effect-scaled effective counts.

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

def paired_counts(pairs: Iterable[tuple[bool, bool]]) -> tuple[int, int, int, int]:
    """The 2x2 table (a, b, c, d) of matched binary outcomes (arm 1 hit?, arm 2 hit?): a = both,
    b = arm 1 only, c = arm 2 only, d = neither — the cells `paired_difference_ci` takes, and
    the discordant (b, c) `mcnemar_exact_p` takes."""
    a = b = c = d = 0
    for hit1, hit2 in pairs:
        if hit1 and hit2:
            a += 1
        elif hit1:
            b += 1
        elif hit2:
            c += 1
        else:
            d += 1
    return a, b, c, d


def newcombe_diff_ci(x1: int | float, n1: int, x2: int | float, n2: int, *, phi: float = 0.0,
                     z: float = 1.96) -> tuple[float, float]:
    """95% hybrid-score interval on p1 - p2 (Newcombe 1998a, method 10): each proportion's Wilson
    half-widths are combined in quadrature, so the interval inherits Wilson's small-sample
    behaviour (never outside [-1, 1], never a zero-width point at 0/n or n/n). `phi` is the
    correlation between the two arms: 0 for independent samples (two models, two runs: trial i
    on one is not trial i on the other); `paired_difference_ci` supplies the continuity-corrected
    phi of a paired 2x2 table. Reference (Newcombe 1998a, Table II): 56/70 vs 48/80 ->
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


def paired_phi(a: int, b: int, c: int, d: int) -> float:
    """Newcombe's continuity-corrected phi for a paired 2x2 table (Newcombe 1998b, Stat Med
    17:2635-2650, § 5 method 10, p. 2639). With the paper's cells e, f, g, h = a, b, c, d:
    phi = (eh - fg) / sqrt((e+f)(g+h)(e+g)(f+h)), 0 when that denominator is 0, except that the
    numerator is replaced by max(eh - fg - n/2, 0) when eh > fg. A negative eh - fg is left
    uncorrected, exactly as published; the correction only ever pulls a positive phi towards 0."""
    n = a + b + c + d
    margins = (a + b) * (c + d) * (a + c) * (b + d)
    if margins == 0:
        return 0.0
    num = float(a * d - b * c)
    if num > 0:
        num = max(num - n / 2, 0.0)
    return num / sqrt(margins)


def paired_difference_ci(*, a: int, b: int, c: int, d: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI on the paired difference p1 - p2 from a 2x2 table of matched trials: a = both
    arms bypassed, b = arm 1 only, c = arm 2 only, d = neither (Newcombe 1998b, method 10:
    "score intervals with continuity corrected phi").

    Why this method: the delta pairs are small (tens of trials) and often one-sided (c = 0 is the
    common case for a guard that only ever helps). The Wald interval on (b - c)/n degenerates to
    zero width when b = c = 0 or when every pair is discordant the same way (b = n or c = n),
    and the independent-arms interval ignores that the same technique/objective/seed fired on
    both sides. Newcombe's paired method keeps the Wilson small-sample behaviour and reduces the
    width by the arms' correlation phi, which is what the pairing buys. It is closed-form (no
    root-finding) and needs only `math`. Tango's (1998) score interval is the usual alternative;
    it needs an iterative solve for no gain at these sizes.

    phi is `paired_phi`: Newcombe's correlation with his continuity correction, which is what
    keeps the interval from collapsing. Without it a table with no discordant pairs and a = d
    (e.g. 5/0/0/5) has phi = 1 and a zero-width interval; the paper's § 6 (iv)(c) notes methods 8
    and 9 do exactly that and method 10 does not. With it, n > 0 never yields a zero width. A
    negative phi widens the interval, the conservative direction. Table III of the paper is the
    reference (see tests/test_isc70_stats.py).

    n == 0 returns the uninformative (-1.0, 1.0); a negative cell raises."""
    if min(a, b, c, d) < 0:
        raise ValueError(f"paired_difference_ci: negative cell count in ({a}, {b}, {c}, {d})")
    n = a + b + c + d
    if n == 0:
        return (-1.0, 1.0)
    return newcombe_diff_ci(a + b, n, a + c, n, phi=paired_phi(a, b, c, d), z=z)


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


def clustered_interval(clusters: list[tuple[int, int]], z: float | None = None) -> ClusteredInterval:
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

    The quantile is t(m - 1) at 97.5%, not 1.96, unless `z` is given: with few clusters the
    cluster-robust variance is itself a noisy estimate and the normal quantile undercovers
    (Cameron & Miller 2015 § VI.A recommend T(G - 1) critical values; Korn & Graubard 1998 use the
    same degrees of freedom). At m = 3 that is 4.30, at m = 10 2.26, and it reaches 1.96 only in
    the limit, so a category with a handful of techniques gets the wide interval its evidence
    supports instead of a liberal one.

    `interval` is None, with `reason`, in the three cases where the estimator does not exist:
    no trials; one cluster (m - 1 = 0); a rate of exactly 0 or 1, which has no between-cluster
    variance to measure. The caller renders the reason and points at the plain Wilson interval,
    which is what the "clustered" column would otherwise silently have been."""
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
        return ClusteredInterval(None, None, None,
                                 f"rate at {'0' if hits == 0 else '100'}%: no between-technique variance")
    cluster_var = (m / (m - 1)) * sum((y - p * n) ** 2 for y, n in clusters) / (total * total)
    deff = max(1.0, cluster_var / binom_var)
    n_eff = total / deff
    q = t_ppf(0.975, m - 1) if z is None else z
    return ClusteredInterval(wilson_interval(hits * n_eff / total, n_eff, q), deff, n_eff, None)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta I_x(a, b) (Lentz's method, as in
    Numerical Recipes § 6.4 `betacf`)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via the continued fraction, using the symmetry
    I_x(a, b) = 1 - I_{1-x}(b, a) where the fraction converges faster."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Student-t CDF with `df` degrees of freedom: for t >= 0, 1 - I_x(df/2, 1/2) / 2 with
    x = df / (df + t^2) (Abramowitz & Stegun 26.7.1); mirrored for t < 0."""
    if df <= 0:
        raise ValueError(f"t_cdf: df must be positive, got {df}")
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_ppf(p: float, df: float) -> float:
    """Inverse Student-t CDF (the t critical value) by bisection on `t_cdf`, so the clustered
    interval needs no scipy. Checked against the standard table (NIST/SEMATECH e-Handbook of
    Statistical Methods § 1.3.6.7.2) in tests. Raises for p outside (0, 1) or df <= 0."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"t_ppf: p must be in (0, 1), got {p}")
    if df <= 0:
        raise ValueError(f"t_ppf: df must be positive, got {df}")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:        # bracket: df = 1 at p = 0.9995 needs t ~ 636
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


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
