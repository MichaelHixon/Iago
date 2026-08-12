"""Statistical helpers for turning trial counts into defensible rates.

A bypass "rate" from a handful of trials is noisy — 1/3 and 30/90 are both 33%
but carry very different confidence. The Wilson score interval gives an honest
95% confidence bound on a binomial proportion, and (unlike the naive normal
approximation) it is small-sample safe: it never runs off the [0, 1] edge and it
does not collapse to a point at 0/N or N/N. That is what turns "we saw it bypass
a third of the time" into a finding you can defend in a report.
"""

from __future__ import annotations

from math import comb, sqrt


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


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% (z=1.96) Wilson score interval for a binomial proportion hits/total.

    Returns (low, high), each clamped to [0, 1]. total <= 0 => (0.0, 0.0).
    """
    if total <= 0:
        return (0.0, 0.0)
    phat = hits / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (phat + z2 / (2 * total)) / denom
    margin = (z / denom) * sqrt(phat * (1 - phat) / total + z2 / (4 * total * total))
    return (max(0.0, center - margin), min(1.0, center + margin))
