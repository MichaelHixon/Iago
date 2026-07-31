"""Statistical helpers for turning trial counts into defensible rates.

A bypass "rate" from a handful of trials is noisy — 1/3 and 30/90 are both 33%
but carry very different confidence. The Wilson score interval gives an honest
95% confidence bound on a binomial proportion, and (unlike the naive normal
approximation) it is small-sample safe: it never runs off the [0, 1] edge and it
does not collapse to a point at 0/N or N/N. That is what turns "we saw it bypass
a third of the time" into a finding you can defend in a report.
"""

from __future__ import annotations

from math import sqrt


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
