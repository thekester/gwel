"""Family-wise error control over the paper's paired comparisons.

This project reports a dozen paired differences with bootstrap intervals and,
until now, no correction for having asked a dozen questions. At the nominal 5%
level, twelve independent tests produce a false positive with probability
``1 - 0.95**12 = 46%``, so "the interval excludes zero" is a much weaker
statement across a family than within one test.

Two pieces are needed. A bootstrap gives an interval rather than a p-value, so
:func:`bootstrap_p_value` converts one by the standard percentile argument: the
two-sided p-value is twice the fraction of resamples that land on the wrong side
of zero. Then :func:`holm_bonferroni` controls the family-wise error rate.

Holm rather than plain Bonferroni because Holm is uniformly more powerful and
needs no independence assumption, which matters here: our tests reuse the same
examples and are strongly dependent.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def bootstrap_p_value(
    differences: Sequence[float],
    *,
    resamples: int = 10000,
    seed: int = 1234,
) -> float:
    """Two-sided p-value for a paired mean difference, by the percentile method.

    Resamples the paired differences and measures how often the resampled mean
    falls on the opposite side of zero from the observed mean. The result is
    clipped away from exactly zero, since a bootstrap with ``B`` resamples
    cannot resolve a p-value below ``1/B``.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one difference is required")

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    observed = float(values.mean())
    if observed == 0.0:
        return 1.0
    wrong_side = float((draws <= 0).mean() if observed > 0 else (draws >= 0).mean())
    return float(min(1.0, max(2.0 * wrong_side, 1.0 / resamples)))


@dataclass(frozen=True)
class Corrected:
    """One test's verdict before and after family-wise correction."""

    name: str
    p_value: float
    adjusted: float
    survives: bool


def holm_bonferroni(
    tests: Sequence[tuple[str, float]], *, alpha: float = 0.05
) -> list[Corrected]:
    """Holm's step-down procedure, controlling the family-wise error rate.

    Sorts p-values ascending and compares the ``i``-th against
    ``alpha / (m - i)``. The first failure stops the procedure: everything from
    there on is rejected too, which is what makes the guarantee hold rather than
    testing each in isolation.

    Returns one :class:`Corrected` per test in the input order, carrying the
    monotone adjusted p-value so a reader can apply their own threshold.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not tests:
        return []

    order = sorted(range(len(tests)), key=lambda i: tests[i][1])
    total = len(tests)
    adjusted = [0.0] * total
    running = 0.0
    still_rejecting = True
    verdicts = [False] * total

    for rank, index in enumerate(order):
        p = tests[index][1]
        scaled = min(1.0, p * (total - rank))
        # Adjusted p-values must be non-decreasing along the sorted order.
        running = max(running, scaled)
        adjusted[index] = running
        if still_rejecting and running <= alpha:
            verdicts[index] = True
        else:
            still_rejecting = False

    return [
        Corrected(name=name, p_value=p, adjusted=adjusted[i], survives=verdicts[i])
        for i, (name, p) in enumerate(tests)
    ]


def family_wise_error(count: int, *, alpha: float = 0.05) -> float:
    """Probability of at least one false positive among ``count`` tests.

    Reported so the cost of not correcting is a number rather than a worry.
    Assumes independence, which is the optimistic case here.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    return float(1.0 - (1.0 - alpha) ** count)
