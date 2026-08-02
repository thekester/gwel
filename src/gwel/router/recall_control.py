"""Thresholds chosen by a recall guarantee rather than by minimising cost.

Our cost-minimising tuner degenerates: on the training fold it discovers that
never escalating is cheapest and returns a threshold no query can cross. That
is a real failure mode, and Ruan et al. (arXiv 2607.06503) give the fix. They
set each gate so that an exact Clopper-Pearson lower bound on the survival rate
of *successful* cases meets a user-chosen recall target, then maximise savings
subject to that constraint. The guarantee is distribution-free and holds with
high confidence over the calibration draw.

Transposed here: escalation errors are asymmetric. Declining to escalate a
query that escalation would have fixed destroys accuracy, while escalating an
unfixable one only wastes compute. A recall-controlled threshold makes the
first kind of error a budget the operator sets, instead of a side effect of
whichever fold the tuner saw.
"""

from dataclasses import dataclass

import numpy as np


def _beta_ppf_bisect(alpha: float, a: float, b: float, *, iterations: int = 200) -> float:
    """Beta quantile by bisection on the regularised incomplete beta function.

    A dependency-free fallback so the core package does not require scipy.
    """
    from math import lgamma

    def betainc(x: float) -> float:
        # Continued-fraction-free series evaluation, adequate for our sizes.
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        log_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
        total = 0.0
        term = 1.0
        for index in range(2000):
            if index > 0:
                term *= (index - b) * x / index
            total += term / (a + index)
            if abs(term / (a + index)) < 1e-15 * max(abs(total), 1e-300):
                break
        return float(np.exp(a * np.log(x) - log_beta) * total)

    low, high = 0.0, 1.0
    for _ in range(iterations):
        mid = (low + high) / 2.0
        if betainc(mid) < alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def lower_bound(successes: int, trials: int, *, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson lower bound on a binomial proportion.

    No normal approximation, so the bound stays valid at the small counts a
    calibration split provides. Uses scipy's Beta quantile when available
    and falls back to bisection otherwise.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0 or successes == 0:
        return 0.0
    if successes == trials:
        return float(alpha ** (1.0 / trials))
    try:
        from scipy.stats import beta

        return float(beta.ppf(alpha, successes, trials - successes + 1))
    except ImportError:
        return _beta_ppf_bisect(alpha, successes, trials - successes + 1)


def certifiable_recall(n_positive: int, *, alpha: float = 0.05) -> float:
    """Highest recall target this many positive examples can certify.

    From the same paper: with ``n`` positives, a one-sided certificate at
    confidence ``1 - alpha`` supports targets only up to ``alpha ** (1/n)``.
    Worth checking before promising a guarantee the data cannot support.
    """
    if n_positive <= 0:
        return 0.0
    return float(alpha ** (1.0 / n_positive))


@dataclass(frozen=True)
class RecallControlledThreshold:
    """Escalation threshold meeting a recall floor on recoverable queries."""

    threshold: float
    achieved_recall: float
    certified_recall: float
    escalation_rate: float

    def should_escalate(self, score: float) -> bool:
        """Escalate when uncertainty reaches the threshold."""
        return score >= self.threshold


def fit_recall_controlled(
    scores: np.ndarray,
    recoverable: np.ndarray,
    *,
    target_recall: float = 0.9,
    alpha: float = 0.05,
) -> RecallControlledThreshold:
    """Highest threshold whose certified recall on recoverable queries meets the target.

    ``scores`` should increase with uncertainty, so escalating means
    ``score >= threshold``. Raising the threshold escalates fewer queries
    and saves more, so the search takes the largest threshold that still clears the
    Clopper-Pearson floor, maximum saving subject to the guarantee.
    """
    scores = np.asarray(scores, dtype=np.float64)
    recoverable = np.asarray(recoverable, dtype=bool)
    if scores.shape != recoverable.shape:
        raise ValueError("scores and recoverable must have the same shape")
    if not recoverable.any():
        raise ValueError("no recoverable examples to control recall over")

    positives = scores[recoverable]
    n_positive = len(positives)
    best = RecallControlledThreshold(
        threshold=float(scores.min()),
        achieved_recall=1.0,
        certified_recall=lower_bound(n_positive, n_positive, alpha=alpha),
        escalation_rate=1.0,
    )

    for candidate in np.unique(scores):
        retained = int((positives >= candidate).sum())
        certified = lower_bound(retained, n_positive, alpha=alpha)
        if certified < target_recall:
            continue
        rate = float((scores >= candidate).mean())
        if rate <= best.escalation_rate:
            best = RecallControlledThreshold(
                threshold=float(candidate),
                achieved_recall=retained / n_positive,
                certified_recall=certified,
                escalation_rate=rate,
            )
    return best
