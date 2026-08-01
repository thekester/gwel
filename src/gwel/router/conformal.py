"""Split-conformal thresholds for a three-way answer / escalate / abstain policy.

Tayebati et al. (arXiv 2502.06884) partition queries into three regimes with
two conformal thresholds on a nonconformity score: single prediction below the
first, a prediction *set* between them, and abstention above the second. The
middle regime is where the model is unsure but not hopeless.

For a vision-language system that can look again, that middle regime should be
an escalation rather than a hedged answer. Substituting it gives a policy with
the same distribution-free coverage guarantee, but which spends compute instead
of widening a prediction set.

The guarantee is the standard split-conformal one: with a calibration set of
size n drawn exchangeably with the test data, thresholding at the
``ceil((n+1)(1-alpha))/n`` empirical quantile bounds the miscoverage rate by
``alpha``, with no assumption on the score's distribution.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Regime(StrEnum):
    """What the policy does with a query."""

    ANSWER = "answer"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The finite-sample-corrected ``1 - alpha`` quantile of calibration scores.

    Uses ``ceil((n + 1)(1 - alpha)) / n`` rather than the plain empirical
    quantile; the correction is what makes the coverage bound exact rather than
    asymptotic.
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0:
        raise ValueError("at least one calibration score is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    n = array.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float(array.max())
    return float(np.sort(array)[rank - 1])


@dataclass(frozen=True)
class ThreeWayConformal:
    """Two thresholds partitioning nonconformity into three regimes."""

    answer_threshold: float
    abstain_threshold: float

    def __post_init__(self) -> None:
        if self.answer_threshold > self.abstain_threshold:
            raise ValueError("answer_threshold must not exceed abstain_threshold")

    def regime(self, score: float) -> Regime:
        """Classify one nonconformity score."""
        if score <= self.answer_threshold:
            return Regime.ANSWER
        if score <= self.abstain_threshold:
            return Regime.ESCALATE
        return Regime.ABSTAIN

    def regimes(self, scores: Sequence[float]) -> list[Regime]:
        return [self.regime(float(s)) for s in scores]


def fit_three_way(
    calibration_scores: Sequence[float],
    *,
    answer_alpha: float,
    abstain_alpha: float,
) -> ThreeWayConformal:
    """Calibrate both thresholds on held-out nonconformity scores.

    ``answer_alpha`` is the miscoverage the operator accepts when answering
    directly; ``abstain_alpha`` is the looser level above which the query is
    declined outright. ``abstain_alpha`` must be the smaller of the two, since
    it corresponds to the higher quantile.
    """
    if not answer_alpha > abstain_alpha:
        raise ValueError("answer_alpha must exceed abstain_alpha")
    return ThreeWayConformal(
        answer_threshold=conformal_quantile(calibration_scores, answer_alpha),
        abstain_threshold=conformal_quantile(calibration_scores, abstain_alpha),
    )


def evaluate_three_way(
    scores: Sequence[float],
    cheap_correct: Sequence[bool],
    escalated_correct: Sequence[bool],
    policy: ThreeWayConformal,
) -> dict[str, float]:
    """Coverage, risk and escalation rate of a calibrated three-way policy.

    Answered queries are served by the cheap pass; escalated ones by the
    expensive pass; abstentions are not counted in risk but reduce coverage.
    """
    regimes = policy.regimes(scores)
    cheap = np.asarray(cheap_correct, dtype=bool)
    escalated = np.asarray(escalated_correct, dtype=bool)
    if not (len(regimes) == cheap.size == escalated.size):
        raise ValueError("all inputs must have the same length")

    served: list[bool] = []
    for regime, ok_cheap, ok_escalated in zip(regimes, cheap, escalated):
        if regime is Regime.ANSWER:
            served.append(bool(ok_cheap))
        elif regime is Regime.ESCALATE:
            served.append(bool(ok_escalated))
    total = len(regimes)
    return {
        "coverage": len(served) / total,
        "risk": 1.0 - (sum(served) / len(served)) if served else float("nan"),
        "answer_rate": sum(r is Regime.ANSWER for r in regimes) / total,
        "escalation_rate": sum(r is Regime.ESCALATE for r in regimes) / total,
        "abstention_rate": sum(r is Regime.ABSTAIN for r in regimes) / total,
    }
