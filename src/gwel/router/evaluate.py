"""Policy evaluation: aggregate metrics, risk–coverage, and Pareto fronts.

Everything here is numpy-only so evaluation of cached runs never needs torch.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..actions import Action


@dataclass(frozen=True)
class PolicyResult:
    """Outcome of a policy on one benchmark example."""

    action: Action
    correct: bool
    latency_ms: float
    energy_mj: float | None
    memory_mb: float | None
    visual_tokens: int
    cost: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics for one policy over a benchmark split."""

    examples: int
    accuracy: float
    mean_cost: float
    mean_latency_ms: float
    mean_visual_tokens: float
    mean_energy_mj: float | None
    action_counts: dict[Action, int]

    @property
    def escalation_rate(self) -> float:
        """Fraction of examples routed beyond the low-res answer."""
        if self.examples == 0:
            return 0.0
        escalations = self.examples - self.action_counts.get(Action.ANSWER_LOW, 0)
        return escalations / self.examples


def summarize(results: Sequence[PolicyResult]) -> EvaluationSummary:
    """Aggregate policy outcomes; empty input is valid for split pipelines."""
    count = len(results)
    if count == 0:
        return EvaluationSummary(0, 0.0, 0.0, 0.0, 0.0, None, {})
    energies = [r.energy_mj for r in results if r.energy_mj is not None]
    return EvaluationSummary(
        examples=count,
        accuracy=sum(r.correct for r in results) / count,
        mean_cost=sum(r.cost for r in results) / count,
        mean_latency_ms=sum(r.latency_ms for r in results) / count,
        mean_visual_tokens=sum(r.visual_tokens for r in results) / count,
        mean_energy_mj=sum(energies) / len(energies) if energies else None,
        action_counts=dict(Counter(r.action for r in results)),
    )


@dataclass(frozen=True)
class RiskCoverageCurve:
    """Selective-prediction curve: risk among the most confident fraction."""

    coverages: tuple[float, ...]
    risks: tuple[float, ...]
    aurc: float


def risk_coverage(
    confidences: Sequence[float],
    corrects: Sequence[bool],
) -> RiskCoverageCurve:
    """Compute the risk–coverage curve and its area (AURC).

    Examples are sorted by decreasing confidence; at coverage k/N the risk is
    the error rate among the k most confident predictions. Ties are broken by
    original order for determinism.
    """
    if len(confidences) != len(corrects):
        raise ValueError("confidences and corrects must have the same length")
    n = len(confidences)
    if n == 0:
        raise ValueError("at least one prediction is required")

    order = np.argsort(-np.asarray(confidences, dtype=np.float64), kind="stable")
    sorted_errors = 1.0 - np.asarray(corrects, dtype=np.float64)[order]
    cumulative_errors = np.cumsum(sorted_errors)
    ks = np.arange(1, n + 1, dtype=np.float64)
    risks = cumulative_errors / ks
    coverages = ks / n
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 fallback
    aurc = float(trapezoid(risks, coverages)) if n > 1 else float(risks[0])
    return RiskCoverageCurve(
        coverages=tuple(float(c) for c in coverages),
        risks=tuple(float(r) for r in risks),
        aurc=aurc,
    )


@dataclass(frozen=True)
class Interval:
    """Point estimate with a percentile bootstrap confidence interval."""

    estimate: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_interval(
    values: Sequence[float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> Interval:
    """Percentile bootstrap CI for the mean of ``values``.

    Policy comparisons at pilot scale are dominated by sampling noise, so every
    headline number should carry one of these rather than a bare point estimate.
    """
    if not values:
        raise ValueError("at least one value is required")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        point = float(array[0])
        return Interval(point, point, point)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return Interval(float(array.mean()), float(low), float(high))


def paired_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> Interval:
    """Bootstrap CI for the paired mean difference ``left - right``.

    Policies are simulated on the same examples, so pairing removes the
    example-difficulty variance that would otherwise swamp the comparison. An
    interval excluding zero is the evidence that one policy really is better.
    """
    if len(left) != len(right):
        raise ValueError("paired inputs must have the same length")
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return bootstrap_interval(
        differences.tolist(), resamples=resamples, confidence=confidence, seed=seed
    )


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve, computed from rank statistics.

    ``scores`` should rank positives above negatives. Ties receive the average
    rank, so a constant score gives exactly 0.5. Returns ``nan`` when one class
    is absent, since AUROC is undefined there.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("at least one observation is required")

    positives = np.asarray(labels, dtype=bool)
    n_pos = int(positives.sum())
    n_neg = len(positives) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    # Average the ranks inside each tied group so ties contribute 0.5 each.
    sorted_values = values[order]
    start = 0
    for index in range(1, len(sorted_values) + 1):
        if index == len(sorted_values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pareto_front(
    costs: Sequence[float],
    accuracies: Sequence[float],
) -> list[int]:
    """Indices of non-dominated (min cost, max accuracy) points.

    A point dominates another when it is at least as cheap and at least as
    accurate, and strictly better on one axis. Returned indices are sorted by
    increasing cost.
    """
    if len(costs) != len(accuracies):
        raise ValueError("costs and accuracies must have the same length")
    points = list(zip(costs, accuracies, range(len(costs))))
    points.sort(key=lambda p: (p[0], -p[1]))

    front: list[int] = []
    best_accuracy = -np.inf
    for cost, accuracy, index in points:
        if accuracy > best_accuracy:
            front.append(index)
            best_accuracy = accuracy
    return front
