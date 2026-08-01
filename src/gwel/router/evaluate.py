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
