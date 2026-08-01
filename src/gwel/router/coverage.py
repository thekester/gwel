"""The risk / coverage / cost frontier.

Selective prediction trades risk against coverage: answer only when confident,
abstain otherwise. Efficient inference trades accuracy against cost: escalate
only when needed. The two literatures use the same confidence signal and never
appear on the same axes.

Escalation is the missing third option. A system facing an uncertain query can
answer anyway, abstain, or *spend more* and re-decide. ReCoVERR (arXiv
2402.15610) takes that route to raise coverage — asking sub-questions and
verifying with an NLI model — but does not report what the extra calls cost.
This module computes all three quantities together so the trade-off is visible.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectivePoint:
    """One operating point of a selective policy."""

    coverage: float      # fraction of queries answered rather than abstained
    risk: float          # error rate among answered queries
    cost: float          # mean cost per query, including escalations
    escalation_rate: float


@dataclass(frozen=True)
class ThreeWayFrontier:
    """Operating points of a selective policy that may also escalate."""

    points: tuple[SelectivePoint, ...]

    def at_coverage(self, target: float) -> SelectivePoint | None:
        """The lowest-risk point achieving at least ``target`` coverage."""
        feasible = [p for p in self.points if p.coverage >= target]
        return min(feasible, key=lambda p: (p.risk, p.cost)) if feasible else None

    def at_risk(self, tolerance: float) -> SelectivePoint | None:
        """The highest-coverage point whose risk stays within ``tolerance``."""
        feasible = [p for p in self.points if p.risk <= tolerance]
        return max(feasible, key=lambda p: (p.coverage, -p.cost)) if feasible else None


def selective_frontier(
    cheap_confidence: Sequence[float],
    cheap_correct: Sequence[bool],
    cheap_cost: float,
    *,
    grid: int = 40,
) -> ThreeWayFrontier:
    """Risk/coverage for answering from the cheap pass or abstaining.

    ``cheap_confidence`` must increase with confidence. This is vanilla
    selective prediction: no escalation, so cost is flat.
    """
    confidence = np.asarray(cheap_confidence, dtype=np.float64)
    correct = np.asarray(cheap_correct, dtype=bool)
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correctness must have the same shape")
    if confidence.size == 0:
        raise ValueError("at least one query is required")

    points: list[SelectivePoint] = []
    for threshold in np.quantile(confidence, np.linspace(0.0, 1.0, grid)):
        answered = confidence >= threshold
        coverage = float(answered.mean())
        if coverage == 0.0:
            continue
        risk = float(1.0 - correct[answered].mean())
        points.append(
            SelectivePoint(coverage=coverage, risk=risk, cost=cheap_cost, escalation_rate=0.0)
        )
    return ThreeWayFrontier(points=tuple(points))


def escalating_frontier(
    cheap_confidence: Sequence[float],
    cheap_correct: Sequence[bool],
    escalated_confidence: Sequence[float],
    escalated_correct: Sequence[bool],
    *,
    cheap_cost: float,
    escalated_cost: float,
    grid: int = 25,
) -> ThreeWayFrontier:
    """Risk/coverage/cost when uncertain queries may escalate before deciding.

    Two thresholds sweep independently: one deciding whether the cheap answer
    is good enough, one deciding whether the escalated answer is. Queries
    failing both abstain. Escalated queries pay both passes, which is the
    accounting the cascade literature is careful about and the selective
    prediction literature does not report at all.
    """
    cheap_c = np.asarray(cheap_confidence, dtype=np.float64)
    cheap_ok = np.asarray(cheap_correct, dtype=bool)
    esc_c = np.asarray(escalated_confidence, dtype=np.float64)
    esc_ok = np.asarray(escalated_correct, dtype=bool)
    shapes = {cheap_c.shape, cheap_ok.shape, esc_c.shape, esc_ok.shape}
    if len(shapes) != 1:
        raise ValueError("all four inputs must have the same shape")
    if cheap_c.size == 0:
        raise ValueError("at least one query is required")

    n = cheap_c.size
    cheap_levels = np.quantile(cheap_c, np.linspace(0.0, 1.0, grid))
    esc_levels = np.quantile(esc_c, np.linspace(0.0, 1.0, grid))

    points: list[SelectivePoint] = []
    for accept in cheap_levels:
        answered_cheap = cheap_c >= accept
        escalated = ~answered_cheap
        for accept_esc in esc_levels:
            answered_esc = escalated & (esc_c >= accept_esc)
            answered = answered_cheap | answered_esc
            coverage = float(answered.mean())
            if coverage == 0.0:
                continue
            correct = np.where(answered_cheap, cheap_ok, esc_ok)
            risk = float(1.0 - correct[answered].mean())
            cost = float(
                (n * cheap_cost + escalated.sum() * escalated_cost) / n
            )
            points.append(
                SelectivePoint(
                    coverage=coverage,
                    risk=risk,
                    cost=cost,
                    escalation_rate=float(escalated.mean()),
                )
            )
    return ThreeWayFrontier(points=tuple(points))


def pareto_filter(frontier: ThreeWayFrontier) -> ThreeWayFrontier:
    """Keep only points not dominated on (coverage up, risk down, cost down)."""
    kept: list[SelectivePoint] = []
    for point in frontier.points:
        dominated = any(
            other.coverage >= point.coverage
            and other.risk <= point.risk
            and other.cost <= point.cost
            and (
                other.coverage > point.coverage
                or other.risk < point.risk
                or other.cost < point.cost
            )
            for other in frontier.points
        )
        if not dominated:
            kept.append(point)
    return ThreeWayFrontier(points=tuple(kept))
