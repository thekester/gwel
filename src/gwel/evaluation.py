"""Metrics for comparing fixed policies and learned routers."""

from collections import Counter
from dataclasses import dataclass

from .router import Action


@dataclass(frozen=True)
class PolicyResult:
    """Outcome of a policy on one benchmark example."""

    action: Action
    correct: bool
    latency_ms: float
    memory_mb: float
    energy_mj: float
    visual_tokens: int

    def cost(self) -> float:
        """Return the unweighted sum used for a simple oracle-gap baseline."""
        return self.latency_ms + self.energy_mj + self.visual_tokens


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics for one policy over a benchmark split."""

    examples: int
    accuracy: float
    mean_latency_ms: float
    mean_memory_mb: float
    mean_energy_mj: float
    mean_visual_tokens: float
    action_counts: dict[Action, int]

    @property
    def escalation_rate(self) -> float:
        """Fraction of examples requiring an action beyond the low-res view."""
        if self.examples == 0:
            return 0.0
        escalations = self.examples - self.action_counts.get(Action.LOW_RES, 0)
        return escalations / self.examples


def summarize(results: list[PolicyResult]) -> EvaluationSummary:
    """Aggregate policy outcomes; empty input is valid for split pipelines."""
    count = len(results)
    if count == 0:
        return EvaluationSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    return EvaluationSummary(
        examples=count,
        accuracy=sum(result.correct for result in results) / count,
        mean_latency_ms=sum(result.latency_ms for result in results) / count,
        mean_memory_mb=sum(result.memory_mb for result in results) / count,
        mean_energy_mj=sum(result.energy_mj for result in results) / count,
        mean_visual_tokens=sum(result.visual_tokens for result in results) / count,
        action_counts=dict(Counter(result.action for result in results)),
    )


def oracle_gap(results: list[PolicyResult], oracle_costs: list[float]) -> float:
    """Return mean excess cost over the per-example oracle.

    Both lists must describe the same examples in the same order. A positive
    value means the policy spends more than the oracle on average.
    """
    if len(results) != len(oracle_costs):
        raise ValueError("results and oracle_costs must have the same length")
    if not results:
        return 0.0
    return sum(result.cost() - oracle for result, oracle in zip(results, oracle_costs)) / len(results)
