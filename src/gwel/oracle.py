"""Offline oracle labeling for the active-perception pilot.

The oracle does not predict actions. It uses observed outcomes to identify the
cheapest action that produced a correct answer, which makes it suitable for
building a routing dataset and for measuring the headroom of learned policies.
"""

from dataclasses import dataclass

from .router import Action


@dataclass(frozen=True)
class CostWeights:
    """Relative importance of measured resource costs."""

    latency: float = 1.0
    memory: float = 0.0
    energy: float = 1.0
    visual_tokens: float = 0.0


@dataclass(frozen=True)
class ActionMeasurement:
    """Observed result of running one visual action on one example."""

    action: Action
    correct: bool
    latency_ms: float
    memory_mb: float
    energy_mj: float
    visual_tokens: int

    def cost(self, weights: CostWeights = CostWeights()) -> float:
        """Return a weighted cost in arbitrary comparable units."""
        return (
            weights.latency * self.latency_ms
            + weights.memory * self.memory_mb
            + weights.energy * self.energy_mj
            + weights.visual_tokens * self.visual_tokens
        )


@dataclass(frozen=True)
class OracleLabel:
    """Minimal sufficient action and all measured candidates for one example."""

    action: Action | None
    measurements: tuple[ActionMeasurement, ...]


def label_minimal_action(
    measurements: list[ActionMeasurement],
    *,
    weights: CostWeights = CostWeights(),
) -> OracleLabel:
    """Label an example with its cheapest correct action.

    ``None`` is returned when no evaluated action was correct. Ties are broken
    deterministically by the order in which measurements were supplied.
    """
    if not measurements:
        raise ValueError("at least one action measurement is required")

    candidates = tuple(measurement for measurement in measurements if measurement.correct)
    selected = min(candidates, key=lambda measurement: measurement.cost(weights), default=None)
    return OracleLabel(
        action=selected.action if selected is not None else None,
        measurements=tuple(measurements),
    )
