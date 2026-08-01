"""Parameterizable cost function J for oracle labeling and evaluation.

``J = error_weight * (1 - correct) + lambda_t * latency + lambda_e * energy
+ lambda_m * memory + lambda_v * visual_tokens``. Weights carry explicit
units (per ms, per mJ, per MB, per token) so budget trade-offs stay legible.
"""

from dataclasses import dataclass

from ..config import CostConfig


@dataclass(frozen=True)
class CostWeights:
    """Relative price of each measured resource, in cost units per unit."""

    error_weight: float = 1.0
    lambda_latency_per_ms: float = 0.00005
    lambda_energy_per_mj: float = 0.000002
    lambda_memory_per_mb: float = 0.0
    lambda_visual_tokens: float = 0.0001

    @classmethod
    def from_config(cls, config: CostConfig) -> "CostWeights":
        return cls(
            error_weight=config.error_weight,
            lambda_latency_per_ms=config.lambda_latency_per_ms,
            lambda_energy_per_mj=config.lambda_energy_per_mj,
            lambda_memory_per_mb=config.lambda_memory_per_mb,
            lambda_visual_tokens=config.lambda_visual_tokens,
        )


def compute_cost(
    *,
    correct: bool,
    latency_ms: float,
    energy_mj: float | None,
    memory_mb: float | None,
    visual_tokens: int,
    weights: CostWeights = CostWeights(),
) -> float:
    """Return J for one measured action outcome.

    Missing measurements (``None`` energy or memory) contribute zero rather
    than guessing, which keeps costs comparable within a single machine but
    not across machines with different instrumentation.
    """
    if latency_ms < 0:
        raise ValueError("latency_ms must be >= 0")
    cost = weights.error_weight * (0.0 if correct else 1.0)
    cost += weights.lambda_latency_per_ms * latency_ms
    if energy_mj is not None:
        cost += weights.lambda_energy_per_mj * energy_mj
    if memory_mb is not None:
        cost += weights.lambda_memory_per_mb * memory_mb
    cost += weights.lambda_visual_tokens * visual_tokens
    return cost


def resource_cost(
    *,
    latency_ms: float,
    energy_mj: float | None,
    memory_mb: float | None,
    visual_tokens: int,
    weights: CostWeights = CostWeights(),
) -> float:
    """Return only the resource part of J (error term excluded)."""
    return compute_cost(
        correct=True,
        latency_ms=latency_ms,
        energy_mj=energy_mj,
        memory_mb=memory_mb,
        visual_tokens=visual_tokens,
        weights=weights,
    )
