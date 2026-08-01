import pytest

from gwel.config import CostConfig
from gwel.oracle.cost import CostWeights, compute_cost, resource_cost


def test_error_term_dominates_when_incorrect() -> None:
    weights = CostWeights(error_weight=1.0, lambda_latency_per_ms=0.001)
    wrong = compute_cost(
        correct=False, latency_ms=10, energy_mj=0, memory_mb=0, visual_tokens=0, weights=weights
    )
    right = compute_cost(
        correct=True, latency_ms=10, energy_mj=0, memory_mb=0, visual_tokens=0, weights=weights
    )
    assert wrong - right == pytest.approx(1.0)


def test_cost_is_linear_in_each_resource() -> None:
    weights = CostWeights(
        error_weight=0.0,
        lambda_latency_per_ms=2.0,
        lambda_energy_per_mj=3.0,
        lambda_memory_per_mb=5.0,
        lambda_visual_tokens=7.0,
    )
    cost = compute_cost(
        correct=True, latency_ms=1, energy_mj=1, memory_mb=1, visual_tokens=1, weights=weights
    )
    assert cost == pytest.approx(2 + 3 + 5 + 7)


def test_missing_measurements_contribute_zero() -> None:
    weights = CostWeights(lambda_energy_per_mj=100.0, lambda_memory_per_mb=100.0)
    cost = compute_cost(
        correct=True, latency_ms=0, energy_mj=None, memory_mb=None, visual_tokens=0, weights=weights
    )
    assert cost == 0.0


def test_negative_latency_rejected() -> None:
    with pytest.raises(ValueError):
        compute_cost(correct=True, latency_ms=-1, energy_mj=0, memory_mb=0, visual_tokens=0)


def test_resource_cost_excludes_error_term() -> None:
    weights = CostWeights(error_weight=10.0, lambda_latency_per_ms=1.0)
    assert resource_cost(
        latency_ms=5, energy_mj=None, memory_mb=None, visual_tokens=0, weights=weights
    ) == pytest.approx(5.0)


def test_weights_from_config_round_trip() -> None:
    config = CostConfig(error_weight=2.0, lambda_latency_per_ms=0.1)
    weights = CostWeights.from_config(config)
    assert weights.error_weight == 2.0
    assert weights.lambda_latency_per_ms == 0.1
