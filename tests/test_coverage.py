import numpy as np
import pytest

from gwel.router.coverage import (
    ThreeWayFrontier,
    escalating_frontier,
    pareto_filter,
    selective_frontier,
)


def _graded(n: int = 200):
    """Confidence that genuinely predicts correctness."""
    rng = np.random.default_rng(0)
    confidence = rng.uniform(size=n)
    correct = rng.uniform(size=n) < confidence
    return confidence, correct


def test_risk_falls_as_coverage_falls() -> None:
    confidence, correct = _graded()
    frontier = selective_frontier(confidence, correct, cheap_cost=1.0)
    ordered = sorted(frontier.points, key=lambda p: p.coverage)
    assert ordered[0].risk < ordered[-1].risk


def test_full_coverage_risk_matches_the_base_error_rate() -> None:
    confidence, correct = _graded()
    frontier = selective_frontier(confidence, correct, cheap_cost=1.0)
    full = max(frontier.points, key=lambda p: p.coverage)
    assert full.coverage == 1.0
    assert full.risk == pytest.approx(1.0 - correct.mean())


def test_vanilla_frontier_never_escalates() -> None:
    confidence, correct = _graded()
    frontier = selective_frontier(confidence, correct, cheap_cost=2.0)
    assert all(p.escalation_rate == 0.0 for p in frontier.points)
    assert all(p.cost == 2.0 for p in frontier.points)


def test_escalating_costs_more_when_it_escalates_more() -> None:
    cheap_c, cheap_ok = _graded()
    esc_c, esc_ok = _graded(200)
    frontier = escalating_frontier(
        cheap_c, cheap_ok, esc_c, esc_ok, cheap_cost=1.0, escalated_cost=3.0
    )
    ordered = sorted(frontier.points, key=lambda p: p.escalation_rate)
    assert ordered[-1].cost > ordered[0].cost


def test_escalation_can_raise_coverage_beyond_the_cheap_pass() -> None:
    # The escalated pass is right where the cheap one is not.
    cheap_c = np.array([0.9, 0.1, 0.1, 0.9])
    cheap_ok = np.array([True, False, False, True])
    esc_c = np.array([0.1, 0.9, 0.9, 0.1])
    esc_ok = np.array([False, True, True, False])

    vanilla = selective_frontier(cheap_c, cheap_ok, cheap_cost=1.0)
    escalating = escalating_frontier(
        cheap_c, cheap_ok, esc_c, esc_ok, cheap_cost=1.0, escalated_cost=1.0
    )
    best_vanilla = vanilla.at_risk(0.0)
    best_escalating = escalating.at_risk(0.0)
    assert best_escalating.coverage > best_vanilla.coverage


def test_lookups_return_none_when_infeasible() -> None:
    confidence, correct = _graded()
    frontier = selective_frontier(confidence, correct, cheap_cost=1.0)
    assert frontier.at_coverage(1.5) is None
    assert frontier.at_risk(-0.1) is None


def test_pareto_filter_removes_dominated_points() -> None:
    from gwel.router.coverage import SelectivePoint

    good = SelectivePoint(coverage=0.9, risk=0.1, cost=1.0, escalation_rate=0.0)
    dominated = SelectivePoint(coverage=0.8, risk=0.2, cost=2.0, escalation_rate=0.0)
    filtered = pareto_filter(ThreeWayFrontier(points=(good, dominated)))
    assert filtered.points == (good,)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        selective_frontier([0.5], [True, False], cheap_cost=1.0)
    with pytest.raises(ValueError):
        selective_frontier([], [], cheap_cost=1.0)
    with pytest.raises(ValueError):
        escalating_frontier([0.5], [True], [0.5, 0.6], [True, True],
                            cheap_cost=1.0, escalated_cost=2.0)
