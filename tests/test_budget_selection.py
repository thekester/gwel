import pytest

from gwel.actions import Action
from gwel.router.budget_selection import (
    ActionStats,
    crossover_points,
    normalise_costs,
    policy_regions,
    select_action,
    select_under_budget,
)

# Cheap action is worse but free; expensive one is better. The classic trade-off.
STATS = [
    ActionStats(Action.ANSWER_LOW, error_rate=0.50, cost=100.0),
    ActionStats(Action.CROP, error_rate=0.35, cost=300.0),
    ActionStats(Action.OCR, error_rate=0.40, cost=500.0),
]


def test_costs_normalise_to_the_unit_interval() -> None:
    normalised = normalise_costs(STATS)
    assert normalised[Action.ANSWER_LOW] == 0.0
    assert normalised[Action.OCR] == 1.0
    assert 0.0 < normalised[Action.CROP] < 1.0


def test_identical_costs_normalise_to_zero() -> None:
    stats = [ActionStats(a, 0.5, 42.0) for a in Action.ordered()]
    assert set(normalise_costs(stats).values()) == {0.0}


def test_zero_lam_picks_the_most_accurate_action() -> None:
    assert select_action(STATS, 0.0) is Action.CROP


def test_large_lam_picks_the_cheapest_action() -> None:
    assert select_action(STATS, 100.0) is Action.ANSWER_LOW


def test_lam_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        select_action(STATS, -0.1)


def test_dominated_action_never_wins() -> None:
    # OCR costs more than CROP and is less accurate, so no lam should select it.
    for lam in (0.0, 0.05, 0.1, 0.3, 1.0, 10.0):
        assert select_action(STATS, lam) is not Action.OCR


def test_regions_partition_the_whole_range() -> None:
    regions = policy_regions(STATS)
    assert regions[0][0] == 0.0
    assert regions[-1][1] == float("inf")
    for (_, high), (low, _) in zip(
        [(r[0], r[1]) for r in regions[:-1]], [(r[0], r[1]) for r in regions[1:]]
    ):
        assert high == low  # contiguous, no gaps
    assert regions[-1][2] is Action.ANSWER_LOW  # cost dominates at the top


def test_regions_agree_with_direct_selection() -> None:
    for low, high, action in policy_regions(STATS):
        probe = low + 1.0 if high == float("inf") else (low + high) / 2.0
        assert select_action(STATS, probe) is action


def test_crossovers_are_positive_and_sorted() -> None:
    points = crossover_points(STATS)
    assert points == sorted(points)
    assert all(p > 0 for p in points)


def test_budget_selection_maximises_accuracy_within_the_budget() -> None:
    accuracy = {0.0: 0.80, 0.5: 0.75, 1.0: 0.60}
    cost = {0.0: 300.0, 0.5: 200.0, 1.0: 100.0}
    assert select_under_budget(accuracy, cost, budget=250.0) == 0.5
    assert select_under_budget(accuracy, cost, budget=1000.0) == 0.0


def test_budget_selection_reports_infeasibility() -> None:
    assert select_under_budget({0.0: 0.8}, {0.0: 300.0}, budget=10.0) is None


def test_empty_pool_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalise_costs([])
