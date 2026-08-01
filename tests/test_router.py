from gwel import (
    Action,
    ActionMeasurement,
    ActionProfile,
    BudgetRouter,
    CostWeights,
    PolicyResult,
    label_minimal_action,
    oracle_gap,
    summarize,
)


def test_router_selects_highest_utility_action_that_fits_budget() -> None:
    router = BudgetRouter(
        [
            ActionProfile(Action.LOW_RES, utility=0.4, latency_ms=20, memory_mb=300, energy_mj=5),
            ActionProfile(Action.CROP, utility=0.8, latency_ms=50, memory_mb=500, energy_mj=12),
            ActionProfile(Action.OCR, utility=0.9, latency_ms=90, memory_mb=700, energy_mj=20),
        ]
    )

    decision = router.route(latency_ms=60, memory_mb=600, energy_mj=15)

    assert decision.action is Action.CROP
    assert [profile.action for profile in decision.candidates] == [Action.LOW_RES, Action.CROP]


def test_router_rejects_impossible_budget() -> None:
    router = BudgetRouter(
        [ActionProfile(Action.LOW_RES, utility=0.4, latency_ms=20, memory_mb=300, energy_mj=5)]
    )

    try:
        router.route(latency_ms=10, memory_mb=200, energy_mj=1)
    except ValueError as error:
        assert str(error) == "no action fits the supplied resource budgets"
    else:
        raise AssertionError("expected an impossible budget to raise ValueError")


def test_oracle_labels_cheapest_correct_action() -> None:
    measurements = [
        ActionMeasurement(Action.LOW_RES, True, 20, 300, 8, 64),
        ActionMeasurement(Action.CROP, True, 35, 450, 10, 96),
        ActionMeasurement(Action.OCR, True, 15, 250, 5, 16),
    ]

    label = label_minimal_action(
        measurements,
        weights=CostWeights(latency=1, memory=0, energy=2, visual_tokens=0.1),
    )

    assert label.action is Action.OCR


def test_oracle_returns_none_when_all_actions_fail() -> None:
    measurements = [
        ActionMeasurement(Action.LOW_RES, False, 20, 300, 8, 64),
        ActionMeasurement(Action.CROP, False, 35, 450, 10, 96),
    ]

    assert label_minimal_action(measurements).action is None


def test_summary_reports_accuracy_and_escalation_rate() -> None:
    results = [
        PolicyResult(Action.LOW_RES, True, 20, 300, 8, 64),
        PolicyResult(Action.CROP, True, 35, 450, 10, 96),
        PolicyResult(Action.OCR, False, 15, 250, 5, 16),
    ]

    summary = summarize(results)

    assert summary.accuracy == 2 / 3
    assert summary.escalation_rate == 2 / 3
    assert summary.action_counts[Action.CROP] == 1


def test_oracle_gap_requires_aligned_inputs() -> None:
    result = PolicyResult(Action.CROP, True, 35, 450, 10, 96)
    assert oracle_gap([result], [50]) == 91

    try:
        oracle_gap([result], [])
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("expected mismatched inputs to raise ValueError")
