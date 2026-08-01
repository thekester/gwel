import pytest

from gwel.actions import Action
from gwel.router.evaluate import PolicyResult, pareto_front, risk_coverage, summarize


def _result(action: Action, correct: bool, cost: float = 1.0) -> PolicyResult:
    return PolicyResult(
        action=action,
        correct=correct,
        latency_ms=100.0,
        energy_mj=10.0,
        memory_mb=500.0,
        visual_tokens=64,
        cost=cost,
    )


def test_summarize_accuracy_and_escalation() -> None:
    results = [
        _result(Action.ANSWER_LOW, True),
        _result(Action.CROP, True),
        _result(Action.OCR, False),
    ]
    summary = summarize(results)
    assert summary.accuracy == pytest.approx(2 / 3)
    assert summary.escalation_rate == pytest.approx(2 / 3)
    assert summary.action_counts[Action.CROP] == 1


def test_summarize_empty_is_valid() -> None:
    summary = summarize([])
    assert summary.examples == 0
    assert summary.escalation_rate == 0.0


def test_risk_coverage_confident_correct_first() -> None:
    curve = risk_coverage(confidences=[0.9, 0.8, 0.1], corrects=[True, True, False])
    assert curve.coverages == (pytest.approx(1 / 3), pytest.approx(2 / 3), pytest.approx(1.0))
    assert curve.risks[0] == 0.0
    assert curve.risks[-1] == pytest.approx(1 / 3)
    perfect = risk_coverage([0.9, 0.8, 0.7], [True, True, True])
    assert curve.aurc > perfect.aurc == 0.0


def test_risk_coverage_penalizes_confident_mistakes() -> None:
    bad = risk_coverage([0.9, 0.1], [False, True])
    good = risk_coverage([0.9, 0.1], [True, False])
    assert bad.aurc > good.aurc


def test_risk_coverage_validates_inputs() -> None:
    with pytest.raises(ValueError):
        risk_coverage([0.5], [True, False])
    with pytest.raises(ValueError):
        risk_coverage([], [])


def test_pareto_front_keeps_non_dominated_points() -> None:
    costs = [1.0, 2.0, 3.0, 2.5]
    accuracies = [0.5, 0.7, 0.9, 0.6]
    front = pareto_front(costs, accuracies)
    assert front == [0, 1, 2]  # index 3 dominated by index 1


def test_pareto_front_drops_equal_cost_lower_accuracy() -> None:
    front = pareto_front([1.0, 1.0], [0.5, 0.9])
    assert front == [1]
