import numpy as np
import pytest

from gwel.router.conformal import (
    Regime,
    ThreeWayConformal,
    conformal_quantile,
    evaluate_three_way,
    fit_three_way,
)


def test_quantile_uses_the_finite_sample_correction() -> None:
    scores = np.arange(1.0, 11.0)  # 1..10
    # ceil(11 * 0.9) = 10 -> the 10th smallest, i.e. the maximum.
    assert conformal_quantile(scores, 0.1) == 10.0
    # ceil(11 * 0.5) = 6 -> the 6th smallest.
    assert conformal_quantile(scores, 0.5) == 6.0


def test_quantile_saturates_when_the_rank_exceeds_the_sample() -> None:
    scores = np.arange(1.0, 6.0)
    assert conformal_quantile(scores, 0.01) == 5.0


def test_quantile_validates_inputs() -> None:
    with pytest.raises(ValueError):
        conformal_quantile([], 0.1)
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], 1.0)


def test_regimes_partition_the_score_line() -> None:
    policy = ThreeWayConformal(answer_threshold=1.0, abstain_threshold=2.0)
    assert policy.regime(0.5) is Regime.ANSWER
    assert policy.regime(1.0) is Regime.ANSWER
    assert policy.regime(1.5) is Regime.ESCALATE
    assert policy.regime(2.0) is Regime.ESCALATE
    assert policy.regime(2.5) is Regime.ABSTAIN


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        ThreeWayConformal(answer_threshold=2.0, abstain_threshold=1.0)


def test_fitting_orders_the_thresholds() -> None:
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=500)
    policy = fit_three_way(scores, answer_alpha=0.5, abstain_alpha=0.1)
    assert policy.answer_threshold < policy.abstain_threshold


def test_fitting_rejects_reversed_alphas() -> None:
    with pytest.raises(ValueError):
        fit_three_way(np.zeros(10), answer_alpha=0.1, abstain_alpha=0.5)


def test_coverage_guarantee_holds_empirically() -> None:
    # Exchangeable calibration and test draws: answering below the 1-alpha
    # quantile should miscover at most alpha of the time.
    rng = np.random.default_rng(1)
    calibration = rng.uniform(size=2000)
    test = rng.uniform(size=2000)
    threshold = conformal_quantile(calibration, 0.2)
    assert (test <= threshold).mean() == pytest.approx(0.8, abs=0.03)


def test_evaluation_accounts_for_all_three_regimes() -> None:
    policy = ThreeWayConformal(answer_threshold=1.0, abstain_threshold=2.0)
    scores = [0.5, 1.5, 2.5, 0.5]
    cheap = [True, False, False, False]
    escalated = [False, True, False, False]

    stats = evaluate_three_way(scores, cheap, escalated, policy)
    assert stats["answer_rate"] == 0.5
    assert stats["escalation_rate"] == 0.25
    assert stats["abstention_rate"] == 0.25
    assert stats["coverage"] == 0.75
    # Served: cheap True, escalated True, cheap False -> 2 of 3 correct.
    assert stats["risk"] == pytest.approx(1 / 3)


def test_evaluation_validates_lengths() -> None:
    policy = ThreeWayConformal(answer_threshold=1.0, abstain_threshold=2.0)
    with pytest.raises(ValueError):
        evaluate_three_way([0.5, 1.5], [True], [True, False], policy)
