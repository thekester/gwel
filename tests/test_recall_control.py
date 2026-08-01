import numpy as np
import pytest

from gwel.router.recall_control import (
    certifiable_recall,
    fit_recall_controlled,
    lower_bound,
)


def test_lower_bound_is_below_the_point_estimate() -> None:
    assert 0.0 < lower_bound(90, 100) < 0.90


def test_lower_bound_tightens_with_more_data() -> None:
    few = lower_bound(9, 10)
    many = lower_bound(900, 1000)
    assert many > few  # same proportion, more evidence


def test_lower_bound_edge_cases() -> None:
    assert lower_bound(0, 10) == 0.0
    assert lower_bound(0, 0) == 0.0
    # All successes: the bound is alpha^(1/n).
    assert lower_bound(20, 20) == pytest.approx(0.05 ** (1 / 20))


def test_lower_bound_validates_counts() -> None:
    with pytest.raises(ValueError):
        lower_bound(11, 10)
    with pytest.raises(ValueError):
        lower_bound(-1, 10)


def test_certifiable_recall_matches_the_published_formula() -> None:
    # Ruan et al.: 114 positives support targets near 0.974.
    assert certifiable_recall(114) == pytest.approx(0.974, abs=0.002)
    assert certifiable_recall(299) > 0.99
    assert certifiable_recall(0) == 0.0


def test_threshold_retains_recoverable_queries() -> None:
    rng = np.random.default_rng(0)
    # Recoverable queries carry higher uncertainty, as our pilot shows.
    scores = np.concatenate([rng.normal(2.0, 0.5, 200), rng.normal(0.5, 0.5, 300)])
    recoverable = np.array([True] * 200 + [False] * 300)

    fitted = fit_recall_controlled(scores, recoverable, target_recall=0.9)
    assert fitted.achieved_recall >= 0.9
    assert fitted.certified_recall >= 0.9
    assert fitted.escalation_rate < 1.0  # it saves something


def test_a_stricter_target_escalates_more() -> None:
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(2.0, 1.0, 300), rng.normal(0.0, 1.0, 300)])
    recoverable = np.array([True] * 300 + [False] * 300)

    loose = fit_recall_controlled(scores, recoverable, target_recall=0.7)
    strict = fit_recall_controlled(scores, recoverable, target_recall=0.95)
    assert strict.escalation_rate >= loose.escalation_rate


def test_never_degenerates_to_never_escalating() -> None:
    # The failure mode of a cost-minimising tuner: it returns a threshold above
    # every score, so nothing escalates. A recall floor forbids that.
    rng = np.random.default_rng(2)
    scores = rng.normal(size=400)
    recoverable = rng.random(400) < 0.3
    fitted = fit_recall_controlled(scores, recoverable, target_recall=0.9)
    assert fitted.escalation_rate > 0.0
    assert fitted.should_escalate(scores.max())


def test_requires_recoverable_examples() -> None:
    with pytest.raises(ValueError):
        fit_recall_controlled(np.zeros(5), np.zeros(5, dtype=bool))
    with pytest.raises(ValueError):
        fit_recall_controlled(np.zeros(5), np.ones(3, dtype=bool))
