import numpy as np
import pytest

from gwel.actions import Action
from gwel.router.features import FEATURE_NAMES, SIGNAL_FEATURES
from gwel.router.policies import group_runs
from gwel.router.zero_probe import (
    FREE_COLUMNS,
    FREE_FEATURES,
    fit_logistic,
    train_zero_probe,
)


def test_free_features_exclude_every_model_signal() -> None:
    assert not set(FREE_FEATURES) & set(SIGNAL_FEATURES)
    assert len(FREE_COLUMNS) == len(FREE_FEATURES)
    assert all(FEATURE_NAMES[i] in FREE_FEATURES for i in FREE_COLUMNS)


def test_logistic_separates_a_linearly_separable_problem() -> None:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    targets = np.array([0.0, 0.0, 1.0, 1.0])
    weights = fit_logistic(features, targets, epochs=2000, lr=0.5, l2=0.0)
    design = np.hstack([features, np.ones((4, 1))])
    predictions = 1.0 / (1.0 + np.exp(-(design @ weights)))
    assert predictions[0] < 0.5 < predictions[-1]


def test_logistic_validates_inputs() -> None:
    with pytest.raises(ValueError):
        fit_logistic(np.zeros((2, 1)), np.zeros(3))
    with pytest.raises(ValueError):
        fit_logistic(np.zeros((0, 1)), np.zeros(0))


def _runs(make_record, n: int = 12):
    """Small images answer correctly, large ones do not: a learnable pattern."""
    records = []
    for i in range(n):
        large = i % 2 == 0
        records.append(
            make_record(
                example_id=f"ex-{i}",
                config_id="lowres_384",
                correct=not large,
                meta={
                    "orig_width": 2000 if large else 300,
                    "orig_height": 1500 if large else 220,
                },
            )
        )
    return group_runs(records)


def test_router_learns_a_free_feature_pattern(make_record) -> None:
    runs = _runs(make_record)
    router = train_zero_probe(runs, probe_config_id="lowres_384")
    choose = router.policy(probe_config_id="lowres_384")
    # Small images (odd indices) were correct, so they should not escalate.
    assert choose(runs[1]) is Action.ANSWER_LOW
    assert choose(runs[0]) is Action.CROP


def test_policy_escalates_when_the_example_is_missing(make_record) -> None:
    runs = _runs(make_record)
    router = train_zero_probe(runs, probe_config_id="lowres_384")
    empty = group_runs([make_record(config_id="crop_r0c0", action=Action.CROP)])[0]
    assert router.policy(probe_config_id="lowres_384")(empty) is Action.CROP


def test_threshold_shifts_the_escalation_rate(make_record) -> None:
    # A noisy pattern keeps predictions off the saturated ends, which is where
    # the threshold has anything to decide.
    records = [
        make_record(
            example_id=f"ex-{i}",
            config_id="lowres_384",
            correct=(i % 3 != 0),
            meta={"orig_width": 300 + 40 * i, "orig_height": 220 + 30 * i},
        )
        for i in range(18)
    ]
    runs = group_runs(records)

    def escalations(threshold: float) -> int:
        router = train_zero_probe(runs, probe_config_id="lowres_384", threshold=threshold)
        choose = router.policy(probe_config_id="lowres_384")
        return sum(choose(run) is Action.CROP for run in runs)

    assert escalations(0.99) > escalations(0.01)


def test_training_requires_matching_records(make_record) -> None:
    runs = group_runs([make_record(config_id="crop_r0c0", action=Action.CROP)])
    with pytest.raises(ValueError):
        train_zero_probe(runs, probe_config_id="lowres_384")
