import numpy as np
import pytest

from gwel.router.evaluate import auroc
from gwel.router.zero_probe import fit_difference_of_means, score_difference_of_means


def test_direction_points_from_negative_to_positive() -> None:
    features = np.array([[0.0, 0.0], [0.0, 1.0], [4.0, 0.0], [4.0, 1.0]])
    targets = np.array([0.0, 0.0, 1.0, 1.0])
    direction, offset = fit_difference_of_means(features, targets)
    assert direction[0] == pytest.approx(4.0)  # separated on the first axis
    assert direction[1] == pytest.approx(0.0)  # not on the second
    assert offset[0] == pytest.approx(2.0)


def test_scores_separate_the_classes() -> None:
    rng = np.random.default_rng(0)
    positive = rng.normal(loc=2.0, size=(60, 5))
    negative = rng.normal(loc=-2.0, size=(60, 5))
    features = np.vstack([positive, negative])
    targets = np.array([1.0] * 60 + [0.0] * 60)

    direction, offset = fit_difference_of_means(features, targets)
    scores = score_difference_of_means(features, direction, offset)
    assert auroc(scores.tolist(), [t > 0.5 for t in targets]) > 0.99


def test_no_signal_gives_chance_performance_on_held_out_data() -> None:
    # Scored on data the direction was not fitted on: fitting and scoring the
    # same points separates pure noise, which is the failure the split avoids.
    rng = np.random.default_rng(1)
    features = rng.normal(size=(400, 8))
    targets = rng.integers(0, 2, 400).astype(float)
    direction, offset = fit_difference_of_means(features[:200], targets[:200])
    scores = score_difference_of_means(features[200:], direction, offset)
    assert 0.35 < auroc(scores.tolist(), [t > 0.5 for t in targets[200:]]) < 0.65


def test_degenerate_direction_scores_zero() -> None:
    features = np.ones((4, 3))
    scores = score_difference_of_means(features, np.zeros(3), np.zeros(3))
    assert np.all(scores == 0.0)


def test_requires_both_classes() -> None:
    features = np.ones((4, 2))
    with pytest.raises(ValueError):
        fit_difference_of_means(features, np.ones(4))
    with pytest.raises(ValueError):
        fit_difference_of_means(features, np.zeros(4))


def test_validates_shapes() -> None:
    with pytest.raises(ValueError):
        fit_difference_of_means(np.zeros((3, 2)), np.zeros(2))
