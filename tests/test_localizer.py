import numpy as np
import pytest

from gwel.router.localizer import (
    RegionLocalizer,
    evaluate_localizer,
    pool_cells,
    train_localizer,
)


def test_pooling_splits_the_grid_row_major() -> None:
    # An 8x8 grid whose value encodes the quadrant it belongs to.
    grid = np.zeros((8, 8, 1))
    grid[:4, :4] = 1  # top-left
    grid[:4, 4:] = 2  # top-right
    grid[4:, :4] = 3  # bottom-left
    grid[4:, 4:] = 4  # bottom-right
    cells = pool_cells(grid, rows=2, cols=2)
    assert cells.shape == (4, 1)
    assert [float(c) for c in cells.ravel()] == [1.0, 2.0, 3.0, 4.0]


def test_pooling_handles_uneven_splits() -> None:
    grid = np.ones((8, 8, 3))
    assert pool_cells(grid, rows=3, cols=3).shape == (9, 3)


def test_pooling_validates_shape_and_layout() -> None:
    with pytest.raises(ValueError):
        pool_cells(np.ones((8, 4, 2)), rows=2, cols=2)
    with pytest.raises(ValueError):
        pool_cells(np.ones((4, 4, 2)), rows=8, cols=8)


def _separable(n: int = 40, cells: int = 4, dim: int = 6, seed: int = 0):
    """One cell per example is useful, and its feature is offset."""
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for _ in range(n):
        block = rng.normal(size=(cells, dim))
        winner = rng.integers(0, cells)
        block[winner] += 4.0
        flags = [i == winner for i in range(cells)]
        features.append(block)
        labels.append(flags)
    return features, labels


def test_localizer_learns_which_cell_answers() -> None:
    features, labels = _separable()
    localizer = train_localizer(features, labels)
    stats = evaluate_localizer(localizer, features, labels)
    assert stats["chosen"] > 0.9
    assert stats["random"] == pytest.approx(0.25, abs=0.02)


def test_evaluation_reports_the_oracle_ceiling() -> None:
    features, labels = _separable()
    # Half the examples have no useful cell at all.
    for i in range(0, len(labels), 2):
        labels[i] = [False] * 4
    localizer = train_localizer(features, labels)
    stats = evaluate_localizer(localizer, features, labels)
    assert stats["oracle"] == pytest.approx(0.5, abs=0.05)
    assert stats["chosen"] <= stats["oracle"]


def test_training_skips_uninformative_examples() -> None:
    features, labels = _separable(n=20)
    features += [np.zeros((4, 6))] * 5
    labels += [[True] * 4] * 5      # all useful: no ranking information
    features += [np.zeros((4, 6))] * 5
    labels += [[False] * 4] * 5     # none useful: likewise
    localizer = train_localizer(features, labels)
    assert np.linalg.norm(localizer.direction) > 0


def test_training_needs_at_least_one_mixed_example() -> None:
    with pytest.raises(ValueError):
        train_localizer([np.zeros((4, 3))], [[True] * 4])


def test_alignment_between_cells_and_labels_is_checked() -> None:
    with pytest.raises(ValueError):
        train_localizer([np.zeros((4, 3))], [[True, False]])


def test_degenerate_direction_scores_zero() -> None:
    localizer = RegionLocalizer(direction=np.zeros(3), offset=np.zeros(3))
    assert np.all(localizer.scores(np.ones((4, 3))) == 0.0)
    assert localizer.choose(np.ones((4, 3))) == 0
