import numpy as np
import pytest

from gwel.router.probes import (
    fisher_separability,
    fit_layer_probe,
    sweep_layers,
)


def _separable(n: int = 60, dim: int = 8, gap: float = 4.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    positive = rng.normal(loc=gap, size=(n, dim))
    negative = rng.normal(loc=-gap, size=(n, dim))
    activations = np.vstack([positive, negative])
    labels = np.array([1.0] * n + [0.0] * n)
    return activations, labels


def test_fisher_rises_with_separation() -> None:
    close, labels = _separable(gap=0.5)
    far, _ = _separable(gap=5.0)
    assert fisher_separability(far, labels) > fisher_separability(close, labels)


def test_fisher_is_zero_for_identical_class_means() -> None:
    activations = np.tile(np.arange(6.0), (20, 1))
    labels = np.array([1.0] * 10 + [0.0] * 10)
    assert fisher_separability(activations, labels) == 0.0


def test_fisher_needs_two_examples_per_class() -> None:
    with pytest.raises(ValueError):
        fisher_separability(np.zeros((3, 2)), np.array([1.0, 0.0, 0.0]))


def test_probe_scores_separate_the_classes() -> None:
    from gwel.router.evaluate import auroc

    activations, labels = _separable()
    probe = fit_layer_probe(activations, labels, layer=3)
    assert probe.layer == 3
    scores = probe.score(activations)
    assert auroc(scores.tolist(), [bool(v) for v in labels]) > 0.99


def test_degenerate_direction_scores_zero() -> None:
    activations = np.ones((4, 3))
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    probe = fit_layer_probe(activations, labels, layer=0)
    assert np.all(probe.score(activations) == 0.0)


def test_sweep_reports_one_row_per_layer() -> None:
    rng = np.random.default_rng(1)
    n, layers, dim = 80, 5, 6
    labels = np.array([1.0] * 40 + [0.0] * 40)
    activations = rng.normal(size=(n, layers, dim))
    # Only the middle layer carries signal, as depth-saturation predicts.
    activations[:40, 2, :] += 5.0

    train = np.arange(0, n, 2)
    test = np.arange(1, n, 2)
    rows = sweep_layers(activations, labels, train, test)

    assert [layer for layer, _, _ in rows] == list(range(layers))
    best = max(rows, key=lambda row: row[1])
    assert best[0] == 2  # the informative layer is found
    assert best[1] > 0.95


def test_sweep_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        sweep_layers(np.zeros((4, 3)), np.zeros(4), np.arange(2), np.arange(2, 4))
