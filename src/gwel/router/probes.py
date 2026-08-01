"""Linear probes on internal activations, and layer selection.

Two literatures converge on the same recipe: read the residual stream at the
last prompt position, fit a linear direction, and use its projection as a
routing score. Moreno Cencerrado et al. (2509.10625) use a difference-of-means
direction and find separability saturates at intermediate layers; Lugoloobi et
al. (2602.09924) train supervised linear probes on the same activations;
NVIDIA's LLM Router (2603.20895) selects layers by Fisher separability.

What is new here is the *target*. Every probe above predicts whether a model
will succeed. We also fit probes for whether an intervention — more pixels —
will change the outcome, which is the quantity that actually spends budget.
"""

from dataclasses import dataclass

import numpy as np


def fisher_separability(activations: np.ndarray, labels: np.ndarray) -> float:
    """Between-class over within-class scatter along the class-mean axis.

    NVIDIA's layer-selection criterion. Higher means the two classes are more
    linearly separable at this layer, and it is cheap enough to evaluate on
    every layer before fitting anything.
    """
    if len(activations) != len(labels):
        raise ValueError("activations and labels must have the same length")
    positive = activations[labels > 0.5]
    negative = activations[labels <= 0.5]
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("both classes need at least two examples")

    mu_pos, mu_neg = positive.mean(axis=0), negative.mean(axis=0)
    direction = mu_pos - mu_neg
    norm = np.linalg.norm(direction)
    if norm == 0:
        return 0.0
    unit = direction / norm

    between = float((mu_pos - mu_neg) @ unit) ** 2
    within = float(np.var(positive @ unit) + np.var(negative @ unit))
    return between / within if within > 0 else float("inf")


@dataclass(frozen=True)
class LayerProbe:
    """A difference-of-means direction fitted at one layer."""

    layer: int
    direction: np.ndarray
    offset: np.ndarray

    def score(self, activations: np.ndarray) -> np.ndarray:
        """Project activations onto the direction; higher means positive class."""
        norm = np.linalg.norm(self.direction)
        if norm == 0:
            return np.zeros(len(activations))
        return (activations - self.offset) @ self.direction / norm


def fit_layer_probe(activations: np.ndarray, labels: np.ndarray, layer: int) -> LayerProbe:
    """Fit the difference-of-means direction at one layer."""
    if len(activations) != len(labels):
        raise ValueError("activations and labels must have the same length")
    positive = activations[labels > 0.5]
    negative = activations[labels <= 0.5]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("both classes must be present")

    mu_pos, mu_neg = positive.mean(axis=0), negative.mean(axis=0)
    return LayerProbe(
        layer=layer,
        direction=mu_pos - mu_neg,
        offset=(mu_pos + mu_neg) / 2.0,
    )


def sweep_layers(
    activations: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> list[tuple[int, float, float]]:
    """Fit and score a probe at every layer.

    ``activations`` is ``(examples, layers, hidden)``. Returns one
    ``(layer, test_auroc, fisher_separability)`` tuple per layer, with the
    probe fitted on ``train_index`` and scored on ``test_index`` so that
    saturation across depth is measured out of sample.
    """
    from .evaluate import auroc

    if activations.ndim != 3:
        raise ValueError("activations must be (examples, layers, hidden)")

    results: list[tuple[int, float, float]] = []
    test_labels = [bool(label) for label in labels[test_index]]
    for layer in range(activations.shape[1]):
        train = activations[train_index, layer, :]
        probe = fit_layer_probe(train, labels[train_index], layer)
        scores = probe.score(activations[test_index, layer, :])
        try:
            separability = fisher_separability(train, labels[train_index])
        except ValueError:
            separability = float("nan")
        results.append((layer, auroc(scores.tolist(), test_labels), separability))
    return results
