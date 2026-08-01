"""Routing without a probe pass.

A confidence-conditioned router must run the cheap pass before deciding, and
that probe dominates the cost once budgets tighten. This module routes from
features available *before* any model pass — question wording and image
geometry — so escalation is decided for free.

The classifier is a strongly regularised logistic regression rather than an
MLP: at pilot scale the extra capacity only buys overfitting, and the question
is what the free features carry, not how well a model can be tuned.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..actions import Action
from .features import FEATURE_NAMES, IMAGE_FEATURES, QUESTION_FEATURES, build_features
from .policies import ExampleRuns

#: Feature names observable without running the model on the image.
FREE_FEATURES: tuple[str, ...] = QUESTION_FEATURES + IMAGE_FEATURES

#: Column indices of the free features inside a full feature vector.
FREE_COLUMNS: tuple[int, ...] = tuple(
    FEATURE_NAMES.index(name) for name in FREE_FEATURES if name in FEATURE_NAMES
)


def fit_difference_of_means(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the difference-of-means direction separating the two classes.

    Returns ``(direction, offset)`` where the score of a feature vector is
    ``(x - offset) @ direction / ||direction||``. Following Moreno Cencerrado
    et al. (arXiv 2509.10625), who use this deliberately minimal probe to test
    whether correctness is *linearly* accessible rather than to maximise
    accuracy. It has no hyperparameters, which matters at pilot sample sizes
    where a regularised classifier mostly measures its own regularisation.
    """
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")
    positive = features[targets > 0.5]
    negative = features[targets <= 0.5]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("both classes must be present")

    mu_true = positive.mean(axis=0)
    mu_false = negative.mean(axis=0)
    direction = mu_true - mu_false
    offset = (mu_true + mu_false) / 2.0
    return direction, offset


def score_difference_of_means(
    features: np.ndarray,
    direction: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Project features onto the correctness direction; higher means correct."""
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.zeros(len(features))
    return (features - offset) @ direction / norm


def fit_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int = 400,
    lr: float = 0.1,
    l2: float = 1e-2,
    seed: int = 0,
) -> np.ndarray:
    """Fit L2-regularised logistic regression by full-batch gradient descent.

    Returns the weight vector, with the bias in the last position.
    """
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")
    if len(features) == 0:
        raise ValueError("at least one training example is required")

    rng = np.random.default_rng(seed)
    design = np.hstack([features, np.ones((len(features), 1))])
    weights = rng.normal(scale=0.01, size=design.shape[1])
    for _ in range(epochs):
        predictions = 1.0 / (1.0 + np.exp(-(design @ weights)))
        gradient = design.T @ (predictions - targets) / len(targets)
        gradient[:-1] += l2 * weights[:-1]
        weights -= lr * gradient
    return weights


@dataclass(frozen=True)
class ZeroProbeRouter:
    """Predicts whether the cheap pass suffices, using only free features."""

    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    threshold: float

    def score(self, full_features: np.ndarray) -> float:
        """Probability that the cheap pass will answer correctly."""
        subset = full_features[list(FREE_COLUMNS)]
        normalized = (subset - self.mean) / self.std
        design = np.append(normalized, 1.0)
        return float(1.0 / (1.0 + np.exp(-(design @ self.weights))))

    def policy(self, *, probe_config_id: str, escalate_to: Action = Action.CROP):
        """Wrap this router as a simulation policy.

        ``probe_config_id`` only identifies which record carries the example's
        metadata; no signal from that record is read, so the policy stays free.
        """

        def choose(run: ExampleRuns) -> Action:
            record = run.by_config.get(probe_config_id)
            if record is None:
                return escalate_to
            features = build_features(record)
            return Action.ANSWER_LOW if self.score(features) >= self.threshold else escalate_to

        return choose


def train_zero_probe(
    runs: Sequence[ExampleRuns],
    *,
    probe_config_id: str,
    threshold: float = 0.5,
    seed: int = 0,
) -> ZeroProbeRouter:
    """Fit a zero-probe router to predict cheap-pass success on ``runs``."""
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for run in runs:
        record = run.by_config.get(probe_config_id)
        if record is None or record.signals is None:
            continue
        rows.append(build_features(record)[list(FREE_COLUMNS)])
        targets.append(float(record.correct))

    if not rows:
        raise ValueError(f"no {probe_config_id} records to train on")

    matrix = np.stack(rows)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    weights = fit_logistic((matrix - mean) / std, np.asarray(targets), seed=seed)
    return ZeroProbeRouter(weights=weights, mean=mean, std=std, threshold=threshold)
