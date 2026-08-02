"""Map a raw confidence signal to a calibrated error probability.

Kotte (arXiv 2605.18796) shows that a raw token-margin signal is a weak routing
score used directly, but becomes near-optimal after a small isotonic fit on a
held-out calibration set, expected calibration error 0.12 to 0.03, and 11%
lower cost than thresholding the raw signal. Their operational summary is to
calibrate first and threshold second, because most of the value comes from
making the score probabilistic rather than from tuning the threshold.

Isotonic regression is the right tool here: it assumes only that the mapping
from uncertainty to error probability is monotone, which a confidence signal
should satisfy by construction, and it needs no parametric form.
"""

from dataclasses import dataclass

import numpy as np


def _pool_adjacent_violators(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Isotonic (non-decreasing) fit by pool-adjacent-violators.

    Implemented directly rather than pulled from scikit-learn to keep the core
    package dependency-light; the algorithm is a few lines and exact.
    """
    fitted = values.astype(np.float64).copy()
    counts = weights.astype(np.float64).copy()
    # Each block holds a running weighted mean; merge whenever order breaks.
    block_starts = list(range(len(fitted)))
    index = 0
    while index < len(fitted) - 1:
        if fitted[index] <= fitted[index + 1]:
            index += 1
            continue
        total = counts[index] + counts[index + 1]
        merged = (fitted[index] * counts[index] + fitted[index + 1] * counts[index + 1]) / total
        fitted[index] = merged
        counts[index] = total
        fitted = np.delete(fitted, index + 1)
        counts = np.delete(counts, index + 1)
        block_starts.pop(index + 1)
        index = max(index - 1, 0)
    # Expand blocks back to one value per original observation.
    expanded = np.empty(int(counts.sum()))
    position = 0
    for value, count in zip(fitted, counts):
        expanded[position : position + int(count)] = value
        position += int(count)
    return expanded


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone map from an uncertainty score to P(error)."""

    thresholds: np.ndarray  # sorted uncertainty values
    probabilities: np.ndarray  # fitted, non-decreasing

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Calibrated error probability for each score, by interpolation."""
        return np.interp(
            np.asarray(scores, dtype=np.float64),
            self.thresholds,
            self.probabilities,
            left=float(self.probabilities[0]),
            right=float(self.probabilities[-1]),
        )


def fit_isotonic(uncertainty: np.ndarray, errors: np.ndarray) -> IsotonicCalibrator:
    """Fit P(error | uncertainty), assuming error probability rises with it.

    ``uncertainty`` should increase as the model becomes less sure, pass raw
    entropy, or the negation of a margin or log-probability.
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if uncertainty.shape != errors.shape:
        raise ValueError("uncertainty and errors must have the same shape")
    if uncertainty.size == 0:
        raise ValueError("at least one calibration example is required")

    order = np.argsort(uncertainty, kind="stable")
    fitted = _pool_adjacent_violators(errors[order], np.ones(len(errors)))
    return IsotonicCalibrator(thresholds=uncertainty[order], probabilities=fitted)


def expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Binned ECE: mean gap between predicted probability and observed rate."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if probabilities.shape != outcomes.shape:
        raise ValueError("probabilities and outcomes must have the same shape")
    if probabilities.size == 0:
        raise ValueError("at least one observation is required")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities > low) & (probabilities <= high)
        if low == 0.0:
            mask |= probabilities == 0.0
        if not mask.any():
            continue
        gap = abs(probabilities[mask].mean() - outcomes[mask].mean())
        total += mask.sum() / len(probabilities) * gap
    return float(total)
