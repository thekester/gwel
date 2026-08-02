"""Per-example latency from visual-token count, replacing a flat per-config cost.

Every policy comparison in this project charges each configuration a single
measured latency. That is safe only if a configuration costs the same on every
image, and it does not: the processor caps its target at the input's longest
side, so a resolution-capped ``full`` pass spends 64 visual tokens on a small
image and 640 on a large one. On our pilot ``full`` averages 424 tokens while
the profiling image happened to yield 320, so a flat cost taken from that image
understates the escalation it is meant to price.

The fix is to price a pass by what it actually spent. Latency is affine in the
visual-token count at fixed answer length --- the encoder and prefill both scale
with the token sequence while decode does not --- so a two-parameter fit over
the profiled configurations transfers to any token count:

    t(v) = base + slope * v.

``base`` is the cost of the same pass with no image, and ``slope`` the marginal
cost per visual token. Both are read from the component profile rather than
assumed, and :func:`fit_token_cost` reports the residual so a bad fit is visible
instead of silent.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenCostModel:
    """Affine map from visual-token count to pass latency, in milliseconds."""

    base_ms: float
    slope_ms_per_token: float
    residual_ms: float  # worst absolute error over the fitted points
    points: int

    def predict(self, visual_tokens: np.ndarray | Sequence[int] | int) -> np.ndarray:
        """Predicted latency, always as a 1-d array so a scalar indexes cleanly."""
        tokens = np.atleast_1d(np.asarray(visual_tokens, dtype=np.float64))
        return self.base_ms + self.slope_ms_per_token * tokens


def fit_token_cost(
    visual_tokens: Sequence[int], latency_ms: Sequence[float]
) -> TokenCostModel:
    """Least-squares fit of latency against visual-token count.

    At least two distinct token counts are required; the residual is returned so
    a caller can refuse the model rather than trust an extrapolation it does not
    support.
    """
    tokens = np.asarray(visual_tokens, dtype=np.float64)
    latency = np.asarray(latency_ms, dtype=np.float64)
    if tokens.shape != latency.shape:
        raise ValueError("visual_tokens and latency_ms must have the same shape")
    if len(np.unique(tokens)) < 2:
        raise ValueError("at least two distinct token counts are required")

    design = np.column_stack([np.ones_like(tokens), tokens])
    (base, slope), *_ = np.linalg.lstsq(design, latency, rcond=None)
    residual = float(np.abs(latency - (base + slope * tokens)).max())
    return TokenCostModel(
        base_ms=float(base),
        slope_ms_per_token=float(slope),
        residual_ms=residual,
        points=len(tokens),
    )


def extrapolation_span(model: TokenCostModel, fitted: Sequence[int], asked: Sequence[int]) -> float:
    """How far beyond the fitted range a prediction reaches, as a fraction.

    Reported because the whole point of the correction is that the profiling
    image under-covered the range actually used; a caller should know when the
    replacement is itself extrapolating.
    """
    fitted_max = float(np.max(fitted))
    asked_max = float(np.max(asked))
    if fitted_max <= 0:
        raise ValueError("fitted token counts must be positive")
    return max(0.0, (asked_max - fitted_max) / fitted_max)
