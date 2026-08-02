"""Escalate by a decision rule derived from the cost model, not a tuned rate.

Every policy in this project so far picks an *escalation rate* and thresholds a
score at that quantile. That is a tuning exercise, not a decision: the rate is
chosen by sweeping, and nothing connects it to the cost function the oracle is
already labelled under.

Kotte (arXiv 2605.18796, UCCI) closes exactly this gap for text cascades:
calibrate the uncertainty signal to a probability, then pick the threshold by
constrained cost minimisation, and threshold policies on the calibrated
probability are cost-optimal (their Theorem 1). Their operational summary is
"calibrate first, threshold second".

**Their Theorem 1 does not transfer to a visual cascade, and this module is the
correction.** Their assumption (ii) is that the strong model attains a fixed
accuracy on the escalated subset, invariant to which queries are escalated.
Under that assumption the escalation value of a query is determined by the weak
model's error probability alone, so calibrating ``P(cheap wrong)`` is enough.
For a VLM it is not: escalation is *non-monotone*, repairing 21% of queries
and damaging 4% of them, so the strong pass is not a fixed accuracy but a second
random outcome. The quantity that decides whether compute is well spent is the
signed gain

    G = 1[cheap wrong and full right] - 1[cheap right and full wrong]

whose conditional expectation ``E[G | x] = P(full right | x) - P(cheap right |
x)`` is what must be calibrated. Calibrating correctness instead is the
calibration-stage version of the target error this paper reports for ranking.

Given the linear cost of ``oracle.cost``, escalating query ``x`` changes
expected cost by

    dJ(x) = -w_e * E[G | x] + lambda_t * dt,

so the optimal rule is to escalate iff ``E[G | x] > dt / V`` where
``V = w_e / lambda_t`` is the latency an operator will spend to buy one extra
correct answer. The break-even gain ``tau = dt / V`` is a decision threshold in
the units of the thing being predicted, and it is *signal-dependent*: a probe
read mid-prefill has a smaller ``dt`` than a signal read after the cheap pass
completes, so the same operator preference justifies escalating on weaker
evidence when the signal is cheaper.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .calibration import IsotonicCalibrator, fit_isotonic

# The signed gain takes values in {-1, 0, +1}; isotonic regression fits a
# probability in [0, 1], so it is fitted on the affine rescaling and mapped back.
_GAIN_LOW, _GAIN_HIGH = -1.0, 1.0


def signed_gain(cheap_correct: Sequence[bool], full_correct: Sequence[bool]) -> np.ndarray:
    """``G`` per example: +1 if escalation repairs, -1 if it damages, else 0."""
    cheap = np.asarray(cheap_correct, dtype=bool)
    full = np.asarray(full_correct, dtype=bool)
    if cheap.shape != full.shape:
        raise ValueError("cheap_correct and full_correct must have the same shape")
    return (~cheap & full).astype(np.float64) - (cheap & ~full).astype(np.float64)


def break_even_gain(delta_ms: float, value_ms_per_correct: float) -> float:
    """Expected gain that justifies spending ``delta_ms`` extra milliseconds.

    ``value_ms_per_correct`` is the single interpretable knob: how much latency
    the operator will pay for one additional correct answer. Following the
    normalisation of Moslem et al. (arXiv 2606.27457), stating the trade-off in
    the units of the outcome makes it readable across devices, where a raw
    per-millisecond weight is not.
    """
    if value_ms_per_correct <= 0:
        raise ValueError("value_ms_per_correct must be positive")
    return float(delta_ms) / float(value_ms_per_correct)


def escalation_delta(costs: dict[str, float], *, read: str) -> float:
    """Extra latency an escalated query pays, relative to answering cheap.

    Reading output entropy requires the cheap pass to finish, so escalation adds
    the whole full-resolution pass. Reading a pre-generation probe abandons the
    remainder of the cheap pass, so the escalated query pays the probe prefix
    plus the full pass and *refunds* the cheap pass it never completed. This is
    Eq. (policy cost) of the paper, differenced.
    """
    cheap, full, probe = costs["cheap"], costs["full"], costs["probe"]
    if read == "entropy":
        return float(full)
    if read == "probe":
        return float(probe + full - cheap)
    if read == "none":
        return float(full - cheap)
    raise ValueError(f"unknown read {read!r}")


@dataclass(frozen=True)
class GainCalibrator:
    """Monotone map from a routing score to the expected signed gain ``E[G|s]``.

    Isotonic regression assumes only that the mapping is monotone, which a score
    that *ranks* escalation value satisfies by construction, the same argument
    UCCI makes for correctness, applied to the gain.
    """

    isotonic: IsotonicCalibrator

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Calibrated expected gain in [-1, +1] for each score."""
        unit = self.isotonic.predict(np.asarray(scores, dtype=np.float64))
        return _GAIN_LOW + unit * (_GAIN_HIGH - _GAIN_LOW)


def fit_gain_calibrator(scores: np.ndarray, gains: np.ndarray) -> GainCalibrator:
    """Fit ``E[G | score]`` by isotonic regression on the rescaled gain.

    ``scores`` must be oriented so that higher means escalation is more
    valuable; ``gains`` is the output of :func:`signed_gain`.
    """
    scores = np.asarray(scores, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    if scores.shape != gains.shape:
        raise ValueError("scores and gains must have the same shape")
    unit = (gains - _GAIN_LOW) / (_GAIN_HIGH - _GAIN_LOW)
    return GainCalibrator(isotonic=fit_isotonic(scores, unit))


@dataclass(frozen=True)
class EscalationRule:
    """A calibrated, cost-derived escalation decision.

    Unlike a quantile threshold, this rule has no free rate: the operating point
    falls out of ``value_ms_per_correct`` and the measured latencies.
    """

    calibrator: GainCalibrator
    tau: float
    delta_ms: float
    value_ms_per_correct: float

    def expected_gain(self, scores: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(scores)

    def escalate(self, scores: np.ndarray) -> np.ndarray:
        """Escalate exactly where the expected gain clears the break-even."""
        return self.expected_gain(scores) > self.tau

    def expected_saving(self, scores: np.ndarray) -> float:
        """Expected cost improvement per query, in milliseconds, over never escalating.

        Positive means the rule believes it is buying accuracy worth more than
        the latency it spends. Reported because a rule that fires on no query is
        a legitimate answer here, and this number says so explicitly.
        """
        gains = self.expected_gain(scores)
        fires = gains > self.tau
        if not fires.any():
            return 0.0
        value = gains[fires] * self.value_ms_per_correct - self.delta_ms
        return float(value.sum() / len(scores))


@dataclass(frozen=True)
class PerQueryEscalationRule:
    """The same rule with a break-even that varies per query.

    A global ``tau`` assumes escalation costs the same everywhere. It does not:
    a resolution-capped pass costs what the image allows, so escalating a large
    image is dearer than escalating a small one. That matters here rather than
    being a detail, because the two effects are correlated in the *worst*
    direction --- the queries more pixels help are the queries with more pixels
    to add, so a global threshold systematically under-charges exactly the
    escalations it most wants to make.

    Charging each query its own ``delta_ms`` restores the comparison the cost
    model intends: escalate iff this query's expected gain repays *this query's*
    extra latency.
    """

    calibrator: GainCalibrator
    value_ms_per_correct: float

    def expected_gain(self, scores: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(scores)

    def tau(self, delta_ms: np.ndarray) -> np.ndarray:
        """Per-query break-even gain."""
        return np.asarray(delta_ms, dtype=np.float64) / self.value_ms_per_correct

    def escalate(self, scores: np.ndarray, delta_ms: np.ndarray) -> np.ndarray:
        return self.expected_gain(scores) > self.tau(delta_ms)


def fit_per_query_gain_rule(
    scores: np.ndarray,
    gains: np.ndarray,
    *,
    value_ms_per_correct: float,
) -> PerQueryEscalationRule:
    """Calibrate the gain once; the threshold is applied per query at serve time."""
    if value_ms_per_correct <= 0:
        raise ValueError("value_ms_per_correct must be positive")
    return PerQueryEscalationRule(
        calibrator=fit_gain_calibrator(scores, gains),
        value_ms_per_correct=float(value_ms_per_correct),
    )


def fit_gain_rule(
    scores: np.ndarray,
    gains: np.ndarray,
    *,
    delta_ms: float,
    value_ms_per_correct: float,
) -> EscalationRule:
    """Calibrate the gain, then threshold at the cost-derived break-even."""
    return EscalationRule(
        calibrator=fit_gain_calibrator(scores, gains),
        tau=break_even_gain(delta_ms, value_ms_per_correct),
        delta_ms=float(delta_ms),
        value_ms_per_correct=float(value_ms_per_correct),
    )


@dataclass(frozen=True)
class LadderRule:
    """Escalate to the cheapest rung that repays itself, not to the top one.

    Published escalation is binary: answer from a thumbnail, or run the full
    image. That collapses a second decision --- *how far* to escalate --- which
    our measurements say carries most of the money. On the pilot only $5.2\\%$ of
    queries need full resolution while $17.7\\%$ are answered by an intermediate
    rung, so a binary policy over-serves three quarters of what it escalates.

    The generalisation is direct. Rung $r$ has its own expected gain over
    answering cheap, $\\mathbb{E}[G_r \\mid x]$, and its own extra latency
    $\\Delta t_r$. The rule picks whichever rung maximises the net value

        V * E[G_r | x] - dt_r,

    and stays at the cheap pass when no rung is positive. With one rung this
    reduces exactly to :class:`EscalationRule`; the binary policy is the special
    case that has deleted the middle of its own ladder.

    Rungs are held in increasing cost order and each carries its own calibrator,
    fitted on the same score.
    """

    rungs: tuple[str, ...]
    calibrators: tuple[GainCalibrator, ...]
    value_ms_per_correct: float

    def __post_init__(self) -> None:
        if len(self.rungs) != len(self.calibrators):
            raise ValueError("each rung needs a calibrator")
        if not self.rungs:
            raise ValueError("at least one rung is required")

    def net_value(self, scores: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """Net milliseconds gained by each rung, as ``(queries, rungs)``.

        ``deltas`` is ``(queries, rungs)`` so a rung may cost a different amount
        on different images, which is what a resolution-capped pass does.
        """
        scores = np.asarray(scores, dtype=np.float64)
        deltas = np.atleast_2d(np.asarray(deltas, dtype=np.float64))
        if deltas.shape[1] != len(self.rungs):
            raise ValueError("deltas must have one column per rung")
        gains = np.column_stack([c.predict(scores) for c in self.calibrators])
        return gains * self.value_ms_per_correct - deltas

    def choose(self, scores: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """Index of the chosen rung per query, or ``-1`` to answer cheap."""
        value = self.net_value(scores, deltas)
        best = np.argmax(value, axis=1)
        worth_it = value[np.arange(len(best)), best] > 0.0
        return np.where(worth_it, best, -1)


def fit_ladder_rule(
    scores: np.ndarray,
    rung_gains: dict[str, np.ndarray],
    *,
    value_ms_per_correct: float,
) -> LadderRule:
    """Calibrate one gain map per rung, over the same routing score.

    ``rung_gains`` maps a rung name to its signed gain against the cheap pass,
    in increasing cost order (Python dictionaries preserve insertion order,
    and the caller is expected to supply them that way).
    """
    if value_ms_per_correct <= 0:
        raise ValueError("value_ms_per_correct must be positive")
    if not rung_gains:
        raise ValueError("at least one rung is required")
    return LadderRule(
        rungs=tuple(rung_gains),
        calibrators=tuple(
            fit_gain_calibrator(scores, gains) for gains in rung_gains.values()
        ),
        value_ms_per_correct=float(value_ms_per_correct),
    )


@dataclass(frozen=True)
class CorrectnessRule:
    """UCCI transposed literally: calibrate ``P(cheap wrong)``, threshold on it.

    This is the baseline the correction is measured against. It assumes, as
    UCCI's Theorem 1 does, that the escalated pass delivers a fixed accuracy
    ``gamma`` regardless of which queries are sent to it, so the expected gain
    of escalating reduces to ``gamma - (1 - p_error)``.
    """

    calibrator: IsotonicCalibrator
    gamma: float
    tau: float

    def expected_gain(self, scores: np.ndarray) -> np.ndarray:
        error = self.calibrator.predict(np.asarray(scores, dtype=np.float64))
        return self.gamma - (1.0 - error)

    def escalate(self, scores: np.ndarray) -> np.ndarray:
        return self.expected_gain(scores) > self.tau


def fit_correctness_rule(
    scores: np.ndarray,
    cheap_correct: Sequence[bool],
    *,
    full_accuracy: float,
    delta_ms: float,
    value_ms_per_correct: float,
) -> CorrectnessRule:
    """Fit the UCCI-style rule on the same scores, for a like-for-like contrast.

    ``full_accuracy`` is the marginal accuracy of the escalated configuration,
    which is exactly the quantity UCCI's assumption (ii) treats as invariant.
    """
    errors = 1.0 - np.asarray(cheap_correct, dtype=np.float64)
    return CorrectnessRule(
        calibrator=fit_isotonic(np.asarray(scores, dtype=np.float64), errors),
        gamma=float(full_accuracy),
        tau=break_even_gain(delta_ms, value_ms_per_correct),
    )


def gain_calibration_error(
    predicted: np.ndarray,
    realised: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Binned calibration error for a signed-gain prediction.

    The analogue of ECE for a quantity on [-1, +1]: bin by predicted gain
    and average the absolute gap to the realised mean gain, weighted by bin mass.
    Reported because a rule that thresholds a *magnitude* is only as trustworthy
    as the magnitude, where a ranking metric would not notice.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    realised = np.asarray(realised, dtype=np.float64)
    if predicted.shape != realised.shape:
        raise ValueError("predicted and realised must have the same shape")
    if predicted.size == 0:
        raise ValueError("at least one observation is required")

    edges = np.linspace(_GAIN_LOW, _GAIN_HIGH, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (predicted > low) & (predicted <= high)
        if low == _GAIN_LOW:
            mask |= predicted == _GAIN_LOW
        if not mask.any():
            continue
        total += mask.sum() / predicted.size * abs(predicted[mask].mean() - realised[mask].mean())
    return float(total)
