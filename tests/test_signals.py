import math

import numpy as np
import pytest

from gwel.modeling.signals import (
    ConfidenceSignals,
    entropy_from_logits,
    signals_from_scores,
    top2_margin_from_logits,
)


def test_uniform_logits_have_max_entropy_and_zero_margin() -> None:
    logits = np.zeros(8)
    assert entropy_from_logits(logits) == pytest.approx(math.log(8))
    assert top2_margin_from_logits(logits) == pytest.approx(0.0)


def test_peaked_logits_have_low_entropy_and_high_margin() -> None:
    logits = np.array([20.0, 0.0, 0.0, 0.0])
    assert entropy_from_logits(logits) == pytest.approx(0.0, abs=1e-6)
    assert top2_margin_from_logits(logits) == pytest.approx(1.0, abs=1e-6)


def test_signals_from_scores_summarizes_steps() -> None:
    confident = np.array([10.0, 0.0, 0.0])
    uncertain = np.array([0.1, 0.0, 0.0])
    signals = signals_from_scores([confident, uncertain], chosen_ids=[0, 0])

    assert signals.num_tokens == 2
    assert signals.first_entropy == pytest.approx(entropy_from_logits(confident), abs=1e-9)
    assert signals.max_entropy == pytest.approx(entropy_from_logits(uncertain), abs=1e-9)
    assert signals.min_margin == pytest.approx(top2_margin_from_logits(uncertain), abs=1e-9)
    assert -signals.mean_logprob > 0  # log-probs are negative


def test_signals_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        signals_from_scores([np.zeros(4)], chosen_ids=[0, 1])
    with pytest.raises(ValueError):
        signals_from_scores([], chosen_ids=[])


def test_signals_dict_round_trip() -> None:
    signals = signals_from_scores([np.array([1.0, 0.5, 0.0])], chosen_ids=[1])
    assert ConfidenceSignals.from_dict(signals.to_dict()) == signals
