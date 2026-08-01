import numpy as np
import pytest

from gwel.router.calibration import (
    expected_calibration_error,
    fit_isotonic,
)


def test_isotonic_output_is_non_decreasing() -> None:
    rng = np.random.default_rng(0)
    uncertainty = np.sort(rng.uniform(0, 1, 200))
    errors = (rng.uniform(size=200) < uncertainty).astype(float)
    calibrator = fit_isotonic(uncertainty, errors)
    predictions = calibrator.predict(np.linspace(0, 1, 50))
    assert np.all(np.diff(predictions) >= -1e-9)


def test_recovers_a_known_monotone_relationship() -> None:
    # True P(error) = uncertainty, so calibrated output should track it.
    rng = np.random.default_rng(1)
    uncertainty = rng.uniform(0, 1, 4000)
    errors = (rng.uniform(size=4000) < uncertainty).astype(float)
    calibrator = fit_isotonic(uncertainty, errors)
    for probe in (0.2, 0.5, 0.8):
        assert calibrator.predict(np.array([probe]))[0] == pytest.approx(probe, abs=0.1)


def test_calibration_reduces_expected_calibration_error() -> None:
    rng = np.random.default_rng(2)
    uncertainty = rng.uniform(0, 1, 2000)
    # Raw signal is systematically overconfident: true error is uncertainty^2.
    errors = (rng.uniform(size=2000) < uncertainty**2).astype(float)
    raw_ece = expected_calibration_error(uncertainty, errors)
    calibrated = fit_isotonic(uncertainty, errors).predict(uncertainty)
    assert expected_calibration_error(calibrated, errors) < raw_ece


def test_predictions_clamp_outside_the_calibration_range() -> None:
    calibrator = fit_isotonic(np.array([0.2, 0.5, 0.8]), np.array([0.0, 0.0, 1.0]))
    assert calibrator.predict(np.array([-5.0]))[0] == calibrator.probabilities[0]
    assert calibrator.predict(np.array([5.0]))[0] == calibrator.probabilities[-1]


def test_perfect_calibration_scores_zero_ece() -> None:
    probabilities = np.array([0.0] * 50 + [1.0] * 50)
    outcomes = np.array([0.0] * 50 + [1.0] * 50)
    assert expected_calibration_error(probabilities, outcomes) == pytest.approx(0.0)


def test_worst_calibration_scores_one() -> None:
    probabilities = np.zeros(100)
    outcomes = np.ones(100)
    assert expected_calibration_error(probabilities, outcomes) == pytest.approx(1.0)


def test_validates_inputs() -> None:
    with pytest.raises(ValueError):
        fit_isotonic(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError):
        fit_isotonic(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        expected_calibration_error(np.zeros(3), np.zeros(2))
