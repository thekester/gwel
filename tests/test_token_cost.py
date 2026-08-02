import numpy as np
import pytest

from gwel.oracle.token_cost import extrapolation_span, fit_token_cost


def test_recovers_an_exact_affine_relationship() -> None:
    tokens = [0, 64, 320, 640]
    latency = [100.0 + 0.3 * t for t in tokens]
    model = fit_token_cost(tokens, latency)
    assert model.base_ms == pytest.approx(100.0)
    assert model.slope_ms_per_token == pytest.approx(0.3)
    assert model.residual_ms == pytest.approx(0.0, abs=1e-9)


def test_predicts_beyond_the_fitted_range() -> None:
    model = fit_token_cost([0, 64, 320], [100.0, 119.2, 196.0])
    assert model.predict(640)[0] == pytest.approx(100.0 + 0.3 * 640, abs=1.0)


def test_residual_exposes_a_bad_fit() -> None:
    # A step function is not affine, so the fit must report a large residual
    # rather than silently returning a line.
    model = fit_token_cost([0, 64, 320, 640], [100.0, 100.0, 300.0, 300.0])
    assert model.residual_ms > 20.0


def test_rejects_a_single_token_count() -> None:
    with pytest.raises(ValueError):
        fit_token_cost([320, 320], [200.0, 201.0])


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        fit_token_cost([0, 64], [100.0])


def test_extrapolation_span_reports_how_far_past_the_fit() -> None:
    model = fit_token_cost([0, 64, 320], [100.0, 119.2, 196.0])
    assert extrapolation_span(model, [0, 64, 320], [640]) == pytest.approx(1.0)
    assert extrapolation_span(model, [0, 64, 320], [200]) == 0.0


def test_predict_accepts_arrays() -> None:
    model = fit_token_cost([0, 320], [100.0, 196.0])
    assert model.predict(np.array([0, 320])).tolist() == pytest.approx([100.0, 196.0])
