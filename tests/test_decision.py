import numpy as np
import pytest

from gwel.router.decision import (
    break_even_gain,
    escalation_delta,
    fit_correctness_rule,
    fit_gain_calibrator,
    fit_gain_rule,
    gain_calibration_error,
    signed_gain,
)

COSTS = {"cheap": 123.4, "full": 206.0, "probe": 20.3}


def test_signed_gain_encodes_repair_and_damage() -> None:
    gains = signed_gain(
        cheap_correct=[False, True, True, False],
        full_correct=[True, False, True, False],
    )
    # repaired, damaged, both right (no change), both wrong (no change)
    assert gains.tolist() == [1.0, -1.0, 0.0, 0.0]


def test_signed_gain_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        signed_gain([True, False], [True])


def test_break_even_falls_as_the_operator_values_accuracy_more() -> None:
    assert break_even_gain(100.0, 200.0) == pytest.approx(0.5)
    assert break_even_gain(100.0, 1000.0) == pytest.approx(0.1)


def test_break_even_rejects_a_non_positive_value() -> None:
    with pytest.raises(ValueError):
        break_even_gain(100.0, 0.0)


def test_probe_escalation_is_cheaper_than_entropy_escalation() -> None:
    """Reading mid-prefill refunds the cheap pass an escalated query abandons."""
    entropy = escalation_delta(COSTS, read="entropy")
    probe = escalation_delta(COSTS, read="probe")
    assert entropy == pytest.approx(206.0)
    assert probe == pytest.approx(20.3 + 206.0 - 123.4)
    assert probe < entropy


def test_escalation_delta_rejects_an_unknown_signal() -> None:
    with pytest.raises(ValueError):
        escalation_delta(COSTS, read="telepathy")


def test_gain_calibrator_is_monotone_and_bounded() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=500)
    # Higher score means escalation is more likely to repair than damage.
    gains = np.where(rng.uniform(size=500) < 1 / (1 + np.exp(-scores)), 1.0, -1.0)
    calibrator = fit_gain_calibrator(scores, gains)
    predicted = calibrator.predict(np.linspace(-3, 3, 50))
    assert np.all(np.diff(predicted) >= -1e-9)
    assert predicted.min() >= -1.0 - 1e-9
    assert predicted.max() <= 1.0 + 1e-9


def test_gain_calibrator_recovers_a_known_expected_gain() -> None:
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 6000)
    # E[G | s] = 2s - 1 by construction.
    gains = np.where(rng.uniform(size=6000) < scores, 1.0, -1.0)
    calibrator = fit_gain_calibrator(scores, gains)
    for point in (0.25, 0.5, 0.75):
        assert calibrator.predict(np.array([point]))[0] == pytest.approx(
            2 * point - 1, abs=0.12
        )


def test_a_rule_never_fires_when_no_gain_can_pay_for_the_latency() -> None:
    rng = np.random.default_rng(2)
    scores = rng.normal(size=300)
    gains = np.where(rng.uniform(size=300) < 0.3, 1.0, 0.0)
    # Valuing a correct answer at less than the latency it costs makes tau > 1,
    # which no expected gain can clear.
    rule = fit_gain_rule(scores, gains, delta_ms=200.0, value_ms_per_correct=100.0)
    assert rule.tau > 1.0
    assert not rule.escalate(scores).any()
    assert rule.expected_saving(scores) == 0.0


def test_a_rule_fires_everywhere_when_accuracy_is_valued_highly() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=300)
    gains = np.where(rng.uniform(size=300) < 0.6, 1.0, 0.0)
    rule = fit_gain_rule(scores, gains, delta_ms=100.0, value_ms_per_correct=1e6)
    assert rule.escalate(scores).all()


def test_the_rule_only_fires_where_predicted_gain_clears_break_even() -> None:
    rng = np.random.default_rng(4)
    scores = rng.uniform(0, 1, 800)
    gains = np.where(rng.uniform(size=800) < scores, 1.0, -1.0)
    rule = fit_gain_rule(scores, gains, delta_ms=200.0, value_ms_per_correct=800.0)
    fires = rule.escalate(scores)
    predicted = rule.expected_gain(scores)
    assert np.all(predicted[fires] > rule.tau)
    assert np.all(predicted[~fires] <= rule.tau)


def test_correctness_rule_over_escalates_when_escalation_cannot_repair() -> None:
    """The UCCI assumption, isolated.

    Here the cheap pass fails on half the queries and escalation repairs none of
    them, so the correct decision is never to escalate. A rule calibrated on the
    signed gain sees that; one calibrated on correctness alone believes the
    escalated pass will deliver its marginal accuracy to every query it is given.
    """
    rng = np.random.default_rng(5)
    n = 600
    scores = rng.normal(size=n)
    cheap_ok = rng.uniform(size=n) < 1 / (1 + np.exp(scores))
    full_ok = cheap_ok.copy()  # escalation changes nothing
    gains = signed_gain(cheap_ok, full_ok)

    kwargs = {"delta_ms": 100.0, "value_ms_per_correct": 800.0}
    gain_rule = fit_gain_rule(scores, gains, **kwargs)
    ucci_rule = fit_correctness_rule(
        scores, cheap_ok, full_accuracy=float(full_ok.mean()), **kwargs
    )
    assert not gain_rule.escalate(scores).any()
    assert ucci_rule.escalate(scores).any()


def test_gain_calibration_error_is_zero_for_a_perfect_predictor() -> None:
    realised = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0])
    assert gain_calibration_error(realised, realised) == pytest.approx(0.0)


def test_gain_calibration_error_penalises_a_biased_predictor() -> None:
    realised = np.zeros(100)
    predicted = np.full(100, 0.5)
    assert gain_calibration_error(predicted, realised) == pytest.approx(0.5)


def test_gain_calibration_error_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        gain_calibration_error(np.zeros(3), np.zeros(4))


def test_per_query_rule_charges_each_query_its_own_break_even() -> None:
    from gwel.router.decision import fit_per_query_gain_rule

    rng = np.random.default_rng(6)
    scores = rng.uniform(0, 1, 600)
    gains = np.where(rng.uniform(size=600) < scores, 1.0, -1.0)
    rule = fit_per_query_gain_rule(scores, gains, value_ms_per_correct=800.0)

    cheap = np.full(600, 100.0)
    dear = np.full(600, 400.0)
    # A dearer escalation demands a larger expected gain, so it fires less often.
    assert rule.escalate(scores, cheap).sum() > rule.escalate(scores, dear).sum()
    assert rule.tau(cheap)[0] == pytest.approx(0.125)
    assert rule.tau(dear)[0] == pytest.approx(0.5)


def test_per_query_rule_matches_the_global_rule_at_a_constant_price() -> None:
    """The refinement must reduce to the original when prices do not vary."""
    from gwel.router.decision import fit_gain_rule, fit_per_query_gain_rule

    rng = np.random.default_rng(7)
    scores = rng.uniform(0, 1, 400)
    gains = np.where(rng.uniform(size=400) < scores, 1.0, -1.0)
    flat = fit_gain_rule(scores, gains, delta_ms=200.0, value_ms_per_correct=800.0)
    per_query = fit_per_query_gain_rule(scores, gains, value_ms_per_correct=800.0)
    assert (
        per_query.escalate(scores, np.full(400, 200.0)) == flat.escalate(scores)
    ).all()


def test_per_query_rule_rejects_a_non_positive_value() -> None:
    from gwel.router.decision import fit_per_query_gain_rule

    with pytest.raises(ValueError):
        fit_per_query_gain_rule(np.zeros(4), np.zeros(4), value_ms_per_correct=0.0)


def _ladder(value: float):
    from gwel.router.decision import fit_ladder_rule

    rng = np.random.default_rng(11)
    scores = rng.uniform(0, 1, 800)
    # The cheap rung repairs proportionally to the score; the dear rung repairs
    # only slightly more, so it is rarely worth its extra price.
    mid = np.where(rng.uniform(size=800) < 0.8 * scores, 1.0, 0.0)
    top = np.maximum(mid, np.where(rng.uniform(size=800) < 0.1 * scores, 1.0, 0.0))
    return scores, fit_ladder_rule(
        scores, {"mid": mid, "top": top}, value_ms_per_correct=value
    )


def test_ladder_prefers_the_cheap_rung_when_the_dear_one_adds_little() -> None:
    scores, rule = _ladder(800.0)
    deltas = np.column_stack([np.full(800, 80.0), np.full(800, 400.0)])
    chosen = rule.choose(scores, deltas)
    assert (chosen == 0).sum() > (chosen == 1).sum()


def test_ladder_takes_the_dear_rung_when_it_becomes_free() -> None:
    scores, rule = _ladder(800.0)
    # Same gain, same price as the middle rung: the argmax should move up.
    equal = np.column_stack([np.full(800, 80.0), np.full(800, 80.0)])
    assert (rule.choose(scores, equal) == 1).sum() > 0


def test_ladder_answers_cheap_when_no_rung_repays_itself() -> None:
    scores, rule = _ladder(10.0)
    deltas = np.column_stack([np.full(800, 80.0), np.full(800, 400.0)])
    assert (rule.choose(scores, deltas) == -1).all()


def test_ladder_with_one_rung_reduces_to_the_binary_rule() -> None:
    from gwel.router.decision import fit_gain_rule, fit_ladder_rule

    rng = np.random.default_rng(12)
    scores = rng.uniform(0, 1, 400)
    gains = np.where(rng.uniform(size=400) < scores, 1.0, -1.0)
    ladder = fit_ladder_rule(scores, {"only": gains}, value_ms_per_correct=800.0)
    binary = fit_gain_rule(scores, gains, delta_ms=200.0, value_ms_per_correct=800.0)
    chosen = ladder.choose(scores, np.full((400, 1), 200.0))
    assert ((chosen == 0) == binary.escalate(scores)).all()


def test_ladder_rejects_a_delta_matrix_of_the_wrong_width() -> None:
    scores, rule = _ladder(800.0)
    with pytest.raises(ValueError):
        rule.choose(scores, np.full((800, 3), 80.0))


def test_ladder_rejects_an_empty_rung_set() -> None:
    from gwel.router.decision import fit_ladder_rule

    with pytest.raises(ValueError):
        fit_ladder_rule(np.zeros(4), {}, value_ms_per_correct=800.0)
