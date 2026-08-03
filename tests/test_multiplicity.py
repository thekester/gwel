import numpy as np
import pytest

from gwel.router.multiplicity import (
    bootstrap_p_value,
    family_wise_error,
    holm_bonferroni,
)


def test_a_clear_effect_gets_a_small_p_value() -> None:
    differences = np.full(200, 0.5) + np.random.default_rng(0).normal(0, 0.05, 200)
    assert bootstrap_p_value(differences.tolist()) < 0.001


def test_the_false_positive_rate_under_the_null_is_near_the_nominal_level() -> None:
    """The property a p-value must have, tested as a rate rather than one draw.

    A single null sample can land anywhere in [0, 1] by construction, so
    asserting one p-value is large tests nothing. What must hold is that across
    many null experiments, the fraction below 0.05 is near 0.05.
    """
    rng = np.random.default_rng(1)
    p_values = [
        bootstrap_p_value(rng.normal(0, 1.0, 200).tolist(), resamples=2000, seed=seed)
        for seed in range(120)
    ]
    rate = float(np.mean([p <= 0.05 for p in p_values]))
    assert 0.0 <= rate <= 0.12, f"false positive rate {rate:.3f} far from nominal 0.05"
    assert np.median(p_values) > 0.25


def test_p_value_cannot_beat_the_resolution_of_the_bootstrap() -> None:
    differences = np.full(50, 10.0)
    assert bootstrap_p_value(differences.tolist(), resamples=1000) >= 1 / 1000


def test_p_value_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_p_value([])


def test_holm_is_stricter_than_no_correction_and_kinder_than_bonferroni() -> None:
    tests = [("a", 0.01), ("b", 0.02), ("c", 0.03), ("d", 0.04)]
    results = holm_bonferroni(tests, alpha=0.05)
    survivors = [r.name for r in results if r.survives]
    # Uncorrected, all four clear 0.05. Plain Bonferroni needs p <= 0.0125 and
    # keeps only "a". Holm keeps "a" and "b": 0.01*4=0.04 and 0.02*3=0.06 -> the
    # step-down stops at b, so only "a" survives here.
    assert survivors == ["a"]
    assert all(r.adjusted >= r.p_value for r in results)


def test_holm_keeps_a_strong_result_in_a_large_family() -> None:
    tests = [("strong", 0.0001)] + [(f"weak{i}", 0.4) for i in range(11)]
    results = {r.name: r for r in holm_bonferroni(tests)}
    assert results["strong"].survives
    assert not results["weak0"].survives


def test_adjusted_p_values_are_monotone_in_the_sorted_order() -> None:
    tests = [("a", 0.001), ("b", 0.20), ("c", 0.01), ("d", 0.5)]
    results = {r.name: r.adjusted for r in holm_bonferroni(tests)}
    ordered = [results[name] for name, _ in sorted(tests, key=lambda t: t[1])]
    assert ordered == sorted(ordered)


def test_step_down_stops_at_the_first_failure() -> None:
    """A later small p-value must not be rescued once the sequence has failed."""
    tests = [("a", 0.04), ("b", 0.041), ("c", 0.042)]
    results = holm_bonferroni(tests, alpha=0.05)
    assert not any(r.survives for r in results)


def test_empty_family_is_handled() -> None:
    assert holm_bonferroni([]) == []


def test_holm_rejects_a_bad_alpha() -> None:
    with pytest.raises(ValueError):
        holm_bonferroni([("a", 0.01)], alpha=1.5)


def test_family_wise_error_grows_with_the_number_of_questions() -> None:
    assert family_wise_error(1) == pytest.approx(0.05)
    assert family_wise_error(12) == pytest.approx(0.46, abs=0.01)
    assert family_wise_error(0) == pytest.approx(0.0)


def test_family_wise_error_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError):
        family_wise_error(-1)
