import pytest

from gwel.router.evaluate import bootstrap_interval, paired_difference


def test_interval_brackets_the_mean() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5] * 10
    interval = bootstrap_interval(values, seed=7)
    assert interval.low <= interval.estimate <= interval.high
    assert interval.estimate == pytest.approx(0.3)


def test_constant_values_give_a_degenerate_interval() -> None:
    interval = bootstrap_interval([2.0] * 20)
    assert interval.low == interval.high == interval.estimate == 2.0


def test_single_observation_is_its_own_interval() -> None:
    assert bootstrap_interval([1.5]) .estimate == 1.5


def test_more_data_narrows_the_interval() -> None:
    pattern = [0.0, 1.0]
    narrow = bootstrap_interval(pattern * 200, seed=3)
    wide = bootstrap_interval(pattern * 10, seed=3)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_paired_difference_detects_a_consistent_gap() -> None:
    left = [0.5 + i * 0.01 for i in range(60)]
    right = [0.3 + i * 0.01 for i in range(60)]
    interval = paired_difference(left, right, seed=11)
    assert interval.estimate == pytest.approx(0.2)
    assert interval.low > 0  # the gap is real, not noise


def test_paired_difference_of_identical_policies_covers_zero() -> None:
    values = [0.1, 0.9, 0.4, 0.6] * 10
    interval = paired_difference(values, values)
    assert interval.low <= 0.0 <= interval.high


def test_paired_difference_requires_aligned_inputs() -> None:
    with pytest.raises(ValueError):
        paired_difference([1.0, 2.0], [1.0])


def test_bootstrap_requires_values() -> None:
    with pytest.raises(ValueError):
        bootstrap_interval([])
