import pytest

from gwel.profiling.stats import repeat_measure, summarize_repeats


def test_summarize_repeats_median_and_iqr() -> None:
    stats = summarize_repeats([1, 2, 3, 4, 5])
    assert stats.n == 5
    assert stats.median == 3
    assert stats.iqr == pytest.approx(2.0)
    assert stats.minimum == 1
    assert stats.maximum == 5
    assert stats.p95 == pytest.approx(4.8)


def test_summarize_repeats_requires_values() -> None:
    with pytest.raises(ValueError):
        summarize_repeats([])


def test_repeat_measure_discards_warmup() -> None:
    calls: list[int] = []

    def fn() -> int:
        calls.append(len(calls))
        return calls[-1]

    results = repeat_measure(fn, repeats=3, warmup=2)
    assert len(calls) == 5
    assert results == [2, 3, 4]  # warmup results discarded


def test_repeat_measure_validates_arguments() -> None:
    with pytest.raises(ValueError):
        repeat_measure(lambda: None, repeats=0)
    with pytest.raises(ValueError):
        repeat_measure(lambda: None, repeats=1, warmup=-1)
