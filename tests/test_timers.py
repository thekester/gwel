import time

import pytest

from gwel.profiling.timers import FirstTokenTimer, Timer


def test_timer_measures_elapsed_time() -> None:
    with Timer() as timer:
        time.sleep(0.02)
    assert 15.0 < timer.elapsed_ms < 200.0


def test_timer_is_zero_before_use() -> None:
    assert Timer().elapsed_ms == 0.0


def test_timer_records_even_when_the_block_raises() -> None:
    timer = Timer()
    with pytest.raises(RuntimeError):
        with timer:
            raise RuntimeError("boom")
    assert timer.elapsed_ms > 0.0


def test_first_token_timer_follows_the_streamer_protocol() -> None:
    # generate() calls put() once with the prompt ids, then once per new token.
    timer = FirstTokenTimer()
    timer.arm()
    time.sleep(0.01)
    timer.put("prompt ids")   # ignored
    assert timer.ttft_ms is None
    time.sleep(0.01)
    timer.put("first token")  # this is the one that counts
    timer.end()
    assert timer.ttft_ms is not None
    assert timer.ttft_ms > 15.0


def test_first_token_time_is_not_overwritten_by_later_tokens() -> None:
    timer = FirstTokenTimer()
    timer.arm()
    timer.put("prompt")
    timer.put("first")
    first = timer.ttft_ms
    time.sleep(0.02)
    timer.put("second")
    timer.put("third")
    assert timer.ttft_ms == first


def test_ttft_is_none_without_arming() -> None:
    timer = FirstTokenTimer()
    timer.put("prompt")
    timer.put("first")
    assert timer.ttft_ms is None


def test_arming_resets_a_previous_measurement() -> None:
    timer = FirstTokenTimer()
    timer.arm()
    timer.put("prompt")
    timer.put("first")
    assert timer.ttft_ms is not None
    timer.arm()
    assert timer.ttft_ms is None
