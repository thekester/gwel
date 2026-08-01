import numpy as np

from gwel.profiling.memory import current_hardware_state, track_memory


def test_tracker_reports_after_the_block() -> None:
    with track_memory(sample_interval_ms=5) as tracker:
        assert tracker.report is None  # not available until the block ends
        _ = np.zeros((512, 512))
    assert tracker.report is not None
    assert tracker.report.ram_peak_mb > 0.0


def test_peak_is_at_least_the_baseline() -> None:
    with track_memory(sample_interval_ms=5) as tracker:
        pass
    report = tracker.report
    assert report.ram_peak_mb >= report.ram_baseline_mb
    assert report.ram_delta_mb >= 0.0


def test_a_large_allocation_shows_up_in_the_peak() -> None:
    with track_memory(sample_interval_ms=2) as tracker:
        block = np.ones((4096, 4096), dtype=np.float64)  # ~128 MB, held live
        assert block[0, 0] == 1.0
    # The sampler runs on a thread; require only that it saw something sizeable.
    assert tracker.report.ram_delta_mb > 10.0


def test_tracking_survives_an_exception() -> None:
    tracker_ref = {}
    try:
        with track_memory(sample_interval_ms=5) as tracker:
            tracker_ref["t"] = tracker
            raise ValueError("boom")
    except ValueError:
        pass
    assert tracker_ref["t"].report is not None


def test_hardware_state_has_the_expected_schema() -> None:
    state = current_hardware_state()
    assert set(state) == {
        "ram_available_mb",
        "ram_used_fraction",
        "cpu_load_fraction",
        "vram_free_mb",
    }
    assert state["ram_available_mb"] > 0
    assert 0.0 <= state["ram_used_fraction"] <= 1.0
    # vram_free_mb is None on machines without CUDA, which is a valid answer.
    assert state["vram_free_mb"] is None or state["vram_free_mb"] > 0
