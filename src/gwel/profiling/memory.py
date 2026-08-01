"""Peak RAM and VRAM tracking around a measured operation.

Host RSS is sampled from a background thread because Python offers no portable
high-water-mark API mid-process. CUDA peaks use ``torch.cuda`` allocator
statistics when torch is available; both are optional and degrade to ``None``.
"""

import threading
import time
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class MemoryReport:
    """Peak memory observed during a tracked section."""

    ram_baseline_mb: float
    ram_peak_mb: float
    vram_peak_mb: float | None

    @property
    def ram_delta_mb(self) -> float:
        return self.ram_peak_mb - self.ram_baseline_mb


class _RssSampler(threading.Thread):
    """Daemon thread recording the peak RSS of the current process."""

    def __init__(self, interval_s: float) -> None:
        super().__init__(daemon=True)
        self._interval_s = interval_s
        self._process = psutil.Process()
        self._stop_event = threading.Event()
        self.peak_bytes = self._process.memory_info().rss

    def run(self) -> None:
        while not self._stop_event.is_set():
            rss = self._process.memory_info().rss
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            self._stop_event.wait(self._interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join()
        rss = self._process.memory_info().rss
        if rss > self.peak_bytes:
            self.peak_bytes = rss


def _cuda_peak_tracker(device: int | None) -> tuple[bool, int | None]:
    """Reset CUDA peak stats if torch+CUDA are usable; return (active, device)."""
    try:
        import torch
    except ImportError:
        return False, None
    if not torch.cuda.is_available():
        return False, None
    index = device if device is not None else torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(index)
    return True, index


class MemoryTracker:
    """Collects the :class:`MemoryReport` after a tracked section ends."""

    def __init__(self) -> None:
        self.report: MemoryReport | None = None


@contextmanager
def track_memory(
    *,
    sample_interval_ms: int = 10,
    cuda_device: int | None = None,
) -> Iterator[MemoryTracker]:
    """Track peak RSS (and CUDA VRAM when available) around a code block."""
    tracker = MemoryTracker()
    baseline_mb = psutil.Process().memory_info().rss / 1e6
    cuda_active, cuda_index = _cuda_peak_tracker(cuda_device)
    sampler = _RssSampler(interval_s=max(sample_interval_ms, 1) / 1000.0)
    sampler.start()
    try:
        yield tracker
    finally:
        sampler.stop()
        vram_peak_mb: float | None = None
        if cuda_active:
            import torch

            vram_peak_mb = torch.cuda.max_memory_allocated(cuda_index) / 1e6
        tracker.report = MemoryReport(
            ram_baseline_mb=baseline_mb,
            ram_peak_mb=sampler.peak_bytes / 1e6,
            vram_peak_mb=vram_peak_mb,
        )


def current_hardware_state() -> dict[str, float | None]:
    """Snapshot of live resource pressure, usable as router input features."""
    memory = psutil.virtual_memory()
    state: dict[str, float | None] = {
        "ram_available_mb": memory.available / 1e6,
        "ram_used_fraction": memory.percent / 100.0,
        "cpu_load_fraction": psutil.cpu_percent(interval=None) / 100.0,
        "vram_free_mb": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, _total = torch.cuda.mem_get_info()
            state["vram_free_mb"] = free_bytes / 1e6
    except ImportError:
        pass
    return state
