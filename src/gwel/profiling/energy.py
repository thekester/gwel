"""Energy measurement backends.

Two real backends are provided and both may run simultaneously:

- :class:`RaplBackend` reads Intel RAPL package counters from
  ``/sys/class/powercap`` (Linux, CPU-side energy, counter-based).
- :class:`NvmlPowerBackend` samples GPU power draw through NVML and
  integrates it over time (trapezoidal rule).

Counters are preferred over sampling when available. Every reading is in
millijoules; a backend returns ``None`` when it cannot produce a value, so
records stay honest about what was actually measured.
"""

import glob
import threading
import time
from abc import ABC, abstractmethod


class EnergyBackend(ABC):
    """One source of energy readings around a measured section."""

    name: str = "abstract"

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> float | None:
        """Return energy in millijoules since :meth:`start`, or ``None``."""


class RaplBackend(EnergyBackend):
    """CPU package energy from Linux powercap RAPL counters."""

    name = "rapl"
    _PATTERN = "/sys/class/powercap/intel-rapl:[0-9]*"

    def __init__(self) -> None:
        self._domains = sorted(glob.glob(self._PATTERN))
        self._start_uj: list[int] = []
        self._ranges_uj: list[int] = []

    @classmethod
    def available(cls) -> bool:
        domains = glob.glob(cls._PATTERN)
        if not domains:
            return False
        try:
            with open(f"{domains[0]}/energy_uj", encoding="ascii") as handle:
                handle.read()
        except OSError:
            return False
        return True

    def _read_uj(self, domain: str) -> int:
        with open(f"{domain}/energy_uj", encoding="ascii") as handle:
            return int(handle.read().strip())

    def _read_range_uj(self, domain: str) -> int:
        try:
            with open(f"{domain}/max_energy_range_uj", encoding="ascii") as handle:
                return int(handle.read().strip())
        except OSError:
            return 0

    def start(self) -> None:
        self._start_uj = [self._read_uj(d) for d in self._domains]
        self._ranges_uj = [self._read_range_uj(d) for d in self._domains]

    def stop(self) -> float | None:
        if not self._start_uj:
            return None
        total_uj = 0
        for domain, start, wrap in zip(self._domains, self._start_uj, self._ranges_uj):
            end = self._read_uj(domain)
            delta = end - start
            if delta < 0 and wrap > 0:  # counter wrapped around
                delta += wrap
            total_uj += max(delta, 0)
        return total_uj / 1000.0


class NvmlPowerBackend(EnergyBackend):
    """GPU energy integrated from NVML power samples."""

    name = "nvml"

    def __init__(self, device_index: int = 0, sample_interval_ms: int = 20) -> None:
        self._device_index = device_index
        self._interval_s = max(sample_interval_ms, 1) / 1000.0
        self._samples: list[tuple[float, float]] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._handle = None

    @classmethod
    def available(cls, device_index: int = 0) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            pynvml.nvmlDeviceGetPowerUsage(handle)
        except Exception:
            return False
        return True

    def _sample_loop(self) -> None:
        import pynvml

        while not self._stop_event.is_set():
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self._samples.append((time.perf_counter(), float(power_mw)))
            self._stop_event.wait(self._interval_s)

    def start(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        self._samples = []
        self._stop_event.clear()
        self._samples.append((time.perf_counter(), float(pynvml.nvmlDeviceGetPowerUsage(self._handle))))
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if self._thread is None:
            return None
        self._stop_event.set()
        self._thread.join()
        self._thread = None

        import pynvml

        self._samples.append((time.perf_counter(), float(pynvml.nvmlDeviceGetPowerUsage(self._handle))))
        if len(self._samples) < 2:
            return None
        energy_mj = 0.0
        for (t0, p0), (t1, p1) in zip(self._samples, self._samples[1:]):
            energy_mj += (t1 - t0) * (p0 + p1) / 2.0  # mW * s == mJ
        return energy_mj


def sample_idle_power_mw(
    device_index: int = 0,
    *,
    duration_s: float = 2.0,
    interval_ms: int = 50,
) -> float | None:
    """Mean GPU power draw (mW) over an idle window, or ``None`` without NVML.

    NVML reports whole-board power including idle draw; sampling a quiet
    window before a run lets analysis subtract the baseline
    (``net_mj = total_mj - idle_mw * duration_s``).
    """
    if not NvmlPowerBackend.available(device_index):
        return None
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    samples: list[float] = []
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        samples.append(float(pynvml.nvmlDeviceGetPowerUsage(handle)))
        time.sleep(max(interval_ms, 1) / 1000.0)
    return sum(samples) / len(samples) if samples else None


class EnergyMeter:
    """Aggregate several backends behind one start/stop pair."""

    def __init__(self, backends: list[EnergyBackend]) -> None:
        self._backends = backends

    @property
    def backend_names(self) -> tuple[str, ...]:
        return tuple(backend.name for backend in self._backends)

    def start(self) -> None:
        for backend in self._backends:
            backend.start()

    def stop(self) -> dict[str, float | None]:
        """Per-backend energy in millijoules, plus a ``total`` over non-null values."""
        readings: dict[str, float | None] = {b.name: b.stop() for b in self._backends}
        values = [v for v in readings.values() if v is not None]
        readings["total"] = sum(values) if values else None
        return readings


def build_energy_meter(
    backends: str | tuple[str, ...] = "auto",
    *,
    sample_interval_ms: int = 20,
    nvml_device_index: int = 0,
) -> EnergyMeter:
    """Build a meter from explicit backend names or by auto-detection."""
    selected: list[EnergyBackend] = []
    if backends == "auto":
        if RaplBackend.available():
            selected.append(RaplBackend())
        if NvmlPowerBackend.available(nvml_device_index):
            selected.append(
                NvmlPowerBackend(nvml_device_index, sample_interval_ms=sample_interval_ms)
            )
    else:
        for name in backends:
            if name == "rapl":
                selected.append(RaplBackend())
            elif name == "nvml":
                selected.append(
                    NvmlPowerBackend(nvml_device_index, sample_interval_ms=sample_interval_ms)
                )
            else:
                raise ValueError(f"unknown energy backend {name!r}")
    return EnergyMeter(selected)
