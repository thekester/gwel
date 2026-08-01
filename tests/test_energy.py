import pytest

from gwel.profiling.energy import (
    EnergyBackend,
    EnergyMeter,
    NvmlPowerBackend,
    RaplBackend,
    build_energy_meter,
)


class _Fake(EnergyBackend):
    """Backend returning a fixed reading, for testing the aggregator."""

    def __init__(self, name: str, value: float | None) -> None:
        self.name = name
        self._value = value
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> float | None:
        return self._value


def test_meter_sums_backends_and_reports_each() -> None:
    meter = EnergyMeter([_Fake("a", 100.0), _Fake("b", 250.0)])
    meter.start()
    readings = meter.stop()
    assert readings["a"] == 100.0
    assert readings["b"] == 250.0
    assert readings["total"] == 350.0


def test_total_ignores_backends_that_returned_nothing() -> None:
    meter = EnergyMeter([_Fake("a", 100.0), _Fake("b", None)])
    assert meter.stop()["total"] == 100.0


def test_total_is_none_when_no_backend_measured_anything() -> None:
    meter = EnergyMeter([_Fake("a", None)])
    assert meter.stop()["total"] is None


def test_empty_meter_reports_no_total() -> None:
    meter = EnergyMeter([])
    meter.start()
    assert meter.stop() == {"total": None}
    assert meter.backend_names == ()


def test_start_reaches_every_backend() -> None:
    backends = [_Fake("a", 1.0), _Fake("b", 2.0)]
    EnergyMeter(backends).start()
    assert all(b.started for b in backends)


def test_explicit_backend_names_are_honoured() -> None:
    assert build_energy_meter(("rapl",)).backend_names == ("rapl",)
    assert build_energy_meter(()).backend_names == ()


def test_unknown_backend_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="quantum"):
        build_energy_meter(("quantum",))


def test_auto_detection_only_selects_available_backends() -> None:
    names = build_energy_meter("auto").backend_names
    assert set(names) <= {"rapl", "nvml"}
    if "rapl" in names:
        assert RaplBackend.available()
    if "nvml" in names:
        assert NvmlPowerBackend.available(0)


def test_rapl_reports_nothing_when_it_never_started() -> None:
    # Windows and macOS have no powercap tree; stopping must not raise.
    assert RaplBackend().stop() is None


def test_nvml_reports_nothing_when_it_never_started() -> None:
    assert NvmlPowerBackend().stop() is None
