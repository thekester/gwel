import pytest


def test_net_energy_subtracts_idle_baseline(make_record) -> None:
    record = make_record(
        latency_ms=2000.0,
        energy_total_mj=50000.0,
        meta={"orig_width": 1024, "orig_height": 768, "idle_power_mw": 20000.0},
    )
    # 20 W idle over 2 s = 40 000 mJ baseline
    assert record.net_energy_mj == pytest.approx(10000.0)


def test_net_energy_clamps_at_zero(make_record) -> None:
    record = make_record(
        latency_ms=2000.0,
        energy_total_mj=1000.0,
        meta={"idle_power_mw": 20000.0},
    )
    assert record.net_energy_mj == 0.0


def test_net_energy_uses_measured_energy_window(make_record) -> None:
    record = make_record(
        latency_ms=1000.0,
        energy_total_mj=50000.0,
        meta={"idle_power_mw": 20000.0, "energy_window_ms": 1500.0},
    )
    assert record.net_energy_mj == pytest.approx(20000.0)


def test_net_energy_without_baseline_is_raw_total(make_record) -> None:
    record = make_record(energy_total_mj=500.0)
    assert record.net_energy_mj == 500.0


def test_net_energy_none_when_unmeasured(make_record) -> None:
    record = make_record(energy_total_mj=None, energy_mj={"total": None})
    assert record.net_energy_mj is None
