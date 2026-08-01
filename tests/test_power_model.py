import pytest

from gwel.profiling.power_model import PowerModel, vision_energy_share


def test_reference_fit_matches_the_published_equation() -> None:
    # Zhan et al. Eq. 4: P = 12.1*S + 42.2
    assert PowerModel.from_parameter_count(1.0).average_power_w == pytest.approx(54.3)
    assert PowerModel.from_parameter_count(0.5).average_power_w == pytest.approx(48.25)


def test_larger_models_draw_more_power() -> None:
    small = PowerModel.from_parameter_count(0.5).average_power_w
    large = PowerModel.from_parameter_count(7.0).average_power_w
    assert large > small


def test_energy_is_power_times_time() -> None:
    model = PowerModel(average_power_w=50.0)
    assert model.energy_mj(200.0) == pytest.approx(10000.0)  # 50 W * 0.2 s = 10 J
    assert model.energy_mj(0.0) == 0.0


def test_idle_subtraction_gives_net_power() -> None:
    model = PowerModel.from_idle_and_busy(idle_w=13.2, busy_w=48.0)
    assert model.average_power_w == pytest.approx(34.8)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        PowerModel.from_parameter_count(0)
    with pytest.raises(ValueError):
        PowerModel.from_idle_and_busy(idle_w=50.0, busy_w=10.0)
    with pytest.raises(ValueError):
        PowerModel(average_power_w=50.0).energy_mj(-1.0)


def test_vision_share_from_the_pilot_measurements() -> None:
    # Median latencies on the 200-example pilot: text-only 160 ms, full 282 ms.
    assert vision_energy_share(160.0, 282.0) == pytest.approx(0.43, abs=0.01)


def test_vision_share_is_zero_when_the_image_is_free() -> None:
    assert vision_energy_share(200.0, 200.0) == 0.0
    assert vision_energy_share(250.0, 200.0) == 0.0  # noise cannot go negative


def test_vision_share_validates_inputs() -> None:
    with pytest.raises(ValueError):
        vision_energy_share(100.0, 0.0)
    with pytest.raises(ValueError):
        vision_energy_share(-1.0, 100.0)
