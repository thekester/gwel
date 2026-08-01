"""A confidence-conditioned policy must pay for the probe it conditions on."""

import pytest

from gwel.actions import Action
from gwel.oracle.cost import CostWeights
from gwel.router.policies import fixed_policy, group_runs, simulate

WEIGHTS = CostWeights(
    error_weight=1.0,
    lambda_latency_per_ms=0.001,
    lambda_energy_per_mj=0.0,
    lambda_memory_per_mb=0.0,
    lambda_visual_tokens=0.0,
)


def _runs(make_record):
    return group_runs(
        [
            make_record(config_id="lowres_384", action=Action.ANSWER_LOW,
                        correct=False, latency_ms=200, visual_tokens=64),
            make_record(config_id="crop_r0c0", action=Action.CROP,
                        correct=True, latency_ms=300, visual_tokens=128),
        ]
    )


def test_escalating_pays_probe_plus_escalation(make_record) -> None:
    results = simulate(
        _runs(make_record),
        fixed_policy(Action.CROP),
        weights=WEIGHTS,
        probe_config_id="lowres_384",
    )
    assert results[0].latency_ms == 500  # 200 probe + 300 crop
    assert results[0].visual_tokens == 192
    assert results[0].cost == pytest.approx(0.5)  # correct, so no error term


def test_answering_from_the_probe_pays_it_once(make_record) -> None:
    results = simulate(
        _runs(make_record),
        fixed_policy(Action.ANSWER_LOW),
        weights=WEIGHTS,
        probe_config_id="lowres_384",
    )
    assert results[0].latency_ms == 200
    assert results[0].visual_tokens == 64


def test_without_a_probe_only_the_action_is_charged(make_record) -> None:
    results = simulate(_runs(make_record), fixed_policy(Action.CROP), weights=WEIGHTS)
    assert results[0].latency_ms == 300
    assert results[0].visual_tokens == 128


def test_cascade_never_looks_cheaper_than_the_escalation_alone(make_record) -> None:
    with_probe = simulate(
        _runs(make_record), fixed_policy(Action.CROP), weights=WEIGHTS,
        probe_config_id="lowres_384",
    )[0]
    without = simulate(_runs(make_record), fixed_policy(Action.CROP), weights=WEIGHTS)[0]
    assert with_probe.cost > without.cost


def test_memory_is_the_peak_not_the_sum(make_record) -> None:
    # Passes run sequentially, so their footprints do not add up.
    records = [
        make_record(config_id="lowres_384", action=Action.ANSWER_LOW, ram_peak_mb=1000.0),
        make_record(config_id="crop_r0c0", action=Action.CROP, ram_peak_mb=1600.0),
    ]
    results = simulate(
        group_runs(records), fixed_policy(Action.CROP), weights=WEIGHTS,
        probe_config_id="lowres_384",
    )
    assert results[0].memory_mb == 1600.0
