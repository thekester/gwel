import pytest

from gwel.actions import Action
from gwel.oracle.cost import CostWeights
from gwel.router.policies import (
    action_of,
    fixed_policy,
    group_runs,
    oracle_policy,
    simulate,
    threshold_policy,
)

# Only latency and the error term are priced, so expected costs stay readable.
WEIGHTS = CostWeights(
    error_weight=1.0,
    lambda_latency_per_ms=0.001,
    lambda_energy_per_mj=0.0,
    lambda_memory_per_mb=0.0,
    lambda_visual_tokens=0.0,
)


def test_action_families_are_recognised() -> None:
    assert action_of("lowres_384") is Action.ANSWER_LOW
    assert action_of("crop_r1c0") is Action.CROP
    assert action_of("ocr_r0c0") is Action.OCR
    assert action_of("ocr_full") is Action.OCR
    assert action_of("no_image") is None
    assert action_of("full") is None


def test_grouping_collects_configs_per_example(make_record) -> None:
    records = [
        make_record(example_id="a", config_id="lowres_384"),
        make_record(example_id="a", config_id="crop_r0c0", action=Action.CROP),
        make_record(example_id="b", config_id="lowres_384"),
    ]
    runs = group_runs(records)
    assert [r.example_id for r in runs] == ["a", "b"]
    assert set(runs[0].by_config) == {"lowres_384", "crop_r0c0"}


def test_policy_pays_the_cheapest_member_of_its_family(make_record) -> None:
    records = [
        make_record(example_id="a", config_id="crop_r0c0", action=Action.CROP,
                    correct=False, latency_ms=500),
        make_record(example_id="a", config_id="crop_r0c1", action=Action.CROP,
                    correct=True, latency_ms=100),
    ]
    runs = group_runs(records)
    chosen = runs[0].realise(Action.CROP, WEIGHTS)
    assert chosen.config_id == "crop_r0c1"  # cheapest, not best-answering


def test_without_a_localizer_the_policy_cannot_peek_at_the_lucky_crop(make_record) -> None:
    # The cheap crop is wrong and the expensive one is right: a deployed
    # router picks one operation, so it must be charged the wrong answer.
    results = simulate(
        group_runs(_mixed_crops(make_record)), fixed_policy(Action.CROP), weights=WEIGHTS
    )
    assert results[0].correct is False


def test_perfect_localizer_recovers_the_correct_region(make_record) -> None:
    results = simulate(
        group_runs(_mixed_crops(make_record)),
        fixed_policy(Action.CROP),
        weights=WEIGHTS,
        region_selection="best",
    )
    assert results[0].correct is True


def test_region_selection_rejects_unknown_modes(make_record) -> None:
    runs = group_runs(_mixed_crops(make_record))
    with pytest.raises(ValueError):
        runs[0].realise(Action.CROP, WEIGHTS, region_selection="magic")


def _mixed_crops(make_record):
    """One wrong-but-cheap crop and one right-but-expensive crop."""
    return [
        make_record(example_id="a", config_id="crop_r0c0", action=Action.CROP,
                    correct=False, latency_ms=100),
        make_record(example_id="a", config_id="crop_r0c1", action=Action.CROP,
                    correct=True, latency_ms=900),
    ]


def test_incorrect_results_pay_the_error_weight(make_record) -> None:
    records = [make_record(config_id="lowres_384", correct=False, latency_ms=100)]
    results = simulate(group_runs(records), fixed_policy(Action.ANSWER_LOW), weights=WEIGHTS)
    assert results[0].cost == pytest.approx(1.0 + 0.1)


def test_oracle_policy_follows_its_labels(make_record) -> None:
    records = [
        make_record(example_id="a", config_id="lowres_384"),
        make_record(example_id="a", config_id="crop_r0c0", action=Action.CROP),
    ]
    runs = group_runs(records)
    choose = oracle_policy({"a": Action.CROP})
    assert choose(runs[0]) is Action.CROP


def test_oracle_policy_falls_back_on_unsolvable_examples(make_record) -> None:
    runs = group_runs([make_record(example_id="a", config_id="lowres_384")])
    assert oracle_policy({"a": None})(runs[0]) is Action.ANSWER_LOW


def test_threshold_policy_escalates_on_high_entropy(make_record) -> None:
    confident = make_record(config_id="lowres_384", signals={
        "mean_logprob": -0.1, "min_logprob": -0.2, "mean_entropy": 0.1,
        "max_entropy": 0.2, "first_entropy": 0.1, "mean_margin": 0.9,
        "min_margin": 0.8, "num_tokens": 3,
    })
    uncertain = make_record(example_id="b", config_id="lowres_384", signals={
        "mean_logprob": -2.0, "min_logprob": -3.0, "mean_entropy": 1.9,
        "max_entropy": 2.5, "first_entropy": 1.8, "mean_margin": 0.1,
        "min_margin": 0.05, "num_tokens": 3,
    })
    choose = threshold_policy("max_entropy", 1.0, cheap_config_id="lowres_384")
    runs = {r.example_id: r for r in group_runs([confident, uncertain])}
    assert choose(runs["ex-1"]) is Action.ANSWER_LOW
    assert choose(runs["b"]) is Action.CROP


def test_examples_without_the_chosen_family_are_skipped(make_record) -> None:
    runs = group_runs([make_record(config_id="lowres_384")])
    assert simulate(runs, fixed_policy(Action.OCR), weights=WEIGHTS) == []
