import pytest

from gwel.actions import Action
from gwel.oracle.cost import CostWeights
from gwel.oracle.label import OracleLabel, derive_labels, label_example


def test_oracle_picks_cheapest_correct_routable_action(make_record) -> None:
    records = [
        make_record(config_id="lowres_256", action=Action.ANSWER_LOW, correct=False, latency_ms=50),
        make_record(config_id="crop_r0c0", action=Action.CROP, correct=True, latency_ms=200),
        make_record(config_id="ocr_full", action=Action.OCR, correct=True, latency_ms=120),
    ]
    label = label_example(records, weights=CostWeights(lambda_latency_per_ms=1.0))
    assert label.action is Action.OCR
    assert label.config_id == "ocr_full"
    assert label.any_correct


def test_oracle_ignores_diagnostic_configs(make_record) -> None:
    records = [
        make_record(config_id="full", action=None, correct=True, latency_ms=1),
        make_record(config_id="crop_r0c0", action=Action.CROP, correct=True, latency_ms=500),
    ]
    label = label_example(records)
    assert label.action is Action.CROP  # "full" is cheaper but not routable


def test_oracle_returns_none_when_nothing_correct(make_record) -> None:
    records = [
        make_record(config_id="lowres_256", correct=False),
        make_record(config_id="crop_r0c0", action=Action.CROP, correct=False),
    ]
    label = label_example(records)
    assert label.action is None
    assert label.config_id is None
    assert label.cost is None
    assert not label.any_correct
    assert set(label.config_costs) == {"lowres_256", "crop_r0c0"}


def test_oracle_tie_breaks_on_config_id(make_record) -> None:
    records = [
        make_record(config_id="crop_r0c1", action=Action.CROP, correct=True, latency_ms=100),
        make_record(config_id="crop_r0c0", action=Action.CROP, correct=True, latency_ms=100),
    ]
    label = label_example(records)
    assert label.config_id == "crop_r0c0"


def test_label_example_rejects_mixed_examples(make_record) -> None:
    with pytest.raises(ValueError):
        label_example([make_record(example_id="a"), make_record(example_id="b")])


def test_derive_labels_groups_by_example_in_order(make_record) -> None:
    records = [
        make_record(example_id="b", config_id="lowres_256", correct=True),
        make_record(example_id="a", config_id="lowres_256", correct=False),
        make_record(example_id="b", config_id="crop_r0c0", action=Action.CROP, correct=True),
        make_record(example_id="a", config_id="crop_r0c0", action=Action.CROP, correct=True),
    ]
    labels = derive_labels(records)
    assert [label.example_id for label in labels] == ["b", "a"]
    assert labels[0].action is Action.ANSWER_LOW
    assert labels[1].action is Action.CROP


def test_cheaper_but_wrong_never_wins(make_record) -> None:
    records = [
        make_record(config_id="lowres_256", correct=False, latency_ms=1, visual_tokens=1),
        make_record(config_id="crop_r0c0", action=Action.CROP, correct=True, latency_ms=1000),
    ]
    label = label_example(records, weights=CostWeights(lambda_latency_per_ms=10.0))
    assert label.action is Action.CROP


def test_label_round_trip_dict(make_record) -> None:
    label = label_example([make_record()])
    assert OracleLabel.from_dict(label.to_dict()) == label
