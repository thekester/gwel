import pytest

from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.data.vqa_metrics import anls


def test_anls_gives_partial_credit_to_near_misses() -> None:
    score = anls("FIELD SALES SURVEY", ["field sales survey may promotions"])
    assert 0.5 <= score < 1.0


def test_anls_is_one_for_exact_normalized_match() -> None:
    assert anls("FIELD SALES SURVEY.", ["field sales survey"]) == pytest.approx(1.0)


def test_anls_zeroes_below_threshold() -> None:
    assert anls("membership", ["american conservative network regional meeting"]) == 0.0


def test_anls_handles_empty_gold() -> None:
    assert anls("anything", []) == 0.0


def test_policy_routes_each_dataset_to_its_metric() -> None:
    policy = ScoringPolicy()
    assert policy.metric_for("docvqa") == "anls"
    assert policy.metric_for("vqav2") == "vqa"
    assert policy.metric_for("vstar") == "exact"
    assert policy.metric_for("unknown_dataset") == "vqa"


def test_rescoring_flips_a_docvqa_near_miss(make_record) -> None:
    record = make_record(
        dataset="docvqa",
        answer="FIELD SALES SURVEY",
        gold_answers=("field sales survey may promotions",),
        correct=False,
        vqa_score=0.0,
    )
    rescored = rescore_records([record])[0]
    assert rescored.correct
    assert rescored.meta["metric"] == "anls"


def test_rescoring_preserves_hardware_measurements(make_record) -> None:
    record = make_record(latency_ms=123.0, visual_tokens=64)
    rescored = rescore_records([record])[0]
    assert rescored.latency_ms == 123.0
    assert rescored.visual_tokens == 64
    assert rescored.energy_mj == record.energy_mj


def test_exact_metric_is_strict(make_record) -> None:
    record = make_record(
        dataset="vstar",
        answer="greenish",
        gold_answers=("green",),
    )
    assert not rescore_records([record])[0].correct
