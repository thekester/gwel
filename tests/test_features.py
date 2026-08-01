import numpy as np
import pytest

from gwel.router.features import FEATURE_NAMES, build_feature_matrix, build_features


def test_feature_vector_matches_declared_names(make_record) -> None:
    vector = build_features(make_record())
    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector.dtype == np.float32


def test_features_are_deterministic(make_record) -> None:
    record = make_record()
    assert np.array_equal(build_features(record), build_features(record))


def test_text_hint_and_wh_features(make_record) -> None:
    record = make_record()  # question: "What does the sign say?"
    vector = build_features(record)
    named = dict(zip(FEATURE_NAMES, vector))
    assert named["question_text_hint"] == 1.0
    assert named["question_wh_what"] == 1.0
    assert named["question_wh_why"] == 0.0


def test_hardware_state_features(make_record) -> None:
    state = {
        "ram_available_mb": 8000.0,
        "ram_used_fraction": 0.5,
        "cpu_load_fraction": 0.25,
        "vram_free_mb": None,
    }
    named = dict(zip(FEATURE_NAMES, build_features(make_record(), hardware_state=state)))
    assert named["ram_available_mb"] == 8000.0
    assert named["vram_free_mb"] == 0.0
    assert named["vram_present"] == 0.0


def test_record_without_signals_rejected(make_record) -> None:
    with pytest.raises(ValueError):
        build_features(make_record(signals=None))


def test_feature_matrix_stacks_rows(make_record) -> None:
    matrix = build_feature_matrix([make_record(), make_record(example_id="ex-2")])
    assert matrix.shape == (2, len(FEATURE_NAMES))
