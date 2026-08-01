"""Checkpoint round-trip for the torch-dependent router model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gwel.router.features import FEATURE_NAMES  # noqa: E402
from gwel.router.model import ACTIONS, ACTION_TO_INDEX, RouterCheckpoint, RouterMLP  # noqa: E402


def _checkpoint(hidden=(16, 8)) -> RouterCheckpoint:
    model = RouterMLP(input_dim=len(FEATURE_NAMES), hidden_dims=hidden, dropout=0.0)
    dim = len(FEATURE_NAMES)
    return RouterCheckpoint(model, np.zeros(dim), np.ones(dim))


def test_action_index_is_a_bijection() -> None:
    assert len(ACTION_TO_INDEX) == len(ACTIONS)
    assert sorted(ACTION_TO_INDEX.values()) == list(range(len(ACTIONS)))


def test_forward_shape_matches_the_action_space() -> None:
    model = RouterMLP(input_dim=len(FEATURE_NAMES))
    out = model(torch.zeros(4, len(FEATURE_NAMES)))
    assert out.shape == (4, len(ACTIONS))


def test_predict_returns_actions_and_normalised_probabilities() -> None:
    checkpoint = _checkpoint()
    features = np.random.default_rng(0).normal(size=(5, len(FEATURE_NAMES))).astype(np.float32)
    actions, probs = checkpoint.predict(features)
    assert len(actions) == 5
    assert probs.shape == (5, len(ACTIONS))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert all(a in ACTIONS for a in actions)


def test_checkpoint_round_trip_preserves_predictions(tmp_path) -> None:
    checkpoint = _checkpoint()
    features = np.random.default_rng(1).normal(size=(6, len(FEATURE_NAMES))).astype(np.float32)
    before = checkpoint.predict(features)[1]

    checkpoint.save(tmp_path)
    reloaded = RouterCheckpoint.load(tmp_path)
    assert np.allclose(before, reloaded.predict(features)[1], atol=1e-6)


def test_normalisation_is_persisted(tmp_path) -> None:
    dim = len(FEATURE_NAMES)
    checkpoint = RouterCheckpoint(RouterMLP(input_dim=dim), np.full(dim, 3.0), np.full(dim, 2.0))
    checkpoint.save(tmp_path)
    reloaded = RouterCheckpoint.load(tmp_path)
    assert np.allclose(reloaded.feature_mean, 3.0)
    assert np.allclose(reloaded.feature_std, 2.0)


def test_loading_rejects_a_stale_feature_schema(tmp_path) -> None:
    import json

    _checkpoint().save(tmp_path)
    metadata = json.loads((tmp_path / "router.json").read_text(encoding="utf-8"))
    metadata["feature_names"] = metadata["feature_names"][:-1]
    (tmp_path / "router.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="feature schema"):
        RouterCheckpoint.load(tmp_path)


def test_architecture_is_persisted(tmp_path) -> None:
    _checkpoint(hidden=(32, 16, 8)).save(tmp_path)
    reloaded = RouterCheckpoint.load(tmp_path)
    assert reloaded.model.hidden_dims == (32, 16, 8)
