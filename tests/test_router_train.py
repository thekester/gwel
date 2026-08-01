"""Training-loop invariants for the distillation router."""

import numpy as np
import pytest

pytest.importorskip("torch")

from gwel.actions import Action  # noqa: E402
from gwel.config import RouterConfig  # noqa: E402
from gwel.oracle.label import OracleLabel  # noqa: E402
from gwel.router.features import FEATURE_NAMES  # noqa: E402
from gwel.router.train import build_routing_dataset, split_by_example, train_router  # noqa: E402


def _dataset(make_record, n: int = 60):
    """Records plus oracle labels where the action is learnable from geometry."""
    records, labels = [], []
    for i in range(n):
        action = Action.ordered()[i % 3]
        wide = action is Action.CROP
        records.append(
            make_record(
                example_id=f"ex-{i}",
                config_id="lowres_384",
                dataset=("docvqa" if i % 2 else "vqav2"),
                meta={"orig_width": 2000 if wide else 300, "orig_height": 400},
            )
        )
        labels.append(
            OracleLabel(
                example_id=f"ex-{i}",
                dataset=("docvqa" if i % 2 else "vqav2"),
                config_id="x",
                action=action,
                cost=1.0,
                config_costs={},
                any_correct=True,
            )
        )
    return build_routing_dataset(records, labels, feature_config_id="lowres_384")


def test_dataset_shape_and_targets(make_record) -> None:
    dataset = _dataset(make_record)
    assert dataset.features.shape == (60, len(FEATURE_NAMES))
    assert set(dataset.targets.tolist()) == {0, 1, 2}
    assert len(dataset.example_ids) == 60


def test_unsolvable_examples_are_dropped(make_record) -> None:
    records = [make_record(example_id="a", config_id="lowres_384")]
    labels = [OracleLabel("a", "vqav2", None, None, None, {}, False)]
    with pytest.raises(ValueError, match="no trainable examples"):
        build_routing_dataset(records, labels, feature_config_id="lowres_384")


def test_missing_feature_config_reports_what_is_available(make_record) -> None:
    records = [make_record(example_id="a", config_id="full", action=None)]
    labels = [OracleLabel("a", "vqav2", "full", Action.CROP, 1.0, {}, True)]
    with pytest.raises(ValueError, match="full"):
        build_routing_dataset(records, labels, feature_config_id="lowres_384")


def test_split_is_a_three_way_partition(make_record) -> None:
    dataset = _dataset(make_record)
    train, val, test = split_by_example(dataset, val_fraction=0.2, seed=1234, test_fraction=0.2)
    combined = sorted(np.concatenate([train, val, test]).tolist())
    assert combined == list(range(60))
    assert len(val) > 0 and len(test) > 0


def test_split_is_deterministic(make_record) -> None:
    dataset = _dataset(make_record)
    first = split_by_example(dataset, val_fraction=0.2, seed=7, test_fraction=0.2)
    second = split_by_example(dataset, val_fraction=0.2, seed=7, test_fraction=0.2)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_training_saves_a_loadable_checkpoint(make_record, tmp_path) -> None:
    from gwel.router.model import RouterCheckpoint

    dataset = _dataset(make_record)
    result = train_router(
        dataset, RouterConfig(epochs=5, hidden_dims=(16,), batch_size=8), out_dir=tmp_path
    )
    assert result.n_train + result.n_val + result.n_test == 60
    assert 0.0 <= result.val_accuracy <= 1.0
    assert 0.0 <= result.test_accuracy <= 1.0
    RouterCheckpoint.load(tmp_path)  # must not raise


def test_training_rejects_a_split_that_leaves_a_fold_empty(make_record, tmp_path) -> None:
    records = [make_record(example_id=f"e{i}", config_id="lowres_384") for i in range(2)]
    labels = [
        OracleLabel(f"e{i}", "vqav2", "x", Action.ANSWER_LOW, 1.0, {}, True) for i in range(2)
    ]
    dataset = build_routing_dataset(records, labels, feature_config_id="lowres_384")
    with pytest.raises(ValueError, match="empty fold"):
        train_router(dataset, RouterConfig(epochs=1, val_fraction=0.5, test_fraction=0.49),
                     out_dir=tmp_path)
