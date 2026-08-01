"""Distillation of oracle labels into the MLP router.

Training data pairs the ANSWER_LOW-pass features of each example with the
oracle's minimal sufficient action. Examples where no action was correct are
dropped (the router cannot fix an unanswerable question). The split is by
example with a fixed seed; class imbalance is handled with inverse-frequency
weights.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import RouterConfig
from ..oracle.label import OracleLabel
from ..oracle.records import RunRecord
from .features import build_feature_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDataset:
    """Aligned features and oracle actions for router training."""

    features: np.ndarray            # (N, D) float32
    targets: np.ndarray             # (N,) int64 action indices
    example_ids: tuple[str, ...]
    datasets: tuple[str, ...]


@dataclass(frozen=True)
class TrainResult:
    """Summary of one router training run."""

    n_train: int
    n_val: int
    train_accuracy: float
    val_accuracy: float
    val_accuracy_per_action: dict[str, float]
    checkpoint_dir: str


def build_routing_dataset(
    records: list[RunRecord],
    labels: list[OracleLabel],
    *,
    feature_config_id: str,
) -> RoutingDataset:
    """Join oracle labels with their example's ANSWER_LOW feature record."""
    from ..actions import Action

    action_to_index = {action: index for index, action in enumerate(Action.ordered())}
    feature_records = {
        record.example_id: record
        for record in records
        if record.config_id == feature_config_id
    }

    kept_records: list[RunRecord] = []
    targets: list[int] = []
    example_ids: list[str] = []
    dataset_names: list[str] = []
    for label in labels:
        if label.action is None:
            continue
        record = feature_records.get(label.example_id)
        if record is None or record.signals is None:
            logger.warning("no %s record for %s; skipping", feature_config_id, label.example_id)
            continue
        kept_records.append(record)
        targets.append(action_to_index[label.action])
        example_ids.append(label.example_id)
        dataset_names.append(label.dataset)

    if not kept_records:
        raise ValueError("no trainable examples: check feature_config_id and labels")

    return RoutingDataset(
        features=build_feature_matrix(kept_records),
        targets=np.asarray(targets, dtype=np.int64),
        example_ids=tuple(example_ids),
        datasets=tuple(dataset_names),
    )


def split_by_example(
    dataset: RoutingDataset,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, val_indices), shuffled with a fixed seed."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n = len(dataset.targets)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    return permutation[n_val:], permutation[:n_val]


def train_router(
    dataset: RoutingDataset,
    config: RouterConfig,
    *,
    out_dir: str | Path,
) -> TrainResult:
    """Train the MLP router on oracle labels and save the best checkpoint."""
    import torch
    from torch import nn

    from .model import ACTIONS, RouterCheckpoint, RouterMLP

    torch.manual_seed(config.seed)
    train_idx, val_idx = split_by_example(
        dataset, val_fraction=config.val_fraction, seed=config.seed
    )

    mean = dataset.features[train_idx].mean(axis=0)
    std = dataset.features[train_idx].std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (dataset.features - mean) / std

    features = torch.from_numpy(normalized.astype(np.float32))
    targets = torch.from_numpy(dataset.targets)

    counts = np.bincount(dataset.targets[train_idx], minlength=len(ACTIONS)).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.sum()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights.astype(np.float32)))

    model = RouterMLP(
        input_dim=features.shape[1],
        hidden_dims=config.hidden_dims,
        num_actions=len(ACTIONS),
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_features, train_targets = features[train_idx], targets[train_idx]
    val_features, val_targets = features[val_idx], targets[val_idx]

    def accuracy(split_features: torch.Tensor, split_targets: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            predictions = model(split_features).argmax(dim=-1)
        return float((predictions == split_targets).float().mean())

    best_val = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    generator = torch.Generator().manual_seed(config.seed)
    for _epoch in range(config.epochs):
        model.train()
        for batch in torch.randperm(len(train_idx), generator=generator).split(config.batch_size):
            optimizer.zero_grad()
            loss = criterion(model(train_features[batch]), train_targets[batch])
            loss.backward()
            optimizer.step()
        val_acc = accuracy(val_features, val_targets)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    checkpoint = RouterCheckpoint(model, mean, std)
    checkpoint.save(out_dir)

    per_action: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        val_predictions = model(val_features).argmax(dim=-1)
    for index, action in enumerate(ACTIONS):
        mask = val_targets == index
        if int(mask.sum()) > 0:
            per_action[action.value] = float(
                (val_predictions[mask] == val_targets[mask]).float().mean()
            )

    return TrainResult(
        n_train=len(train_idx),
        n_val=len(val_idx),
        train_accuracy=accuracy(train_features, train_targets),
        val_accuracy=best_val,
        val_accuracy_per_action=per_action,
        checkpoint_dir=str(out_dir),
    )
