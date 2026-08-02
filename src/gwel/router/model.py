"""Lightweight MLP router and its (de)serialization.

This module imports torch at module level; import it only from training
or deployment code paths, never from offline analysis.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..actions import Action
from .features import FEATURE_NAMES

ACTIONS: tuple[Action, ...] = Action.ordered()
ACTION_TO_INDEX: dict[Action, int] = {action: index for index, action in enumerate(ACTIONS)}


class RouterMLP(nn.Module):
    """MLP mapping the feature vector to action logits."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        hidden_dims: tuple[int, ...] = (128, 64),
        num_actions: int = len(ACTIONS),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, num_actions))
        self.network = nn.Sequential(*layers)
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class RouterCheckpoint:
    """Bundle of a trained router, its input normalization, and metadata."""

    def __init__(
        self,
        model: RouterMLP,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> None:
        self.model = model
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)

    def predict(self, features: np.ndarray) -> tuple[list[Action], np.ndarray]:
        """Return predicted actions and softmax probabilities for (N, D) input."""
        self.model.eval()
        normalized = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            logits = self.model(torch.from_numpy(normalized.astype(np.float32)))
            probs = torch.softmax(logits, dim=-1).numpy()
        indices = probs.argmax(axis=-1)
        return [ACTIONS[i] for i in indices], probs

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "router.pt")
        metadata = {
            "feature_names": list(FEATURE_NAMES),
            "actions": [action.value for action in ACTIONS],
            "hidden_dims": list(self.model.hidden_dims),
            "dropout": self.model.dropout,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
        }
        (directory / "router.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "RouterCheckpoint":
        directory = Path(directory)
        metadata = json.loads((directory / "router.json").read_text(encoding="utf-8"))
        if metadata["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("checkpoint feature schema does not match current FEATURE_NAMES")
        model = RouterMLP(
            input_dim=len(metadata["feature_names"]),
            hidden_dims=tuple(metadata["hidden_dims"]),
            num_actions=len(metadata["actions"]),
            dropout=metadata["dropout"],
        )
        model.load_state_dict(torch.load(directory / "router.pt", weights_only=True))
        return cls(
            model=model,
            feature_mean=np.asarray(metadata["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(metadata["feature_std"], dtype=np.float32),
        )
