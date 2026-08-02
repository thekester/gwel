"""Choosing *where* to crop, from the cheap pass's own visual tokens.

Our measurements say region choice dominates action choice, picking the right
crop is worth far more than picking the right action family. AwaRes (arXiv
2603.16932) solves this with cold-start SFT, multi-turn GRPO and a LLaMA-3.3-70B
judge for supervision. That is not available to a 500M model on edge hardware.

This localizer needs none of it. SmolVLM lays its visual tokens out in a square
grid, so the hidden states of the tokens covering a candidate crop can be pooled
into a per-cell feature and scored by a linear probe, trained on which cells
actually answered correctly, which the oracle run already recorded. No extra
forward pass, no judge model, no reinforcement learning.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def pool_cells(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Average visual-token states within each cell of a ``rows x cols`` grid.

    ``grid`` is ``(side, side, hidden)`` as returned by
    :meth:`gwel.modeling.smolvlm.SmolVlmEngine.extract_visual_grid`. Returns
    ``(rows * cols, hidden)`` in row-major order, matching the crop-box order
    of :func:`gwel.modeling.imaging.crop_grid`.
    """
    if grid.ndim != 3 or grid.shape[0] != grid.shape[1]:
        raise ValueError("grid must be (side, side, hidden)")
    side = grid.shape[0]
    if rows < 1 or cols < 1 or rows > side or cols > side:
        raise ValueError(f"a {rows}x{cols} layout does not fit a {side}x{side} grid")

    row_edges = np.linspace(0, side, rows + 1).round().astype(int)
    col_edges = np.linspace(0, side, cols + 1).round().astype(int)
    cells = []
    for r in range(rows):
        for c in range(cols):
            block = grid[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1], :]
            cells.append(block.reshape(-1, grid.shape[2]).mean(axis=0))
    return np.stack(cells)


@dataclass(frozen=True)
class RegionLocalizer:
    """Scores candidate crop cells; higher means more likely to answer."""

    direction: np.ndarray
    offset: np.ndarray

    def scores(self, cells: np.ndarray) -> np.ndarray:
        """Score every cell of one example. ``cells`` is (n_cells, hidden)."""
        norm = np.linalg.norm(self.direction)
        if norm == 0:
            return np.zeros(len(cells))
        return (cells - self.offset) @ self.direction / norm

    def choose(self, cells: np.ndarray) -> int:
        """Index of the cell to crop."""
        return int(np.argmax(self.scores(cells)))


def train_localizer(
    cells_per_example: Sequence[np.ndarray],
    correct_per_example: Sequence[Sequence[bool]],
) -> RegionLocalizer:
    """Fit a difference-of-means direction separating useful cells from useless.

    Each example contributes one pooled feature per candidate cell, labelled by
    whether cropping there produced a correct answer. Examples where every cell
    fails, or every cell succeeds, carry no ranking information and are skipped.
    """
    positive, negative = [], []
    for cells, correct in zip(cells_per_example, correct_per_example):
        if len(cells) != len(correct):
            raise ValueError("cells and labels must align per example")
        flags = np.asarray(correct, dtype=bool)
        if flags.all() or not flags.any():
            continue
        positive.append(cells[flags])
        negative.append(cells[~flags])

    if not positive:
        raise ValueError("no example has a mix of useful and useless cells")

    mu_pos = np.concatenate(positive).mean(axis=0)
    mu_neg = np.concatenate(negative).mean(axis=0)
    return RegionLocalizer(direction=mu_pos - mu_neg, offset=(mu_pos + mu_neg) / 2.0)


def evaluate_localizer(
    localizer: RegionLocalizer,
    cells_per_example: Sequence[np.ndarray],
    correct_per_example: Sequence[Sequence[bool]],
) -> dict[str, float]:
    """Hit rate of the chosen cell, against random and oracle baselines.

    ``chosen`` is the fraction of examples where the localizer's pick answers
    correctly; ``random`` is what picking uniformly would give; ``oracle`` is
    the fraction where *some* cell works, the ceiling any localizer can reach.
    """
    chosen = random = oracle = 0
    total = 0
    for cells, correct in zip(cells_per_example, correct_per_example):
        flags = np.asarray(correct, dtype=bool)
        if len(flags) == 0:
            continue
        total += 1
        chosen += bool(flags[localizer.choose(cells)])
        random += float(flags.mean())
        oracle += bool(flags.any())
    if total == 0:
        raise ValueError("no examples to evaluate")
    return {
        "chosen": chosen / total,
        "random": random / total,
        "oracle": oracle / total,
        "examples": float(total),
    }
