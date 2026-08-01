"""Deterministic, dataset-stratified train/val/test splits.

Splits are computed from example ids alone, so the same example lands in the
same fold no matter which script asks, which run produced the records, or how
many examples were measured. Stratifying by dataset keeps each fold covering
all four question regimes even at pilot sizes.
"""

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    """Example ids assigned to each fold."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def fold_of(self, example_id: str) -> str:
        if example_id in set(self.train):
            return "train"
        if example_id in set(self.val):
            return "val"
        if example_id in set(self.test):
            return "test"
        raise KeyError(f"{example_id!r} is not in this split")

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _stable_fraction(example_id: str, seed: int) -> float:
    """Map an example id to a stable value in [0, 1).

    Hashing rather than shuffling means adding examples never reshuffles the
    ones already assigned, so a split stays comparable as the pilot grows.
    """
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def make_split(
    example_ids: Sequence[str],
    datasets: Sequence[str] | None = None,
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 1234,
) -> Split:
    """Assign each example to train, val, or test.

    When ``datasets`` is supplied the fractions are applied within each dataset
    so every fold stays balanced across question regimes.
    """
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must leave room for training")
    if datasets is not None and len(datasets) != len(example_ids):
        raise ValueError("datasets and example_ids must have the same length")

    grouped: dict[str, list[str]] = defaultdict(list)
    for index, example_id in enumerate(example_ids):
        key = datasets[index] if datasets is not None else ""
        grouped[key].append(example_id)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for key in sorted(grouped):
        # Rank within the stratum so small strata still get every fold filled,
        # which a raw threshold on the hash value cannot guarantee.
        ranked = sorted(grouped[key], key=lambda e: _stable_fraction(e, seed))
        n = len(ranked)
        n_test = round(n * test_fraction)
        n_val = round(n * val_fraction)
        test.extend(ranked[:n_test])
        val.extend(ranked[n_test : n_test + n_val])
        train.extend(ranked[n_test + n_val :])

    return Split(train=tuple(sorted(train)), val=tuple(sorted(val)), test=tuple(sorted(test)))
