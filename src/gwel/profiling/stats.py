"""Robust summary statistics for repeated hardware measurements."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class RepeatStats:
    """Median-centred summary of one repeated scalar measurement."""

    n: int
    median: float
    iqr: float
    p95: float
    minimum: float
    maximum: float
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "median": self.median,
            "iqr": self.iqr,
            "p95": self.p95,
            "min": self.minimum,
            "max": self.maximum,
            "values": list(self.values),
        }


def summarize_repeats(values: Sequence[float]) -> RepeatStats:
    """Summarize repeated measurements with median, IQR, and p95."""
    if len(values) == 0:
        raise ValueError("at least one measurement is required")
    array = np.asarray(values, dtype=np.float64)
    q25, median, q75, p95 = np.percentile(array, [25, 50, 75, 95])
    return RepeatStats(
        n=int(array.size),
        median=float(median),
        iqr=float(q75 - q25),
        p95=float(p95),
        minimum=float(array.min()),
        maximum=float(array.max()),
        values=tuple(float(v) for v in array),
    )


def repeat_measure(
    fn: Callable[[], T],
    *,
    repeats: int = 1,
    warmup: int = 0,
) -> list[T]:
    """Call ``fn`` ``warmup + repeats`` times and return the measured results.

    Warmup calls are executed but discarded, which absorbs one-time costs such
    as CUDA kernel compilation or page-cache population.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        fn()
    return [fn() for _ in range(repeats)]
