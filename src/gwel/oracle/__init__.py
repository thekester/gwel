"""Oracle pipeline: run all configurations, then label minimal actions."""

from .cost import CostWeights, compute_cost, resource_cost
from .label import OracleLabel, derive_labels, label_example, read_labels, record_cost, write_labels
from .records import (
    RunRecord,
    append_records,
    load_done_keys,
    read_records,
    records_to_parquet,
)

__all__ = [
    "CostWeights",
    "OracleLabel",
    "RunRecord",
    "append_records",
    "compute_cost",
    "derive_labels",
    "label_example",
    "load_done_keys",
    "read_labels",
    "read_records",
    "record_cost",
    "records_to_parquet",
    "resource_cost",
    "write_labels",
]


def __getattr__(name: str):
    """Lazily expose the runner so importing gwel.oracle never pulls PIL-heavy deps."""
    if name in ("OracleRunner", "PreparedOp", "prepare_ops"):
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
