"""Gwel: budget-aware active perception for sub-1B vision-language models.

Top-level exports stay dependency-light (numpy, PIL, psutil, yaml only);
torch, transformers, OCR engines, and ``datasets`` are imported lazily by
the subpackages that need them.
"""

from .actions import Action
from .config import GwelConfig, load_config
from .oracle import (
    CostWeights,
    OracleLabel,
    RunRecord,
    compute_cost,
    derive_labels,
    label_example,
    read_labels,
    read_records,
    record_cost,
    write_labels,
)
from .router import pareto_front, risk_coverage, summarize

__all__ = [
    "Action",
    "CostWeights",
    "GwelConfig",
    "OracleLabel",
    "RunRecord",
    "compute_cost",
    "derive_labels",
    "label_example",
    "load_config",
    "pareto_front",
    "read_labels",
    "read_records",
    "record_cost",
    "risk_coverage",
    "summarize",
    "write_labels",
]

__version__ = "0.3.0"
