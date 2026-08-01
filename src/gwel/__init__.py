"""Core primitives for budget-aware visual routing."""

from .router import Action, ActionProfile, BudgetRouter, RoutingDecision
from .oracle import ActionMeasurement, CostWeights, OracleLabel, label_minimal_action
from .evaluation import EvaluationSummary, PolicyResult, oracle_gap, summarize
from .benchmark import BenchmarkExample, read_jsonl, write_jsonl

__all__ = [
    "Action",
    "ActionProfile",
    "ActionMeasurement",
    "BenchmarkExample",
    "BudgetRouter",
    "CostWeights",
    "EvaluationSummary",
    "OracleLabel",
    "PolicyResult",
    "RoutingDecision",
    "label_minimal_action",
    "oracle_gap",
    "read_jsonl",
    "summarize",
    "write_jsonl",
]

__version__ = "0.1.0"
