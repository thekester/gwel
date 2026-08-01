"""Core primitives for budget-aware visual routing."""

from .router import Action, ActionProfile, BudgetRouter, RoutingDecision
from .oracle import ActionMeasurement, CostWeights, OracleLabel, label_minimal_action

__all__ = [
    "Action",
    "ActionProfile",
    "ActionMeasurement",
    "BudgetRouter",
    "CostWeights",
    "OracleLabel",
    "RoutingDecision",
    "label_minimal_action",
]

__version__ = "0.1.0"
