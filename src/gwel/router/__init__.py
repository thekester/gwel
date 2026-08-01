"""Supervised router: features, MLP, distillation training, evaluation."""

from .budget import ActionProfile, BudgetRouter, RoutingDecision
from .evaluate import (
    EvaluationSummary,
    PolicyResult,
    RiskCoverageCurve,
    pareto_front,
    risk_coverage,
    summarize,
)
from .features import FEATURE_NAMES, build_feature_matrix, build_features

__all__ = [
    "ActionProfile",
    "BudgetRouter",
    "EvaluationSummary",
    "FEATURE_NAMES",
    "PolicyResult",
    "RiskCoverageCurve",
    "RoutingDecision",
    "build_feature_matrix",
    "build_features",
    "pareto_front",
    "risk_coverage",
    "summarize",
]


def __getattr__(name: str):
    """Lazily expose torch-dependent training/model APIs."""
    if name in ("RouterMLP", "RouterCheckpoint", "ACTIONS", "ACTION_TO_INDEX"):
        from . import model

        return getattr(model, name)
    if name in ("train_router", "build_routing_dataset", "split_by_example", "TrainResult", "RoutingDataset"):
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
