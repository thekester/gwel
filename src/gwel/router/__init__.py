"""Routing: features, policies, probes, calibration, and evaluation.

Everything here is numpy-only except `model` and `train`, which are exposed
lazily so importing the package never pulls torch.
"""

from .budget import ActionProfile, BudgetRouter, RoutingDecision
from .budget_selection import (
    ActionStats,
    crossover_points,
    normalise_costs,
    policy_regions,
    select_action,
    select_under_budget,
)
from .calibration import IsotonicCalibrator, expected_calibration_error, fit_isotonic
from .conformal import (
    Regime,
    ThreeWayConformal,
    conformal_quantile,
    evaluate_three_way,
    fit_three_way,
)
from .coverage import (
    SelectivePoint,
    ThreeWayFrontier,
    escalating_frontier,
    pareto_filter,
    selective_frontier,
)
from .evaluate import (
    EvaluationSummary,
    Interval,
    PolicyResult,
    RiskCoverageCurve,
    auroc,
    bootstrap_interval,
    paired_difference,
    pareto_front,
    risk_coverage,
    summarize,
)
from .features import FEATURE_NAMES, build_feature_matrix, build_features
from .policies import (
    ExampleRuns,
    action_of,
    fixed_policy,
    group_runs,
    oracle_policy,
    simulate,
    threshold_policy,
    tune_threshold,
)
from .probes import LayerProbe, fisher_separability, fit_layer_probe, sweep_layers
from .recall_control import (
    RecallControlledThreshold,
    certifiable_recall,
    fit_recall_controlled,
    lower_bound,
)
from .splits import Split, make_split
from .zero_probe import (
    ZeroProbeRouter,
    fit_difference_of_means,
    fit_logistic,
    score_difference_of_means,
    train_zero_probe,
)

__all__ = [
    "ActionProfile",
    "ActionStats",
    "BudgetRouter",
    "EvaluationSummary",
    "ExampleRuns",
    "FEATURE_NAMES",
    "Interval",
    "IsotonicCalibrator",
    "LayerProbe",
    "PolicyResult",
    "RecallControlledThreshold",
    "Regime",
    "RiskCoverageCurve",
    "RoutingDecision",
    "SelectivePoint",
    "Split",
    "ThreeWayConformal",
    "ThreeWayFrontier",
    "ZeroProbeRouter",
    "action_of",
    "auroc",
    "bootstrap_interval",
    "build_feature_matrix",
    "build_features",
    "certifiable_recall",
    "conformal_quantile",
    "crossover_points",
    "escalating_frontier",
    "evaluate_three_way",
    "expected_calibration_error",
    "fisher_separability",
    "fit_difference_of_means",
    "fit_isotonic",
    "fit_layer_probe",
    "fit_recall_controlled",
    "fit_three_way",
    "fit_logistic",
    "fixed_policy",
    "group_runs",
    "lower_bound",
    "make_split",
    "normalise_costs",
    "oracle_policy",
    "paired_difference",
    "pareto_filter",
    "pareto_front",
    "policy_regions",
    "risk_coverage",
    "score_difference_of_means",
    "select_action",
    "select_under_budget",
    "selective_frontier",
    "simulate",
    "summarize",
    "sweep_layers",
    "threshold_policy",
    "train_zero_probe",
    "tune_threshold",
]


def __getattr__(name: str):
    """Lazily expose torch-dependent training/model APIs."""
    if name in ("RouterMLP", "RouterCheckpoint", "ACTIONS", "ACTION_TO_INDEX"):
        from . import model

        return getattr(model, name)
    if name in (
        "train_router",
        "build_routing_dataset",
        "split_by_example",
        "TrainResult",
        "RoutingDataset",
    ):
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
