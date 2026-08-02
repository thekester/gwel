"""Emit pgfplots coordinates for the decision-rule figures.

Figures in the paper are drawn from numbers, and the numbers should come from
the same code path as the claims. This prints ready-to-paste coordinate lists
so no figure can drift from the run records behind it.

Usage: PYTHONPATH=scripts python scripts/make_figure_data.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.router.decision import (
    escalation_delta,
    fit_correctness_rule,
    fit_gain_calibrator,
    fit_gain_rule,
    signed_gain,
)
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split

from evaluate_decision_rule import VALUE_GRID, component_costs, load_arrays, policy_cost
from resplit_dominance import RATES, saving_at_matched_accuracy

BINS = 6


def coords(pairs) -> str:
    return " ".join(f"({x:.4g},{y:.4g})" for x, y in pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--activations", default="results/activations_full.npz")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()

    config = load_config(args.config)
    cheap_ok, full_ok, entropy, matrix, folds = load_arrays(
        config, args.activations, args.layer
    )
    train, test = folds["train"], folds["test"]
    gains = signed_gain(cheap_ok, full_ok)
    costs = component_costs()
    delta = escalation_delta(costs, read="probe")

    probe = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float), args.layer)
    score = probe.score(matrix)

    # ---- Figure: gain reliability -----------------------------------------
    calibrator = fit_gain_calibrator(score[train], gains[train])
    predicted = calibrator.predict(score[test])
    edges = np.quantile(predicted, np.linspace(0, 1, BINS + 1))
    edges[0] -= 1e-9
    reliability = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (predicted > low) & (predicted <= high)
        if mask.sum() >= 5:
            reliability.append((float(predicted[mask].mean()), float(gains[test][mask].mean())))
    print("% gain reliability: (predicted E[G], realised mean G)")
    print(coords(reliability))
    span = [min(p for p, _ in reliability), max(p for p, _ in reliability)]
    print(f"% diagonal from {span[0]:.3f} to {span[1]:.3f}\n")

    # ---- Figure: decision rule on the accuracy/latency plane ---------------
    def point(fires):
        return (
            float(policy_cost(fires, costs, read="probe").mean()),
            float(np.where(fires, full_ok[test], cheap_ok[test]).mean()),
        )

    swept = []
    for rate in RATES:
        cut = np.quantile(score[train], 1.0 - rate)
        swept.append(point(score[test] >= cut))
    print("% probe, tuned rate sweep (latency, accuracy)")
    print(coords(swept))

    full_accuracy = float(full_ok[train].mean())
    gain_points, ucci_points = [], []
    for value in VALUE_GRID:
        gain_rule = fit_gain_rule(
            score[train], gains[train], delta_ms=delta, value_ms_per_correct=value
        )
        ucci_rule = fit_correctness_rule(
            score[train],
            cheap_ok[train],
            full_accuracy=full_accuracy,
            delta_ms=delta,
            value_ms_per_correct=value,
        )
        gain_points.append(point(gain_rule.escalate(score[test])))
        ucci_points.append(point(ucci_rule.escalate(score[test])))
    print("% cost-derived gain rule, untuned (latency, accuracy)")
    print(coords(gain_points))
    print("% UCCI correctness rule, same values of V")
    print(coords(ucci_points))
    print(
        f"% fixed: cheap ({costs['cheap']:.1f},{cheap_ok[test].mean():.3f}) "
        f"full ({costs['full']:.1f},{full_ok[test].mean():.3f})\n"
    )

    # ---- Figure: distribution of the saving across resplits ---------------
    stored = np.load(args.activations, allow_pickle=True)
    ids = [str(e) for e in stored["example_ids"]]
    from collections import defaultdict

    from gwel.data.scoring import ScoringPolicy, rescore_records
    from gwel.oracle.records import deduplicate_records, read_records

    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    usable = [
        e
        for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    datasets = [grouped[e]["lowres_384"].dataset for e in usable]
    order = {e: i for i, e in enumerate(usable)}

    savings = []
    for trial in range(args.trials):
        split = make_split(
            usable,
            datasets,
            val_fraction=config.router.val_fraction,
            test_fraction=config.router.test_fraction,
            seed=10_000 + trial,
        )
        tr = np.array([order[e] for e in split.train])
        te = np.array([order[e] for e in split.test])
        local = fit_layer_probe(matrix[tr], (gains[tr] > 0).astype(float), args.layer)
        local_score = local.score(matrix)
        curves = {}
        for name, values, read in (
            ("entropy", entropy, "entropy"),
            ("probe", local_score, "probe"),
        ):
            points = []
            for rate in RATES:
                cut = np.quantile(values[tr], 1.0 - rate)
                fires = values[te] >= cut
                points.append(
                    (
                        float(np.where(fires, full_ok[te], cheap_ok[te]).mean()),
                        float(policy_cost(fires, costs, read=read).mean()),
                    )
                )
            curves[name] = points
        saving = saving_at_matched_accuracy(curves["probe"], curves["entropy"])
        if saving is not None:
            savings.append(saving)

    array = np.array(savings)
    hist_edges = np.arange(0.0, 0.55, 0.05)
    counts, _ = np.histogram(array, bins=hist_edges)
    print("% resplit saving histogram (bin centre in %, count)")
    print(
        coords(
            [
                (100 * (lo + hi) / 2, int(c))
                for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], counts)
            ]
        )
    )
    print(
        f"% n={len(array)}, median {np.median(array):+.1%}, "
        f"p05 {np.percentile(array, 5):+.1%}, p95 {np.percentile(array, 95):+.1%}"
    )

    Path("results/figure_data.json").write_text(
        json.dumps(
            {
                "reliability": reliability,
                "swept": swept,
                "gain_rule": gain_points,
                "ucci_rule": ucci_points,
                "saving_hist": [
                    [float((lo + hi) / 2), int(c)]
                    for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], counts)
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
