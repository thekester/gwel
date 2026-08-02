"""Does the Pareto-dominance headline survive a different split?

The paper's strongest claim, every entropy operating point is dominated by a
probe operating point, is measured on a single held-out fold of 200 examples.
A single fold is exactly the sample size at which this project has already been
burned once: `FINDINGS.md` documents train and test folds disagreeing about
which fixed policy is better.

So the claim is re-run under many resplits. Everything downstream of the split
is refitted each time (the probe direction, the calibration, the thresholds) so nothing leaks across the resampling. What is reported is the *frequency* of
dominance, not a single verdict, plus the distribution of the latency saving at
matched accuracy.

Usage: python scripts/resplit_dominance.py --trials 200
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.router.decision import escalation_delta, fit_gain_rule, signed_gain
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split

from evaluate_decision_rule import VALUE_GRID, component_costs, load_arrays, policy_cost

RATES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


def frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points not dominated on (accuracy up, latency down)."""
    return [
        (a, c)
        for a, c in points
        if not any(
            a2 >= a - 1e-9 and c2 <= c + 1e-9 and (a2 > a or c2 < c) for a2, c2 in points
        )
    ]


def is_dominated(point: tuple[float, float], pool: list[tuple[float, float]]) -> bool:
    a, c = point
    return any(
        a2 >= a - 1e-9 and c2 <= c + 1e-9 and (a2 > a or c2 < c) for a2, c2 in pool
    )


def saving_at_matched_accuracy(
    probe_points: list[tuple[float, float]],
    entropy_points: list[tuple[float, float]],
) -> float | None:
    """Latency saving of the probe at the best accuracy both signals reach.

    Interpolation would invent operating points neither policy has, so the
    comparison is made at the highest accuracy each signal actually attains
    at or above the other's target.
    """
    best = min(max(a for a, _ in probe_points), max(a for a, _ in entropy_points))
    probe = min(c for a, c in probe_points if a >= best - 1e-9)
    entropy = min(c for a, c in entropy_points if a >= best - 1e-9)
    if entropy <= 0:
        return None
    return (entropy - probe) / entropy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--activations", default="results/activations_full.npz")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--out", default="results/resplit_dominance.json")
    args = parser.parse_args()

    config = load_config(args.config)
    cheap_ok, full_ok, entropy, matrix, _ = load_arrays(config, args.activations, args.layer)
    gains = signed_gain(cheap_ok, full_ok)
    costs = component_costs()
    delta = {read: escalation_delta(costs, read=read) for read in ("entropy", "probe")}

    # Re-derive example ids and dataset labels for stratified resplitting.
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

    dominated_all = 0
    savings: list[float] = []
    probe_front_share: list[float] = []
    rule_undominated: list[float] = []
    survivor_counts: list[int] = []
    survivor_margins: list[float] = []

    for trial in range(args.trials):
        split = make_split(
            usable,
            datasets,
            val_fraction=config.router.val_fraction,
            test_fraction=config.router.test_fraction,
            seed=10_000 + trial,
        )
        train = np.array([order[e] for e in split.train])
        test = np.array([order[e] for e in split.test])
        if len(train) < 50 or len(test) < 50:
            continue

        probe = fit_layer_probe(
            matrix[train], (gains[train] > 0).astype(float), args.layer
        )
        probe_score = probe.score(matrix)

        curves: dict[str, list[tuple[float, float]]] = {}
        for name, score, read in (
            ("entropy", entropy, "entropy"),
            ("probe", probe_score, "probe"),
        ):
            points = []
            for rate in RATES:
                cut = np.quantile(score[train], 1.0 - rate)
                fires = score[test] >= cut
                points.append(
                    (
                        float(np.where(fires, full_ok[test], cheap_ok[test]).mean()),
                        float(policy_cost(fires, costs, read=read).mean()),
                    )
                )
            curves[name] = points

        pool = curves["entropy"] + curves["probe"]
        survivors = [p for p in curves["entropy"] if not is_dominated(p, curves["probe"])]
        dominated_all += not survivors
        survivor_counts.append(len(survivors))
        # When an entropy point does survive, how much accuracy does it actually
        # buy over the cheapest probe point at no greater latency? A "surviving"
        # point that wins by one query is not a substantive exception.
        for accuracy, latency in survivors:
            cheaper = [a for a, c in curves["probe"] if c <= latency + 1e-9]
            if cheaper:
                survivor_margins.append(accuracy - max(cheaper))
        front = frontier(pool)
        probe_front_share.append(
            sum(1 for p in front if p in curves["probe"]) / max(len(front), 1)
        )
        saving = saving_at_matched_accuracy(curves["probe"], curves["entropy"])
        if saving is not None:
            savings.append(saving)

        # The untuned cost-derived rule, on the same resplit.
        undominated = 0
        for value in VALUE_GRID:
            rule = fit_gain_rule(
                probe_score[train],
                gains[train],
                delta_ms=delta["probe"],
                value_ms_per_correct=value,
            )
            fires = rule.escalate(probe_score[test])
            point = (
                float(np.where(fires, full_ok[test], cheap_ok[test]).mean()),
                float(policy_cost(fires, costs, read="probe").mean()),
            )
            undominated += not is_dominated(point, curves["probe"])
        rule_undominated.append(undominated / len(VALUE_GRID))

    trials = len(probe_front_share)
    savings_array = np.array(savings)
    margins = np.array(survivor_margins)
    summary = {
        "trials": trials,
        "rates_per_signal": len(RATES),
        "dominance_rate": dominated_all / trials,
        "probe_share_of_front_mean": float(np.mean(probe_front_share)),
        "surviving_entropy_points_mean": float(np.mean(survivor_counts)),
        "surviving_entropy_points_max": int(np.max(survivor_counts)),
        "survivor_margin_median": float(np.median(margins)) if margins.size else None,
        "survivor_margin_p90": float(np.percentile(margins, 90)) if margins.size else None,
        "survivor_margin_within_2pt": float((margins <= 0.02).mean()) if margins.size else None,
        "saving_median": float(np.median(savings_array)),
        "saving_p05": float(np.percentile(savings_array, 5)),
        "saving_p95": float(np.percentile(savings_array, 95)),
        "saving_positive_rate": float((savings_array > 0).mean()),
        "untuned_rule_undominated_mean": float(np.mean(rule_undominated)),
    }

    print(f"{trials} resplits, probe layer {args.layer}, {len(RATES)} rates per signal\n")
    print(f"every entropy point dominated  : {summary['dominance_rate']:.0%} of splits")
    print(
        f"entropy points surviving       : {summary['surviving_entropy_points_mean']:.2f} "
        f"of {len(RATES)} on average, at most {summary['surviving_entropy_points_max']}"
    )
    if margins.size:
        print(
            f"  and when one survives it wins by {summary['survivor_margin_median']:+.3f} "
            f"accuracy (median); {summary['survivor_margin_within_2pt']:.0%} win by <= 0.02"
        )
    print(f"probe share of the Pareto front: {summary['probe_share_of_front_mean']:.0%}")
    print(
        f"latency saving at matched accuracy: median {summary['saving_median']:+.1%}, "
        f"90% range [{summary['saving_p05']:+.1%}, {summary['saving_p95']:+.1%}]"
    )
    print(f"saving is positive in          : {summary['saving_positive_rate']:.0%} of splits")
    print(
        f"untuned cost-derived rule on front: "
        f"{summary['untuned_rule_undominated_mean']:.0%} of its operating points"
    )

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
