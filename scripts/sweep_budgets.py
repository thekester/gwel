"""Sweep the cost weights to trace the budget/accuracy Pareto front.

"Budget-aware" only means something if different budgets produce different
policies. This sweeps one lambda at a time over cached measurements, re-deriving
the oracle at each setting, and reports how the chosen action mix shifts.

Usage: python scripts/sweep_budgets.py --config configs/pilot200.yaml
"""

import argparse
from collections import Counter
from dataclasses import replace

from gwel.actions import Action
from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.cost import CostWeights
from gwel.oracle.label import derive_labels
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.evaluate import pareto_front, summarize
from gwel.router.policies import fixed_policy, group_runs, oracle_policy, simulate

#: Multipliers applied to the configured weight, spanning three decades.
MULTIPLIERS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)

SWEEPABLE = {
    "latency": "lambda_latency_per_ms",
    "energy": "lambda_energy_per_mj",
    "tokens": "lambda_visual_tokens",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot200.yaml")
    parser.add_argument("--axis", choices=sorted(SWEEPABLE), default="tokens")
    parser.add_argument(
        "--region-selection", choices=("cheapest", "best", "first"), default="best"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    base = CostWeights.from_config(config.cost)
    field = SWEEPABLE[args.axis]
    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    runs = group_runs(records)

    print(f"sweeping {field} over {len(runs)} examples")
    print(f"region selection: {args.region_selection}\n")
    print(
        f"{'x':>6}{'weight':>11}{'oracle':>9}{'best fixed':>12}{'gain':>8}"
        f"{'vtok':>7}  action mix"
    )

    accuracies: list[float] = []
    costs: list[float] = []
    mixes: list[str] = []
    for multiplier in MULTIPLIERS:
        weights = replace(base, **{field: getattr(base, field) * multiplier})
        labels = {
            label.example_id: label.action
            for label in derive_labels(records, weights=weights)
        }
        oracle = summarize(
            simulate(
                runs,
                oracle_policy(labels),
                weights=weights,
                region_selection=args.region_selection,
            )
        )
        # The strongest single action at this budget: routing is only worth
        # anything if the oracle beats whatever fixed policy wins here.
        fixed_costs = {}
        for action in Action.ordered():
            summary = summarize(
                simulate(
                    runs,
                    fixed_policy(action),
                    weights=weights,
                    region_selection=args.region_selection,
                )
            )
            if summary.examples:
                fixed_costs[action] = summary.mean_cost
        best_fixed = min(fixed_costs.values())
        gain = (best_fixed - oracle.mean_cost) / best_fixed if best_fixed else 0.0

        mix = Counter(labels.values())
        mix_text = " ".join(
            f"{action.value.split('_')[0]}={mix.get(action, 0)}" for action in Action.ordered()
        )
        accuracies.append(oracle.accuracy)
        costs.append(oracle.mean_cost)
        mixes.append(mix_text)
        print(
            f"{multiplier:>6.1f}{getattr(weights, field):>11.1e}{oracle.mean_cost:>9.4f}"
            f"{best_fixed:>12.4f}{gain:>8.1%}{oracle.mean_visual_tokens:>7.0f}  {mix_text}"
        )

    front = pareto_front(costs, accuracies)
    print(
        f"\nnon-dominated settings: "
        f"{[MULTIPLIERS[i] for i in sorted(front, key=lambda i: costs[i])]}"
    )
    distinct = len(set(mixes))
    print(f"distinct action mixes across the sweep: {distinct}/{len(MULTIPLIERS)}")
    if distinct == 1:
        print("  the budget does not change the policy: the routing premise is not exercised")


if __name__ == "__main__":
    main()
