"""Compare every routing policy on the same cached measurements.

Simulates fixed policies, an entropy-threshold baseline, the learned router,
and the oracle over identical run records, then reports the accuracy/cost
Pareto front and the fraction of the oracle gap each policy closes.

Usage: python scripts/evaluate_router.py --config configs/pilot200.yaml
"""

import argparse
from dataclasses import replace

from gwel.actions import Action
from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.cost import CostWeights
from gwel.oracle.label import read_labels
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.evaluate import bootstrap_interval, paired_difference, pareto_front, summarize
from gwel.router.splits import make_split
from gwel.router.policies import (
    ExampleRuns,
    fixed_policy,
    group_runs,
    oracle_policy,
    simulate,
    threshold_policy,
    tune_threshold,
)


def learned_policy(checkpoint_dir: str, feature_config_id: str):
    """Wrap a trained router checkpoint as a simulation policy."""
    import numpy as np

    from gwel.router.features import build_features
    from gwel.router.model import RouterCheckpoint

    checkpoint = RouterCheckpoint.load(checkpoint_dir)

    def choose(run: ExampleRuns) -> Action:
        record = run.by_config.get(feature_config_id)
        if record is None or record.signals is None:
            return Action.ANSWER_LOW
        features = build_features(record)[np.newaxis, :]
        return checkpoint.predict(features)[0][0]

    return choose


def _best_zero_probe(
    train_runs,
    *,
    probe_config_id: str,
    weights,
    region_selection: str,
):
    """Fit a zero-probe router and pick its threshold on the training fold."""
    from gwel.router.zero_probe import train_zero_probe

    best = None
    best_cost = float("inf")
    for threshold in [i / 20 for i in range(1, 20)]:
        router = train_zero_probe(
            train_runs, probe_config_id=probe_config_id, threshold=threshold
        )
        results = simulate(
            train_runs,
            router.policy(probe_config_id=probe_config_id),
            weights=weights,
            region_selection=region_selection,
        )
        if not results:
            continue
        cost = sum(r.cost for r in results) / len(results)
        if cost < best_cost:
            best, best_cost = router, cost
    if best is None:
        raise ValueError("could not fit a zero-probe router on the training fold")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot200.yaml")
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=1.0,
        help="max_entropy above which the baseline escalates to CROP",
    )
    parser.add_argument(
        "--region-selection",
        choices=("cheapest", "best", "first"),
        default="cheapest",
        help="how the region within a CROP/OCR family is picked; 'best' assumes a perfect localizer",
    )
    parser.add_argument(
        "--fold",
        choices=("all", "train", "val", "test"),
        default="all",
        help="restrict evaluation to one split fold; use 'test' for reportable numbers",
    )
    parser.add_argument(
        "--cost-scale",
        type=float,
        default=1.0,
        help="multiply every resource weight: >1 tightens the budget relative to accuracy",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    weights = CostWeights.from_config(config.cost)
    if args.cost_scale != 1.0:
        weights = replace(
            weights,
            lambda_latency_per_ms=weights.lambda_latency_per_ms * args.cost_scale,
            lambda_energy_per_mj=weights.lambda_energy_per_mj * args.cost_scale,
            lambda_memory_per_mb=weights.lambda_memory_per_mb * args.cost_scale,
            lambda_visual_tokens=weights.lambda_visual_tokens * args.cost_scale,
        )
    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    runs = group_runs(records)
    labels = {label.example_id: label.action for label in read_labels(config.paths.labels)}

    split = make_split(
        [run.example_id for run in runs],
        [run.dataset for run in runs],
        val_fraction=config.router.val_fraction,
        test_fraction=config.router.test_fraction,
        seed=config.router.seed,
    )
    # The scalar threshold is fitted on training examples only, so it faces
    # the same held-out test fold as the learned router.
    train_runs = [run for run in runs if run.example_id in set(split.train)]
    tuned_threshold, _ = tune_threshold(
        train_runs,
        signal="mean_entropy",
        cheap_config_id=config.router.feature_config_id,
        weights=weights,
        region_selection=args.region_selection,
        probe_config_id=config.router.feature_config_id,
    )

    if args.fold != "all":
        keep = set(getattr(split, args.fold))
        runs = [run for run in runs if run.example_id in keep]

    policies: list[tuple[str, object]] = [
        ("always ANSWER_LOW", fixed_policy(Action.ANSWER_LOW)),
        ("always CROP", fixed_policy(Action.CROP)),
        ("always OCR", fixed_policy(Action.OCR)),
        (
            f"entropy > {args.entropy_threshold} (fixed)",
            threshold_policy(
                "mean_entropy",
                args.entropy_threshold,
                cheap_config_id=config.router.feature_config_id,
            ),
        ),
        (
            f"entropy > {tuned_threshold:.2f} (tuned)",
            threshold_policy(
                "mean_entropy",
                tuned_threshold,
                cheap_config_id=config.router.feature_config_id,
            ),
        ),
    ]
    try:
        policies.append(
            (
                "learned router",
                learned_policy(config.paths.router_dir, config.router.feature_config_id),
            )
        )
    except (FileNotFoundError, OSError):
        print(f"(no router checkpoint in {config.paths.router_dir}; skipping learned policy)")

    # Zero-probe: decides from question wording and image geometry alone, so it
    # is never charged for a probe pass. Its threshold is tuned on train only.
    zero_probe = _best_zero_probe(
        train_runs,
        probe_config_id=config.router.feature_config_id,
        weights=weights,
        region_selection=args.region_selection,
    )
    policies.append(
        (
            f"zero-probe (t={zero_probe.threshold:.2f})",
            zero_probe.policy(probe_config_id=config.router.feature_config_id),
        )
    )
    policies.append(("oracle", oracle_policy(labels)))

    print(f"{len(runs)} examples ({args.fold} fold), cost weights from {args.config}")
    print(f"region selection: {args.region_selection}\n")
    print(f"{'policy':<22}{'accuracy [95% CI]':>26}{'cost [95% CI]':>26}{'escal':>7}")

    # Confidence-conditioned policies must run the cheap pass before deciding,
    # so they are charged for it; fixed policies and the oracle are not.
    probe = config.router.feature_config_id
    needs_probe = {name for name, _ in policies if "entropy" in name or name == "learned router"}

    names: list[str] = []
    accuracies: list[float] = []
    costs: list[float] = []
    per_example_cost: dict[str, list[float]] = {}
    for name, choose in policies:
        results = simulate(
            runs,
            choose,
            weights=weights,
            region_selection=args.region_selection,
            probe_config_id=probe if name in needs_probe else None,
        )
        summary = summarize(results)
        correct = [float(r.correct) for r in results]
        cost_values = [r.cost for r in results]
        per_example_cost[name] = cost_values

        names.append(name)
        accuracies.append(summary.accuracy)
        costs.append(summary.mean_cost)
        print(
            f"{name:<22}{str(bootstrap_interval(correct)):>26}"
            f"{str(bootstrap_interval(cost_values)):>26}{summary.escalation_rate:>7.2f}"
        )

    front = set(pareto_front(costs, accuracies))
    print("\nPareto front (min cost, max accuracy):")
    for index in sorted(front, key=lambda i: costs[i]):
        print(f"  {names[index]:<22} acc={accuracies[index]:.3f} cost={costs[index]:.4f}")

    # How much of the fixed-policy-to-oracle gap does each policy close?
    oracle_cost = costs[names.index("oracle")]
    best_fixed = min(
        (n for n in names if n.startswith("always")), key=lambda n: costs[names.index(n)]
    )
    span = costs[names.index(best_fixed)] - oracle_cost
    if span > 0:
        print(f"\nOracle gap closed ({best_fixed} = 0%, oracle = 100%):")
        for name in names:
            if name != "oracle" and not name.startswith("always"):
                closed = (costs[names.index(best_fixed)] - costs[names.index(name)]) / span
                print(f"  {name:<22}{closed:>8.1%}")

    # Paired comparisons against the strongest fixed policy: an interval that
    # excludes zero is the evidence that routing beats not routing.
    print(f"\nPaired cost difference vs {best_fixed} (negative favours routing):")
    for name in names:
        if name == best_fixed:
            continue
        difference = paired_difference(per_example_cost[name], per_example_cost[best_fixed])
        verdict = "significant" if difference.high < 0 or difference.low > 0 else "not significant"
        print(f"  {name:<22}{str(difference):>26}  {verdict}")


if __name__ == "__main__":
    main()
