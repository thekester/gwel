"""Summarize an oracle run: per-config accuracy, label distribution, savings.

Answers the go/no-go questions for the pilot: do several actions actually win,
and how much cost does the oracle save over fixed policies?

Usage: python scripts/analyze_oracle.py --config configs/pilot20.yaml
"""

import argparse
from collections import Counter, defaultdict

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.cost import CostWeights
from gwel.oracle.label import derive_labels, record_cost
from gwel.oracle.records import RunRecord, deduplicate_records, read_records


def per_config_table(records: list[RunRecord]) -> None:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.config_id].append(record)

    print(f"{'config':<14}{'n':>4}{'acc':>7}{'lat ms':>9}{'vtok':>7}{'net mJ':>10}")
    for config_id, group in sorted(grouped.items()):
        accuracy = sum(r.correct for r in group) / len(group)
        latency = sum(r.latency_ms for r in group) / len(group)
        tokens = sum(r.visual_tokens for r in group) / len(group)
        energies = [r.net_energy_mj for r in group if r.net_energy_mj is not None]
        energy = sum(energies) / len(energies) if energies else float("nan")
        print(
            f"{config_id:<14}{len(group):>4}{accuracy:>7.2f}{latency:>9.0f}"
            f"{tokens:>7.0f}{energy:>10.0f}"
        )


def fixed_policy_cost(
    records: list[RunRecord], config_id: str, weights: CostWeights
) -> tuple[float, float] | None:
    """(accuracy, mean cost) of always running one configuration."""
    selected = [r for r in records if r.config_id == config_id]
    if not selected:
        return None
    accuracy = sum(r.correct for r in selected) / len(selected)
    cost = sum(record_cost(r, weights) for r in selected) / len(selected)
    return accuracy, cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot20.yaml")
    parser.add_argument("--records", default=None)
    parser.add_argument("--metric", choices=("per-dataset", "vqa"), default="per-dataset")
    args = parser.parse_args()

    config = load_config(args.config)
    records = deduplicate_records(read_records(args.records or config.paths.records))
    policy = (
        ScoringPolicy() if args.metric == "per-dataset" else ScoringPolicy(dataset_metrics={})
    )
    records = rescore_records(records, policy)
    weights = CostWeights.from_config(config.cost)
    examples = {r.example_id for r in records}
    print(f"{len(records)} records, {len(examples)} examples\n")

    print("== per-config accuracy and mean measured cost ==")
    per_config_table(records)

    print("\n== per-dataset accuracy of key configs ==")
    by_dataset: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_dataset[record.dataset].append(record)
    for dataset, group in sorted(by_dataset.items()):
        parts = []
        for config_id in ("no_image", "lowres_256", "full", "ocr_full"):
            selected = [r for r in group if r.config_id == config_id]
            if selected:
                parts.append(f"{config_id}={sum(r.correct for r in selected)}/{len(selected)}")
        print(f"  {dataset:<10} {'  '.join(parts)}")

    labels = derive_labels(records, weights=weights)
    print("\n== oracle label distribution ==")
    counts = Counter(
        label.action.value if label.action is not None else "unsolvable" for label in labels
    )
    for name, count in sorted(counts.items()):
        print(f"  {name:<12} {count:>3} ({count / len(labels):.0%})")

    print("\n== action families: best config per example ==")
    families = {
        "ANSWER_LOW": lambda c: c.startswith("lowres_"),
        "CROP": lambda c: c.startswith("crop_"),
        "OCR region": lambda c: c.startswith("ocr_r"),
        "OCR page": lambda c: c == "ocr_full",
    }
    by_example: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_example[record.example_id].append(record)
    for family, matches in families.items():
        solved = sum(
            any(r.correct for r in group if matches(r.config_id)) for group in by_example.values()
        )
        cheapest = [
            min(
                (record_cost(r, weights) for r in group if matches(r.config_id) and r.correct),
                default=None,
            )
            for group in by_example.values()
        ]
        costs = [c for c in cheapest if c is not None]
        mean_cost = sum(costs) / len(costs) if costs else float("nan")
        print(
            f"  {family:<12} solves {solved:>3}/{len(by_example)} "
            f"| mean cost when it solves {mean_cost:.4f}"
        )

    solvable = [label for label in labels if label.cost is not None]
    if solvable:
        oracle_cost = sum(label.cost for label in solvable) / len(solvable)
        print(f"\n== mean cost, solvable examples (n={len(solvable)}) ==")
        print(f"  oracle:       {oracle_cost:.4f}")
        for config_id in ("lowres_256", "full", "ocr_full"):
            policy = fixed_policy_cost(
                [r for r in records if r.example_id in {l.example_id for l in solvable}],
                config_id,
                weights,
            )
            if policy is not None:
                accuracy, cost = policy
                print(f"  always {config_id:<11} acc={accuracy:.2f} cost={cost:.4f}")


if __name__ == "__main__":
    main()
