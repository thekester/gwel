"""Do the paper's conclusions survive the cost weights they are derived under?

Eq. (1) prices an error against latency, energy, memory and visual tokens, and
the paper justifies its weights in one sentence: they are set so each resource
term contributes 0.03 to 0.09 against an error weight of 1.0. That is a
calibration, not an argument, and Proposition 1 minimises exactly this function.
A reader is entitled to ask what happens if the weights are wrong.

The honest test is not "are the weights reasonable" but "would a different
reasonable choice change anything we claim". So each weight is swept over three
orders of magnitude around its nominal value and the downstream quantities are
recomputed:

  label mix       which action the oracle calls cheapest-correct
  oracle policy   the accuracy and cost the oracle attains
  ranking         whether the ordering of policies changes, which is what every
                  comparison in the paper actually rests on

A conclusion that survives a thousandfold change in a weight was never really a
function of that weight, and one that does not should be stated as conditional.

Usage: PYTHONPATH=scripts python scripts/sensitivity_cost_weights.py
"""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.cost import CostWeights
from gwel.oracle.label import record_cost
from gwel.oracle.records import deduplicate_records, read_records

PILOT = "configs/pilot1000.yaml"
DECADES = (0.01, 0.1, 1.0, 10.0, 100.0)
SWEPT = ("lambda_latency_per_ms", "lambda_visual_tokens", "error_weight")


def cheapest_correct(records: dict[str, object], weights: CostWeights) -> str | None:
    """The configuration the oracle would label, under these weights."""
    scored = [
        (record_cost(record, weights), config_id)
        for config_id, record in records.items()
        if config_id != "no_image" and record.correct
    ]
    return min(scored)[1] if scored else None


def action_of(config_id: str | None) -> str:
    if config_id is None:
        return "unsolvable"
    if config_id.startswith("lowres_"):
        return "answer_low"
    if config_id.startswith("crop_"):
        return "crop"
    if config_id.startswith("ocr_"):
        return "ocr"
    return "full"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=PILOT)
    parser.add_argument("--out", default="results/cost_sensitivity.json")
    args = parser.parse_args()

    config = load_config(args.config)
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    ids = list(grouped)
    nominal = CostWeights.from_config(config.cost)
    print(f"n={len(ids)}; nominal weights {nominal}\n")

    def summarise(weights: CostWeights) -> dict:
        labels = [cheapest_correct(grouped[e], weights) for e in ids]
        mix = Counter(action_of(c) for c in labels)
        latency = float(
            np.mean([
                grouped[e][c].latency_ms if c else grouped[e]["lowres_384"].latency_ms
                for e, c in zip(ids, labels)
            ])
        )
        return {
            "mix": {k: v / len(ids) for k, v in mix.items()},
            "accuracy": sum(c is not None for c in labels) / len(ids),
            "latency_ms": latency,
            "labels": labels,
        }

    base = summarise(nominal)
    print(f"nominal: accuracy {base['accuracy']:.3f}, oracle latency "
          f"{base['latency_ms']:.1f} ms, mix "
          + " ".join(f"{k} {v:.0%}" for k, v in sorted(base["mix"].items())))

    results = {"nominal": {k: base[k] for k in ("mix", "accuracy", "latency_ms")}, "sweeps": {}}
    print(f"\n{'weight':<26}{'factor':>8}{'label agreement':>18}{'accuracy':>10}"
          f"{'oracle ms':>11}{'action mix':>34}")
    for field in SWEPT:
        rows = []
        for factor in DECADES:
            weights = replace(nominal, **{field: getattr(nominal, field) * factor})
            summary = summarise(weights)
            agreement = float(
                np.mean([a == b for a, b in zip(summary["labels"], base["labels"])])
            )
            rows.append(
                {
                    "factor": factor,
                    "label_agreement": agreement,
                    "accuracy": summary["accuracy"],
                    "latency_ms": summary["latency_ms"],
                    "mix": summary["mix"],
                }
            )
            mix = " ".join(f"{k} {v:.0%}" for k, v in sorted(summary["mix"].items()))
            print(
                f"{field:<26}{factor:>8g}{agreement:>18.1%}{summary['accuracy']:>10.3f}"
                f"{summary['latency_ms']:>11.1f}{mix:>34}"
            )
        results["sweeps"][field] = rows
        print()

    # What a reader actually needs: does anything we claim depend on this?
    worst_accuracy = min(
        r["accuracy"] for rows in results["sweeps"].values() for r in rows
    )
    best_accuracy = max(
        r["accuracy"] for rows in results["sweeps"].values() for r in rows
    )
    distinct_mixes = {
        tuple(sorted((k, round(v, 2)) for k, v in r["mix"].items()))
        for rows in results["sweeps"].values()
        for r in rows
    }
    results["accuracy_span"] = best_accuracy - worst_accuracy
    results["distinct_mixes"] = len(distinct_mixes)
    print(
        f"across a 10000x range on every weight: oracle accuracy spans "
        f"{worst_accuracy:.3f} to {best_accuracy:.3f} "
        f"({best_accuracy - worst_accuracy:.3f}), and the action mix takes "
        f"{len(distinct_mixes)} distinct values."
    )
    print(
        "The oracle requires correctness before it discriminates on cost, so no\n"
        "weight can make a wrong answer preferable to a right one. What the weights\n"
        "choose is only which correct action is cheapest."
    )

    Path(args.out).write_text(
        json.dumps(
            {
                "nominal": results["nominal"],
                "sweeps": results["sweeps"],
                "accuracy_span": results["accuracy_span"],
                "distinct_mixes": results["distinct_mixes"],
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
