"""How far to escalate, not just whether: a multi-rung policy against a binary one.

Every escalation method we have read is binary --- answer from a downsampled
image, or run the full one. Our ladder says that collapses the decision that
carries most of the money. Taking the cheapest configuration that answers each
query correctly:

    38.3% answered at 384 px      17.7% need an intermediate rung
     5.2% need full resolution    38.8% answered by nothing

So a binary policy over-serves three quarters of what it escalates. The two
rungs are also not equally efficient: 384 to 768 buys $13.3$ points for $79$ ms
($+1.69$ points per second) while 768 to full buys $3.3$ for $38$ ms ($+0.86$).
The binary jump sits between them at $+1.42$, which is what averaging a good
rung with a bad one produces.

This evaluates :class:`LadderRule` against the binary rule it generalises, on
output entropy, inside each domain, with each rung priced per example. The
comparison is like-for-like: same signal, same folds, same operator preference,
the only difference being whether the middle of the ladder exists.

Usage: PYTHONPATH=scripts python scripts/evaluate_ladder.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.oracle.token_cost import fit_token_cost
from gwel.router.decision import fit_gain_rule, fit_ladder_rule, signed_gain
from gwel.router.evaluate import bootstrap_interval
from gwel.router.splits import make_split

CHEAP = "lowres_384"
RUNGS = ("lowres_768", "full")
RESAMPLES = 40
VALUE_GRID = (400.0, 800.0, 1600.0, 3200.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--latency", default="results/component_latency.json")
    parser.add_argument("--out", default="results/ladder.json")
    args = parser.parse_args()

    config = load_config(args.config)
    profile = json.loads(Path(args.latency).read_text())
    cost_model = fit_token_cost(
        [r["visual_tokens"] for r in profile], [r["total_ms"] for r in profile]
    )

    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    ids = [
        e
        for e in grouped
        if all(c in grouped[e] for c in (CHEAP, *RUNGS)) and grouped[e][CHEAP].signals
    ]
    order = {e: i for i, e in enumerate(ids)}

    correct = {
        c: np.array([grouped[e][c].correct for e in ids]) for c in (CHEAP, *RUNGS)
    }
    latency = {
        c: cost_model.predict(np.array([grouped[e][c].visual_tokens for e in ids]))
        for c in (CHEAP, *RUNGS)
    }
    entropy = np.array([float(grouped[e][CHEAP].signals["mean_entropy"]) for e in ids])
    datasets = np.array([grouped[e][CHEAP].dataset for e in ids])

    gains = {r: signed_gain(correct[CHEAP], correct[r]) for r in RUNGS}
    deltas = np.column_stack([latency[r] - latency[CHEAP] for r in RUNGS])
    print(
        "rung prices over the cheap pass: "
        + ", ".join(f"{r} {(latency[r] - latency[CHEAP]).mean():+.0f} ms" for r in RUNGS)
    )

    # A rung only exists where it spends more than the one below it. The
    # processor caps its target at the input's longest side, so on a small image
    # `full` and the intermediate rung are literally the same configuration.
    tokens = {
        c: np.array([grouped[e][c].visual_tokens for e in ids], dtype=float)
        for c in RUNGS
    }
    distinct = tokens[RUNGS[1]] > tokens[RUNGS[0]]
    print(f"\n{'dataset':<10}{'top rung exists':>17}{'n':>6}")
    existence = {}
    for dataset in sorted(set(datasets)):
        mask = datasets == dataset
        existence[dataset] = float(distinct[mask].mean())
        print(f"{dataset:<10}{distinct[mask].mean():>17.0%}{mask.sum():>6}")
    print(f"{'ALL':<10}{distinct.mean():>17.0%}{len(ids):>6}")
    print(
        "\nWhere the share is zero the two rungs are the same configuration, so a\n"
        "ladder policy has nothing to find and should show no effect.\n"
    )

    results: dict[str, object] = {"datasets": {}}
    ladder_minus_binary: list[tuple[float, float]] = []

    for dataset in sorted(set(datasets)):
        members = [ids[i] for i in np.where(datasets == dataset)[0]]
        rows: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"accuracy": [], "latency": [], "top": [], "mid": []}
        )
        for seed in range(RESAMPLES):
            local = make_split(
                members, [dataset] * len(members),
                val_fraction=0.0, test_fraction=0.3, seed=9000 + seed,
            )
            train = np.array([order[e] for e in local.train])
            test = np.array([order[e] for e in local.test])
            if any(len(set((gains[r][train] > 0).tolist())) < 2 for r in RUNGS):
                continue

            for value in VALUE_GRID:
                # Binary: the published shape, escalating only to full resolution.
                binary = fit_gain_rule(
                    entropy[train], gains["full"][train],
                    delta_ms=float(deltas[train, -1].mean()),
                    value_ms_per_correct=value,
                ).escalate(entropy[test])
                accuracy = np.where(binary, correct["full"][test], correct[CHEAP][test])
                cost = np.where(binary, latency["full"][test], latency[CHEAP][test])
                rows[f"binary V={value:.0f}"]["accuracy"].append(float(accuracy.mean()))
                rows[f"binary V={value:.0f}"]["latency"].append(float(cost.mean()))
                rows[f"binary V={value:.0f}"]["top"].append(float(binary.mean()))
                rows[f"binary V={value:.0f}"]["mid"].append(0.0)

                # Ladder: same signal, same preference, middle rung restored.
                ladder = fit_ladder_rule(
                    entropy[train],
                    {r: gains[r][train] for r in RUNGS},
                    value_ms_per_correct=value,
                )
                chosen = ladder.choose(entropy[test], deltas[test])
                accuracy = np.where(
                    chosen < 0,
                    correct[CHEAP][test],
                    np.where(chosen == 0, correct[RUNGS[0]][test], correct[RUNGS[1]][test]),
                )
                cost = np.where(
                    chosen < 0,
                    latency[CHEAP][test],
                    np.where(chosen == 0, latency[RUNGS[0]][test], latency[RUNGS[1]][test]),
                )
                rows[f"ladder V={value:.0f}"]["accuracy"].append(float(accuracy.mean()))
                rows[f"ladder V={value:.0f}"]["latency"].append(float(cost.mean()))
                rows[f"ladder V={value:.0f}"]["top"].append(float((chosen == 1).mean()))
                rows[f"ladder V={value:.0f}"]["mid"].append(float((chosen == 0).mean()))

        summary = {
            name: {k: float(np.mean(v)) for k, v in fields.items()}
            for name, fields in rows.items()
        }
        results["datasets"][dataset] = summary

        print(f"== {dataset} (n={len(members)}) ==")
        print(f"{'policy':<18}{'mid rung':>10}{'top rung':>10}{'accuracy':>10}{'latency':>10}")
        for name in sorted(summary, key=lambda k: (k.split()[1], k)):
            row = summary[name]
            print(
                f"{name:<18}{row['mid']:>10.0%}{row['top']:>10.0%}"
                f"{row['accuracy']:>10.3f}{row['latency']:>10.1f}"
            )
        for value in VALUE_GRID:
            a = summary[f"ladder V={value:.0f}"]
            b = summary[f"binary V={value:.0f}"]
            ladder_minus_binary.append(
                (a["accuracy"] - b["accuracy"], a["latency"] - b["latency"])
            )
        print()

    accuracy_delta = bootstrap_interval([d for d, _ in ladder_minus_binary])
    latency_delta = bootstrap_interval([c for _, c in ladder_minus_binary])
    print("ladder minus binary, paired over four domains and four preferences:")
    print(f"  accuracy {accuracy_delta}")
    print(f"  latency  {latency_delta} ms")
    results["ladder_vs_binary"] = {
        "accuracy": [accuracy_delta.estimate, accuracy_delta.low, accuracy_delta.high],
        "latency": [latency_delta.estimate, latency_delta.low, latency_delta.high],
        "pairs": len(ladder_minus_binary),
    }

    dominates = sum(
        1 for a, c in ladder_minus_binary if a >= -1e-9 and c <= 1e-9 and (a > 0 or c < 0)
    )
    results["ladder_dominates"] = dominates
    results["top_rung_exists"] = existence
    print(
        f"\nthe ladder dominates the binary rule in {dominates}/"
        f"{len(ladder_minus_binary)} paired comparisons"
    )

    # The effect should appear only where the rung is real, which is the
    # sharpest version of the claim and the one worth checking.
    print(f"\n{'dataset':<10}{'top rung exists':>17}{'accuracy':>11}{'latency':>10}")
    per_domain = {}
    for dataset, summary in results["datasets"].items():
        accuracy = float(np.mean([
            summary[f"ladder V={v:.0f}"]["accuracy"] - summary[f"binary V={v:.0f}"]["accuracy"]
            for v in VALUE_GRID
        ]))
        cost = float(np.mean([
            summary[f"ladder V={v:.0f}"]["latency"] - summary[f"binary V={v:.0f}"]["latency"]
            for v in VALUE_GRID
        ]))
        per_domain[dataset] = {
            "top_rung_exists": existence[dataset],
            "accuracy_delta": accuracy,
            "latency_delta": cost,
        }
        print(
            f"{dataset:<10}{existence[dataset]:>17.0%}{accuracy:>+11.3f}{cost:>+10.1f}"
        )
    results["per_domain"] = per_domain

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
