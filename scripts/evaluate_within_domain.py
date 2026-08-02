"""Does the decision rule survive its own signal being discredited?

`analyze_domain_confound.py` showed the pre-generation probe is largely a domain
detector: within a single dataset it ranks at chance while output entropy holds
$0.664$. Every number reported for the calibrated escalation rule was measured
with the probe, on a four-dataset mixture. Two questions follow, and the paper
has no positive contribution left unless the answers are yes.

  1. Is the *method* signal-agnostic? Run the same rule on output entropy,
     inside one dataset, where entropy is the signal that works.
  2. Does per-query costing change the policy, not just the accounting?
     `recost_policies.py` established that escalation prices vary by $16\\%$ on
     average and that the probe selects the dear tail. A rule with a single
     break-even cannot express that; one with a per-query break-even can.

Baselines are matched throughout: the tuned rate sweep (what the paper reports),
the UCCI-style correctness rule, and both variants of the gain rule. Everything
is refit inside each dataset over many resamples, so no between-domain variance
can leak in.

Usage: PYTHONPATH=scripts python scripts/evaluate_within_domain.py
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
from gwel.router.decision import (
    fit_correctness_rule,
    fit_gain_rule,
    fit_per_query_gain_rule,
    signed_gain,
)
from gwel.router.evaluate import bootstrap_interval
from gwel.router.splits import make_split

RESAMPLES = 40
VALUE_GRID = (400.0, 800.0, 1600.0)
RATES = (0.10, 0.20, 0.30, 0.40, 0.50)
PROBE_MS = 20.3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--latency", default="results/component_latency.json")
    parser.add_argument("--out", default="results/within_domain.json")
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
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    order = {e: i for i, e in enumerate(ids)}

    cheap_ok = np.array([grouped[e]["lowres_384"].correct for e in ids])
    full_ok = np.array([grouped[e]["full"].correct for e in ids])
    entropy = np.array(
        [float(grouped[e]["lowres_384"].signals["mean_entropy"]) for e in ids]
    )
    datasets = np.array([grouped[e]["lowres_384"].dataset for e in ids])
    gains = signed_gain(cheap_ok, full_ok)

    cheap_ms = cost_model.predict(
        np.array([grouped[e]["lowres_384"].visual_tokens for e in ids])
    )
    full_ms = cost_model.predict(np.array([grouped[e]["full"].visual_tokens for e in ids]))
    # Reading entropy needs the whole cheap pass, so escalation adds the full
    # pass; the per-query part is that the full pass costs what the image allows.
    delta_ms = full_ms
    print(
        f"per-query escalation price: mean {delta_ms.mean():.1f} ms, "
        f"range [{delta_ms.min():.1f}, {delta_ms.max():.1f}] "
        f"({delta_ms.max() / delta_ms.min():.1f}x spread)\n"
    )

    def cost(fires: np.ndarray, index: np.ndarray) -> np.ndarray:
        return cheap_ms[index] + fires * full_ms[index]

    results: dict[str, object] = {"datasets": {}}
    for dataset in sorted(set(datasets)):
        members = [ids[i] for i in np.where(datasets == dataset)[0]]
        collected: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"accuracy": [], "latency": [], "rate": []}
        )
        for seed in range(RESAMPLES):
            local = make_split(
                members, [dataset] * len(members),
                val_fraction=0.0, test_fraction=0.3, seed=9000 + seed,
            )
            train = np.array([order[e] for e in local.train])
            test = np.array([order[e] for e in local.test])
            if len(set((gains[train] > 0).tolist())) < 2:
                continue

            def record(name: str, fires: np.ndarray) -> None:
                collected[name]["accuracy"].append(
                    float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
                )
                collected[name]["latency"].append(float(cost(fires, test).mean()))
                collected[name]["rate"].append(float(fires.mean()))

            record("always cheap", np.zeros(len(test), dtype=bool))
            record("always full", np.ones(len(test), dtype=bool))
            for rate in RATES:
                cut = np.quantile(entropy[train], 1.0 - rate)
                record(f"tuned rate {rate:.0%}", entropy[test] >= cut)

            mean_delta = float(delta_ms[train].mean())
            for value in VALUE_GRID:
                record(
                    f"gain rule V={value:.0f}",
                    fit_gain_rule(
                        entropy[train], gains[train],
                        delta_ms=mean_delta, value_ms_per_correct=value,
                    ).escalate(entropy[test]),
                )
                record(
                    f"per-query V={value:.0f}",
                    fit_per_query_gain_rule(
                        entropy[train], gains[train], value_ms_per_correct=value
                    ).escalate(entropy[test], delta_ms[test]),
                )
                record(
                    f"UCCI V={value:.0f}",
                    fit_correctness_rule(
                        entropy[train], cheap_ok[train],
                        full_accuracy=float(full_ok[train].mean()),
                        delta_ms=mean_delta, value_ms_per_correct=value,
                    ).escalate(entropy[test]),
                )

        rows = {
            name: {
                "accuracy": float(np.mean(v["accuracy"])),
                "latency": float(np.mean(v["latency"])),
                "rate": float(np.mean(v["rate"])),
            }
            for name, v in collected.items()
        }
        results["datasets"][dataset] = rows

        # A policy is on the front if nothing else is at least as accurate
        # and at least as cheap, with one strict.
        def dominated(name: str) -> bool:
            a, c = rows[name]["accuracy"], rows[name]["latency"]
            return any(
                other["accuracy"] >= a - 1e-9
                and other["latency"] <= c + 1e-9
                and (other["accuracy"] > a or other["latency"] < c)
                for key, other in rows.items()
                if key != name
            )

        print(f"== {dataset} (n={len(members)}, {RESAMPLES} resamples) ==")
        print(f"{'policy':<22}{'escalates':>11}{'accuracy':>10}{'latency':>10}{'front':>8}")
        for name in sorted(rows, key=lambda k: rows[k]["latency"]):
            mark = "" if dominated(name) else "  *"
            print(
                f"{name:<22}{rows[name]['rate']:>11.0%}{rows[name]['accuracy']:>10.3f}"
                f"{rows[name]['latency']:>10.1f}{mark:>8}"
            )
        print()

    # --- does the per-query break-even beat the global one? ----------------
    print("per-query vs global break-even, paired across datasets and V:")
    deltas_acc, deltas_lat = [], []
    for dataset, rows in results["datasets"].items():
        for value in VALUE_GRID:
            g = rows[f"gain rule V={value:.0f}"]
            p = rows[f"per-query V={value:.0f}"]
            deltas_acc.append(p["accuracy"] - g["accuracy"])
            deltas_lat.append(p["latency"] - g["latency"])
            print(
                f"  {dataset:<9} V={value:>6.0f}  accuracy {p['accuracy'] - g['accuracy']:+.3f}"
                f"   latency {p['latency'] - g['latency']:+7.1f} ms"
            )
    acc_ci = bootstrap_interval(deltas_acc)
    lat_ci = bootstrap_interval(deltas_lat)
    print(f"\n  mean accuracy delta {acc_ci}")
    print(f"  mean latency  delta {lat_ci} ms")
    results["per_query_vs_global"] = {
        "accuracy": [acc_ci.estimate, acc_ci.low, acc_ci.high],
        "latency": [lat_ci.estimate, lat_ci.low, lat_ci.high],
    }

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
