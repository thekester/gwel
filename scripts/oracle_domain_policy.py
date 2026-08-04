"""If the signal is a domain detector, what does a perfect one buy?

The confound audit concludes that the probe's pooled advantage rides on dataset
identity. That invites a control the paper owed and did not run: give a policy
the dataset label outright, let it pick the best escalation rate per dataset
with hindsight, and see where it lands against the signals that have to infer
it. This is an upper bound on everything domain identity can be worth, not a
deployable policy, and it is the right ceiling to place the probe and the free
image descriptor against.

Two policies are reported. The oracle-domain policy knows the label and tunes
one rate per dataset on the test fold itself, so it cannot be beaten by any
signal whose information is the label. The per-domain rule instead applies one
unchanged operator preference inside each dataset, which is deployable wherever
the label is known, and separates "knowing the domain" from "tuning on it".

Usage: PYTHONPATH=src:scripts python scripts/oracle_domain_policy.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.decision import fit_gain_rule, signed_gain
from gwel.router.evaluate import bootstrap_interval
from gwel.router.probes import fit_layer_probe

from baseline_convexity import hull_accuracy
from evaluate_decision_rule import component_costs, load_arrays

RESAMPLES = 30
HELDOUT = 300
VALUES = (400.0, 800.0, 1600.0, 3200.0)
RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LAYER = 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--activations", default="results/activations_full.npz")
    parser.add_argument("--out", default="results/domain_policy.json")
    args = parser.parse_args()

    config = load_config(args.config)
    cheap_ok, full_ok, entropy, matrix, _ = load_arrays(config, args.activations, LAYER)
    gains = signed_gain(cheap_ok, full_ok)
    costs = component_costs()

    stored = np.load(args.activations, allow_pickle=True)
    ids = [str(e) for e in stored["example_ids"]]
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    usable = [
        e for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    dataset = np.array([grouped[e]["lowres_384"].dataset for e in usable])
    n = len(usable)

    # A label is metadata, so a policy reading it skips the cheap pass on the
    # queries it escalates, exactly like the free image descriptor.
    def free_cost(fires, index):
        return np.where(fires, costs["full"], costs["cheap"])

    rows: dict[str, list] = defaultdict(list)
    chosen_rates: dict[str, list] = defaultdict(list)
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(11000 + seed)
        order = rng.permutation(n)
        test, train = order[:HELDOUT], order[HELDOUT:]
        fixed = [
            (costs["cheap"], float(cheap_ok[test].mean())),
            (costs["full"], float(full_ok[test].mean())),
        ]
        probe = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float), LAYER)
        score = probe.score(matrix)

        for value in VALUES:
            # What dataset identity alone is worth: inside each dataset,
            # escalate a uniformly random fraction, with that fraction tuned per
            # dataset on the test fold. Selection within a dataset must stay
            # random, or this measures a per-query oracle rather than the label.
            fires = np.zeros(len(test), bool)
            for name in np.unique(dataset[test]):
                mask = dataset[test] == name
                pick_order = rng.permutation(int(mask.sum()))
                best_rate, best_value = 0.0, -np.inf
                for rate in RATES:
                    pick = np.zeros(mask.sum(), bool)
                    pick[pick_order[: int(round(rate * mask.sum()))]] = True
                    acc = np.where(pick, full_ok[test][mask], cheap_ok[test][mask])
                    objective = value * acc.mean() - free_cost(pick, None).mean()
                    if objective > best_value:
                        best_rate, best_value = rate, objective
                sub = np.zeros(mask.sum(), bool)
                sub[pick_order[: int(round(best_rate * mask.sum()))]] = True
                fires[mask] = sub
                chosen_rates[f"{name} V={value:.0f}"].append(best_rate)
            record_row(rows, "domain label, tuned rate", value, fires, test,
                       cheap_ok, full_ok, free_cost, fixed)

            # Per-domain rule: one preference, fitted inside each dataset on
            # the training fold, applied to the test fold. Deployable.
            fires = np.zeros(len(test), bool)
            for name in np.unique(dataset[test]):
                tr = train[dataset[train] == name]
                te = dataset[test] == name
                if len(tr) < 20:
                    continue
                rule = fit_gain_rule(
                    entropy[tr], gains[tr],
                    delta_ms=costs["full"], value_ms_per_correct=value,
                )
                fires[te] = rule.escalate(entropy[test][te])
            record_row(rows, "per-domain entropy rule", value, fires, test,
                       cheap_ok, full_ok,
                       lambda f, i: costs["cheap"] + f * costs["full"], fixed)

            # The probe, for reference, on the same folds and prices as before.
            fires = fit_gain_rule(
                score[train], gains[train],
                delta_ms=costs["probe"] + costs["full"] - costs["cheap"],
                value_ms_per_correct=value,
            ).escalate(score[test])
            record_row(rows, "probe", value, fires, test, cheap_ok, full_ok,
                       lambda f, i: np.where(f, costs["probe"] + costs["full"],
                                             costs["cheap"]), fixed)

    print(f"{'policy':<26}{'accuracy':>10}{'latency':>10}{'gap to hull [95% CI]':>26}")
    out = {}
    for name, triples in rows.items():
        gap = bootstrap_interval([t[2] for t in triples])
        out[name] = {
            "accuracy": float(np.mean([t[0] for t in triples])),
            "latency": float(np.mean([t[1] for t in triples])),
            "gap": [gap.estimate, gap.low, gap.high],
        }
        print(
            f"{name:<26}{out[name]['accuracy']:>10.3f}"
            f"{out[name]['latency']:>10.1f}{str(gap):>26}"
        )

    print("\nrates the oracle picks per dataset (mean over resamples):")
    rates = {k: float(np.mean(v)) for k, v in sorted(chosen_rates.items())}
    for key, value in rates.items():
        print(f"  {key:<28}{value:>6.0%}")

    Path(args.out).write_text(
        json.dumps({"policies": out, "oracle_rates": rates}, indent=2)
    )
    print(
        "\nThe oracle domain policy tunes on the test fold and cannot be\n"
        "deployed. It bounds what dataset identity is worth, which is the\n"
        "quantity the probe was found to be reading."
    )
    print(f"wrote {args.out}")


def record_row(rows, name, value, fires, test, cheap_ok, full_ok, cost_fn, fixed):
    acc = float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
    lat = float(np.asarray(cost_fn(fires, test)).mean())
    hull = hull_accuracy(fixed, lat)
    rows[f"{name} V={value:.0f}"].append(
        (acc, lat, acc - (hull if hull is not None else acc))
    )


if __name__ == "__main__":
    main()
