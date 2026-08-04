"""The cheapest baselines an escalation policy has to beat, measured.

Three comparators the paper argued past rather than ran.

Raw image size, read from the file header. The confound audit found it scoring
0.748 against the probe's 0.761, and a signal that needs no forward pass does
not merely rank almost as well: it prices differently. A policy reading it
decides before anything runs, so an escalated query never pays for a cheap pass
it will discard. That is the same cost structure as randomising between fixed
configurations, which makes it the sharpest possible test of whether any signal
is worth reading at all.

Random escalation with a mid-prefill abort. The paper attributes the probe's
margin over the hull to its abort rather than to its ranking. If that is the
whole story, a policy that aborts on a coin flip should clear the hull too, and
the probe's margin measures nothing about the direction. This separates the
cost mechanism from the signal.

Both costings. The hull table in the paper prices configurations flat, while
the cost audit establishes that per-example pricing is the honest one. Every
comparison here is reported under both, because a result that only survives the
costing the paper itself rejects is not a result.

Usage: PYTHONPATH=src:scripts python scripts/baseline_free_signal.py
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
from gwel.router.decision import fit_gain_rule, signed_gain
from gwel.router.evaluate import auroc, bootstrap_interval
from gwel.router.probes import fit_layer_probe

from baseline_convexity import hull_accuracy
from evaluate_decision_rule import component_costs, load_arrays

RESAMPLES = 30
HELDOUT = 300
VALUES = (400.0, 800.0, 1600.0, 3200.0)
RATES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
LAYER = 6


def image_sizes(config, ids: list[str]) -> np.ndarray:
    """Longest edge of the source image, in pixels, per example.

    This is metadata: it is available from the file header before the model is
    loaded, so a policy reading it costs nothing and can skip the cheap pass
    entirely on the queries it escalates.
    """
    meta: dict[str, float] = {}
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        if record.config_id != "lowres_384":
            continue
        info = record.meta or {}
        width, height = info.get("orig_width"), info.get("orig_height")
        if width and height:
            meta[record.example_id] = float(max(width, height))
    missing = [e for e in ids if e not in meta]
    if missing:
        raise ValueError(f"{len(missing)} examples carry no source dimensions")
    return np.array([meta[e] for e in ids])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--activations", default="results/activations_full.npz")
    parser.add_argument("--latency", default="results/component_latency.json")
    parser.add_argument("--out", default="results/free_signal.json")
    args = parser.parse_args()

    config = load_config(args.config)
    cheap_ok, full_ok, entropy, matrix, _ = load_arrays(config, args.activations, LAYER)
    gains = signed_gain(cheap_ok, full_ok)
    flat = component_costs(args.latency)

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
    size = image_sizes(config, usable)
    n = len(usable)
    if n != len(cheap_ok):
        raise ValueError(f"{n} sized examples against {len(cheap_ok)} scored")

    profile = json.loads(Path(args.latency).read_text())
    model = fit_token_cost(
        [r["visual_tokens"] for r in profile], [r["total_ms"] for r in profile]
    )
    per_full = model.predict(
        np.array([grouped[e]["full"].visual_tokens for e in usable], float)
    )
    per_cheap = model.predict(
        np.array([grouped[e]["lowres_384"].visual_tokens for e in usable], float)
    )

    print(f"n = {n}; image size ranges {size.min():.0f} to {size.max():.0f} px")
    print(
        f"escalation price: flat {flat['full']:.1f} ms against per-example mean "
        f"{per_full.mean():.1f} ms ({per_full.mean() / flat['full'] - 1:+.0%})"
    )
    print(
        f"image size predicts the escalation gain at AUROC "
        f"{auroc(size.tolist(), [bool(g > 0) for g in gains]):.3f} "
        f"(pooled, whole pilot)"
    )

    def priced(escalates, index, *, read, per_example):
        cheap = per_cheap[index] if per_example else np.full(len(index), flat["cheap"])
        full = per_full[index] if per_example else np.full(len(index), flat["full"])
        if read == "entropy":  # must finish the cheap pass before it can decide
            return cheap + escalates * full
        if read == "probe":  # abandons the cheap pass mid-prefill
            return np.where(escalates, flat["probe"] + full, cheap)
        if read == "free":  # decides before anything runs
            return np.where(escalates, full, cheap)
        raise ValueError(read)

    results: dict = {}
    for per_example in (False, True):
        label = "per-example" if per_example else "flat"
        rows_v: dict[str, list] = defaultdict(list)
        rows_r: dict[str, list] = defaultdict(list)
        for seed in range(RESAMPLES):
            rng = np.random.default_rng(9000 + seed)
            order = rng.permutation(n)
            test, train = order[:HELDOUT], order[HELDOUT:]
            fixed = [
                (float(priced(np.zeros(len(test), bool), test, read="free",
                              per_example=per_example).mean()),
                 float(cheap_ok[test].mean())),
                (float(priced(np.ones(len(test), bool), test, read="free",
                              per_example=per_example).mean()),
                 float(full_ok[test].mean())),
            ]
            probe = fit_layer_probe(
                matrix[train], (gains[train] > 0).astype(float), LAYER
            )
            score = probe.score(matrix)
            noise = rng.standard_normal(n)

            signals = (
                ("probe", score, "probe"),
                ("entropy", entropy, "entropy"),
                ("image size", size, "free"),
                ("random, abort", noise, "probe"),
            )
            for name, sig, read in signals:
                delta = (
                    flat["probe"] + flat["full"] - flat["cheap"] if read == "probe"
                    else flat["full"] if read == "entropy"
                    else flat["full"] - flat["cheap"]
                )
                for value in VALUES:
                    fires = fit_gain_rule(
                        sig[train], gains[train],
                        delta_ms=delta, value_ms_per_correct=value,
                    ).escalate(sig[test])
                    acc = float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
                    lat = float(priced(fires, test, read=read,
                                       per_example=per_example).mean())
                    hull = hull_accuracy(fixed, lat)
                    rows_v[f"{name} V={value:.0f}"].append(
                        (acc, lat, acc - (hull if hull is not None else acc))
                    )
                for rate in RATES:
                    cut = np.quantile(sig[train], 1.0 - rate)
                    fires = sig[test] >= cut
                    acc = float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
                    lat = float(priced(fires, test, read=read,
                                       per_example=per_example).mean())
                    hull = hull_accuracy(fixed, lat)
                    rows_r[f"{name} @{rate:.0%}"].append(
                        (acc, lat, acc - (hull if hull is not None else acc))
                    )

        results[label] = {
            "preference swept": summarise(rows_v, f"{label} costs, preference swept"),
            "rate swept": summarise(rows_r, f"{label} costs, rate swept"),
            "free minus probe": paired(rows_v, "image size", "probe", VALUES),
        }

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(
        "\nA policy reading image size decides before the cheap pass runs, so it\n"
        "prices exactly like randomising between fixed configurations. Any gap it\n"
        "opens over the hull is bought by the signal alone. The random policy\n"
        "aborts mid-prefill on a coin flip, so its gap measures what the abort\n"
        "mechanism is worth with no signal at all."
    )
    print(f"\nwrote {args.out}")


def paired(rows: dict, left: str, right: str, values) -> dict:
    """Difference in hull gap between two policies, paired over resamples.

    The per-policy intervals above are not paired against each other, so they
    cannot say whether the free signal matches the probe. This does, resample
    by resample, at each operator preference.
    """
    print(f"\n=== {left} minus {right}, paired over resamples ===")
    out = {}
    for value in values:
        a = rows[f"{left} V={value:.0f}"]
        b = rows[f"{right} V={value:.0f}"]
        delta = [x[2] - y[2] for x, y in zip(a, b, strict=True)]
        interval = bootstrap_interval(delta)
        out[f"V={value:.0f}"] = [interval.estimate, interval.low, interval.high]
        verdict = (
            "free signal wins" if interval.low > 0
            else "probe wins" if interval.high < 0
            else "indistinguishable"
        )
        print(f"  V={value:<6.0f} gap difference {str(interval):>24}  {verdict}")
    return out


def summarise(rows: dict, label: str) -> dict:
    print(f"\n=== {label} ===")
    print(f"{'policy':<22}{'accuracy':>10}{'latency':>10}{'gap to hull [95% CI]':>26}")
    out = {}
    for name, triples in rows.items():
        gap = bootstrap_interval([t[2] for t in triples])
        out[name] = {
            "accuracy": float(np.mean([t[0] for t in triples])),
            "latency": float(np.mean([t[1] for t in triples])),
            "gap": [gap.estimate, gap.low, gap.high],
        }
        print(
            f"{name:<22}{out[name]['accuracy']:>10.3f}"
            f"{out[name]['latency']:>10.1f}{str(gap):>26}"
        )
    return out


if __name__ == "__main__":
    main()
