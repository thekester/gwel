"""Adaptive routing against the baseline that needs no signal at all.

Any point on the segment between two fixed policies is reachable by
randomisation: serve a fraction p of queries at the dear configuration and the
rest at the cheap one, choosing p to hit the budget. The upper convex hull of
the fixed policies is therefore the floor an adaptive policy must clear, and no
comparison against a single fixed endpoint establishes it. This paper compared
adaptive policies to each other and to the endpoints; this script adds the
missing comparator.

For each pilot, at each operator preference, the adaptive policy is refit on 30
resampled folds and its accuracy is compared with the hull accuracy at the same
expected latency, computed from the fixed policies measured on the same test
fold. The gap is reported paired over resamples.

Two costings, because the two adaptive readings pay differently: the probe
aborts the cheap pass mid-prefill on escalated queries, entropy must finish it.
The hull is always computed in the same costing as the policy it benchmarks.

Usage: PYTHONPATH=scripts python scripts/baseline_convexity.py
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
from gwel.router.probes import fit_layer_probe

from evaluate_decision_rule import component_costs, load_arrays, policy_cost

RESAMPLES = 30
HELDOUT = 300
VALUES = (400.0, 800.0, 1600.0, 3200.0)
MIN_BUCKET = 50
LAYER = 6


def hull_accuracy(points: list[tuple[float, float]], budget: float) -> float | None:
    """Best accuracy reachable at ``budget`` by randomising between fixed policies."""
    best = None
    for i, (c0, a0) in enumerate(points):
        if c0 <= budget + 1e-9 and (best is None or a0 > best):
            best = a0  # a pure policy inside the budget
        for c1, a1 in points[i + 1 :]:
            lo, hi = (c0, c1) if c0 <= c1 else (c1, c0)
            if lo - 1e-9 <= budget <= hi + 1e-9 and hi > lo:
                w = (budget - c0) / (c1 - c0)
                if 0.0 <= w <= 1.0:
                    v = a0 + w * (a1 - a0)
                    if best is None or v > best:
                        best = v
    return best


def mixture(results: dict) -> None:
    """The four-dataset pilot: probe with mid-prefill abort, flat costs."""
    config = load_config("configs/pilot1000.yaml")
    cheap_ok, full_ok, entropy, matrix, _ = load_arrays(
        config, "results/activations_full.npz", LAYER
    )
    gains = signed_gain(cheap_ok, full_ok)
    costs = component_costs()
    n = len(cheap_ok)

    rows = {f"probe V={v:.0f}": [] for v in VALUES}
    rows.update({f"entropy V={v:.0f}": [] for v in VALUES})
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(7000 + seed)
        p = rng.permutation(n)
        test, train = p[:HELDOUT], p[HELDOUT:]
        fixed = [
            (costs["cheap"], float(cheap_ok[test].mean())),
            (costs["full"], float(full_ok[test].mean())),
        ]
        probe = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float), LAYER)
        score = probe.score(matrix)
        for name, sig, read, delta in (
            ("probe", score, "probe", costs["probe"] + costs["full"] - costs["cheap"]),
            ("entropy", entropy, "entropy", costs["full"]),
        ):
            for v in VALUES:
                fires = fit_gain_rule(
                    sig[train], gains[train], delta_ms=delta, value_ms_per_correct=v
                ).escalate(sig[test])
                acc = float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
                lat = float(policy_cost(fires, costs, read=read).mean())
                hull = hull_accuracy(fixed, lat)
                rows[f"{name} V={v:.0f}"].append((acc, lat, acc - (hull or acc)))
    results["mixture"] = summarise(rows, "mixture (probe aborts mid-prefill)")


def mixture_rates(results: dict) -> None:
    """The same chord test under the frontier figure's parameterisation.

    Quantile-rate thresholds instead of operator preferences: the reported
    fold shows two entropy points visually above the chord, and this measures
    whether that impression survives resampling.
    """
    config = load_config("configs/pilot1000.yaml")
    cheap_ok, full_ok, entropy, matrix, _ = load_arrays(
        config, "results/activations_full.npz", LAYER
    )
    gains = signed_gain(cheap_ok, full_ok)
    costs = component_costs()
    n = len(cheap_ok)
    rates = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)

    def chord(budget, a_cheap, a_full):
        c0, c1 = costs["cheap"], costs["full"]
        if budget <= c0:
            return a_cheap
        if budget >= c1:
            return a_full
        return a_cheap + (budget - c0) / (c1 - c0) * (a_full - a_cheap)

    rows = {f"entropy @{r:.0%}": [] for r in rates}
    rows.update({f"probe @{r:.0%}": [] for r in rates})
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(7000 + seed)
        p = rng.permutation(n)
        test, train = p[:HELDOUT], p[HELDOUT:]
        a_cheap, a_full = float(cheap_ok[test].mean()), float(full_ok[test].mean())
        probe = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float), LAYER)
        score = probe.score(matrix)
        for name, sig, read in (
            ("entropy", entropy, "entropy"),
            ("probe", score, "probe"),
        ):
            for r in rates:
                cut = np.quantile(sig[train], 1.0 - r)
                fires = sig[test] >= cut
                acc = float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
                lat = float(policy_cost(fires, costs, read=read).mean())
                rows[f"{name} @{r:.0%}"].append(
                    (acc, lat, acc - chord(lat, a_cheap, a_full))
                )
    results["mixture_rates"] = summarise(rows, "mixture, rate-swept thresholds")


def single_domain(results: dict) -> None:
    """DocVQA: entropy over the full ladder, per-example costs."""
    config = load_config("configs/docvqa1200.yaml")
    grouped: dict[str, dict] = defaultdict(dict)
    for r in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[r.example_id][r.config_id] = r
    cheap, rungs = "lowres_384", ("lowres_768", "lowres_1152", "full")
    ids = [
        e for e in grouped
        if all(c in grouped[e] for c in (cheap, *rungs)) and grouped[e][cheap].signals
    ]
    ok = {c: np.array([grouped[e][c].correct for e in ids]) for c in (cheap, *rungs)}
    tok = {
        c: np.array([grouped[e][c].visual_tokens for e in ids], float)
        for c in (cheap, *rungs)
    }
    ent = np.array([float(grouped[e][cheap].signals["mean_entropy"]) for e in ids])
    by: dict[int, list[float]] = defaultdict(list)
    for e in ids:
        for r in grouped[e].values():
            by[int(r.visual_tokens)].append(r.latency_ms)
    good = [t for t in sorted(by) if len(by[t]) >= MIN_BUCKET]
    model = fit_token_cost(good, [float(np.median(by[t])) for t in good])
    ms = {c: model.predict(tok[c]) for c in (cheap, *rungs)}
    gains = {r: signed_gain(ok[cheap], ok[r]) for r in rungs}
    deltas = np.column_stack([ms[r] - ms[cheap] for r in rungs])

    rows = {f"ladder V={v:.0f}": [] for v in VALUES}
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(7000 + seed)
        p = rng.permutation(len(ids))
        test, train = p[:HELDOUT], p[HELDOUT:]
        fixed = [
            (float(ms[c][test].mean()), float(ok[c][test].mean()))
            for c in (cheap, *rungs)
        ]
        for v in VALUES:
            chosen = fit_ladder_rule(
                ent[train], {r: gains[r][train] for r in rungs}, value_ms_per_correct=v
            ).choose(ent[test], deltas[test])
            acc = np.where(chosen < 0, ok[cheap][test], 0.0)
            lat = np.where(chosen < 0, ms[cheap][test], 0.0)
            for level, r in enumerate(rungs):
                mask = chosen == level
                acc = np.where(mask, ok[r][test], acc)
                lat = np.where(mask, ms[r][test], lat)
            hull = hull_accuracy(fixed, float(lat.mean()))
            rows[f"ladder V={v:.0f}"].append(
                (float(acc.mean()), float(lat.mean()), float(acc.mean()) - (hull or 0.0))
            )
    results["docvqa"] = summarise(rows, "DocVQA (entropy, per-example costs)")


def summarise(rows: dict, label: str) -> dict:
    print(f"\n=== {label} ===")
    print(f"{'policy':<18}{'accuracy':>10}{'latency':>10}{'gap to hull [95% CI]':>26}")
    out = {}
    for name, triples in rows.items():
        acc = float(np.mean([t[0] for t in triples]))
        lat = float(np.mean([t[1] for t in triples]))
        gap = bootstrap_interval([t[2] for t in triples])
        out[name] = {
            "accuracy": acc,
            "latency": lat,
            "gap": [gap.estimate, gap.low, gap.high],
        }
        print(f"{name:<18}{acc:>10.3f}{lat:>10.1f}{str(gap):>26}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/convexity.json")
    args = parser.parse_args()

    results: dict = {}
    mixture(results)
    mixture_rates(results)
    single_domain(results)

    print(
        "\nA positive gap means the adaptive policy delivers accuracy that no"
        "\nrandomisation between fixed configurations reaches at that latency;"
        "\na gap at or below zero means a coin flip between fixed settings does"
        "\nas well, and the signal bought nothing."
    )
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
