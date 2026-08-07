"""Emit the coordinates for the hull diagram, on one stated fold.

The paper reports a gap as the mean over resampled folds of
(accuracy_s - hull(latency_s)). That is the right quantity, because each fold
compares a policy against the hull its own fixed points trace. It is also not
readable off a scatter plot: the hull is steeply curved between the first two
rungs, so the mean of the gaps is not the gap of the means, and an arrow drawn
between an averaged policy point and an averaged hull would misstate it by a
factor of several.

A figure therefore has to pick one fold and be honest that it did. This script
takes a stated seed, reproduces that fold exactly as scripts/free_signal_single
_domain.py does, and prints the coordinates plus the quantities the caption may
claim. Anyone can rerun it and get the same picture.

Usage: PYTHONPATH=src:scripts python scripts/make_hull_figure.py [--seed 0]
"""

import argparse
from collections import defaultdict

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.oracle.token_cost import fit_token_cost
from gwel.router.decision import fit_ladder_rule, signed_gain

from baseline_convexity import hull_accuracy

CHEAP = "lowres_384"
RUNGS = ("lowres_768", "lowres_1152", "full")
LADDER = (CHEAP, *RUNGS)
HELDOUT = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/docvqa1200.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value", type=float, default=400.0)
    args = parser.parse_args()

    config = load_config(args.config)
    grouped: dict[str, dict] = defaultdict(dict)
    for row in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[row.example_id][row.config_id] = row
    ids = [e for e in grouped if all(c in grouped[e] for c in LADDER)]

    ok = {c: np.array([grouped[e][c].correct for e in ids], float) for c in LADDER}
    tok = {c: np.array([grouped[e][c].visual_tokens for e in ids], float) for c in LADDER}

    # The measurement prices every pass by the affine token-cost model when
    # that model passes its residual test, so the figure must too: a hull drawn
    # in measured milliseconds is not the hull the gap was computed against.
    buckets: dict[int, list[float]] = defaultdict(list)
    for e in ids:
        for row in grouped[e].values():
            buckets[int(row.visual_tokens)].append(row.latency_ms)
    usable = [t for t in sorted(buckets) if len(buckets[t]) >= 20]
    model = fit_token_cost(usable, [float(np.median(buckets[t])) for t in usable])
    ms = {c: model.predict(tok[c]) for c in LADDER}
    entropy = np.array(
        [grouped[e][CHEAP].signals.get("mean_entropy", 0.0) for e in ids], float
    )
    gains = {r: signed_gain(ok[CHEAP], ok[r]) for r in RUNGS}
    deltas = np.stack([ms[r] - ms[CHEAP] for r in RUNGS], axis=1)

    # The same fold construction as the measurement, at a stated seed.
    rng = np.random.default_rng(13000 + args.seed)
    order = rng.permutation(len(ids))
    test, train = order[:HELDOUT], order[HELDOUT:]
    fixed = [(float(ms[c][test].mean()), float(ok[c][test].mean())) for c in LADDER]

    chosen = fit_ladder_rule(
        entropy[train],
        {r: gains[r][train] for r in RUNGS},
        value_ms_per_correct=args.value,
    ).choose(entropy[test], deltas[test])
    accuracy = np.where(chosen < 0, ok[CHEAP][test], 0.0)
    latency = np.where(chosen < 0, ms[CHEAP][test], 0.0)
    for level, rung in enumerate(RUNGS):
        mask = chosen == level
        accuracy = np.where(mask, ok[rung][test], accuracy)
        latency = np.where(mask, ms[rung][test], latency)

    policy = (float(latency.mean()), float(accuracy.mean()))
    reference = hull_accuracy(fixed, policy[0])
    gap = policy[1] - (reference if reference is not None else policy[1])

    print(f"fold seed {args.seed}, held-out {len(test)} of {len(ids)}, V = {args.value:.0f}\n")
    print("fixed rungs (latency ms, accuracy):")
    for name, (lat, acc) in zip(LADDER, fixed):
        print(f"  ({lat:.1f},{acc:.3f})   {name}")
    print(f"\nentropy ladder: ({policy[0]:.1f},{policy[1]:.3f})")
    print(f"hull beneath it: {reference:.3f}")
    print(f"gap on this fold: {gap:+.3f}")
    print(
        "\nThe paper's headline gap is the mean over thirty such folds of each fold's\n"
        "own gap, which is not the gap between the averaged points: the hull is\n"
        "steeply curved here. A figure must therefore show one fold and say so."
    )
    print("\ntikz coordinates:")
    print("  " + "".join(f"({lat:.1f},{acc:.3f})" for lat, acc in fixed))
    print(f"  policy ({policy[0]:.1f},{policy[1]:.3f})   hull ({policy[0]:.1f},{reference:.3f})")


if __name__ == "__main__":
    main()
