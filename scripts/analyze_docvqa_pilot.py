"""The single-domain pilot: settle the confound, and price a verified ladder.

Two questions the four-dataset mixture cannot answer, because in it every
within-domain fit gets 210 examples where the pooled fit gets 600, and because
for 56% of it the top rung is not a distinct configuration.

**Q1. Is the probe's within-domain collapse real, or starvation?** The
decisive form is a learning curve. Fit the probe inside DocVQA at growing
training sizes and watch what AUROC does. If the signal exists and the earlier
result was starvation, AUROC climbs toward the pooled figure as n grows. If the
direction carries no within-domain information, it stays flat at chance however
much data it is given --- and output entropy, which is fitted on nothing, marks
the level a real signal reaches here.

**Q2. Does the ladder pay when every rung is genuinely dearer?** This pilot's
rungs were chosen by measuring SmolVLM's patch-grid buckets, so each costs
strictly more than the one below on every image. The mixture could not test
this.

Usage: PYTHONPATH=scripts python scripts/analyze_docvqa_pilot.py
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
from gwel.router.evaluate import auroc, bootstrap_interval
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split

CHEAP = "lowres_384"
RUNGS = ("lowres_768", "lowres_1152", "full")
TRAIN_SIZES = (100, 200, 400, 600, 900)
HELDOUT = 300  # fixed, so only the training size varies across the curve
LAYERS = (1, 3, 6, 12, 20, 32)
RESAMPLES = 30
VALUE_GRID = (400.0, 800.0, 1600.0, 3200.0)
MIN_BUCKET = 50  # passes a token bucket needs before it may set the cost line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/docvqa1200.yaml")
    parser.add_argument("--activations", default="results/activations_docvqa1200.npz")
    parser.add_argument("--latency", default="results/component_latency.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--out", default="results/docvqa_pilot.json")
    args = parser.parse_args()

    config = load_config(args.config)
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record

    stored = np.load(args.activations, allow_pickle=True)
    activations = stored["activations"]
    ids = [str(e) for e in stored["example_ids"]]
    usable = [
        e
        for e in ids
        if all(c in grouped[e] for c in (CHEAP, *RUNGS)) and grouped[e][CHEAP].signals
    ]
    position = {e: i for i, e in enumerate(ids)}
    matrix = activations[[position[e] for e in usable]]
    order = {e: i for i, e in enumerate(usable)}

    correct = {
        c: np.array([grouped[e][c].correct for e in usable]) for c in (CHEAP, *RUNGS)
    }
    tokens = {
        c: np.array([grouped[e][c].visual_tokens for e in usable], dtype=float)
        for c in (CHEAP, *RUNGS)
    }
    entropy = np.array([float(grouped[e][CHEAP].signals["mean_entropy"]) for e in usable])
    gains = {r: signed_gain(correct[CHEAP], correct[r]) for r in RUNGS}
    labels = gains["full"] > 0

    results: dict[str, object] = {"n": len(usable)}

    # The component profile tops out at 320 visual tokens because it was taken
    # on a small image; this ladder reaches 1088. Extrapolating 3.4x beyond a
    # fit is how this project acquired its previous cost errors, so the model is
    # fitted on *this run's own* measured (tokens, latency) pairs instead, which
    # cover the whole range by construction. The extrapolated version is
    # reported alongside so the difference is visible rather than assumed.
    # Fitted on the median latency within each distinct token count, not on raw
    # passes: the first passes of any run carry warm-up that is not a property
    # of the configuration, and a least-squares fit on raw values would let a
    # handful of them set the slope.
    by_tokens: dict[int, list[float]] = defaultdict(list)
    for example in usable:
        for record in grouped[example].values():
            by_tokens[int(record.visual_tokens)].append(record.latency_ms)

    # Sparse buckets are noise, not measurements: a bucket seen twice can sit
    # 300 ms off the line and drag the slope with it. The model's own residual
    # is what exposes this, so it is reported either way.
    observed_tokens = sorted(t for t in by_tokens if len(by_tokens[t]) >= MIN_BUCKET)
    dropped = sorted(t for t in by_tokens if len(by_tokens[t]) < MIN_BUCKET)
    observed_ms = [float(np.median(by_tokens[t])) for t in observed_tokens]
    cost_model = fit_token_cost(observed_tokens, observed_ms)
    print(
        "token buckets fitted: "
        + ", ".join(f"{t} (n={len(by_tokens[t])})" for t in observed_tokens)
    )
    if dropped:
        print(
            "  dropped as under-sampled: "
            + ", ".join(f"{t} (n={len(by_tokens[t])})" for t in dropped)
        )
    print(f"  worst residual {cost_model.residual_ms:.1f} ms")

    profile = json.loads(Path(args.latency).read_text())
    extrapolated = fit_token_cost(
        [r["visual_tokens"] for r in profile], [r["total_ms"] for r in profile]
    )
    print(
        f"cost model from this run: {cost_model.base_ms:.1f} + "
        f"{cost_model.slope_ms_per_token:.3f} ms/token "
        f"over {max(observed_tokens)} tokens of range"
    )
    print(
        f"  the component profile would have said {extrapolated.base_ms:.1f} + "
        f"{extrapolated.slope_ms_per_token:.3f}, fitted only to "
        f"{max(r['visual_tokens'] for r in profile)} tokens"
    )
    top = max(observed_tokens)
    print(
        f"  at {top} tokens they differ by "
        f"{float(cost_model.predict(top)[0] - extrapolated.predict(top)[0]):+.0f} ms\n"
    )
    results["cost_model"] = {
        "base_ms": cost_model.base_ms,
        "slope_ms_per_token": cost_model.slope_ms_per_token,
        "residual_ms": cost_model.residual_ms,
        "max_tokens": int(top),
        "extrapolated_base_ms": extrapolated.base_ms,
        "extrapolated_slope": extrapolated.slope_ms_per_token,
        "disagreement_at_top_ms": float(
            cost_model.predict(top)[0] - extrapolated.predict(top)[0]
        ),
    }
    latency = {c: cost_model.predict(tokens[c]) for c in (CHEAP, *RUNGS)}

    print(f"{len(usable)} DocVQA examples with a complete ladder\n")

    print(f"{'rung':<14}{'mean tokens':>13}{'predicted ms':>14}{'accuracy':>10}"
          f"{'dearer than below':>19}")
    ladder_rows = []
    previous = CHEAP
    for c in (CHEAP, *RUNGS):
        dearer = "" if c == CHEAP else f"{(tokens[c] > tokens[previous]).mean():.0%}"
        ladder_rows.append(
            {
                "config": c,
                "mean_tokens": float(tokens[c].mean()),
                "mean_ms": float(latency[c].mean()),
                "accuracy": float(correct[c].mean()),
                "strictly_dearer": None if c == CHEAP else float(
                    (tokens[c] > tokens[previous]).mean()
                ),
            }
        )
        print(
            f"{c:<14}{tokens[c].mean():>13.0f}{latency[c].mean():>14.1f}"
            f"{correct[c].mean():>10.3f}{dearer:>19}"
        )
        if c != CHEAP:
            previous = c
    results["ladder"] = ladder_rows

    # --- which rung does each query need? ----------------------------------
    rung_order = (CHEAP, *RUNGS)
    # --- where does escalation value saturate? -----------------------------
    # The rung-to-rung gain, with an interval, so "stops helping" is a measured
    # statement and not an eyeballed plateau.
    print(f"\n{'step':<26}{'net gain [95% CI]':>28}{'cost':>8}{'points/s':>11}")
    steps = []
    for low, high in zip(rung_order[:-1], rung_order[1:]):
        gain = (correct[high] & ~correct[low]).astype(float) - (
            correct[low] & ~correct[high]
        ).astype(float)
        interval = bootstrap_interval(gain.tolist())
        extra = float((latency[high] - latency[low]).mean())
        steps.append(
            {
                "from": low,
                "to": high,
                "net_gain": interval.estimate,
                "low": interval.low,
                "high": interval.high,
                "extra_ms": extra,
                "points_per_second": interval.estimate / extra * 1000.0,
            }
        )
        label = f"{low.replace('lowres_', '')} -> {high.replace('lowres_', '')}"
        print(
            f"{label:<26}{str(interval):>28}{extra:>8.0f}"
            f"{interval.estimate / extra * 1000:>+11.2f}"
        )
    results["steps"] = steps

    saturated = [s for s in steps if s["low"] <= 0.0 <= s["high"]]
    if saturated:
        first = saturated[0]
        print(
            f"\nescalation value saturates at {first['from']}: the step above it is "
            f"indistinguishable from zero while costing {first['extra_ms']:.0f} ms"
        )
        results["saturation_rung"] = first["from"]

    cheapest = np.full(len(usable), -1)
    for index in range(len(usable)):
        for level, config_id in enumerate(rung_order):
            if correct[config_id][index]:
                cheapest[index] = level
                break
    print(f"\n{'cheapest rung that answers':<30}{'share':>8}")
    need = {}
    for level, config_id in enumerate(rung_order):
        need[config_id] = float((cheapest == level).mean())
        print(f"{config_id:<30}{need[config_id]:>8.1%}")
    print(f"{'nothing we run':<30}{(cheapest == -1).mean():>8.1%}")
    escalated = cheapest > 0
    if escalated.any():
        below_top = float((cheapest[escalated] < len(rung_order) - 1).mean())
        print(f"\nof what escalation repairs, {below_top:.0%} is repaired below the top rung")
        results["repaired_below_top"] = below_top
    results["needs"] = need
    results["unsolvable"] = float((cheapest == -1).mean())

    # --- Q1: the learning curve --------------------------------------------
    print(f"\n=== Q1: does the probe find a within-domain signal, given data? ===")
    print(f"{'train n':>9}{'probe AUROC':>22}{'entropy AUROC':>22}")
    curve = []
    for size in TRAIN_SIZES:
        probe_scores, entropy_scores = [], []
        for seed in range(RESAMPLES):
            rng = np.random.default_rng(4000 + seed)
            shuffled = rng.permutation(len(usable))
            # The held-out set is the same size and drawn the same way at every
            # training size, so the curve measures training data alone.
            test, pool = shuffled[:HELDOUT], shuffled[HELDOUT:]
            if len(pool) < size:
                continue
            train = pool[:size]
            if len(test) < 50 or len(set(labels[train].tolist())) < 2:
                continue
            truth = [bool(x) for x in labels[test]]
            if len(set(truth)) < 2:
                continue
            fitted = fit_layer_probe(
                matrix[train, args.layer, :], labels[train].astype(float), args.layer
            )
            probe_scores.append(
                auroc(fitted.score(matrix[test, args.layer, :]).tolist(), truth)
            )
            entropy_scores.append(auroc(entropy[test].tolist(), truth))
        if not probe_scores:
            continue
        probe_ci = bootstrap_interval(probe_scores)
        entropy_ci = bootstrap_interval(entropy_scores)
        curve.append(
            {
                "train_n": size,
                "probe": probe_ci.estimate,
                "probe_low": probe_ci.low,
                "probe_high": probe_ci.high,
                "entropy": entropy_ci.estimate,
            }
        )
        print(f"{size:>9}{str(probe_ci):>22}{str(entropy_ci):>22}")
    results["learning_curve"] = curve
    if len(curve) >= 2:
        slope = curve[-1]["probe"] - curve[0]["probe"]
        print(
            f"\nprobe AUROC moves {slope:+.3f} as training data grows "
            f"{TRAIN_SIZES[-1] / TRAIN_SIZES[0]:.1f}x; "
            f"entropy sits at {curve[-1]['entropy']:.3f} having been fitted on nothing"
        )
        results["probe_slope"] = float(slope)

    # --- depth, at the largest training size -------------------------------
    print(f"\n{'layer':>7}{'probe AUROC at n=' + str(TRAIN_SIZES[-1]):>28}")
    depth = []
    for layer in LAYERS:
        values = []
        for seed in range(RESAMPLES):
            rng = np.random.default_rng(4000 + seed)
            shuffled = rng.permutation(len(usable))
            test, pool = shuffled[:HELDOUT], shuffled[HELDOUT:]
            if len(pool) < TRAIN_SIZES[-1]:
                continue
            train = pool[: TRAIN_SIZES[-1]]
            truth = [bool(x) for x in labels[test]]
            if len(test) < 50 or len(set(truth)) < 2:
                continue
            fitted = fit_layer_probe(
                matrix[train, layer, :], labels[train].astype(float), layer
            )
            values.append(auroc(fitted.score(matrix[test, layer, :]).tolist(), truth))
        depth.append({"layer": layer, "auroc": float(np.mean(values))})
        print(f"{layer:>7}{np.mean(values):>28.3f}")
    results["depth"] = depth

    # --- Q2: the verified ladder against a binary policy -------------------
    print(f"\n=== Q2: does a verified four-rung ladder pay? ===")
    deltas = np.column_stack([latency[r] - latency[CHEAP] for r in RUNGS])
    print(f"{'policy':<18}{'rung mix':>26}{'accuracy':>10}{'latency':>10}")
    comparison = []
    for value in VALUE_GRID:
        binary_acc, binary_ms, ladder_acc, ladder_ms, mixes = [], [], [], [], []
        for seed in range(RESAMPLES):
            rng = np.random.default_rng(5000 + seed)
            shuffled = rng.permutation(len(usable))
            test, pool = shuffled[:HELDOUT], shuffled[HELDOUT:]
            train = pool
            if len(test) < 50:
                continue
            if any(len(set((gains[r][train] > 0).tolist())) < 2 for r in RUNGS):
                continue

            fires = fit_gain_rule(
                entropy[train], gains["full"][train],
                delta_ms=float(deltas[train, -1].mean()), value_ms_per_correct=value,
            ).escalate(entropy[test])
            binary_acc.append(
                float(np.where(fires, correct["full"][test], correct[CHEAP][test]).mean())
            )
            binary_ms.append(
                float(np.where(fires, latency["full"][test], latency[CHEAP][test]).mean())
            )

            chosen = fit_ladder_rule(
                entropy[train], {r: gains[r][train] for r in RUNGS},
                value_ms_per_correct=value,
            ).choose(entropy[test], deltas[test])
            picked_correct = np.where(chosen < 0, correct[CHEAP][test], 0.0)
            picked_cost = np.where(chosen < 0, latency[CHEAP][test], 0.0)
            for level, rung in enumerate(RUNGS):
                mask = chosen == level
                picked_correct = np.where(mask, correct[rung][test], picked_correct)
                picked_cost = np.where(mask, latency[rung][test], picked_cost)
            ladder_acc.append(float(picked_correct.mean()))
            ladder_ms.append(float(picked_cost.mean()))
            mixes.append([float((chosen == level).mean()) for level in range(len(RUNGS))])

        if not binary_acc:
            continue
        mix = np.mean(mixes, axis=0)
        # Pair within an operating point, over the resamples that produced it.
        # Bootstrapping the four preference means instead would resample four
        # numbers and report an interval four points wide, which is what an
        # earlier version of this script did.
        accuracy_delta = bootstrap_interval(
            [a - b for a, b in zip(ladder_acc, binary_acc)]
        )
        latency_delta = bootstrap_interval([a - b for a, b in zip(ladder_ms, binary_ms)])
        comparison.append(
            {
                "value": value,
                "resamples": len(binary_acc),
                "binary_accuracy": float(np.mean(binary_acc)),
                "binary_ms": float(np.mean(binary_ms)),
                "ladder_accuracy": float(np.mean(ladder_acc)),
                "ladder_ms": float(np.mean(ladder_ms)),
                "ladder_mix": mix.tolist(),
                "accuracy_delta": [
                    accuracy_delta.estimate, accuracy_delta.low, accuracy_delta.high
                ],
                "latency_delta": [
                    latency_delta.estimate, latency_delta.low, latency_delta.high
                ],
            }
        )
        print(
            f"{'binary V=' + format(value, '.0f'):<18}{'-- top rung only':>26}"
            f"{np.mean(binary_acc):>10.3f}{np.mean(binary_ms):>10.1f}"
        )
        print(
            f"{'ladder V=' + format(value, '.0f'):<18}"
            f"{'/'.join(f'{m:.0%}' for m in mix):>26}"
            f"{np.mean(ladder_acc):>10.3f}{np.mean(ladder_ms):>10.1f}"
        )
        print(
            f"{'  paired delta':<18}{'':>26}"
            f"{accuracy_delta.estimate:>+10.3f}{latency_delta.estimate:>+10.1f}"
            f"   accuracy [{accuracy_delta.low:+.3f}, {accuracy_delta.high:+.3f}]"
            f"  latency [{latency_delta.low:+.1f}, {latency_delta.high:+.1f}]"
        )
    results["ladder_vs_binary"] = comparison

    cheaper = [
        row for row in comparison if row["latency_delta"][2] < 0.0
    ]
    print(
        f"\nthe ladder is significantly cheaper at {len(cheaper)}/{len(comparison)} "
        f"operating points"
    )
    results["ladder_cheaper_points"] = len(cheaper)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
