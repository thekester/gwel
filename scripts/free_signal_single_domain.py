"""Does the free descriptor also route inside one workload?

On the four-dataset mixture, routing on raw image size clears the randomisation
hull and matches a probe on activations. That result is only half a thesis. If
the same free descriptor also routed inside a single workload, nothing in this
paper would justify reading the model at all; if it does not, the two regimes
separate cleanly, and the separation is the practical claim: a free descriptor
suffices exactly where traffic is heterogeneous, and a graded action space is
what pays where it is not.

The comparison is run on the same $1200$ DocVQA pages and the same verified
rungs as the ladder, with per-example prices, against a hull built from all
four fixed rungs. Four policies share that hull: the ladder on output entropy,
the ladder on image size, and the binary rule on each.

Usage: PYTHONPATH=src:scripts python scripts/free_signal_single_domain.py
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

from baseline_convexity import hull_accuracy

RESAMPLES = 30
HELDOUT = 300
VALUES = (400.0, 800.0, 1600.0, 3200.0)
MIN_BUCKET = 50
CHEAP = "lowres_384"
RUNGS = ("lowres_768", "lowres_1152", "full")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/docvqa1200.yaml")
    parser.add_argument("--out", default="results/free_signal_docvqa.json")
    args = parser.parse_args()

    config = load_config(args.config)
    grouped: dict[str, dict] = defaultdict(dict)
    for row in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[row.example_id][row.config_id] = row
    ids = [
        e for e in grouped
        if all(c in grouped[e] for c in (CHEAP, *RUNGS)) and grouped[e][CHEAP].signals
    ]
    ok = {c: np.array([grouped[e][c].correct for e in ids]) for c in (CHEAP, *RUNGS)}
    tok = {
        c: np.array([grouped[e][c].visual_tokens for e in ids], float)
        for c in (CHEAP, *RUNGS)
    }
    entropy = np.array(
        [float(grouped[e][CHEAP].signals["mean_entropy"]) for e in ids]
    )
    size = np.array(
        [
            float(max(grouped[e][CHEAP].meta["orig_width"],
                      grouped[e][CHEAP].meta["orig_height"]))
            for e in ids
        ]
    )

    by: dict[int, list[float]] = defaultdict(list)
    for e in ids:
        for row in grouped[e].values():
            by[int(row.visual_tokens)].append(row.latency_ms)
    good = [t for t in sorted(by) if len(by[t]) >= MIN_BUCKET]
    model = fit_token_cost(good, [float(np.median(by[t])) for t in good])
    ms = {c: model.predict(tok[c]) for c in (CHEAP, *RUNGS)}
    gains = {r: signed_gain(ok[CHEAP], ok[r]) for r in RUNGS}
    deltas = np.column_stack([ms[r] - ms[CHEAP] for r in RUNGS])
    top_gain = signed_gain(ok[CHEAP], ok["full"])

    print(f"n = {len(ids)} pages, longest edge {size.min():.0f} to {size.max():.0f} px")
    print(
        f"within this corpus, image size predicts the top-rung gain at AUROC "
        f"{auroc(size.tolist(), [bool(g > 0) for g in top_gain]):.3f}, "
        f"entropy at "
        f"{auroc(entropy.tolist(), [bool(g > 0) for g in top_gain]):.3f}"
    )

    rows: dict[str, list] = defaultdict(list)
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(13000 + seed)
        order = rng.permutation(len(ids))
        test, train = order[:HELDOUT], order[HELDOUT:]
        fixed = [
            (float(ms[c][test].mean()), float(ok[c][test].mean()))
            for c in (CHEAP, *RUNGS)
        ]
        for name, signal in (("entropy", entropy), ("image size", size)):
            for value in VALUES:
                chosen = fit_ladder_rule(
                    signal[train],
                    {r: gains[r][train] for r in RUNGS},
                    value_ms_per_correct=value,
                ).choose(signal[test], deltas[test])
                accuracy = np.where(chosen < 0, ok[CHEAP][test], 0.0)
                latency = np.where(chosen < 0, ms[CHEAP][test], 0.0)
                for level, rung in enumerate(RUNGS):
                    mask = chosen == level
                    accuracy = np.where(mask, ok[rung][test], accuracy)
                    latency = np.where(mask, ms[rung][test], latency)
                record(rows, f"ladder, {name}", value, accuracy, latency, fixed)

                fires = fit_gain_rule(
                    signal[train], top_gain[train],
                    delta_ms=float((ms["full"] - ms[CHEAP]).mean()),
                    value_ms_per_correct=value,
                ).escalate(signal[test])
                record(
                    rows, f"binary, {name}", value,
                    np.where(fires, ok["full"][test], ok[CHEAP][test]),
                    np.where(fires, ms["full"][test], ms[CHEAP][test]),
                    fixed,
                )

    print(f"\n{'policy':<24}{'accuracy':>10}{'latency':>10}{'gap to hull [95% CI]':>26}")
    out = {}
    for name, triples in rows.items():
        gap = bootstrap_interval([t[2] for t in triples])
        out[name] = {
            "accuracy": float(np.mean([t[0] for t in triples])),
            "latency": float(np.mean([t[1] for t in triples])),
            "gap": [gap.estimate, gap.low, gap.high],
            "gap_vector": [float(t[2]) for t in triples],
        }
        print(
            f"{name:<24}{out[name]['accuracy']:>10.3f}"
            f"{out[name]['latency']:>10.1f}{str(gap):>26}"
        )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        "\nOn the mixture the free descriptor matched a probe on activations.\n"
        "Whether it does so here decides whether any signal has to be read from\n"
        "the model at all, or only where traffic is heterogeneous."
    )
    print(f"wrote {args.out}")


def record(rows, name, value, accuracy, latency, fixed) -> None:
    mean_accuracy = float(np.mean(accuracy))
    mean_latency = float(np.mean(latency))
    hull = hull_accuracy(fixed, mean_latency)
    rows[f"{name} V={value:.0f}"].append(
        (mean_accuracy, mean_latency, mean_accuracy - (hull if hull is not None else mean_accuracy))
    )


if __name__ == "__main__":
    main()
