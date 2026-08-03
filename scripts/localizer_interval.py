"""Does the localizer's best margin survive resampling, or is it one example?

`test_localizer.py` reports a single split. On a $80$-example test fold a margin
of $+1.8$ points is $1.4$ examples, which is not a number anyone should read as
evidence either way. The published negative is stated at three granularities and
several depths, so the one configuration that looks slightly positive deserves
an interval rather than a shrug.

Resamples the train/test split, refits the localizer each time, and reports the
paired difference against random cell choice. Random is recomputed on the same
test fold rather than assumed, since the cell-correctness rate varies by fold.

Usage: PYTHONPATH=scripts python scripts/localizer_interval.py --layer 32
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.evaluate import bootstrap_interval
from gwel.router.localizer import pool_cells, train_localizer

RESAMPLES = 60
TEST_FRACTION = 0.25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid4.yaml")
    parser.add_argument("--layers", default="3,6,20,32")
    parser.add_argument("--out", default="results/localizer_interval.json")
    parser.add_argument("--include-coarse", action="store_true", default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    rows, cols = config.runner.crop.rows, config.runner.crop.cols
    cell_ids = [f"crop_r{r}c{c}" for r in range(rows) for c in range(cols)]

    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record

    results = {}
    # The published 2x2 claim was made on a single split, so it is re-measured
    # here under the same protocol rather than quoted from a different one.
    coarse = Path("results/visual_grids_multi.npz")
    if coarse.exists() and args.include_coarse:
        stored = np.load(coarse, allow_pickle=True)
        grids2, ids2 = stored["grids"], [str(e) for e in stored["example_ids"]]
        layers2 = list(stored["layers"])
        cells2 = [f"crop_r{r}c{c}" for r in range(2) for c in range(2)]
        coarse_grouped: dict[str, dict] = defaultdict(dict)
        for record in rescore_records(
            deduplicate_records(read_records("results/runs/pilot1000_records.jsonl")),
            ScoringPolicy(),
        ):
            coarse_grouped[record.example_id][record.config_id] = record
        labels2 = np.array([[coarse_grouped[e][c].correct for c in cells2] for e in ids2])
        block = {}
        for k, layer in enumerate(layers2):
            feats = [pool_cells(grids2[i, k], 2, 2) for i in range(len(ids2))]
            margins = []
            for seed in range(RESAMPLES):
                rng = np.random.default_rng(13000 + seed)
                order = rng.permutation(len(ids2))
                cut = int(len(ids2) * TEST_FRACTION)
                test, train = order[:cut], order[cut:]
                loc = train_localizer(
                    [feats[i] for i in train], [list(labels2[i]) for i in train]
                )
                picked = [int(np.argmax(loc.scores(feats[i]))) for i in test]
                margins.append(
                    float(np.mean([labels2[i][p] for i, p in zip(test, picked)]))
                    - float(labels2[test].mean())
                )
            interval = bootstrap_interval(margins)
            block[int(layer)] = {
                "margin": [interval.estimate, interval.low, interval.high]
            }
        results["2x2"] = {
            "n": len(ids2),
            "oracle": float(labels2.any(axis=1).mean()),
            "layers": block,
        }
        beats2 = [l for l, r in block.items() if r["margin"][1] > 0.0]
        print(
            f"2x2 (n={len(ids2)}): {len(beats2)}/{len(block)} depths beat random; "
            f"best margin {max(r['margin'][0] for r in block.values()):+.3f}"
        )
        print()

    fine = {}
    for layer in [int(x) for x in args.layers.split(",")]:
        cache = Path(
            f"results/visual_grids_4.npz" if layer == 6
            else f"results/visual_grids_4_L{layer}.npz"
        )
        if not cache.exists():
            print(f"layer {layer}: no cache at {cache}, skipping")
            continue
        stored = np.load(cache, allow_pickle=True)
        grids, ids = stored["grids"], [str(e) for e in stored["example_ids"]]
        features = [pool_cells(g, rows, cols) for g in grids]
        labels = np.array([[grouped[e][c].correct for c in cell_ids] for e in ids])

        margins, chosen_rates, random_rates = [], [], []
        for seed in range(RESAMPLES):
            rng = np.random.default_rng(13000 + seed)
            order = rng.permutation(len(ids))
            cut = int(len(ids) * TEST_FRACTION)
            test, train = order[:cut], order[cut:]
            localizer = train_localizer(
                [features[i] for i in train], [list(labels[i]) for i in train]
            )
            picked = [int(np.argmax(localizer.scores(features[i]))) for i in test]
            chosen = float(np.mean([labels[i][p] for i, p in zip(test, picked)]))
            # Random on the same fold: the mean cell-correctness rate.
            random_rate = float(labels[test].mean())
            margins.append(chosen - random_rate)
            chosen_rates.append(chosen)
            random_rates.append(random_rate)

        interval = bootstrap_interval(margins)
        oracle = float(labels.any(axis=1).mean())
        fine[layer] = {
            "chosen": float(np.mean(chosen_rates)),
            "random": float(np.mean(random_rates)),
            "oracle": oracle,
            "margin": [interval.estimate, interval.low, interval.high],
        }
        print(
            f"layer {layer:>2}: localizer {np.mean(chosen_rates):.1%} vs random "
            f"{np.mean(random_rates):.1%}, margin {interval} (oracle {oracle:.1%})"
        )

    results[f"{rows}x{cols}"] = {"n": len(ids), "oracle": oracle, "layers": fine}
    beats = [l for l, r in fine.items() if r["margin"][1] > 0.0]
    print(
        f"\n{rows}x{cols} (n={len(ids)}): {len(beats)}/{len(fine)} depths beat random "
        f"with an interval excluding zero, best margin "
        f"{max(r['margin'][0] for r in fine.values()):+.3f} against a headroom of "
        f"{oracle - float(labels.mean()):.3f}, over {RESAMPLES} resampled splits."
    )
    Path(args.out).write_text(
        json.dumps({"resamples": RESAMPLES, "grids": results,
                    "beats_random_fine": beats}, indent=2)
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
