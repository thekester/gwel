"""Can the cheap pass say *where* to crop, from its own visual tokens?

Region choice dominates action choice in our measurements, and AwaRes solves it
with SFT, multi-turn GRPO and a 70B judge. This asks whether the low-resolution
pass already knows, by pooling the hidden states of the visual tokens covering
each candidate cell and ranking them with a linear probe.

The baselines that matter are random cell choice (what a router with no
localizer gets) and the oracle ceiling (the fraction of examples where any cell
works at all).

Usage: python scripts/test_localizer.py --config configs/pilot1000.yaml
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from gwel.config import load_config
from gwel.data.loaders import read_manifest
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.modeling.imaging import downscale
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.localizer import evaluate_localizer, pool_cells, train_localizer
from gwel.router.splits import make_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--cache", default="results/visual_grids.npz")
    args = parser.parse_args()

    config = load_config(args.config)
    rows, cols = config.runner.crop.rows, config.runner.crop.cols
    cell_ids = [f"crop_r{r}c{c}" for r in range(rows) for c in range(cols)]

    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    grouped: dict[str, dict] = defaultdict(dict)
    for record in records:
        grouped[record.example_id][record.config_id] = record

    manifest = {e.example_id: e for e in read_manifest(config.paths.pilot_manifest)}
    usable = [
        e for e in manifest
        if all(c in grouped[e] for c in cell_ids)
    ][: args.limit]

    cache = Path(args.cache)
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        grids, usable = stored["grids"], list(stored["example_ids"])
        print(f"loaded cached visual grids for {len(usable)} examples")
    else:
        from gwel.modeling.smolvlm import SmolVlmEngine

        engine = SmolVlmEngine(config.model)
        engine.ensure_loaded()
        size = config.runner.lowres_sizes[0]
        collected = []
        for index, example_id in enumerate(usable):
            with Image.open(manifest[example_id].image_path) as raw:
                image = downscale(raw.convert("RGB"), size)
            collected.append(
                engine.extract_visual_grid(image, manifest[example_id].question, layer=args.layer)
            )
            if (index + 1) % 100 == 0:
                print(f"  {index + 1}/{len(usable)}")
        grids = np.stack(collected)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, grids=grids, example_ids=usable)

    features = [pool_cells(g, rows, cols) for g in grids]
    labels = [[grouped[e][c].correct for c in cell_ids] for e in usable]

    split = make_split(
        usable,
        [grouped[e][cell_ids[0]].dataset for e in usable],
        val_fraction=config.router.val_fraction,
        test_fraction=config.router.test_fraction,
        seed=config.router.seed,
    )
    order = {e: i for i, e in enumerate(usable)}
    train = [order[e] for e in split.train if e in order]
    test = [order[e] for e in split.test if e in order]

    localizer = train_localizer([features[i] for i in train], [labels[i] for i in train])
    stats = evaluate_localizer(
        localizer, [features[i] for i in test], [labels[i] for i in test]
    )

    print(f"\nvisual grid {grids.shape[1]}x{grids.shape[2]} at layer {args.layer}, "
          f"{rows}x{cols} crop cells")
    print(f"train {len(train)} examples, test {int(stats['examples'])}\n")
    print(f"  {'policy':<28}{'hit rate':>10}")
    print(f"  {'random cell (no localizer)':<28}{stats['random']:>10.1%}")
    print(f"  {'learned localizer':<28}{stats['chosen']:>10.1%}")
    print(f"  {'oracle (best cell known)':<28}{stats['oracle']:>10.1%}")

    span = stats["oracle"] - stats["random"]
    if span > 0:
        closed = (stats["chosen"] - stats["random"]) / span
        print(f"\n  closes {closed:.0%} of the gap between random and oracle")

    # Per-dataset, since region choice should matter most where detail does.
    print(f"\n  {'dataset':<10}{'random':>9}{'localizer':>11}{'oracle':>9}{'n':>5}")
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for i in test:
        by_dataset[grouped[usable[i]][cell_ids[0]].dataset].append(i)
    for dataset, indices in sorted(by_dataset.items()):
        if len(indices) < 10:
            continue
        sub = evaluate_localizer(
            localizer, [features[i] for i in indices], [labels[i] for i in indices]
        )
        print(f"  {dataset:<10}{sub['random']:>9.1%}{sub['chosen']:>11.1%}"
              f"{sub['oracle']:>9.1%}{len(indices):>5}")


if __name__ == "__main__":
    main()
