"""Where downsampling is free and where it is fatal, by dataset.

The domain-confound section establishes statistically that the probe's pooled
AUROC rides on the dataset boundary. This artefact carries the observation the
statistics rest on: accuracy at 384 px against full resolution, per dataset,
plus the share of queries whose correctness changes between the two. On
photograph-centric benchmarks the thumbnail loses almost nothing; on documents
it collapses. The same split is reported for the no-image control, which bounds
how much of each benchmark is answerable blind.

Usage: PYTHONPATH=src python scripts/make_domain_bars.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records

CONFIGS = ("no_image", "lowres_384", "full")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--out", default="results/domain_bars.json")
    args = parser.parse_args()

    config = load_config(args.config)
    grouped: dict[str, dict] = defaultdict(dict)
    dataset: dict[str, str] = {}
    for r in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[r.example_id][r.config_id] = r
        dataset[r.example_id] = r.dataset

    by_ds: dict[str, list[str]] = defaultdict(list)
    for e in grouped:
        if all(c in grouped[e] for c in CONFIGS):
            by_ds[dataset[e]].append(e)

    out = {}
    for ds, ids in sorted(by_ds.items()):
        ok = {c: [grouped[e][c].correct for e in ids] for c in CONFIGS}
        changed = sum(
            a != b for a, b in zip(ok["lowres_384"], ok["full"], strict=True)
        )
        out[ds] = {
            "n": len(ids),
            "no_image": sum(ok["no_image"]) / len(ids),
            "lowres_384": sum(ok["lowres_384"]) / len(ids),
            "full": sum(ok["full"]) / len(ids),
            "changed": changed / len(ids),
        }
        print(
            f"{ds:<10} n={len(ids):<4} blind={out[ds]['no_image']:.3f} "
            f"384px={out[ds]['lowres_384']:.3f} full={out[ds]['full']:.3f} "
            f"changed={out[ds]['changed']:.3f}"
        )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
