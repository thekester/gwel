"""Each benchmark ships its images at its own characteristic size.

The paper reports that raw image size predicts which dataset a query came from
at 0.976 inside a stratum, and reads the free descriptor's value on a mixture
as dataset identification. It never says *why* size identifies the dataset. The
answer is mundane and worth stating: benchmarks are distributed pre-resized,
each to its own convention, so size is close to a dataset label printed on
every image.

Preparing a fourth single-domain corpus made this unavoidable. TextVQA as
distributed has 494 of 500 images at exactly 1024 px on the longest edge, an
interquartile range of zero. A free size descriptor cannot rank anything there,
so a "the descriptor falls to chance inside a workload" result on that corpus
would be true by construction rather than because the between-domain axis was
removed. That is a precondition our own analysis should test and did not.

Usage: PYTHONPATH=src:scripts python scripts/benchmark_size_signature.py
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MANIFESTS = {
    "pilot1000 (mixture)": "data/processed/pilot1000/manifest.jsonl",
    "docvqa1200": "data/processed/docvqa1200/manifest.jsonl",
    "infovqa500": "data/processed/infovqa500/manifest.jsonl",
    "chartqa500": "data/processed/chartqa500/manifest.jsonl",
    "textvqa500": "data/processed/textvqa500/manifest.jsonl",
}


def longest_edges(path: Path) -> dict[str, list[int]]:
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        try:
            with Image.open(row["image_path"]) as image:
                by_dataset[row.get("dataset", "?")].append(max(image.size))
        except (OSError, KeyError):
            continue
    return by_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/size_signature.json")
    args = parser.parse_args()

    out: dict[str, dict] = {}
    print(f"{'corpus / dataset':<28}{'n':>5}{'median':>8}{'IQR':>7}{'modal share':>13}{'distinct':>10}")
    for label, path in MANIFESTS.items():
        target = Path(path)
        if not target.exists():
            continue
        for dataset, edges in sorted(longest_edges(target).items()):
            values = np.array(edges, float)
            if values.size < 20:
                continue
            counts = Counter(edges)
            modal_value, modal_count = counts.most_common(1)[0]
            iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
            row = {
                "n": int(values.size),
                "median": float(np.median(values)),
                "iqr": iqr,
                "modal_edge": int(modal_value),
                "modal_share": modal_count / values.size,
                "distinct": len(counts),
            }
            name = dataset if label.startswith("pilot") else f"{label}"
            key = f"{label}:{dataset}"
            out[key] = row
            print(
                f"{name:<28}{row['n']:>5}{row['median']:>8.0f}{iqr:>7.0f}"
                f"{row['modal_share']:>12.1%}{row['distinct']:>10}"
            )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        "\nA corpus whose modal share is near one carries no size signal at all, so\n"
        "a free size descriptor is at chance there by construction rather than by\n"
        "measurement. Across a mixture the same fact works the other way: distinct\n"
        "characteristic sizes make the descriptor a dataset label."
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
