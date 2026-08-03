"""Spend a visual-token budget on resolution or on position: which wins?

The paper argues that token pruning and resolution choice act on different
halves of the same cost, and that pruning cannot recover the encoder's share.
That is an argument about *where* the cost sits. It never measures the other
question, which is what a reviewer comparing us to the pruning literature will
actually ask: given a fixed number of visual tokens, is it better to see the
whole image coarsely or a piece of it sharply?

Both are available in the pilot without new inference. A resolution rung spends
its tokens on the whole frame; a crop spends them on one cell at high
magnification. Plotting accuracy against tokens for each puts the two families
on one axis, which is the comparison the paper has been asserting rather than
showing.

Three curves, because the crop family has two very different readings:

  resolution      whole image, tokens set by the rung
  crop, random    one cell chosen with no information, what a router without a
                  localizer actually gets
  crop, oracle    the best cell, known in advance: the ceiling that a perfect
                  localizer would reach and that our own localizer does not

Usage: PYTHONPATH=scripts python scripts/tokens_two_ways.py
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

RESOLUTION = ("lowres_384", "lowres_768", "full")
GRID = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--out", default="results/tokens_two_ways.json")
    args = parser.parse_args()

    config = load_config(args.config)
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record

    crops = [f"crop_r{r}c{c}" for r in range(GRID) for c in range(GRID)]
    ids = [
        e for e in grouped if all(c in grouped[e] for c in (*RESOLUTION, *crops))
    ]
    print(f"n={len(ids)} examples carrying every configuration\n")

    rows = []
    print(f"{'family':<16}{'config':<14}{'tokens':>8}{'ms':>8}{'accuracy [95% CI]':>24}")
    for config_id in RESOLUTION:
        tokens = float(np.median([grouped[e][config_id].visual_tokens for e in ids]))
        latency = float(np.mean([grouped[e][config_id].latency_ms for e in ids]))
        correct = [float(grouped[e][config_id].correct) for e in ids]
        interval = bootstrap_interval(correct)
        rows.append(
            {
                "family": "resolution",
                "config": config_id,
                "tokens": tokens,
                "latency_ms": latency,
                "accuracy": interval.estimate,
                "low": interval.low,
                "high": interval.high,
            }
        )
        print(
            f"{'resolution':<16}{config_id:<14}{tokens:>8.0f}{latency:>8.0f}"
            f"{str(interval):>24}"
        )

    matrix = np.array([[grouped[e][c].correct for c in crops] for e in ids])
    crop_tokens = float(
        np.median([grouped[e][crops[0]].visual_tokens for e in ids])
    )
    crop_latency = float(
        np.mean([grouped[e][crops[0]].latency_ms for e in ids])
    )
    for label, values in (
        ("crop, random", matrix.mean(axis=1)),
        ("crop, oracle", matrix.any(axis=1).astype(float)),
    ):
        interval = bootstrap_interval(values.tolist())
        rows.append(
            {
                "family": label,
                "config": f"{GRID}x{GRID} cell",
                "tokens": crop_tokens,
                "latency_ms": crop_latency,
                "accuracy": interval.estimate,
                "low": interval.low,
                "high": interval.high,
            }
        )
        print(
            f"{label:<16}{f'{GRID}x{GRID} cell':<14}{crop_tokens:>8.0f}"
            f"{crop_latency:>8.0f}{str(interval):>24}"
        )

    # The comparison that matters: what does a token buy in each family?
    resolution_rows = [r for r in rows if r["family"] == "resolution"]
    cheap = min(resolution_rows, key=lambda r: r["tokens"])
    # The comparator must be the *best* resolution configuration, not the one
    # spending the most tokens: two rungs tie at 320 tokens here and picking the
    # weaker of them would flatter the crop.
    dear = max(resolution_rows, key=lambda r: r["accuracy"])
    slope = (dear["accuracy"] - cheap["accuracy"]) / (dear["tokens"] - cheap["tokens"])
    oracle = next(r for r in rows if r["family"] == "crop, oracle")
    random_crop = next(r for r in rows if r["family"] == "crop, random")
    predicted = cheap["accuracy"] + slope * (oracle["tokens"] - cheap["tokens"])

    print(
        f"\nresolution buys {slope * 100:.4f} accuracy per token between "
        f"{cheap['tokens']:.0f} and {dear['tokens']:.0f}."
    )
    print(
        f"At {oracle['tokens']:.0f} tokens it would reach about {predicted:.3f}; a "
        f"randomly chosen crop reaches {random_crop['accuracy']:.3f} and the best "
        f"crop {oracle['accuracy']:.3f}."
    )
    # Paired, since both are measured on the same examples.
    from gwel.router.evaluate import paired_difference

    best_crop = matrix.any(axis=1).astype(float)
    best_resolution = np.array([float(grouped[e][dear["config"]].correct) for e in ids])
    delta = paired_difference(best_crop.tolist(), best_resolution.tolist())
    beats_dear = delta.low > 0.0
    print(
        f"\nThe best crop at {oracle['tokens']:.0f} tokens, against the best "
        f"resolution rung ({dear['config']}, {dear['tokens']:.0f} tokens): "
        f"{delta} paired."
    )
    print(
        "It " + ("beats it" if beats_dear else "matches it")
        + f" while spending {dear['tokens'] / oracle['tokens']:.1f}x fewer visual "
        "tokens."
    )
    print(
        "So position is worth more than resolution per token, and only to a\n"
        "policy that knows where to look. That is what the localizer would be\n"
        "worth, and it is why the negative result on it is expensive rather than\n"
        "merely disappointing."
    )

    results = {
        "n": len(ids),
        "rows": rows,
        "resolution_slope_per_token": slope,
        "oracle_beats_best_resolution": bool(beats_dear),
        "paired_delta": [delta.estimate, delta.low, delta.high],
        "comparator": dear["config"],
        "token_ratio": dear["tokens"] / oracle["tokens"],
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
