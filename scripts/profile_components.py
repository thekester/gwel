"""Attribute pass latency to vision encoder, projector, prefill and decode.

Our headline "vision costs X% of a pass" was obtained by subtracting a
text-only pass from an image pass, which conflates three components that scale
differently. This measures them separately, following the decomposition of
Shin et al. (arXiv 2607.08029).

The distinction matters for positioning: post-encoder token pruning can only
recover the LLM's share of the visual cost, while reducing input resolution
recovers the encoder's share as well.

Usage: python scripts/profile_components.py --config configs/pilot1000.yaml
"""

import argparse
import json
from pathlib import Path
from statistics import median

from PIL import Image

from gwel.config import load_config
from gwel.data.loaders import read_manifest
from gwel.modeling.imaging import downscale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--examples", type=int, default=5, help="images to average over")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="results/component_latency.json")
    args = parser.parse_args()

    from gwel.modeling.smolvlm import SmolVlmEngine

    config = load_config(args.config)
    engine = SmolVlmEngine(config.model)
    engine.ensure_loaded()
    examples = read_manifest(config.paths.pilot_manifest)[: args.examples]

    sizes = [None, *config.runner.lowres_sizes, config.runner.full_longest_side]
    collected: dict[str, list] = {}

    for example in examples:
        with Image.open(example.image_path) as raw:
            image = raw.convert("RGB")
            for size in sizes:
                label = "no_image" if size is None else f"longest_{size}"
                timing = engine.profile_components(
                    None if size is None else [downscale(image, size)],
                    example.question,
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                collected.setdefault(label, []).append(timing)

    print(f"medians over {len(examples)} images, {args.repeats} repeats each\n")
    header = f"{'config':<14}{'vtok':>6}{'encoder':>9}{'proj':>7}{'prefill':>9}{'decode':>8}{'total':>8}"
    print(header)

    rows: list[dict[str, object]] = []
    baseline_prefill = None
    for label, timings in collected.items():
        row = {
            "config": label,
            "visual_tokens": timings[0].visual_tokens,
            "vision_encoder_ms": median(t.vision_encoder_ms for t in timings),
            "projector_ms": median(t.projector_ms for t in timings),
            "prefill_ms": median(t.prefill_ms for t in timings),
            "decode_ms": median(t.decode_ms for t in timings),
        }
        row["total_ms"] = (
            row["vision_encoder_ms"] + row["projector_ms"] + row["prefill_ms"] + row["decode_ms"]
        )
        if label == "no_image":
            baseline_prefill = row["prefill_ms"]
        rows.append(row)
        print(
            f"{label:<14}{row['visual_tokens']:>6}{row['vision_encoder_ms']:>9.1f}"
            f"{row['projector_ms']:>7.1f}{row['prefill_ms']:>9.1f}"
            f"{row['decode_ms']:>8.1f}{row['total_ms']:>8.1f}"
        )

    if baseline_prefill is not None:
        print("\n== cost attributable to vision, split by where it is paid ==")
        print(f"  {'config':<14}{'encoder':>9}{'extra prefill':>15}{'vision total':>14}{'share':>8}")
        for row in rows:
            if row["config"] == "no_image":
                continue
            encoder = row["vision_encoder_ms"] + row["projector_ms"]
            extra_prefill = max(row["prefill_ms"] - baseline_prefill, 0.0)
            vision_total = encoder + extra_prefill
            row["extra_prefill_ms"] = extra_prefill
            row["vision_total_ms"] = vision_total
            row["vision_share"] = vision_total / row["total_ms"] if row["total_ms"] else 0.0
            print(
                f"  {row['config']:<14}{encoder:>9.1f}{extra_prefill:>15.1f}"
                f"{vision_total:>14.1f}{row['vision_share']:>8.0%}"
            )
        print(
            "\n  encoder share is what resolution reduction saves and token pruning does not;\n"
            "  extra prefill is what both save."
        )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
