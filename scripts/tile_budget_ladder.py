"""The token axis, with the pixels held fixed.

The fixed-budget control varies input pixels while InternVL3-1B's token spend
stays near constant, and finds the gain stopping one rung earlier than it does
in models whose tokeniser follows resolution. That leaves two readings open,
because pixels and tokens still move together in those models. This is the
complementary experiment, and it closes the pair: InternVL selects its patch
grid up to a `max_patches` bound, so raising that bound multiplies the visual
tokens spent on an image whose pixels never change.

Fixing the input at full resolution and sweeping the bound therefore measures
what sequence length alone buys on this corpus. Read against the fixed-budget
ladder, the two say which of the two quantities the 768 to 1152 px band was
actually paying for.

Usage: PYTHONPATH=src python scripts/tile_budget_ladder.py --limit 500
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy
from gwel.modeling.smolvlm import SmolVlmEngine

BUDGETS = (4, 12, 24)
BOOT = 3000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/docvqa1200_internvl1b.yaml")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--edge", type=int, default=2048)
    parser.add_argument("--out", default="results/tile_budget.json")
    parser.add_argument("--records", default="results/runs/tile_budget_records.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = [
        json.loads(line)
        for line in Path(config.paths.pilot_manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]

    engine = SmolVlmEngine(config.model)
    engine.ensure_loaded()
    processor = engine._processor.image_processor
    policy = ScoringPolicy()

    sink = Path(args.records)
    sink.parent.mkdir(parents=True, exist_ok=True)
    handle = sink.open("w", encoding="utf-8")

    per_budget: dict[int, dict[str, bool]] = defaultdict(dict)
    tokens: dict[int, list[int]] = defaultdict(list)
    started = time.perf_counter()
    for index, row in enumerate(manifest):
        source = Image.open(row["image_path"]).convert("RGB")
        scale = args.edge / max(source.size)
        image = (
            source.resize(
                (round(source.width * scale), round(source.height * scale)),
                Image.LANCZOS,
            )
            if scale < 1.0
            else source
        )
        for budget in BUDGETS:
            processor.max_patches = budget
            out = engine.generate([image], row["question"])
            score = policy.score(row["dataset"], out.answer, row["answers"])
            correct = score >= policy.correct_threshold
            per_budget[budget][row["example_id"]] = bool(correct)
            tokens[budget].append(out.visual_tokens)
            handle.write(
                json.dumps(
                    {
                        "example_id": row["example_id"],
                        "max_patches": budget,
                        "visual_tokens": out.visual_tokens,
                        "generate_ms": out.generate_ms,
                        "answer": out.answer,
                        "correct": bool(correct),
                    }
                )
                + "\n"
            )
        if (index + 1) % 50 == 0:
            rate = (index + 1) / (time.perf_counter() - started)
            print(
                f"  {index + 1}/{len(manifest)} images, "
                f"{(len(manifest) - index - 1) / rate / 60:.0f} min left",
                flush=True,
            )
    handle.close()

    ids = sorted(set.intersection(*(set(per_budget[b]) for b in BUDGETS)))
    rng = np.random.default_rng(20260803)
    accuracy = {
        b: float(np.mean([per_budget[b][e] for e in ids])) for b in BUDGETS
    }
    median_tokens = {b: float(np.median(tokens[b])) for b in BUDGETS}
    print(f"\nn = {len(ids)} pages, input fixed at {args.edge} px")
    for budget in BUDGETS:
        print(
            f"  max_patches={budget:<3} {median_tokens[budget]:>6.0f} tokens  "
            f"accuracy {accuracy[budget]:.3f}"
        )

    steps = []
    for lower, upper in zip(BUDGETS, BUDGETS[1:], strict=False):
        delta = np.array(
            [float(per_budget[upper][e]) - float(per_budget[lower][e]) for e in ids]
        )
        draws = delta[rng.integers(0, len(delta), (BOOT, len(delta)))].mean(axis=1)
        low, high = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
        steps.append(
            {
                "from": lower,
                "to": upper,
                "extra_tokens": median_tokens[upper] - median_tokens[lower],
                "gain": float(delta.mean()),
                "low": low,
                "high": high,
                "half_width": (high - low) / 2.0,
                "null": low <= 0.0 <= high,
            }
        )
        print(
            f"  {lower} -> {upper} patches ({median_tokens[upper] - median_tokens[lower]:+.0f}"
            f" tokens): {delta.mean():+.3f} [{low:+.3f}, {high:+.3f}]"
        )

    Path(args.out).write_text(
        json.dumps(
            {
                "model": config.model.model_id,
                "n": len(ids),
                "input_edge_px": args.edge,
                "accuracy": {str(k): v for k, v in accuracy.items()},
                "median_tokens": {str(k): v for k, v in median_tokens.items()},
                "steps": steps,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
