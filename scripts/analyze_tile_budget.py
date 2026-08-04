"""What extra visual tokens buy when the pixels do not change.

The companion to the fixed-budget ladder. There, input pixels moved and the
token spend did not; here the input is pinned at full resolution and the tile
bound moves instead. Together they separate two quantities that travel together
in every model whose tokeniser follows resolution.

The nominal bound is not the experiment, and this script exists because reading
it as one would repeat the error this paper documents four times. InternVL
picks the aspect-ratio-closest tile grid *under* the bound, so raising the
bound from 12 to 24 raises the token count on part of the corpus and leaves the
rest untouched. Every comparison here is therefore conditioned on the token
count that was actually spent, with the unchanged subset kept as an internal
control: where the tokens did not move, nothing should.

Usage: PYTHONPATH=src python scripts/analyze_tile_budget.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BOOT = 3000
NULL_PRECISION = 0.05


def interval(delta: np.ndarray, rng) -> dict:
    draws = delta[rng.integers(0, len(delta), (BOOT, len(delta)))].mean(axis=1)
    low, high = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    return {
        "n": int(len(delta)),
        "gain": float(delta.mean()),
        "low": low,
        "high": high,
        "half_width": (high - low) / 2.0,
        "null": low <= 0.0 <= high,
        "informative": (high - low) / 2.0 <= NULL_PRECISION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="results/runs/tile_budget_records.jsonl")
    parser.add_argument("--out", default="results/tile_budget_analysis.json")
    args = parser.parse_args()

    correct: dict[str, dict[int, bool]] = defaultdict(dict)
    tokens: dict[str, dict[int, int]] = defaultdict(dict)
    for line in Path(args.records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        correct[row["example_id"]][row["max_patches"]] = bool(row["correct"])
        tokens[row["example_id"]][row["max_patches"]] = int(row["visual_tokens"])
    budgets = sorted({b for v in tokens.values() for b in v})
    ids = [e for e in tokens if len(tokens[e]) == len(budgets)]
    rng = np.random.default_rng(20260803)

    print(f"n = {len(ids)} pages, input pinned at full resolution")
    for budget in budgets:
        spend = np.array([tokens[e][budget] for e in ids], float)
        accuracy = np.mean([correct[e][budget] for e in ids])
        print(
            f"  bound {budget:<3} median {np.median(spend):>6.0f} tokens "
            f"(mean {spend.mean():>6.0f})  accuracy {accuracy:.3f}"
        )

    out: dict = {"n": len(ids), "steps": []}
    for lower, upper in zip(budgets, budgets[1:], strict=False):
        moved = np.array([tokens[e][upper] > tokens[e][lower] for e in ids])
        delta = np.array(
            [float(correct[e][upper]) - float(correct[e][lower]) for e in ids]
        )
        extra = np.array([tokens[e][upper] - tokens[e][lower] for e in ids], float)
        row = {
            "from": lower,
            "to": upper,
            "share_where_tokens_rose": float(moved.mean()),
            "mean_extra_tokens_where_rose": (
                float(extra[moved].mean()) if moved.any() else 0.0
            ),
            "all": interval(delta, rng),
            "tokens_rose": interval(delta[moved], rng) if moved.sum() > 20 else None,
            "tokens_unchanged": (
                interval(delta[~moved], rng) if (~moved).sum() > 20 else None
            ),
        }
        out["steps"].append(row)
        print(
            f"\nbound {lower} -> {upper}: tokens rise on "
            f"{row['share_where_tokens_rose']:.0%} of pages "
            f"(+{row['mean_extra_tokens_where_rose']:.0f} there)"
        )
        for key in ("all", "tokens_rose", "tokens_unchanged"):
            block = row[key]
            if block is None:
                print(f"  {key:<18} too few pages to estimate")
                continue
            print(
                f"  {key:<18} n={block['n']:<4} {block['gain']:+.3f} "
                f"[{block['low']:+.3f}, {block['high']:+.3f}] "
                f"half-width {block['half_width']:.3f}"
                f"{'  (null)' if block['null'] else ''}"
            )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        "\nThe subset where the bound did not change the token count is the\n"
        "internal control: it receives an identical input and an identical\n"
        "sequence, so any difference there is decoding noise and bounds how\n"
        "much of the other subset's difference can be believed."
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
