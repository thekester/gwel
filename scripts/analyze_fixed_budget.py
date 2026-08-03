"""Pixels and tokens, separated, on a model whose token budget barely moves.

Everywhere else in this paper the two travel together: raising the pixel target
raises the visual-token count, so "more pixels stop helping" and "more tokens
stop helping" are the same sentence and the accuracy curve cannot tell them
apart. InternVL3-1B breaks the tie by accident of its architecture. Its tiling
selects a patch grid from the aspect ratio, so a page downsampled to 384 px is
upscaled again to fill the same tiles as the 2048 px original, and the ladder
runs at a near-constant token budget.

The script reports three things and refuses to conflate them. How constant the
budget actually is, since the premise is measured rather than assumed. Where
accuracy stops climbing when only pixels move. And what that implies for the
claim the other models support, stated with the confound it carries: this is
one model, so a ceiling that arrives earlier here is equally consistent with
the pixel information saturating earlier and with this model reading less well
than the ones it is compared to.

Usage: PYTHONPATH=src python scripts/analyze_fixed_budget.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records

RUNGS = ("lowres_384", "lowres_768", "lowres_1152", "full")
PIXELS = {"lowres_384": 384, "lowres_768": 768, "lowres_1152": 1152, "full": 2048}
BOOT = 3000
NULL_PRECISION = 0.05


def interval(delta: np.ndarray, rng) -> dict:
    means = delta[rng.integers(0, len(delta), (BOOT, len(delta)))].mean(axis=1)
    low, high = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    return {
        "gain": float(delta.mean()),
        "low": low,
        "high": high,
        "half_width": (high - low) / 2.0,
        "null": low <= 0.0 <= high,
        "informative": (high - low) / 2.0 <= NULL_PRECISION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", default="results/runs/docvqa1200_internvl1b_records.jsonl"
    )
    parser.add_argument("--out", default="results/fixed_budget.json")
    args = parser.parse_args()

    grouped: dict[str, dict] = defaultdict(dict)
    for r in rescore_records(
        deduplicate_records(read_records(args.records)), ScoringPolicy()
    ):
        grouped[r.example_id][r.config_id] = r
    ids = [e for e in grouped if all(c in grouped[e] for c in RUNGS)]
    ok = {c: np.array([grouped[e][c].correct for e in ids]) for c in RUNGS}
    tok = {
        c: np.array([grouped[e][c].visual_tokens for e in ids], float) for c in RUNGS
    }
    rng = np.random.default_rng(20260803)

    identical = float(
        np.mean([len({tok[c][i] for c in RUNGS}) == 1 for i in range(len(ids))])
    )
    spend = {c: float(tok[c].mean()) for c in RUNGS}
    token_spread = max(spend.values()) / min(spend.values())
    pixel_spread = PIXELS["full"] / PIXELS["lowres_384"]
    print(f"n = {len(ids)} pages")
    print(
        f"token budget: all four rungs identical on {identical:.0%} of pages; "
        f"mean spend {min(spend.values()):.0f} to {max(spend.values()):.0f} "
        f"({token_spread:.2f}x) against {pixel_spread:.1f}x in input pixels"
    )

    steps = []
    for lower, upper in zip(RUNGS, RUNGS[1:], strict=False):
        row = interval(ok[upper].astype(float) - ok[lower].astype(float), rng)
        row.update(
            {
                "from": lower,
                "to": upper,
                "extra_tokens": float(tok[upper].mean() - tok[lower].mean()),
            }
        )
        steps.append(row)
        flag = "null" if row["null"] else "gain"
        print(
            f"  {lower:>11} -> {upper:<11} {row['gain']:+.3f} "
            f"[{row['low']:+.3f}, {row['high']:+.3f}] half-width "
            f"{row['half_width']:.3f}  {flag}  "
            f"({row['extra_tokens']:+.0f} tokens)"
        )

    ceiling = RUNGS[0]
    for row in steps:
        if row["low"] > 0.0:
            ceiling = row["to"]
    undetermined = [
        f"{r['from']}->{r['to']}" for r in steps if r["null"] and not r["informative"]
    ]

    print(f"\nceiling with the token budget held near constant: {ceiling}")
    if undetermined:
        print(f"UNDETERMINED steps (interval too wide): {undetermined}")

    accuracy = {c: float(ok[c].mean()) for c in RUNGS}
    Path(args.out).write_text(
        json.dumps(
            {
                "model": "OpenGVLab/InternVL3-1B-hf",
                "n": len(ids),
                "accuracy": accuracy,
                "mean_tokens": spend,
                "rungs_token_identical": identical,
                "token_spread": token_spread,
                "pixel_spread": pixel_spread,
                "steps": steps,
                "ceiling": ceiling,
                "undetermined": undetermined,
            },
            indent=2,
        )
    )
    print(
        "\nRead with care. A ceiling reached earlier here than on the models whose\n"
        "token count grows with resolution is consistent with two different\n"
        "things: pixel information saturating before 1152 px, or this model\n"
        "reading less of the page than they do. One model cannot separate them."
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
