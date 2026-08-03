"""How many pages does it take to locate a corpus's resolution ceiling?

The paper's most transferable object is no longer the number 1152, it is the
procedure that finds it on a corpus. A procedure is only usable if it says how
much data it needs, and that is measurable rather than assumable: subsample the
DocVQA pilot at increasing n, recompute the paired top-step interval, and read
where its half-width crosses the precision bar a null has to clear.

Two quantities are reported per size. The half-width answers "is this null
tight enough to count". The verdict-stability column answers the sharper
question: how often a run at that size would name the same ceiling rung as the
full corpus does, which is what a practitioner actually cares about.

Usage: PYTHONPATH=src python scripts/ceiling_sample_size.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records

RUNGS = ("lowres_384", "lowres_768", "lowres_1152", "full")
SIZES = (100, 200, 300, 500, 800, 1200)
DRAWS = 200
BOOT = 400
NULL_PRECISION = 0.05


def paired_interval(delta: np.ndarray, rng: np.random.Generator) -> tuple:
    """Bootstrap mean and 95% interval of a paired difference."""
    n = len(delta)
    means = delta[rng.integers(0, n, (BOOT, n))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(delta.mean()), float(low), float(high)


def ceiling_rung(ok: dict, index: np.ndarray, rng: np.random.Generator) -> str:
    """Highest rung whose step gain is significantly positive on this sample."""
    best = RUNGS[0]
    for lower, upper in zip(RUNGS, RUNGS[1:], strict=False):
        delta = ok[upper][index].astype(float) - ok[lower][index].astype(float)
        _, low, _ = paired_interval(delta, rng)
        if low > 0.0:
            best = upper
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="results/runs/docvqa1200_records.jsonl")
    parser.add_argument("--out", default="results/ceiling_sample_size.json")
    args = parser.parse_args()

    grouped: dict[str, dict] = defaultdict(dict)
    for r in rescore_records(
        deduplicate_records(read_records(args.records)), ScoringPolicy()
    ):
        grouped[r.example_id][r.config_id] = r
    ids = [e for e in grouped if all(c in grouped[e] for c in RUNGS)]
    ok = {c: np.array([grouped[e][c].correct for e in ids]) for c in RUNGS}
    n_all = len(ids)

    rng = np.random.default_rng(20260803)
    truth = ceiling_rung(ok, np.arange(n_all), rng)
    top = ok["full"].astype(float) - ok["lowres_1152"].astype(float)

    rows = []
    for size in SIZES:
        if size > n_all:
            continue
        widths, agrees = [], 0
        for _ in range(DRAWS):
            index = rng.choice(n_all, size=size, replace=False)
            _, low, high = paired_interval(top[index], rng)
            widths.append((high - low) / 2.0)
            agrees += ceiling_rung(ok, index, rng) == truth
        row = {
            "n": size,
            "median_half_width": float(np.median(widths)),
            "p90_half_width": float(np.percentile(widths, 90)),
            "verdict_agreement": agrees / DRAWS,
        }
        rows.append(row)
        print(
            f"n={size:<5} half-width median {row['median_half_width']:.3f} "
            f"p90 {row['p90_half_width']:.3f}   names the same ceiling "
            f"{row['verdict_agreement']:.0%} of the time"
        )

    # The half-width of a paired mean shrinks as one over the square root of n;
    # fitting the constant turns the table into a sample-size rule.
    constant = float(
        np.median([r["median_half_width"] * np.sqrt(r["n"]) for r in rows])
    )
    needed = int(np.ceil((constant / NULL_PRECISION) ** 2))
    print(
        f"\nhalf-width is about {constant:.2f}/sqrt(n); reaching the "
        f"{NULL_PRECISION} precision bar takes about {needed} pages"
    )
    print(f"full-corpus ceiling on this pilot: {truth} (n={n_all})")

    Path(args.out).write_text(
        json.dumps(
            {
                "rows": rows,
                "sqrt_constant": constant,
                "pages_for_null_precision": needed,
                "null_precision": NULL_PRECISION,
                "ceiling": truth,
                "n": n_all,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
