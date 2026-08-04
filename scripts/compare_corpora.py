"""Algorithm 3 on a corpus it has never seen, and a prediction it fails.

The paper states that $1152$\\,px is DocVQA's legibility ceiling and not a
universal one, and predicts explicitly that a benchmark of denser type should
place it higher. InfographicVQA is that benchmark: same task, same ANLS metric,
pages that carry a far wider range of type sizes. Running the stated procedure
on it tests the prediction rather than illustrating the claim.

Two quantities are compared per corpus and they come apart. The ceiling is the
highest rung whose paired gain excludes zero, and the token count spent at that
ceiling is what the model pays to reach it. If the ceiling is a pixel property
the first should be portable and the second should not.

Usage: PYTHONPATH=src python scripts/compare_corpora.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records

RUNGS = ("lowres_384", "lowres_768", "lowres_1152", "full")
PIXELS = {"lowres_384": 384, "lowres_768": 768, "lowres_1152": 1152, "full": 2048}
BOOT = 3000
NULL_PRECISION = 0.05


def summarise(config_path: str, label: str, rng) -> dict:
    config = load_config(config_path)
    grouped: dict[str, dict] = defaultdict(dict)
    for row in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[row.example_id][row.config_id] = row
    ids = [e for e in grouped if all(c in grouped[e] for c in RUNGS)]
    ok = {c: np.array([grouped[e][c].correct for e in ids], float) for c in RUNGS}
    tok = {
        c: float(np.median([grouped[e][c].visual_tokens for e in ids])) for c in RUNGS
    }

    steps, ceiling, undetermined = [], RUNGS[0], []
    for lower, upper in zip(RUNGS, RUNGS[1:], strict=False):
        delta = ok[upper] - ok[lower]
        draws = delta[rng.integers(0, len(delta), (BOOT, len(delta)))].mean(axis=1)
        low, high = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
        row = {
            "from": lower,
            "to": upper,
            "gain": float(delta.mean()),
            "low": low,
            "high": high,
            "half_width": (high - low) / 2.0,
            "null": low <= 0.0 <= high,
            "extra_tokens": tok[upper] - tok[lower],
            "vector": [float(d) for d in delta],
        }
        steps.append(row)
        if low > 0.0:
            ceiling = upper
        elif row["half_width"] > NULL_PRECISION:
            undetermined.append(f"{lower}->{upper}")

    return {
        "label": label,
        "config": config_path,
        "n": len(ids),
        "accuracy": {c: float(ok[c].mean()) for c in RUNGS},
        "median_tokens": tok,
        "steps": steps,
        "ceiling_rung": ceiling,
        "ceiling_px": PIXELS[ceiling],
        "tokens_at_ceiling": tok[ceiling],
        "undetermined": undetermined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", nargs="*", default=[
        "configs/docvqa1200.yaml=DocVQA",
        "configs/infovqa500.yaml=InfographicVQA",
    ])
    parser.add_argument("--out", default="results/corpus_ceilings.json")
    args = parser.parse_args()

    rng = np.random.default_rng(20260804)
    runs = []
    for entry in args.corpora:
        path, _, label = entry.partition("=")
        if not Path(path).exists():
            print(f"skipping {path}: no config")
            continue
        try:
            runs.append(summarise(path, label or path, rng))
        except FileNotFoundError as error:
            print(f"skipping {path}: missing {error.filename}")

    header = f"{'corpus':<18}{'n':>6}" + "".join(f"{PIXELS[c]:>9}" for c in RUNGS)
    print(header)
    for run in runs:
        line = f"{run['label']:<18}{run['n']:>6}"
        line += "".join(f"{run['accuracy'][c]:>9.3f}" for c in RUNGS)
        print(line)
    print(f"\n{'median visual tokens':<24}" + "".join(f"{PIXELS[c]:>9}" for c in RUNGS))
    for run in runs:
        print(
            f"{run['label']:<24}"
            + "".join(f"{run['median_tokens'][c]:>9.0f}" for c in RUNGS)
        )

    print()
    for run in runs:
        print(
            f"{run['label']:<18} ceiling {run['ceiling_px']}\\,px, "
            f"{run['tokens_at_ceiling']:.0f} tokens there"
            + (f", UNDETERMINED {run['undetermined']}" if run["undetermined"] else "")
        )
        for step in run["steps"]:
            print(
                f"    {PIXELS[step['from']]:>4} -> {PIXELS[step['to']]:<5}"
                f"{step['gain']:+.3f} [{step['low']:+.3f}, {step['high']:+.3f}]"
                f"  ({step['extra_tokens']:+.0f} tokens)"
            )

    same_px = len({r["ceiling_px"] for r in runs}) == 1
    tokens = {r["tokens_at_ceiling"] for r in runs}
    spread = max(tokens) / min(tokens) if tokens else 1.0
    print(
        f"\nsame ceiling in pixels: {same_px}; token spend at that ceiling differs "
        f"by {spread:.2f}x across corpora"
    )
    Path(args.out).write_text(
        json.dumps(
            {"runs": runs, "same_ceiling_px": same_px, "token_spread": spread},
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
