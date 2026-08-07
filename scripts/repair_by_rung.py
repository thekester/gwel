"""Where on the ladder escalation actually repairs a query.

The section's headline is that most of what escalation buys is bought below the
top rung, which the paper states as two percentages and never shows. The
distribution behind them is more informative than either: for each query the
top rung repairs, the cheapest rung that already repairs it, plus the queries
escalation damages, which a repair-only view hides.

Usage: PYTHONPATH=src:scripts python scripts/repair_by_rung.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records

CHEAP = "lowres_384"
RUNGS = ("lowres_768", "lowres_1152", "full")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/docvqa1200.yaml")
    parser.add_argument("--out", default="results/repair_by_rung.json")
    args = parser.parse_args()

    config = load_config(args.config)
    grouped: dict[str, dict] = defaultdict(dict)
    for row in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[row.example_id][row.config_id] = row
    ladder = (CHEAP, *RUNGS)
    ids = [e for e in grouped if all(c in grouped[e] for c in ladder)]
    ok = np.array([[grouped[e][c].correct for c in ladder] for e in ids], bool)

    cheap_ok = ok[:, 0]
    # A query is repairable if any rung above cheap gets it right.
    repairable = (~cheap_ok) & ok[:, 1:].any(axis=1)
    # The cheapest rung that does it.
    first = np.full(len(ids), -1)
    for level in range(len(RUNGS)):
        hit = repairable & (first < 0) & ok[:, level + 1]
        first[hit] = level

    damaged = cheap_ok & (~ok[:, -1])
    counts = {RUNGS[level]: int((first == level).sum()) for level in range(len(RUNGS))}
    total = int(repairable.sum())

    out = {
        "n": len(ids),
        "repairable": total,
        "damaged_by_top": int(damaged.sum()),
        "first_repairing_rung": counts,
        "share": {k: v / total for k, v in counts.items()},
        "share_below_top": sum(counts[r] for r in RUNGS[:-1]) / total,
    }

    print(f"n = {len(ids)} pages; {total} repairable, {int(damaged.sum())} damaged by the top rung\n")
    print(f"{'cheapest rung that repairs':<30}{'count':>7}{'share':>9}")
    for rung, count in counts.items():
        print(f"{rung:<30}{count:>7}{count / total:>9.1%}")
    print(f"\nrepaired below the top rung: {out['share_below_top']:.1%}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
