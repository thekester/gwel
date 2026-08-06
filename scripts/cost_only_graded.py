"""The graded version of the cost-only baseline, which is the one that binds.

`cost_only_baseline.py` escalates to the top rung or not at all, while every
gap it is compared against comes from a graded ladder. That makes its number a
lower bound on the comparator's slack, and a lower bound is not what a claim of
attribution needs: our three surviving clearances clear the bound rather than
the quantity. This script closes that.

The policy here is allowed the same action space as ours and still reads no
signal. It buys rung upgrades in order of their per-example price, cheapest
first, until the budget is spent. That is the greedy solution to "how many rung
upgrades can I buy" and it is exactly the strategy Proposition 2's corollary
warns about, now with the graded action space rather than the binary one.

If our surviving margins hold against this, the attribution is sound. If they
do not, they go the way of the two the binary version already took.

Usage: PYTHONPATH=src:scripts python scripts/cost_only_graded.py
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

from baseline_convexity import hull_accuracy

RESAMPLES = 30
CHEAP = "lowres_384"
# Fractions of the total upgrade spend available to the policy.
BUDGETS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

PAIRS = (
    ("configs/docvqa1200.yaml", "DocVQA, SmolVLM-500M", ("lowres_768", "lowres_1152", "full")),
    ("configs/docvqa1200_256m.yaml", "DocVQA, SmolVLM-256M", ("lowres_768", "lowres_1152", "full")),
    ("configs/docvqa1200_qwen2b.yaml", "DocVQA, Qwen2-VL-2B", ("lowres_768", "lowres_1152", "full")),
    ("configs/docvqa1200_llavaov.yaml", "DocVQA, LLaVA-OV-0.5B", ("lowres_768", "full")),
    ("configs/infovqa500.yaml", "InfoVQA, SmolVLM-500M", ("lowres_768", "lowres_1152", "full")),
    ("configs/chartqa500_llavaov.yaml", "ChartQA, LLaVA-OV-0.5B", ("lowres_768", "full")),
    ("configs/docvqa1200_2b.yaml", "DocVQA, SmolVLM2-2.2B", ("lowres_768", "lowres_1152", "full")),
    ("configs/infovqa500_qwen2b.yaml", "InfoVQA, Qwen2-VL-2B", ("lowres_768", "lowres_1152", "full")),
)


def greedy_allocation(step_cost: np.ndarray, budget: float, rng) -> np.ndarray:
    """Rung index per example, buying the cheapest upgrade steps first.

    `step_cost[i, k]` is the extra latency of moving example i from rung k to
    rung k+1. Steps are bought in global price order, and a step is only
    affordable once the ones below it on the same example have been bought,
    which the cumulative check enforces. No correctness enters anywhere.
    """
    n, steps = step_cost.shape
    # Random tie-break: SmolVLM's patch grid gives two distinct first-step
    # prices over 1200 pages, so almost every comparison is a tie and a stable
    # sort would order by example index instead of by price.
    jitter = rng.random(step_cost.shape) * 1e-6
    order = np.dstack(
        np.unravel_index(np.argsort(step_cost + jitter, axis=None), step_cost.shape)
    )[0]
    level = np.zeros(n, int)
    spent = 0.0
    # A cheap step above an expensive one has to wait for it, so we sweep
    # repeatedly until no affordable step is reachable.
    changed = True
    while changed:
        changed = False
        for i, k in order:
            if level[i] != k:
                continue
            price = step_cost[i, k]
            if spent + price > budget:
                continue
            level[i] = k + 1
            spent += price
            changed = True
    return level


def measure(config_path: str, rungs: tuple[str, ...]) -> dict | None:
    config = load_config(config_path)
    grouped: dict[str, dict] = defaultdict(dict)
    try:
        records = rescore_records(
            deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
        )
    except FileNotFoundError:
        return None
    for row in records:
        grouped[row.example_id][row.config_id] = row
    ladder = (CHEAP, *rungs)
    ids = [e for e in grouped if all(c in grouped[e] for c in ladder)]
    if len(ids) < 100:
        return None

    correct = np.array(
        [[grouped[e][c].correct for c in ladder] for e in ids], float
    )
    cost = np.array(
        [[grouped[e][c].latency_ms for c in ladder] for e in ids], float
    )
    # The policy orders by predicted cost, not by measured cost. Visual-token
    # count is a deterministic function of the image and the processor, so a
    # server has it before it decides; measured latency is a single-shot
    # timing, and sorting on it selects the examples whose timing noise ran
    # favourably, which inflates the gap without any policy being responsible.
    tokens = np.array(
        [[grouped[e][c].visual_tokens for c in ladder] for e in ids], float
    )
    step_cost = np.clip(np.diff(tokens, axis=1), 0.0, None)

    heldout = min(300, max(50, len(ids) // 2))
    rows: dict[float, list[float]] = defaultdict(list)
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(23000 + seed)
        test = rng.permutation(len(ids))[:heldout]
        fixed = [
            (float(cost[test, k].mean()), float(correct[test, k].mean()))
            for k in range(len(ladder))
        ]
        full_spend = float(step_cost[test].sum())
        for share in BUDGETS:
            level = greedy_allocation(step_cost[test], share * full_spend, rng)
            picked = np.arange(len(test))
            accuracy = float(correct[test][picked, level].mean())
            latency = float(cost[test][picked, level].mean())
            hull = hull_accuracy(fixed, latency)
            rows[share].append(accuracy - (hull if hull is not None else accuracy))

    best = max(rows, key=lambda r: float(np.mean(rows[r])))
    interval = bootstrap_interval(rows[best])
    return {
        "n": len(ids),
        "best_budget": best,
        "gap": [interval.estimate, interval.low, interval.high],
        "gap_vector": [float(v) for v in rows[best]],
        "by_budget": {f"{b:.0%}": float(np.mean(v)) for b, v in sorted(rows.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/cost_only_graded.json")
    args = parser.parse_args()

    out = {}
    print(f"{'corpus, serving model':<26}{'n':>6}{'budget':>9}{'largest gap [95% CI]':>28}")
    for path, label, rungs in PAIRS:
        if not Path(path).exists():
            continue
        row = measure(path, rungs)
        if row is None:
            print(f"{label:<26}{'skipped: no records':>45}")
            continue
        out[label] = row
        estimate, low, high = row["gap"]
        summary = f"{estimate:+.3f} [{low:+.3f}, {high:+.3f}]"
        print(f"{label:<26}{row['n']:>6}{row['best_budget']:>8.0%}{summary:>28}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        "\nThis is the slack with the graded action space, which is the one our\n"
        "own policies use. A reported gap is attributable to its signal only\n"
        "above this number."
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
