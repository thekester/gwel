"""How much of a hull gap is reachable by reading cost and nothing else.

Proposition 2 says a policy above the randomisation hull carries information
about the pair (cost, correctness). The hull we compare against is built from
the mean cost of each fixed configuration, so a policy that knows only the
per-example escalation price can clear it by escalating the cheap instances,
with no information whatever about which instances would benefit. We observed
exactly that: a free image descriptor at chance on the escalation target clears
on every serving model that prices resolution steeply.

That makes the comparator a relaxation of the floor it should be, and the slack
has to be measured rather than described. This script measures it. For each
corpus-model pair it sweeps a policy that escalates the cheapest queries up to
a budget, reads its gap to the same hull the signal policies are scored
against, and reports the largest gap reachable that way. Every positive gap
elsewhere in the paper should be read against this number: what exceeds it is
attributable to the signal, what falls below it is not.

The cost vector is the one a server can estimate before deciding, since these
models price a pass by its visual-token count, so this is a policy that could
be deployed and not only an analytic device.

Usage: PYTHONPATH=src:scripts python scripts/cost_only_baseline.py
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
RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CHEAP = "lowres_384"

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
    ids = [e for e in grouped if all(c in grouped[e] for c in (CHEAP, *rungs))]
    if len(ids) < 100:
        return None

    top = rungs[-1]
    ok_cheap = np.array([grouped[e][CHEAP].correct for e in ids], float)
    ok_top = np.array([grouped[e][top].correct for e in ids], float)
    cost = {
        c: np.array([grouped[e][c].latency_ms for e in ids], float)
        for c in (CHEAP, *rungs)
    }
    # Predicted cost, not measured: see cost_only_graded.py. Sorting on a
    # single-shot latency selects favourable timing noise rather than cheap
    # queries, and the visual-token count is what a server has in advance.
    tokens = {
        c: np.array([grouped[e][c].visual_tokens for e in ids], float)
        for c in (CHEAP, top)
    }
    extra = tokens[top] - tokens[CHEAP]

    heldout = min(300, max(50, len(ids) // 2))
    rows: dict[float, list[float]] = defaultdict(list)
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(17000 + seed)
        test = rng.permutation(len(ids))[:heldout]
        fixed = [
            (float(cost[c][test].mean()), float(np.mean([grouped[e][c].correct for e in np.array(ids)[test]])))
            for c in (CHEAP, *rungs)
        ]
        # Cheapest escalations first, ties broken at random.
        order = np.argsort(extra[test] + rng.random(len(test)) * 1e-6)
        for rate in RATES:
            fires = np.zeros(len(test), bool)
            fires[order[: int(round(rate * len(test)))]] = True
            accuracy = float(np.where(fires, ok_top[test], ok_cheap[test]).mean())
            latency = float(np.where(fires, cost[top][test], cost[CHEAP][test]).mean())
            hull = hull_accuracy(fixed, latency)
            rows[rate].append(accuracy - (hull if hull is not None else accuracy))

    best_rate = max(rows, key=lambda r: float(np.mean(rows[r])))
    interval = bootstrap_interval(rows[best_rate])
    return {
        "n": len(ids),
        "best_rate": best_rate,
        "gap": [interval.estimate, interval.low, interval.high],
        "gap_vector": [float(v) for v in rows[best_rate]],
        "by_rate": {
            f"{r:.0%}": float(np.mean(v)) for r, v in sorted(rows.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/cost_only.json")
    args = parser.parse_args()

    out = {}
    print(f"{'corpus, serving model':<26}{'n':>6}{'best rate':>11}{'largest gap [95% CI]':>28}")
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
        print(f"{label:<26}{row['n']:>6}{row['best_rate']:>10.0%}{summary:>28}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        "\nThis is the slack in the comparator: the gap a policy reaches by\n"
        "escalating cheap queries with no information about which would benefit.\n"
        "A signal's gap is attributable to the signal only above this number."
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
