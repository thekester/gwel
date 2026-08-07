"""Is the cost-oracle comparator real, or an artefact of timing once?

The paper reports two readings of its own comparator's slack. The deployable
one orders queries by predicted visual-token count and reaches at most +0.008.
The oracle one orders by the realised escalation latency, which no server can
know in advance, and reached up to +0.053. That second figure came from
single-shot timings, and the paper marks it as not recomputed on the re-timed
subsample. This closes that marker.

The re-timed run (configs/docvqa_timing.yaml) has three repeats per pass after
a discarded warmup, so the same policy can be scored twice: once ordering by a
single timing, once ordering by the mean of three. If the oracle slack is real
cost information the two agree; if it was noise harvesting the first is larger.

The subsample is one corpus-model pair and 150 pages, so this bounds the
question on the pair we can check rather than settling it everywhere. That is
stated rather than glossed.

Usage: PYTHONPATH=src:scripts python scripts/oracle_slack_retimed.py
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

CHEAP = "lowres_384"
RUNGS = ("lowres_768", "lowres_1152", "full")
LADDER = (CHEAP, *RUNGS)
RESAMPLES = 40
RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def sweep(cost_seen: np.ndarray, cost_paid: np.ndarray, correct: np.ndarray,
          heldout: int, seed_base: int) -> dict:
    """Escalate the cheapest queries by `cost_seen`, pay and score by `cost_paid`.

    Separating the two is the whole point: an oracle over cost decides using a
    quantity it could not have, and is still charged what the pass really cost.
    """
    rows: dict[float, list[float]] = defaultdict(list)
    extra = cost_seen[:, -1] - cost_seen[:, 0]
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(seed_base + seed)
        test = rng.permutation(len(correct))[:heldout]
        fixed = [
            (float(cost_paid[test, k].mean()), float(correct[test, k].mean()))
            for k in range(cost_paid.shape[1])
        ]
        order = np.argsort(extra[test] + rng.random(len(test)) * 1e-9)
        for rate in RATES:
            fires = np.zeros(len(test), bool)
            fires[order[: int(round(rate * len(test)))]] = True
            accuracy = float(
                np.where(fires, correct[test, -1], correct[test, 0]).mean()
            )
            latency = float(
                np.where(fires, cost_paid[test, -1], cost_paid[test, 0]).mean()
            )
            hull = hull_accuracy(fixed, latency)
            rows[rate].append(accuracy - (hull if hull is not None else accuracy))

    best = max(rows, key=lambda r: float(np.mean(rows[r])))
    interval = bootstrap_interval(rows[best])
    return {
        "best_rate": best,
        "gap": [interval.estimate, interval.low, interval.high],
        "gap_vector": [float(v) for v in rows[best]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/docvqa_timing.yaml")
    parser.add_argument("--out", default="results/oracle_slack_retimed.json")
    args = parser.parse_args()

    config = load_config(args.config)
    grouped: dict[str, dict] = defaultdict(dict)
    for row in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[row.example_id][row.config_id] = row
    ids = [e for e in grouped if all(c in grouped[e] for c in LADDER)]
    if len(ids) < 60:
        raise SystemExit(f"only {len(ids)} complete ladders")
    if any(grouped[e][c].latency_stats is None for e in ids for c in LADDER):
        raise SystemExit("this config must be run with repeats > 1")

    correct = np.array([[grouped[e][c].correct for c in LADDER] for e in ids], float)
    tokens = np.array([[grouped[e][c].visual_tokens for c in LADDER] for e in ids], float)
    single = np.array([[grouped[e][c].latency_ms for c in LADDER] for e in ids], float)
    averaged = np.array(
        [[np.mean(grouped[e][c].latency_stats["values"]) for c in LADDER] for e in ids],
        float,
    )
    heldout = max(60, len(ids) // 2)

    # All three policies are charged the averaged cost, which is our best
    # estimate of what a pass really takes. Only what they may *read* differs.
    out = {
        "n": len(ids),
        "heldout": heldout,
        "oracle_single_shot": sweep(single, averaged, correct, heldout, 31000),
        "oracle_averaged": sweep(averaged, averaged, correct, heldout, 32000),
        "deployable_tokens": sweep(tokens, averaged, correct, heldout, 33000),
    }
    out["noise_share"] = (
        out["oracle_single_shot"]["gap"][0] - out["oracle_averaged"]["gap"][0]
    )

    print(f"DocVQA / SmolVLM-500M, {len(ids)} re-timed pages, held-out {heldout}\n")
    print(f"{'policy orders by':<34}{'rate':>7}{'gap to hull [95% CI]':>28}")
    for key, label in (
        ("oracle_single_shot", "measured latency, one timing"),
        ("oracle_averaged", "measured latency, mean of three"),
        ("deployable_tokens", "predicted visual tokens"),
    ):
        row = out[key]
        estimate, low, high = row["gap"]
        summary = f"{estimate:+.3f} [{low:+.3f}, {high:+.3f}]"
        print(f"{label:<34}{row['best_rate']:>6.0%}{summary:>28}")

    print(
        f"\nOrdering by one timing rather than three is worth "
        f"{out['noise_share']:+.3f} of gap.\nWhat remains above the deployable row is "
        "cost information a server cannot have."
    )
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
