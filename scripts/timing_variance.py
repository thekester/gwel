"""How much of the per-example cost variation this paper reads is measurement.

Every comparison here is a cost comparison, and the single-domain pilots time
each pass once. We justified that by the accuracy of the affine token-cost
model on bucket medians, which was the wrong justification: a model that
predicts a bucket's median to within 1.7 ms says nothing about the dispersion
of individual passes, and it is the dispersion that a per-example cost-aware
policy selects on. Check CV13 shows what that costs. Ordering the cheapest
tenth of DocVQA by measured escalation latency reports 93 ms spent for +0.517
accuracy; ordering the same queries by predicted visual tokens reports 188 ms
for +0.400. The first is the same policy scored against its own timing noise.

This script quantifies the noise directly from a re-timed subsample
(configs/docvqa_timing.yaml, three repeats after one warmup) and answers three
questions:

  1. what the within-example spread of a rung's latency is, against the
     between-example spread the paper's policies read as signal;
  2. how much of the 18.1% of non-positive first-step latencies survives
     averaging, since a genuine free upgrade survives and noise does not;
  3. whether the escalation price a policy sees is stable enough to rank on.

Usage: PYTHONPATH=src:scripts python scripts/timing_variance.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.oracle.records import deduplicate_records, read_records

LADDER = ("lowres_384", "lowres_768", "lowres_1152", "full")


def collect(config_path: str) -> dict[str, dict[str, object]] | None:
    config = load_config(config_path)
    try:
        records = deduplicate_records(read_records(config.paths.records))
    except FileNotFoundError:
        return None
    grouped: dict[str, dict[str, object]] = defaultdict(dict)
    for row in records:
        grouped[row.example_id][row.config_id] = row
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/docvqa_timing.yaml")
    parser.add_argument("--out", default="results/timing_variance.json")
    args = parser.parse_args()

    grouped = collect(args.config)
    if grouped is None:
        raise SystemExit(f"no records for {args.config}; run scripts/run_oracle.py first")
    ids = [e for e in grouped if all(c in grouped[e] for c in LADDER)]
    if len(ids) < 30:
        raise SystemExit(f"only {len(ids)} complete ladders; the run is not finished")

    out: dict[str, object] = {"n": len(ids), "config": args.config}
    rungs: dict[str, dict[str, float]] = {}

    for rung in LADDER:
        stats = [grouped[e][rung].latency_stats for e in ids]
        if any(s is None for s in stats):
            raise SystemExit(f"{rung} has no latency_stats; the config needs repeats > 1")
        values = [np.asarray(s["values"], float) for s in stats]
        mean = np.array([v.mean() for v in values], float)
        # Sample standard deviation over the repeats of one example, which is
        # the measurement error a policy reading that example's price faces.
        spread = np.array([v.std(ddof=1) for v in values], float)
        rungs[rung] = {
            "mean_latency_ms": float(mean.mean()),
            "within_example_sd_ms": float(spread.mean()),
            "between_example_sd_ms": float(mean.std()),
            # The ratio a per-example policy actually faces: how much of the
            # variation it ranks on is real.
            "signal_to_noise": float(mean.std() / max(spread.mean(), 1e-9)),
        }

    # The first ladder step, which is where CV13 found 18.1% non-positive.
    def averaged(rung: str) -> np.ndarray:
        return np.array(
            [np.mean(grouped[e][rung].latency_stats["values"]) for e in ids], float
        )

    cheap_mean, next_mean = averaged(LADDER[0]), averaged(LADDER[1])
    cheap_once = np.array([grouped[e][LADDER[0]].latency_ms for e in ids], float)
    next_once = np.array([grouped[e][LADDER[1]].latency_ms for e in ids], float)

    # The step is what a policy ranks, and it is a difference of two noisy
    # numbers. Pair the repeats index by index to get its measurement spread.
    def repeats(rung: str) -> np.ndarray:
        return np.array(
            [np.asarray(grouped[e][rung].latency_stats["values"], float) for e in ids]
        )

    step_repeats = repeats(LADDER[1]) - repeats(LADDER[0])
    step_mean = step_repeats.mean(axis=1)
    step_noise = float(step_repeats.std(axis=1, ddof=1).mean())
    tokens_cheap = np.array([grouped[e][LADDER[0]].visual_tokens for e in ids], float)
    tokens_next = np.array([grouped[e][LADDER[1]].visual_tokens for e in ids], float)
    free_by_tokens = (tokens_next - tokens_cheap) <= 0

    out["rungs"] = rungs
    out["first_step"] = {
        "non_positive_share_single_shot": float((next_once - cheap_once <= 0).mean()),
        "non_positive_share_averaged": float((next_mean - cheap_mean <= 0).mean()),
        "median_step_ms": float(np.median(step_mean)),
        "step_noise_sd_ms": step_noise,
        "step_between_sd_ms": float(step_mean.std()),
        "step_signal_to_noise": float(step_mean.std() / max(step_noise, 1e-9)),
        # A step that costs no extra visual tokens is free by construction, not
        # by luck. This is the share the patch grid explains.
        "free_by_token_count_share": float(free_by_tokens.mean()),
        "non_positive_and_free_by_tokens": float(
            ((step_mean <= 0) & free_by_tokens).mean()
        ),
    }

    print(f"re-timed subsample: n={len(ids)} complete ladders, three repeats after one warmup\n")
    print(f"{'rung':<14}{'mean ms':>10}{'within sd':>12}{'between sd':>12}{'ratio':>9}")
    for rung, row in rungs.items():
        print(
            f"{rung:<14}{row['mean_latency_ms']:>10.1f}{row['within_example_sd_ms']:>12.1f}"
            f"{row['between_example_sd_ms']:>12.1f}{row['signal_to_noise']:>9.2f}"
        )
    step = out["first_step"]
    print(
        f"\nfirst ladder step: median {step['median_step_ms']:.1f} ms, "
        f"measurement sd {step['step_noise_sd_ms']:.1f} ms, "
        f"between-example sd {step['step_between_sd_ms']:.1f} ms "
        f"(ratio {step['step_signal_to_noise']:.2f})"
    )
    print(
        f"non-positive on {step['non_positive_share_single_shot']:.1%} of examples from one "
        f"timing and {step['non_positive_share_averaged']:.1%} from three, while "
        f"{step['free_by_token_count_share']:.1%} cost no extra visual tokens at all"
    )
    print(
        "\nA non-positive step that survives averaging is not noise. Where it also\n"
        "costs no extra visual tokens it is the patch grid, and the rung is free."
    )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
