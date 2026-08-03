"""One ablation table, every component measured on the same footing.

The paper's components are each justified where they are introduced, across five
sections, on folds and cost models that were not always the same. That is how a
paper acquires an inconsistency it cannot see. This recomputes every one of them
against a single reference policy, on one fold, with one cost model, so the
column is comparable down its length and a reader can ask "what does this part
buy?" and get an answer in the same units throughout.

The reference is the paper's recommended configuration: a joint-target signal,
isotonically calibrated to the expected gain, thresholded at the cost-implied
break-even, over the full resolution ladder, with escalation priced per example.
Each row removes exactly one of those and reports the damage.

Usage: PYTHONPATH=scripts python scripts/ablate_policy.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.oracle.token_cost import fit_token_cost
from gwel.router.decision import (
    fit_correctness_rule,
    fit_gain_rule,
    fit_ladder_rule,
    fit_per_query_gain_rule,
    signed_gain,
)
from gwel.router.evaluate import bootstrap_interval

CHEAP = "lowres_384"
RUNGS = ("lowres_768", "lowres_1152", "full")
RESAMPLES = 30
HELDOUT = 300
# Each variant is swept over the operator preference so it produces a frontier,
# not a point. Comparing variants at one fixed V would compare policies at
# different costs, and any variant that escalates more would look "better" for
# no reason other than spending more, which is the error this paper spends a
# section warning about.
VALUE_GRID = (200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0)
BUDGETS = (400.0, 500.0)  # latencies at which variants are compared
MIN_BUCKET = 50


def load(config_path: str):
    config = load_config(config_path)
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    ids = [
        e
        for e in grouped
        if all(c in grouped[e] for c in (CHEAP, *RUNGS)) and grouped[e][CHEAP].signals
    ]
    correct = {c: np.array([grouped[e][c].correct for e in ids]) for c in (CHEAP, *RUNGS)}
    tokens = {
        c: np.array([grouped[e][c].visual_tokens for e in ids], float)
        for c in (CHEAP, *RUNGS)
    }
    entropy = np.array([float(grouped[e][CHEAP].signals["mean_entropy"]) for e in ids])

    by: dict[int, list[float]] = defaultdict(list)
    for e in ids:
        for record in grouped[e].values():
            by[int(record.visual_tokens)].append(record.latency_ms)
    good = [t for t in sorted(by) if len(by[t]) >= MIN_BUCKET]
    model = fit_token_cost(good, [float(np.median(by[t])) for t in good])
    latency = {c: model.predict(tokens[c]) for c in (CHEAP, *RUNGS)}
    return ids, correct, tokens, entropy, latency, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/docvqa1200.yaml")
    parser.add_argument("--out", default="results/ablation.json")
    args = parser.parse_args()

    ids, correct, tokens, entropy, latency, model = load(args.config)
    gains = {r: signed_gain(correct[CHEAP], correct[r]) for r in RUNGS}
    deltas = np.column_stack([latency[r] - latency[CHEAP] for r in RUNGS])
    n = len(ids)
    print(
        f"n={n}, one fold convention, one cost model "
        f"({model.base_ms:.1f} + {model.slope_ms_per_token:.3f}/token), "
        f"each variant swept over {len(VALUE_GRID)} preferences\n"
    )

    def outcome(chosen: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Accuracy and latency vectors for a rung choice (-1 means answer cheap)."""
        acc = np.where(chosen < 0, correct[CHEAP][index], False)
        cost = np.where(chosen < 0, latency[CHEAP][index], 0.0)
        for level, rung in enumerate(RUNGS):
            mask = chosen == level
            acc = np.where(mask, correct[rung][index], acc)
            cost = np.where(mask, latency[rung][index], cost)
        return acc.astype(float), cost

    # Each variant is a function of the preference V, so it traces a frontier.
    frontiers: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(7000 + seed)
        shuffled = rng.permutation(n)
        test, train = shuffled[:HELDOUT], shuffled[HELDOUT:]
        if any(len(set((gains[r][train] > 0).tolist())) < 2 for r in RUNGS):
            continue
        mean_delta = float(deltas[train, -1].mean())
        flat_deltas = np.full_like(deltas, deltas[train].mean(axis=0))
        points: dict[str, list[tuple[float, float]]] = defaultdict(list)

        def record(name: str, chosen: np.ndarray) -> None:
            acc, cost = outcome(chosen, test)
            points[name].append((float(cost.mean()), float(acc.mean())))

        for value in VALUE_GRID:
            ladder = fit_ladder_rule(
                entropy[train], {r: gains[r][train] for r in RUNGS},
                value_ms_per_correct=value,
            )
            record("reference", ladder.choose(entropy[test], deltas[test]))
            record("no per-example pricing", ladder.choose(entropy[test], flat_deltas[test]))

            binary = fit_gain_rule(
                entropy[train], gains["full"][train],
                delta_ms=mean_delta, value_ms_per_correct=value,
            ).escalate(entropy[test])
            record("no ladder (binary)", np.where(binary, len(RUNGS) - 1, -1))

            ucci = fit_correctness_rule(
                entropy[train], correct[CHEAP][train],
                full_accuracy=float(correct["full"][train].mean()),
                delta_ms=mean_delta, value_ms_per_correct=value,
            ).escalate(entropy[test])
            record("no gain target (UCCI)", np.where(ucci, len(RUNGS) - 1, -1))

        # The uncalibrated alternative has no V; it has a rate, swept instead.
        for rate in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            if rate == 0.0:
                tuned = np.zeros(len(test), dtype=bool)
            else:
                cut = np.quantile(entropy[train], 1.0 - rate)
                tuned = entropy[test] >= cut
            record("no calibration (tuned rate)", np.where(tuned, len(RUNGS) - 1, -1))

        # No signal at all: the two endpoints a policy must beat between.
        record("no signal", np.full(len(test), -1))
        record("no signal", np.full(len(test), len(RUNGS) - 1))

        for name, curve in points.items():
            frontiers[name].append(curve)

    def accuracy_at(curve: list[tuple[float, float]], budget: float) -> float | None:
        """Best accuracy any operating point reaches within a latency budget."""
        affordable = [a for c, a in curve if c <= budget + 1e-9]
        return max(affordable) if affordable else None

    print(f"{'variant':<28}" + "".join(f"{'acc @' + str(int(b)) + 'ms':>18}" for b in BUDGETS))
    rows = []
    reference_curves = frontiers["reference"]
    for name, curves in frontiers.items():
        entry: dict[str, object] = {"variant": name}
        cells = []
        for budget in BUDGETS:
            paired = [
                (accuracy_at(c, budget), accuracy_at(r, budget))
                for c, r in zip(curves, reference_curves)
            ]
            usable = [(a, b) for a, b in paired if a is not None and b is not None]
            if not usable:
                cells.append("unreachable")
                entry[f"acc_at_{int(budget)}"] = None
                continue
            value = float(np.mean([a for a, _ in usable]))
            entry[f"acc_at_{int(budget)}"] = value
            if name == "reference":
                cells.append(f"{value:.3f}")
            else:
                delta = bootstrap_interval([a - b for a, b in usable])
                entry[f"delta_at_{int(budget)}"] = [delta.estimate, delta.low, delta.high]
                cells.append(f"{value:.3f} ({delta.estimate:+.3f})")
        rows.append(entry)
        print(f"{name:<28}" + "".join(f"{c:>18}" for c in cells))

    print()
    print(
        "Each variant is swept to a frontier and read at a fixed budget, so a"
        " variant that merely escalates more cannot look better for that reason."
    )

    Path(args.out).write_text(
        json.dumps(
            {
                "n": n,
                "value_grid": list(VALUE_GRID),
                "budgets": list(BUDGETS),
                "resamples": RESAMPLES,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
