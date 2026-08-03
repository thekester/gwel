"""Apply family-wise error control to every paired claim the paper makes.

The paper reports a dozen paired differences with bootstrap intervals and, in
its limitations, concedes that no correction is applied. That concession is
cheap to replace with a result: reconstruct each comparison from the run
records, convert its bootstrap to a two-sided p-value, and run Holm's step-down
over the family.

The point is not to inflate the count of surviving claims. It is to say which
ones a reader should believe after accounting for how many questions were asked,
and to name the ones that do not survive rather than leaving them stated at
nominal significance.

Usage: PYTHONPATH=scripts python scripts/correct_multiplicity.py
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
    escalation_delta,
    fit_correctness_rule,
    fit_gain_rule,
    fit_ladder_rule,
    signed_gain,
)
from gwel.router.multiplicity import bootstrap_p_value, family_wise_error, holm_bonferroni
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split

from evaluate_decision_rule import component_costs, load_arrays, policy_cost

MIXTURE = "configs/pilot1000.yaml"
SINGLE = "configs/docvqa1200.yaml"
LAYER = 6
MIN_BUCKET = 50


def mixture_tests() -> list[tuple[str, list[float]]]:
    """Paired difference vectors for the claims made on the four-dataset pilot."""
    config = load_config(MIXTURE)
    cheap_ok, full_ok, entropy, matrix, folds = load_arrays(
        config, "results/activations_full.npz", LAYER
    )
    train, test = folds["train"], folds["test"]
    gains = signed_gain(cheap_ok, full_ok)
    costs = component_costs()
    delta = escalation_delta(costs, read="probe")

    probe = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float), LAYER)
    score = probe.score(matrix)
    tests: list[tuple[str, list[float]]] = []

    # D1: calibrating the gain against calibrating correctness, same signal.
    gain_rule = fit_gain_rule(
        score[train], gains[train], delta_ms=delta, value_ms_per_correct=800.0
    )
    ucci_rule = fit_correctness_rule(
        score[train], cheap_ok[train],
        full_accuracy=float(full_ok[train].mean()),
        delta_ms=delta, value_ms_per_correct=800.0,
    )
    g_fires, u_fires = gain_rule.escalate(score[test]), ucci_rule.escalate(score[test])
    tests.append((
        "D1 latency: gain rule under UCCI rule",
        (policy_cost(g_fires, costs, read="probe")
         - policy_cost(u_fires, costs, read="probe")).tolist(),
    ))

    # Probe policy against the entropy policy at matched escalation rates.
    for rate in (0.20, 0.30, 0.40):
        cut_e = np.quantile(entropy[train], 1 - rate)
        cut_p = np.quantile(score[train], 1 - rate)
        fires_e, fires_p = entropy[test] >= cut_e, score[test] >= cut_p
        tests.append((
            f"probe vs entropy latency @{rate:.0%}",
            (policy_cost(fires_p, costs, read="probe")
             - policy_cost(fires_e, costs, read="entropy")).tolist(),
        ))
        tests.append((
            f"probe vs entropy accuracy @{rate:.0%}",
            (np.where(fires_p, full_ok[test], cheap_ok[test]).astype(float)
             - np.where(fires_e, full_ok[test], cheap_ok[test]).astype(float)).tolist(),
        ))
    return tests


def single_domain_tests() -> list[tuple[str, list[float]]]:
    """Paired vectors for the claims made on the 1200-page DocVQA pilot."""
    config = load_config(SINGLE)
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    cheap, rungs = "lowres_384", ("lowres_768", "lowres_1152", "full")
    ids = [
        e for e in grouped
        if all(c in grouped[e] for c in (cheap, *rungs)) and grouped[e][cheap].signals
    ]
    correct = {c: np.array([grouped[e][c].correct for e in ids]) for c in (cheap, *rungs)}
    tokens = {
        c: np.array([grouped[e][c].visual_tokens for e in ids], float)
        for c in (cheap, *rungs)
    }
    entropy = np.array([float(grouped[e][cheap].signals["mean_entropy"]) for e in ids])

    by = defaultdict(list)
    for e in ids:
        for r in grouped[e].values():
            by[int(r.visual_tokens)].append(r.latency_ms)
    good = [t for t in sorted(by) if len(by[t]) >= MIN_BUCKET]
    model = fit_token_cost(good, [float(np.median(by[t])) for t in good])
    latency = {c: model.predict(tokens[c]) for c in (cheap, *rungs)}
    gains = {r: signed_gain(correct[cheap], correct[r]) for r in rungs}
    deltas = np.column_stack([latency[r] - latency[cheap] for r in rungs])

    tests: list[tuple[str, list[float]]] = []
    # R1: each rung step, as a paired gain vector.
    for low, high in zip((cheap, *rungs)[:-1], rungs):
        vector = (correct[high] & ~correct[low]).astype(float) - (
            correct[low] & ~correct[high]
        ).astype(float)
        tests.append((f"R1 rung gain {low} to {high}", vector.tolist()))

    # R6: ladder against binary, at each operating point.
    rng = np.random.default_rng(5000)
    shuffled = rng.permutation(len(ids))
    test_index, train_index = shuffled[:300], shuffled[300:]
    for value in (800.0, 1600.0, 3200.0):
        fires = fit_gain_rule(
            entropy[train_index], gains["full"][train_index],
            delta_ms=float(deltas[train_index, -1].mean()), value_ms_per_correct=value,
        ).escalate(entropy[test_index])
        binary_ms = np.where(fires, latency["full"][test_index], latency[cheap][test_index])
        chosen = fit_ladder_rule(
            entropy[train_index], {r: gains[r][train_index] for r in rungs},
            value_ms_per_correct=value,
        ).choose(entropy[test_index], deltas[test_index])
        ladder_ms = np.where(chosen < 0, latency[cheap][test_index], 0.0)
        for level, rung in enumerate(rungs):
            ladder_ms = np.where(chosen == level, latency[rung][test_index], ladder_ms)
        tests.append((
            f"R6 ladder vs binary latency V={value:.0f}", (ladder_ms - binary_ms).tolist()
        ))
    return tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="results/multiplicity.json")
    args = parser.parse_args()

    family: list[tuple[str, list[float]]] = []
    for source in (mixture_tests, single_domain_tests):
        try:
            family.extend(source())
        except FileNotFoundError as error:
            print(f"skipping {source.__name__}: missing {error.filename}")

    scored = [(name, bootstrap_p_value(values)) for name, values in family]
    corrected = holm_bonferroni(scored, alpha=args.alpha)

    print(
        f"{len(family)} paired tests. Uncorrected, the chance of at least one "
        f"false positive is {family_wise_error(len(family), alpha=args.alpha):.0%}.\n"
    )
    width = max(len(name) for name, _ in scored) + 2
    print(f"{'test':<{width}}{'p':>10}{'Holm p':>10}{'survives':>10}")
    for row in sorted(corrected, key=lambda r: r.p_value):
        print(
            f"{row.name:<{width}}{row.p_value:>10.4f}{row.adjusted:>10.4f}"
            f"{('yes' if row.survives else 'no'):>10}"
        )

    survivors = [r for r in corrected if r.survives]
    nominal = [r for r in corrected if r.p_value <= args.alpha]
    print(
        f"\n{len(nominal)}/{len(corrected)} clear the nominal level; "
        f"{len(survivors)}/{len(corrected)} survive Holm at alpha={args.alpha}"
    )
    lost = [r.name for r in corrected if r.p_value <= args.alpha and not r.survives]
    if lost:
        print("lost to the correction: " + "; ".join(lost))

    Path(args.out).write_text(
        json.dumps(
            {
                "alpha": args.alpha,
                "tests": [
                    {
                        "name": r.name,
                        "p_value": r.p_value,
                        "adjusted": r.adjusted,
                        "survives": r.survives,
                    }
                    for r in corrected
                ],
                "nominal": len(nominal),
                "survivors": len(survivors),
                "family_wise_error_uncorrected": family_wise_error(
                    len(corrected), alpha=args.alpha
                ),
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
