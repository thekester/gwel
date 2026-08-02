"""Abstention as a third action: what is it worth, on each axis?

The paper defines three routing targets and uses two. $Q_3$, is the query
answerable at all?, separates queries no visual operation can rescue,
and $30.6\\%$ of the pilot falls there. The obvious move is a third action: answer,
escalate, or decline.

Two literatures make opposite promises about it. Selective prediction
(2402.15610, 2604.14799) treats abstention as the way to hold risk down under a
coverage constraint. Efficiency work treats every escalation as spend, so
declining a hopeless query should be free money. This script measures both.

  coverage axis   how much coverage does abstaining buy at matched risk,
                  against spending the same budget on escalation instead?
  budget axis     of the compute an escalation policy spends, how much goes to
                  queries that no configuration can answer: and what would a
                  perfect abstention gate recover?

Usage: PYTHONPATH=scripts python scripts/evaluate_abstention.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.modeling.signals import ConfidenceSignals
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.conformal import evaluate_three_way, fit_three_way
from gwel.router.coverage import escalating_frontier, selective_frontier
from gwel.router.decision import escalation_delta, fit_gain_rule, signed_gain
from gwel.router.evaluate import bootstrap_interval
from gwel.router.probes import fit_layer_probe

from evaluate_decision_rule import component_costs, load_arrays, policy_cost

VALUE_GRID = (400.0, 800.0, 1600.0)


def solvability(config, example_ids: list[str]) -> np.ndarray:
    """Whether *any* routable configuration answered the query correctly."""
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    return np.array(
        [
            any(
                record.correct
                for config_id, record in grouped[example].items()
                if config_id != "no_image"
            )
            for example in example_ids
        ]
    ), grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--activations", default="results/activations_full.npz")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--out", default="results/abstention.json")
    args = parser.parse_args()

    config = load_config(args.config)
    cheap_ok, full_ok, entropy, matrix, folds = load_arrays(
        config, args.activations, args.layer
    )
    train, test = folds["train"], folds["test"]
    gains = signed_gain(cheap_ok, full_ok)

    stored = np.load(args.activations, allow_pickle=True)
    ids = [str(e) for e in stored["example_ids"]]
    grouped_probe: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped_probe[record.example_id][record.config_id] = record
    usable = [
        e
        for e in ids
        if "lowres_384" in grouped_probe[e]
        and "full" in grouped_probe[e]
        and grouped_probe[e]["lowres_384"].signals
    ]
    solvable, grouped = solvability(config, usable)

    results: dict[str, object] = {}
    print(f"unsolvable at every configuration: {1 - solvable.mean():.1%} of the pilot")
    print(f"                    on the test fold: {1 - solvable[test].mean():.1%}\n")

    # --- budget axis --------------------------------------------------------
    costs = component_costs()
    delta = escalation_delta(costs, read="probe")
    probe = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float), args.layer)
    score = probe.score(matrix)
    base_rate = float(1 - solvable[test].mean())

    print("where does the escalation budget go?")
    print(
        f"{'V':>7}{'escalates':>11}{'unsolvable':>12}{'vs base':>10}"
        f"{'zero gain':>11}{'gate saves':>12}"
    )
    budget_rows = []
    for value in VALUE_GRID:
        rule = fit_gain_rule(
            score[train], gains[train], delta_ms=delta, value_ms_per_correct=value
        )
        fires = rule.escalate(score[test])
        if not fires.any():
            continue
        hopeless = float((~solvable[test][fires]).mean())
        zero_gain = float((gains[test][fires] == 0).mean())
        # A perfect gate declines to escalate anything unsolvable. It cannot
        # lose accuracy, because those queries are wrong under every action.
        gated = fires & solvable[test]
        cost_now = float(policy_cost(fires, costs, read="probe").mean())
        cost_gated = float(policy_cost(gated, costs, read="probe").mean())
        accuracy_now = float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
        accuracy_gated = float(np.where(gated, full_ok[test], cheap_ok[test]).mean())
        saving = (cost_now - cost_gated) / cost_now
        budget_rows.append(
            {
                "value_ms_per_correct": value,
                "escalation_rate": float(fires.mean()),
                "unsolvable_share": hopeless,
                "base_rate": base_rate,
                "zero_gain_share": zero_gain,
                "cost_ms": cost_now,
                "gated_cost_ms": cost_gated,
                "gate_saving": saving,
                "accuracy": accuracy_now,
                "gated_accuracy": accuracy_gated,
            }
        )
        print(
            f"{value:>7.0f}{fires.mean():>11.0%}{hopeless:>12.0%}"
            f"{hopeless - base_rate:>+10.0%}{zero_gain:>11.0%}{saving:>12.1%}"
        )
    results["budget"] = budget_rows

    print(
        "\nthe rule escalates unsolvable queries *below* their base rate, so a "
        "perfect\nabstention gate recovers only the margin; accuracy is unchanged "
        "by construction."
    )

    # --- coverage axis ------------------------------------------------------
    def confidence(example: str, config_id: str) -> float:
        return -ConfidenceSignals.from_dict(grouped[example][config_id].signals).mean_entropy

    with_signals = [
        e for e in usable if grouped[e]["lowres_384"].signals and grouped[e]["full"].signals
    ]
    cheap_c = [confidence(e, "lowres_384") for e in with_signals]
    cheap_correct = [grouped[e]["lowres_384"].correct for e in with_signals]
    esc_c = [confidence(e, "full") for e in with_signals]
    esc_correct = [grouped[e]["full"].correct for e in with_signals]

    vanilla = selective_frontier(cheap_c, cheap_correct, cheap_cost=costs["cheap"])
    escalating = escalating_frontier(
        cheap_c,
        cheap_correct,
        esc_c,
        esc_correct,
        cheap_cost=costs["cheap"],
        escalated_cost=costs["full"],
    )
    print(f"\n{'risk <=':>9}{'abstain only':>15}{'or escalate':>14}{'gain':>8}{'cost':>10}")
    coverage_rows = []
    for tolerance in (0.30, 0.40, 0.50):
        a = vanilla.at_risk(tolerance)
        b = escalating.at_risk(tolerance)
        if b is None:
            continue
        coverage_rows.append(
            {
                "risk": tolerance,
                "abstain_coverage": a.coverage if a else 0.0,
                "escalate_coverage": b.coverage,
                "cost_ms": b.cost,
                "escalation_rate": b.escalation_rate,
            }
        )
        shown = f"{a.coverage:.0%}" if a else "unreachable"
        print(
            f"{tolerance:>9.0%}{shown:>15}{b.coverage:>14.0%}"
            f"{b.coverage - (a.coverage if a else 0.0):>+8.0%}{b.cost:>10.1f}"
        )
    results["coverage"] = coverage_rows

    # --- a calibrated three-way policy --------------------------------------
    # Nonconformity is the negated confidence, so a high score means the cheap
    # pass is unreliable; the middle regime escalates rather than hedging.
    order = {e: i for i, e in enumerate(usable)}
    calib = np.array([order[e] for e in usable if order[e] in set(train.tolist())])
    scores = -np.asarray(cheap_c if len(cheap_c) == len(usable) else [0.0])
    if len(cheap_c) == len(usable):
        print(f"\n{'answer a':>9}{'abstain a':>11}{'answer':>9}{'escalate':>10}"
              f"{'abstain':>9}{'coverage':>10}{'risk':>7}")
        three_way_rows = []
        for answer_alpha, abstain_alpha in ((0.70, 0.30), (0.60, 0.20), (0.50, 0.10)):
            policy = fit_three_way(
                scores[calib].tolist(),
                answer_alpha=answer_alpha,
                abstain_alpha=abstain_alpha,
            )
            summary = evaluate_three_way(
                scores[test].tolist(),
                [bool(x) for x in cheap_ok[test]],
                [bool(x) for x in full_ok[test]],
                policy,
            )
            three_way_rows.append({"answer_alpha": answer_alpha,
                                   "abstain_alpha": abstain_alpha, **summary})
            print(
                f"{answer_alpha:>9.2f}{abstain_alpha:>11.2f}"
                f"{summary['answer_rate']:>9.0%}{summary['escalation_rate']:>10.0%}"
                f"{summary['abstention_rate']:>9.0%}{summary['coverage']:>10.0%}"
                f"{summary['risk']:>7.0%}"
            )
        results["three_way"] = three_way_rows

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
