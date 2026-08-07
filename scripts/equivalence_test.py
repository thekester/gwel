"""An equivalence test for the two calibration targets, not a non-detection.

Table~\\ref{tab:rule} reports an accuracy difference of $0.000$ with a $95\\%$
interval spanning $\\pm 0.025$ and called it "the same answers". That is absence
of evidence read as evidence of absence: a true loss of two accuracy points sits
comfortably inside the interval, and two points would change the trade the table
is selling.

The correct instrument is a two one-sided test. For a paired mean difference,
equivalence at level alpha against a margin delta holds exactly when the
(1 - 2*alpha) interval lies inside (-delta, +delta). So rather than pick a
margin and answer yes or no, we report the smallest margin the data support:
the widest bound of the 90% interval. Any reader with a margin in mind can
compare it against that number.

The comparison is the same one D1 of scripts/correct_multiplicity.py scores for
latency, run here on accuracy: one probe score, one operating point, two
calibration targets, paired per example on the held-out fold.

Usage: PYTHONPATH=src:scripts python scripts/equivalence_test.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.router.decision import (
    escalation_delta,
    fit_correctness_rule,
    fit_gain_rule,
    signed_gain,
)
from gwel.router.probes import fit_layer_probe

from correct_multiplicity import LAYER, MIXTURE
from evaluate_decision_rule import component_costs, load_arrays

RESAMPLES = 10000
VALUE_MS = 800.0


def paired_differences() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-example accuracy differences on the held-out fold, gain minus UCCI."""
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

    gain_rule = fit_gain_rule(
        score[train], gains[train], delta_ms=delta, value_ms_per_correct=VALUE_MS
    )
    ucci_rule = fit_correctness_rule(
        score[train],
        cheap_ok[train],
        full_accuracy=float(full_ok[train].mean()),
        delta_ms=delta,
        value_ms_per_correct=VALUE_MS,
    )
    g_fires = gain_rule.escalate(score[test])
    u_fires = ucci_rule.escalate(score[test])

    accuracy = (
        np.where(g_fires, full_ok[test], cheap_ok[test]).astype(float)
        - np.where(u_fires, full_ok[test], cheap_ok[test]).astype(float)
    )
    disagree = float((g_fires != u_fires).mean())
    return accuracy, g_fires, u_fires, disagree


def tost(differences: np.ndarray, seed: int = 4321) -> dict:
    """Smallest equivalence margin supported at alpha = 0.05.

    Equivalence at alpha against margin delta is exactly the (1 - 2*alpha)
    interval falling inside (-delta, delta), so the smallest supported margin is
    the widest bound of the 90% interval. Reporting it beats reporting a verdict
    against a margin we would have chosen ourselves.
    """
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(RESAMPLES, differences.size), replace=True)
    means = draws.mean(axis=1)
    low90, high90 = (float(v) for v in np.percentile(means, [5.0, 95.0]))
    low95, high95 = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    return {
        "n": int(differences.size),
        "estimate": float(differences.mean()),
        "ci95": [low95, high95],
        "ci90": [low90, high90],
        "smallest_margin": max(abs(low90), abs(high90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/equivalence.json")
    parser.add_argument(
        "--margins",
        type=float,
        nargs="*",
        default=[0.01, 0.02, 0.05],
        help="margins to report a verdict against, for readers who have one",
    )
    args = parser.parse_args()

    accuracy, g_fires, u_fires, disagree = paired_differences()
    result = tost(accuracy)
    result["escalation_rate_gain"] = float(np.mean(g_fires))
    result["escalation_rate_ucci"] = float(np.mean(u_fires))
    result["disagreement_rate"] = disagree
    # Stored so the family-wise correction can score this comparison by the
    # same bootstrap as the rest, rather than from a reported width.
    result["difference_vector"] = [float(v) for v in accuracy]
    result["verdicts"] = {
        f"{m:.3f}": bool(result["smallest_margin"] < m) for m in args.margins
    }

    print(f"paired accuracy difference, gain rule minus UCCI, n={result['n']}")
    print(f"  estimate      {result['estimate']:+.4f}")
    print(f"  95% interval  [{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}]")
    print(f"  90% interval  [{result['ci90'][0]:+.4f}, {result['ci90'][1]:+.4f}]")
    print(f"\nequivalence established at any margin above {result['smallest_margin']:.4f}")
    for margin, ok in result["verdicts"].items():
        print(f"  margin {margin}: {'equivalent' if ok else 'NOT established'}")
    print(
        f"\nThe two rules escalate {result['escalation_rate_gain']:.0%} and "
        f"{result['escalation_rate_ucci']:.0%} of queries and disagree on "
        f"{disagree:.0%} of them, so this is equivalence of aggregate accuracy "
        "and not of answers."
    )

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
