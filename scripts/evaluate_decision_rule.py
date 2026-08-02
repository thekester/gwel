"""Does a cost-derived decision rule beat a tuned escalation rate?

Every policy this project has reported picks an escalation *rate* and thresholds
a score at that quantile. Kotte (2605.18796) argues the rate should not be tuned
at all: calibrate the signal to a probability and let the cost model choose the
threshold. This script implements that, and the correction it needs.

Three rules are compared on identical scores, identical folds and identical
measured latencies:

  rate sweep     the paper's current policy, thresholding at a chosen quantile
  UCCI           calibrate P(cheap wrong), assume escalation delivers a fixed
                 accuracy, escalate where the implied gain clears break-even
  gain rule      calibrate E[G | score] directly, escalate where it clears the
                 same break-even

The UCCI row is the interesting one: its fixed-accuracy assumption is exactly
what non-monotone escalation violates, so it should over-escalate confident
failures that more pixels cannot repair.

Usage: python scripts/evaluate_decision_rule.py --config configs/pilot1000.yaml
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
from gwel.router.calibration import expected_calibration_error, fit_isotonic
from gwel.router.decision import (
    escalation_delta,
    fit_correctness_rule,
    fit_gain_calibrator,
    fit_gain_rule,
    gain_calibration_error,
    signed_gain,
)
from gwel.router.evaluate import bootstrap_interval, paired_difference
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split

# Latency an operator will spend to buy one additional correct answer. Swept to
# trace the frontier; the mid value is used for the headline table.
VALUE_GRID = (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
HEADLINE_VALUE = 800.0


def component_costs(path: str = "results/component_latency.json") -> dict[str, float]:
    rows = {r["config"]: r for r in json.loads(Path(path).read_text())}
    cheap, full = rows["longest_384"], rows["longest_1536"]
    layers = 32
    return {
        "cheap": cheap["total_ms"],
        "full": full["total_ms"],
        "probe": cheap["vision_encoder_ms"]
        + cheap["projector_ms"]
        + cheap["prefill_ms"] * 6 / layers,
    }


def policy_cost(escalates: np.ndarray, costs: dict[str, float], *, read: str) -> np.ndarray:
    cheap, full, probe = costs["cheap"], costs["full"], costs["probe"]
    if read == "entropy":
        return cheap + escalates * full
    if read == "probe":
        return np.where(escalates, probe + full, cheap)
    if read == "none":
        return np.where(escalates, full, cheap)
    raise ValueError(f"unknown read {read!r}")


def load_arrays(config, activations_path: str, layer: int):
    """Per-example correctness, entropy and probe score on the held-out fold."""
    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    grouped: dict[str, dict] = defaultdict(dict)
    for record in records:
        grouped[record.example_id][record.config_id] = record

    stored = np.load(activations_path, allow_pickle=True)
    activations, ids = stored["activations"], list(stored["example_ids"])
    usable = [
        e
        for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    position = {e: i for i, e in enumerate(ids)}

    cheap_ok = np.array([grouped[e]["lowres_384"].correct for e in usable])
    full_ok = np.array([grouped[e]["full"].correct for e in usable])
    entropy = np.array(
        [
            ConfidenceSignals.from_dict(grouped[e]["lowres_384"].signals).mean_entropy
            for e in usable
        ]
    )
    matrix = activations[[position[e] for e in usable]][:, layer, :]

    split = make_split(
        usable,
        [grouped[e]["lowres_384"].dataset for e in usable],
        val_fraction=config.router.val_fraction,
        test_fraction=config.router.test_fraction,
        seed=config.router.seed,
    )
    order = {e: i for i, e in enumerate(usable)}
    folds = {
        name: np.array([order[e] for e in getattr(split, name)])
        for name in ("train", "val", "test")
    }
    return cheap_ok, full_ok, entropy, matrix, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--activations", default="results/activations_full.npz")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--out", default="results/decision_rule.json")
    args = parser.parse_args()

    config = load_config(args.config)
    cheap_ok, full_ok, entropy, matrix, folds = load_arrays(
        config, args.activations, args.layer
    )
    train, test = folds["train"], folds["test"]
    gains = signed_gain(cheap_ok, full_ok)

    # The probe is fitted on the joint target, which the paper already
    # establishes is the one a policy needs; calibration reuses the same fold.
    probe = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float), args.layer)
    probe_score = probe.score(matrix)

    costs = component_costs()
    delta = {read: escalation_delta(costs, read=read) for read in ("entropy", "probe")}
    print(
        f"cheap {costs['cheap']:.1f} ms | probe {costs['probe']:.1f} ms | "
        f"full {costs['full']:.1f} ms"
    )
    print(
        f"escalation delta: entropy {delta['entropy']:.1f} ms, "
        f"probe {delta['probe']:.1f} ms "
        f"({delta['probe'] / delta['entropy']:.2f}x)\n"
    )

    signals = {"entropy": (entropy, "entropy"), "probe": (probe_score, "probe")}
    results: dict[str, object] = {
        "layer": args.layer,
        "costs": costs,
        "delta_ms": delta,
        "headline_value": HEADLINE_VALUE,
    }

    # --- calibration quality -------------------------------------------------
    calibration_rows = []
    for name, (score, _) in signals.items():
        gain_cal = fit_gain_calibrator(score[train], gains[train])
        predicted = gain_cal.predict(score[test])
        gce = gain_calibration_error(predicted, gains[test])
        # UCCI's own target, for reference: P(cheap wrong).
        err_cal = fit_isotonic(score[train], 1.0 - cheap_ok[train].astype(float))
        ece = expected_calibration_error(
            err_cal.predict(score[test]), 1.0 - cheap_ok[test].astype(float)
        )
        calibration_rows.append((name, gce, ece))
    print(f"{'signal':<10}{'gain calib. error':>20}{'correctness ECE':>18}")
    for name, gce, ece in calibration_rows:
        print(f"{name:<10}{gce:>20.3f}{ece:>18.3f}")
    results["calibration"] = [
        {"signal": n, "gain_calibration_error": g, "correctness_ece": e}
        for n, g, e in calibration_rows
    ]

    # --- the two rules, swept over operator value ----------------------------
    print(f"\n{'rule':<22}{'V (ms/correct)':>16}{'tau':>8}{'escalates':>11}"
          f"{'accuracy':>10}{'latency':>10}")
    sweep = []
    full_accuracy = float(full_ok[train].mean())
    for value in VALUE_GRID:
        for name, (score, read) in signals.items():
            gain_rule = fit_gain_rule(
                score[train], gains[train], delta_ms=delta[read], value_ms_per_correct=value
            )
            ucci_rule = fit_correctness_rule(
                score[train],
                cheap_ok[train],
                full_accuracy=full_accuracy,
                delta_ms=delta[read],
                value_ms_per_correct=value,
            )
            for label, rule in (("gain", gain_rule), ("UCCI", ucci_rule)):
                fires = rule.escalate(score[test])
                accuracy = float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
                latency = float(policy_cost(fires, costs, read=read).mean())
                row = {
                    "rule": label,
                    "signal": name,
                    "value_ms_per_correct": value,
                    "tau": float(rule.tau),
                    "escalation_rate": float(fires.mean()),
                    "accuracy": accuracy,
                    "latency_ms": latency,
                }
                sweep.append(row)
                print(
                    f"{label + ' / ' + name:<22}{value:>16.0f}{rule.tau:>8.3f}"
                    f"{fires.mean():>11.0%}{accuracy:>10.3f}{latency:>10.1f}"
                )
    results["sweep"] = sweep

    # --- headline comparison at one operator preference ----------------------
    print(f"\nat V = {HEADLINE_VALUE:.0f} ms per correct answer, paired against "
          f"the UCCI rule on the same signal:")
    for name, (score, read) in signals.items():
        gain_rule = fit_gain_rule(
            score[train], gains[train], delta_ms=delta[read], value_ms_per_correct=HEADLINE_VALUE
        )
        ucci_rule = fit_correctness_rule(
            score[train],
            cheap_ok[train],
            full_accuracy=full_accuracy,
            delta_ms=delta[read],
            value_ms_per_correct=HEADLINE_VALUE,
        )
        gain_fires, ucci_fires = gain_rule.escalate(score[test]), ucci_rule.escalate(score[test])
        gain_ok = np.where(gain_fires, full_ok[test], cheap_ok[test]).astype(float)
        ucci_ok = np.where(ucci_fires, full_ok[test], cheap_ok[test]).astype(float)
        gain_cost = policy_cost(gain_fires, costs, read=read)
        ucci_cost = policy_cost(ucci_fires, costs, read=read)
        print(f"  {name}: accuracy {paired_difference(gain_ok.tolist(), ucci_ok.tolist())}")
        print(f"  {name}: latency  {paired_difference(gain_cost.tolist(), ucci_cost.tolist())} ms")
        print(
            f"  {name}: escalates {gain_fires.mean():.0%} (gain) "
            f"vs {ucci_fires.mean():.0%} (UCCI)"
        )
        print(f"  {name}: accuracy of gain rule {bootstrap_interval(gain_ok.tolist())}")

    # --- does an untuned rule land where the sweep would have put it? --------
    # The rate sweep is the policy the paper currently reports. If the
    # cost-derived rule is worth anything, its self-selected operating points
    # should sit on that frontier rather than inside it.
    print("\nagainst the tuned rate sweep on the same fold (probe signal):")
    score, read = signals["probe"]
    swept = []
    for rate in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        cut = np.quantile(score[train], 1.0 - rate)
        fires = score[test] >= cut
        swept.append(
            (
                float(np.where(fires, full_ok[test], cheap_ok[test]).mean()),
                float(policy_cost(fires, costs, read=read).mean()),
            )
        )
    results["swept_frontier"] = [{"accuracy": a, "latency_ms": c} for a, c in swept]

    print(f"{'V':>8}{'accuracy':>10}{'latency':>10}{'dominated by sweep?':>22}")
    dominated_count = 0
    for value in VALUE_GRID:
        rule = fit_gain_rule(
            score[train], gains[train], delta_ms=delta[read], value_ms_per_correct=value
        )
        fires = rule.escalate(score[test])
        accuracy = float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
        latency = float(policy_cost(fires, costs, read=read).mean())
        # Dominated means some swept point is at least as good on both axes,
        # and strictly better on one.
        dominated = any(
            a >= accuracy - 1e-9 and c <= latency + 1e-9 and (a > accuracy or c < latency)
            for a, c in swept
        )
        dominated_count += dominated
        print(f"{value:>8.0f}{accuracy:>10.3f}{latency:>10.1f}{str(dominated):>22}")
    results["dominated_by_sweep"] = dominated_count
    print(
        f"{dominated_count}/{len(VALUE_GRID)} self-selected points are dominated by "
        f"the tuned sweep"
    )

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
