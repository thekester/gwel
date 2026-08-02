"""End-to-end policy evaluation: does the probe actually route better?

An AUROC says a signal orders queries well. It does not say a policy built on
it serves better answers per unit of compute. This builds the policy, charges
it honestly, and compares it against the alternatives on the same held-out
examples.

Cost accounting follows the paper's policy-cost model: a query that escalates
pays the probe (or the full cheap pass, for entropy) plus the escalation; a
query that does not escalate pays the complete cheap pass either way, because
that is what produces the answer.

Usage: python scripts/evaluate_probe_policy.py --config configs/pilot1000.yaml
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
from gwel.router.evaluate import bootstrap_interval, paired_difference
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split


def component_costs(path: str = "results/component_latency.json") -> dict[str, float]:
    """Latency of a cheap pass, a probe read, and a full-resolution pass."""
    rows = {r["config"]: r for r in json.loads(Path(path).read_text())}
    cheap, full = rows["longest_384"], rows["longest_1536"]
    layers = 32
    return {
        "cheap": cheap["total_ms"],
        "full": full["total_ms"],
        "probe": cheap["vision_encoder_ms"] + cheap["projector_ms"]
        + cheap["prefill_ms"] * 6 / layers,
    }


def policy_cost(escalates: np.ndarray, costs: dict[str, float], *, read: str) -> np.ndarray:
    """Per-query latency under a given escalation decision.

    ``read`` is the signal the policy conditions on. Reading entropy needs the
    whole cheap pass; reading the probe needs only its prefix, but a query that
    declines to escalate must then finish the cheap pass to produce an answer.
    """
    cheap, full, probe = costs["cheap"], costs["full"], costs["probe"]
    if read == "entropy":
        return cheap + escalates * full
    if read == "probe":
        return np.where(escalates, probe + full, cheap)
    if read == "none":  # fixed policies condition on nothing
        return np.where(escalates, full, cheap)
    raise ValueError(f"unknown read {read!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--activations", default="results/activations_full.npz")
    parser.add_argument("--layer", type=int, default=6)
    args = parser.parse_args()

    config = load_config(args.config)
    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    grouped: dict[str, dict] = defaultdict(dict)
    for record in records:
        grouped[record.example_id][record.config_id] = record

    stored = np.load(args.activations, allow_pickle=True)
    activations, ids = stored["activations"], list(stored["example_ids"])
    usable = [
        e for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    position = {e: i for i, e in enumerate(ids)}

    cheap_ok = np.array([grouped[e]["lowres_384"].correct for e in usable])
    full_ok = np.array([grouped[e]["full"].correct for e in usable])
    entropy = np.array(
        [ConfidenceSignals.from_dict(grouped[e]["lowres_384"].signals).mean_entropy
         for e in usable]
    )
    matrix = activations[[position[e] for e in usable]][:, args.layer, :]

    split = make_split(
        usable, [grouped[e]["lowres_384"].dataset for e in usable],
        val_fraction=config.router.val_fraction,
        test_fraction=config.router.test_fraction, seed=config.router.seed,
    )
    order = {e: i for i, e in enumerate(usable)}
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])

    # The probe is fitted on the escalation-value target, over training-fold
    # failures only, then applied to every test query.
    failures = train[~cheap_ok[train]]
    probe = fit_layer_probe(
        matrix[failures], full_ok[failures].astype(float), args.layer
    )
    probe_score = -probe.score(matrix)  # higher means escalation is worth it

    costs = component_costs()
    print(f"cheap pass {costs['cheap']:.1f} ms | probe read {costs['probe']:.1f} ms | "
          f"full pass {costs['full']:.1f} ms\n")

    def outcome(escalates: np.ndarray) -> np.ndarray:
        return np.where(escalates, full_ok[test], cheap_ok[test])

    def thresholds(scores: np.ndarray, rate: float) -> np.ndarray:
        """Escalate the top `rate` fraction by score, calibrated on train."""
        cut = np.quantile(scores[train], 1.0 - rate)
        return scores[test] >= cut

    rows = []
    for rate in (0.20, 0.30, 0.40):
        for name, scores, read in (
            (f"entropy @{rate:.0%}", entropy, "entropy"),
            (f"probe @{rate:.0%}", probe_score, "probe"),
        ):
            escalates = thresholds(scores, rate)
            rows.append((name, outcome(escalates),
                         policy_cost(escalates, costs, read=read), escalates.mean()))

    for name, always in (("always cheap", False), ("always full", True)):
        escalates = np.full(len(test), always)
        rows.append((name, outcome(escalates),
                     policy_cost(escalates, costs, read="none"), float(always)))

    oracle_escalates = (~cheap_ok[test]) & full_ok[test]
    rows.append(("oracle", outcome(oracle_escalates),
                 policy_cost(oracle_escalates, costs, read="none"),
                 oracle_escalates.mean()))

    print(f"{'policy':<18}{'accuracy [95% CI]':>24}{'latency ms':>14}{'escalated':>11}")
    reference = None
    for name, correct, cost, rate in rows:
        accuracy = bootstrap_interval(correct.astype(float).tolist())
        mean_cost = float(cost.mean())
        if name.startswith("entropy @30"):
            reference = (correct, cost)
        print(f"{name:<18}{str(accuracy):>24}{mean_cost:>14.1f}{rate:>11.0%}")

    if reference is not None:
        print("\npaired against the entropy policy at the same escalation rate:")
        for name, correct, cost, _ in rows:
            if not name.startswith("probe @30"):
                continue
            acc_delta = paired_difference(
                correct.astype(float).tolist(), reference[0].astype(float).tolist()
            )
            cost_delta = paired_difference(cost.tolist(), reference[1].tolist())
            print(f"  accuracy {acc_delta}")
            print(f"  latency  {cost_delta} ms")


if __name__ == "__main__":
    main()
