"""Re-derive the headline policy numbers with a per-example cost model.

The paper prices an escalated query at a flat $206.0$ ms, taken from profiling
one image under the ``longest_1536`` configuration. That image's longest side
was below $1536$ px, so the processor capped it and the pass spent $320$ visual
tokens. On the pilot, ``full`` averages $424$ tokens and exceeds the $768$
configuration on $44\\%$ of examples. The flat cost therefore prices an
escalation that did not happen.

This recomputes every policy comparison charging each example for the tokens it
actually spent, and reports what moves. The interesting part is what does *not*
move: the probe's saving is a difference between two cheap-side costs, so it is
algebraically independent of the escalation price, while every percentage
expressed against a total is not.

Usage: PYTHONPATH=scripts python scripts/recost_policies.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.oracle.token_cost import extrapolation_span, fit_token_cost
from gwel.router.decision import signed_gain
from gwel.router.evaluate import paired_difference
from gwel.router.probes import fit_layer_probe

from evaluate_decision_rule import component_costs, load_arrays

RATES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
PROBE_LAYER, LAYERS = 6, 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--activations", default="results/activations_full.npz")
    parser.add_argument("--latency", default="results/component_latency.json")
    parser.add_argument("--out", default="results/recost.json")
    args = parser.parse_args()

    profile = json.loads(Path(args.latency).read_text())
    model = fit_token_cost(
        [r["visual_tokens"] for r in profile], [r["total_ms"] for r in profile]
    )
    print(
        f"latency model: {model.base_ms:.1f} + {model.slope_ms_per_token:.3f} ms/token "
        f"over {model.points} profiled configs, worst residual {model.residual_ms:.1f} ms"
    )

    config = load_config(args.config)
    cheap_ok, full_ok, entropy, matrix, folds = load_arrays(
        config, args.activations, args.layer if hasattr(args, "layer") else PROBE_LAYER
    )
    train, test = folds["train"], folds["test"]
    gains = signed_gain(cheap_ok, full_ok)

    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    stored = np.load(args.activations, allow_pickle=True)
    ids = [str(e) for e in stored["example_ids"]]
    usable = [
        e
        for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    full_tokens = np.array([grouped[e]["full"].visual_tokens for e in usable])
    cheap_tokens = np.array([grouped[e]["lowres_384"].visual_tokens for e in usable])

    span = extrapolation_span(
        model, [r["visual_tokens"] for r in profile], full_tokens.tolist()
    )
    print(
        f"full pass on the pilot: {full_tokens.mean():.0f} tokens on average, "
        f"max {full_tokens.max()}, against {max(r['visual_tokens'] for r in profile)} profiled "
        f"({span:.0%} beyond the fitted range)\n"
    )

    flat = component_costs(args.latency)
    per_example_full = model.predict(full_tokens)
    per_example_cheap = model.predict(cheap_tokens)
    print(
        f"escalation price: flat {flat['full']:.1f} ms vs "
        f"per-example mean {per_example_full.mean():.1f} ms "
        f"({per_example_full.mean() / flat['full'] - 1:+.0%})"
    )
    # The probe read is a prefix of the cheap pass, so it scales with the cheap
    # pass's tokens, not the escalation's.
    probe_ms = flat["probe"]

    def costs(escalates: np.ndarray, index: np.ndarray, *, read: str, flat_cost: bool):
        cheap = np.full(len(index), flat["cheap"]) if flat_cost else per_example_cheap[index]
        full = np.full(len(index), flat["full"]) if flat_cost else per_example_full[index]
        if read == "entropy":
            return cheap + escalates * full
        if read == "probe":
            return np.where(escalates, probe_ms + full, cheap)
        return np.where(escalates, full, cheap)

    probe = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float), PROBE_LAYER)
    score = probe.score(matrix)

    print(f"\n{'policy':<20}{'accuracy':>10}{'flat ms':>10}{'per-example ms':>17}{'shift':>9}")
    rows = []
    for rate in (0.20, 0.30, 0.40):
        for name, values, read in (("entropy", entropy, "entropy"), ("probe", score, "probe")):
            cut = np.quantile(values[train], 1.0 - rate)
            fires = values[test] >= cut
            accuracy = float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
            flat_ms = float(costs(fires, test, read=read, flat_cost=True).mean())
            real_ms = float(costs(fires, test, read=read, flat_cost=False).mean())
            rows.append(
                {
                    "policy": f"{name} @{rate:.0%}",
                    "accuracy": accuracy,
                    "flat_ms": flat_ms,
                    "per_example_ms": real_ms,
                }
            )
            print(
                f"{name + ' @' + format(rate, '.0%'):<20}{accuracy:>10.3f}{flat_ms:>10.1f}"
                f"{real_ms:>17.1f}{real_ms / flat_ms - 1:>+9.0%}"
            )

    # --- what survives the correction -------------------------------------
    print("\nthe probe's saving over entropy at a matched escalation rate:")
    survives = []
    for rate in (0.20, 0.30, 0.40):
        cut_e = np.quantile(entropy[train], 1.0 - rate)
        cut_p = np.quantile(score[train], 1.0 - rate)
        fires_e, fires_p = entropy[test] >= cut_e, score[test] >= cut_p
        for label, flat_cost in (("flat", True), ("per-example", False)):
            e = costs(fires_e, test, read="entropy", flat_cost=flat_cost)
            p = costs(fires_p, test, read="probe", flat_cost=flat_cost)
            delta = paired_difference(p.tolist(), e.tolist())
            share = float(p.mean() / e.mean() - 1)
            survives.append(
                {
                    "rate": rate,
                    "costing": label,
                    "delta_ms": delta.estimate,
                    "delta_low": delta.low,
                    "delta_high": delta.high,
                    "relative": share,
                }
            )
            print(f"  @{rate:.0%} {label:<12} {delta} ms  ({share:+.0%} relative)")

    Path(args.out).write_text(
        json.dumps(
            {
                "model": {
                    "base_ms": model.base_ms,
                    "slope_ms_per_token": model.slope_ms_per_token,
                    "residual_ms": model.residual_ms,
                },
                "flat_full_ms": flat["full"],
                "per_example_full_ms": float(per_example_full.mean()),
                "mean_full_tokens": float(full_tokens.mean()),
                "extrapolation_span": span,
                "policies": rows,
                "savings": survives,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
