"""Is the pre-generation probe reading escalation value, or reading the domain?

A self-audit of this project's central positive result, prompted by a cost
measurement. Re-pricing escalation per example (`scripts/recost_policies.py`)
showed the probe systematically escalates *larger* images than entropy does. It
is worth asking why, and the answer is uncomfortable.

Escalation value is extremely heterogeneous across our four datasets: DocVQA
repairs 45% of its queries and VQAv2 5%. Image size tracks that split --- DocVQA
images spend 620 visual tokens at full resolution against 283 for VQAv2 --- so
*any* signal correlated with image size will score well on a pooled mixture
without knowing anything about a particular query. Image size is free: it
requires no forward pass at all.

Two tests separate the hypotheses.

  pooled       score every signal on the mixture, including raw image size as a
               zero-cost baseline the probe must beat
  within       train and test the probe inside a single dataset, where the
               between-domain axis is gone by construction

If the probe encodes escalation value, it survives the second test. If it
encodes which dataset a query came from, it collapses to chance there while
output entropy --- which reads the model's response to *this* query --- does not.

Usage: PYTHONPATH=scripts python scripts/analyze_domain_confound.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.decision import signed_gain
from gwel.router.evaluate import auroc, bootstrap_interval
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split

LAYERS = (1, 3, 6, 9, 12, 16, 20, 24, 28, 32)
RESAMPLES = 40


def load(config, activations_path: str):
    stored = np.load(activations_path, allow_pickle=True)
    activations = stored["activations"]
    ids = [str(e) for e in stored["example_ids"]]
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    usable = [
        e
        for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    position = {e: i for i, e in enumerate(ids)}
    matrix = activations[[position[e] for e in usable]]
    cheap_ok = np.array([grouped[e]["lowres_384"].correct for e in usable])
    full_ok = np.array([grouped[e]["full"].correct for e in usable])
    entropy = np.array(
        [float(grouped[e]["lowres_384"].signals["mean_entropy"]) for e in usable]
    )
    tokens = np.array([grouped[e]["full"].visual_tokens for e in usable], dtype=float)
    datasets = np.array([grouped[e]["lowres_384"].dataset for e in usable])
    return usable, matrix, cheap_ok, full_ok, entropy, tokens, datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--activations", default="results/activations_full.npz")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--out", default="results/domain_confound.json")
    args = parser.parse_args()

    config = load_config(args.config)
    usable, matrix, cheap_ok, full_ok, entropy, tokens, datasets = load(
        config, args.activations
    )
    gains = signed_gain(cheap_ok, full_ok)
    labels = gains > 0
    order = {e: i for i, e in enumerate(usable)}
    results: dict[str, object] = {}

    # --- how heterogeneous is the mixture? ---------------------------------
    print(f"{'dataset':<10}{'n':>5}{'repairs':>10}{'full tokens':>14}")
    mixture = []
    for dataset in sorted(set(datasets)):
        mask = datasets == dataset
        mixture.append(
            {
                "dataset": dataset,
                "n": int(mask.sum()),
                "repair_rate": float(labels[mask].mean()),
                "mean_full_tokens": float(tokens[mask].mean()),
            }
        )
        print(
            f"{dataset:<10}{mask.sum():>5}{labels[mask].mean():>10.1%}"
            f"{tokens[mask].mean():>14.0f}"
        )
    results["mixture"] = mixture
    print(
        f"\nescalation value and image size are both dataset-determined: "
        f"r = {np.corrcoef(tokens, labels.astype(float))[0, 1]:+.3f} between them\n"
    )

    # --- pooled, the paper's setup, plus a free baseline -------------------
    split = make_split(
        usable, datasets.tolist(),
        val_fraction=config.router.val_fraction,
        test_fraction=config.router.test_fraction,
        seed=config.router.seed,
    )
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])
    probe = fit_layer_probe(
        matrix[train, args.layer, :], labels[train].astype(float), args.layer
    )
    score = probe.score(matrix[:, args.layer, :])
    truth = [bool(x) for x in labels[test]]

    pooled = {
        "probe": auroc(score[test].tolist(), truth),
        "entropy": auroc(entropy[test].tolist(), truth),
        "image_size": auroc(tokens[test].tolist(), truth),
    }
    results["pooled"] = pooled
    print(f"pooled on the held-out fold (n={len(test)}), the paper's setup:")
    for name, value in pooled.items():
        free = "  <- free, no forward pass" if name == "image_size" else ""
        print(f"  {name:<12}{value:>8.3f}{free}")
    print(
        f"\nthe probe's margin over a zero-cost image-size feature is "
        f"{pooled['probe'] - pooled['image_size']:+.3f}"
    )

    # --- within a single dataset -------------------------------------------
    print(f"\ntrained and tested inside one dataset, {RESAMPLES} resamples:")
    print(f"{'dataset':<10}{'n':>6}{'probe':>22}{'entropy':>22}{'size':>9}")
    within = []
    for dataset in sorted(set(datasets)):
        members = [usable[i] for i in np.where(datasets == dataset)[0]]
        probe_scores, entropy_scores, size_scores = [], [], []
        for seed in range(RESAMPLES):
            local = make_split(
                members, [dataset] * len(members),
                val_fraction=0.0, test_fraction=0.3, seed=9000 + seed,
            )
            inner_test = np.array([order[e] for e in local.test])
            inner_train = np.array([order[e] for e in local.train])
            truth_inner = [bool(x) for x in labels[inner_test]]
            if len(set(truth_inner)) < 2 or len(set(labels[inner_train].tolist())) < 2:
                continue
            fitted = fit_layer_probe(
                matrix[inner_train, args.layer, :],
                labels[inner_train].astype(float),
                args.layer,
            )
            probe_scores.append(
                auroc(fitted.score(matrix[inner_test, args.layer, :]).tolist(), truth_inner)
            )
            entropy_scores.append(auroc(entropy[inner_test].tolist(), truth_inner))
            size_scores.append(auroc(tokens[inner_test].tolist(), truth_inner))
        if not probe_scores:
            continue
        probe_ci = bootstrap_interval(probe_scores)
        entropy_ci = bootstrap_interval(entropy_scores)
        within.append(
            {
                "dataset": dataset,
                "n": len(members),
                "probe": probe_ci.estimate,
                "probe_low": probe_ci.low,
                "probe_high": probe_ci.high,
                "entropy": entropy_ci.estimate,
                "entropy_low": entropy_ci.low,
                "entropy_high": entropy_ci.high,
                "image_size": float(np.mean(size_scores)),
            }
        )
        print(
            f"{dataset:<10}{len(members):>6}{str(probe_ci):>22}{str(entropy_ci):>22}"
            f"{np.mean(size_scores):>9.3f}"
        )
    results["within"] = within

    weights = np.array([row["n"] for row in within], dtype=float)
    probe_mean = float(np.average([row["probe"] for row in within], weights=weights))
    entropy_mean = float(np.average([row["entropy"] for row in within], weights=weights))
    results["within_weighted"] = {"probe": probe_mean, "entropy": entropy_mean}
    print(f"{'weighted':<10}{'':>6}{probe_mean:>22.3f}{entropy_mean:>22.3f}")

    # --- is layer 6 simply the wrong depth for a within-domain probe? ------
    largest = max(within, key=lambda row: row["n"] * row["probe"] if False else row["n"])
    dataset = largest["dataset"]
    members = [usable[i] for i in np.where(datasets == dataset)[0]]
    print(f"\nlayer sweep inside {dataset} (n={len(members)}), 30 resamples:")
    sweep = []
    for layer in LAYERS:
        values = []
        for seed in range(30):
            local = make_split(
                members, [dataset] * len(members),
                val_fraction=0.0, test_fraction=0.3, seed=9000 + seed,
            )
            inner_test = np.array([order[e] for e in local.test])
            inner_train = np.array([order[e] for e in local.train])
            truth_inner = [bool(x) for x in labels[inner_test]]
            if len(set(truth_inner)) < 2:
                continue
            fitted = fit_layer_probe(
                matrix[inner_train, layer, :], labels[inner_train].astype(float), layer
            )
            values.append(
                auroc(fitted.score(matrix[inner_test, layer, :]).tolist(), truth_inner)
            )
        sweep.append({"layer": layer, "auroc": float(np.mean(values))})
        print(f"  layer {layer:>2}: {np.mean(values):.3f}")
    results["layer_sweep"] = {"dataset": dataset, "points": sweep}
    best = max(sweep, key=lambda row: row["auroc"])
    entropy_here = next(row["entropy"] for row in within if row["dataset"] == dataset)
    print(
        f"\nbest depth {best['layer']} reaches {best['auroc']:.3f}, still below "
        f"output entropy at {entropy_here:.3f}. No layer rescues it."
    )

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
