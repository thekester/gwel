"""Is the linear probe a choice, or a limitation we never tested?

The paper reads its escalation signal with a difference-of-means direction and
defends that as deliberate: the question is whether the signal is linearly
accessible, not how well a classifier can be tuned. That is a reasonable framing
and it is also, unexamined, a way of never asking whether more capacity would
find more. It matters here more than usual, because the paper's central negative
result is that the direction carries little within-domain signal. If a
non-linear probe recovers that signal, the conclusion changes; if it does not,
the negative is about the representation rather than about linear models.

Four probe families on identical folds and targets:

  difference of means   the paper's probe, unfitted beyond two centroids
  logistic              a fitted linear boundary, L2-regularised
  random features       logistic on 512 random Fourier features, a standard
                        non-parametric expansion that can fit smooth non-linear
                        boundaries without a bespoke architecture
  shrunk centroid       difference of means on covariance-whitened activations,
                        which is the linear-discriminant correction the plain
                        centroid difference omits

Evaluated pooled on the four-dataset mixture and within DocVQA on the
single-domain pilot, because those are the two regimes whose disagreement is the
paper's main finding.

Usage: PYTHONPATH=scripts python scripts/justify_probe_family.py
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
from gwel.router.zero_probe import fit_logistic

RESAMPLES = 20
HELDOUT = 300
RANDOM_FEATURES = 512


def _standardise(train: np.ndarray, test: np.ndarray):
    mean, std = train.mean(axis=0), train.std(axis=0) + 1e-8
    return (train - mean) / std, (test - mean) / std


def difference_of_means(train, labels, test, layer):
    probe = fit_layer_probe(train, labels, layer)
    return probe.score(test)


def logistic(train, labels, test, layer):
    a, b = _standardise(train, test)
    weights = fit_logistic(a, labels)
    return b @ weights[:-1] + weights[-1]


def random_features(train, labels, test, layer, *, seed=0):
    """Logistic on random Fourier features: a smooth non-linear boundary."""
    a, b = _standardise(train, test)
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(a.shape[1])
    projection = rng.normal(scale=scale, size=(a.shape[1], RANDOM_FEATURES))
    phase = rng.uniform(0, 2 * np.pi, RANDOM_FEATURES)
    expand = lambda x: np.cos(x @ projection + phase)  # noqa: E731
    weights = fit_logistic(expand(a), labels)
    return expand(b) @ weights[:-1] + weights[-1]


def shrunk_centroid(train, labels, test, layer, *, shrinkage=0.2):
    """Difference of means after whitening, i.e. regularised linear discriminant."""
    positive, negative = train[labels > 0.5], train[labels <= 0.5]
    if len(positive) < 2 or len(negative) < 2:
        return np.zeros(len(test))
    direction = positive.mean(axis=0) - negative.mean(axis=0)
    centred = train - train.mean(axis=0)
    covariance = centred.T @ centred / max(len(train) - 1, 1)
    covariance += shrinkage * np.trace(covariance) / covariance.shape[0] * np.eye(
        covariance.shape[0]
    )
    weights = np.linalg.solve(covariance, direction)
    return test @ weights


FAMILIES = {
    "difference of means": difference_of_means,
    "logistic": logistic,
    "random features": random_features,
    "shrunk centroid": shrunk_centroid,
}


def evaluate(matrix, labels, layer, entropy):
    """AUROC per family over resampled folds, with entropy as the reference."""
    scores = defaultdict(list)
    n = len(labels)
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(9000 + seed)
        shuffled = rng.permutation(n)
        test, train = shuffled[:HELDOUT], shuffled[HELDOUT:]
        truth = [bool(x) for x in labels[test]]
        if len(set(truth)) < 2 or len(set(labels[train].tolist())) < 2:
            continue
        for name, fit in FAMILIES.items():
            values = fit(matrix[train], labels[train].astype(float), matrix[test], layer)
            scores[name].append(auroc(np.asarray(values).ravel().tolist(), truth))
        scores["output entropy"].append(auroc(entropy[test].tolist(), truth))
    return scores


def load_mixture(layer):
    config = load_config("configs/pilot1000.yaml")
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    stored = np.load("results/activations_full.npz", allow_pickle=True)
    ids = [str(e) for e in stored["example_ids"]]
    usable = [
        e for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    position = {e: i for i, e in enumerate(ids)}
    matrix = stored["activations"][[position[e] for e in usable]][:, layer, :]
    cheap = np.array([grouped[e]["lowres_384"].correct for e in usable])
    full = np.array([grouped[e]["full"].correct for e in usable])
    entropy = np.array(
        [float(grouped[e]["lowres_384"].signals["mean_entropy"]) for e in usable]
    )
    return matrix, signed_gain(cheap, full) > 0, entropy


def load_single_domain(layer):
    config = load_config("configs/docvqa1200.yaml")
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    stored = np.load("results/activations_docvqa1200.npz", allow_pickle=True)
    ids = [str(e) for e in stored["example_ids"]]
    usable = [
        e for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    position = {e: i for i, e in enumerate(ids)}
    matrix = stored["activations"][[position[e] for e in usable]][:, layer, :]
    cheap = np.array([grouped[e]["lowres_384"].correct for e in usable])
    full = np.array([grouped[e]["full"].correct for e in usable])
    entropy = np.array(
        [float(grouped[e]["lowres_384"].signals["mean_entropy"]) for e in usable]
    )
    return matrix, signed_gain(cheap, full) > 0, entropy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--out", default="results/probe_family.json")
    args = parser.parse_args()

    results = {}
    for label, loader in (
        ("pooled mixture", load_mixture),
        ("within DocVQA", load_single_domain),
    ):
        matrix, labels, entropy = loader(args.layer)
        scores = evaluate(matrix, labels, args.layer, entropy)
        print(f"\n{label} (n={len(labels)}, layer {args.layer}, {RESAMPLES} resamples)")
        print(f"{'probe family':<24}{'AUROC [95% CI]':>26}")
        block = {}
        for name, values in scores.items():
            interval = bootstrap_interval(values)
            block[name] = [interval.estimate, interval.low, interval.high]
            print(f"{name:<24}{str(interval):>26}")
        results[label] = block

    print()
    for label, block in results.items():
        linear = block["difference of means"][0]
        best_fitted = max(
            block[name][0] for name in FAMILIES if name != "difference of means"
        )
        print(
            f"{label}: capacity buys {best_fitted - linear:+.3f} over the paper's probe"
        )
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
