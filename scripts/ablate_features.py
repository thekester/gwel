"""How much does the probe pass actually buy the router?

A confidence-conditioned router must run a cheap pass before it can decide,
and that probe is not free. This ablation trains the same classifier on
feature subsets — question text only (zero probe cost), probe signals only,
and both — to quantify what the probe adds over free features.

If question-only features come close, a zero-probe router is possible and the
cascade's probe cost disappears.

Usage: python scripts/ablate_features.py --config configs/pilot200.yaml
"""

import argparse

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.cost import CostWeights
from gwel.oracle.label import derive_labels
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.evaluate import auroc
from gwel.router.features import (
    HARDWARE_FEATURES,
    IMAGE_FEATURES,
    QUESTION_FEATURES,
    RUN_FEATURES,
    SIGNAL_FEATURES,
    FEATURE_NAMES,
    build_features,
)
from gwel.router.splits import make_split

#: Feature groups and whether they are available before running any model pass.
GROUPS: dict[str, tuple[tuple[str, ...], bool]] = {
    "question only": (QUESTION_FEATURES, True),
    "image geometry only": (IMAGE_FEATURES, True),
    "question + geometry": (QUESTION_FEATURES + IMAGE_FEATURES, True),
    "probe signals only": (SIGNAL_FEATURES, False),
    "probe signals + tokens": (SIGNAL_FEATURES + RUN_FEATURES, False),
    "everything": (FEATURE_NAMES, False),
}


def fit_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int = 400,
    lr: float = 0.1,
    l2: float = 1e-2,
    seed: int = 0,
) -> np.ndarray:
    """Fit L2-regularised logistic regression by gradient descent.

    Deliberately simple and strongly regularised: at pilot scale a bigger model
    only measures overfitting, and the question here is what the features carry.
    """
    rng = np.random.default_rng(seed)
    design = np.hstack([features, np.ones((len(features), 1))])
    weights = rng.normal(scale=0.01, size=design.shape[1])
    for _ in range(epochs):
        logits = design @ weights
        predictions = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (predictions - targets) / len(targets)
        gradient[:-1] += l2 * weights[:-1]
        weights -= lr * gradient
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot200.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    weights_cfg = CostWeights.from_config(config.cost)
    labels = {
        label.example_id: label.action
        for label in derive_labels(records, weights=weights_cfg)
    }

    probe_id = config.router.feature_config_id
    probes = [r for r in records if r.config_id == probe_id and r.signals is not None]

    # Target: does the cheap pass suffice? That is the escalate/don't decision.
    matrix = np.stack([build_features(r) for r in probes])
    targets = np.array([float(r.correct) for r in probes])
    example_ids = [r.example_id for r in probes]
    datasets = [r.dataset for r in probes]

    split = make_split(
        example_ids,
        datasets,
        val_fraction=config.router.val_fraction,
        test_fraction=config.router.test_fraction,
        seed=config.router.seed,
    )
    index_of = {e: i for i, e in enumerate(example_ids)}
    train_idx = np.array([index_of[e] for e in split.train])
    test_idx = np.array([index_of[e] for e in split.test])

    print(f"target: will the cheap pass ({probe_id}) answer correctly?")
    print(f"train n={len(train_idx)}, test n={len(test_idx)}, ")
    print(f"base rate correct = {targets.mean():.2f}\n")
    print(f"{'feature group':<24}{'free?':>7}{'test AUROC':>12}")

    for name, (group, is_free) in GROUPS.items():
        columns = [FEATURE_NAMES.index(f) for f in group if f in FEATURE_NAMES]
        if not columns:
            continue
        subset = matrix[:, columns]
        mean = subset[train_idx].mean(axis=0)
        std = subset[train_idx].std(axis=0)
        std[std < 1e-6] = 1.0
        normalized = (subset - mean) / std

        model = fit_logistic(normalized[train_idx], targets[train_idx])
        design = np.hstack([normalized[test_idx], np.ones((len(test_idx), 1))])
        scores = design @ model
        area = auroc(scores.tolist(), [bool(t) for t in targets[test_idx]])
        print(f"{name:<24}{'yes' if is_free else 'no':>7}{area:>12.3f}")

    # Raw single-signal reference: no fitting at all, just the entropy.
    entropy_index = FEATURE_NAMES.index("mean_entropy")
    raw = auroc(
        (-matrix[test_idx, entropy_index]).tolist(),
        [bool(t) for t in targets[test_idx]],
    )
    print(f"{'raw mean_entropy':<24}{'no':>7}{raw:>12.3f}")

    solvable = sum(1 for a in labels.values() if a is not None)
    print(f"\n({solvable}/{len(labels)} examples are solvable by some routable action)")


if __name__ == "__main__":
    main()
