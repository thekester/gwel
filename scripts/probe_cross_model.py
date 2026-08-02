"""Can a smaller model's activations decide a larger model's escalations?

NVIDIA's LLM Router (arXiv 2603.20895) calls this Encoder-Target Decoupling:
open-weight encoders predict the performance of a different, larger target
model, and in several cases beat the target's own hidden states. They do it to
route across a pool of closed-source text LLMs.

The visual analogue has a sharper motivation. If SmolVLM-256M's activations
predict whether SmolVLM-500M needs a high-resolution pass, the escalation
decision is made by a model that is cheaper to run *and* separable from the
serving path, so the probe never touches the model that answers.

Usage: python scripts/probe_cross_model.py --source configs/smol256.yaml
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from gwel.config import load_config
from gwel.data.loaders import read_manifest
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.modeling.imaging import downscale
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.evaluate import auroc
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split


def bootstrap_auroc(scores, labels, *, resamples: int = 2000, seed: int = 0):
    point = auroc(list(scores), [bool(v) for v in labels])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        index = rng.integers(0, len(labels), len(labels))
        if len(set(np.asarray(labels)[index].tolist())) < 2:
            continue
        draws.append(auroc(np.asarray(scores)[index].tolist(),
                           [bool(v) for v in np.asarray(labels)[index]]))
    low, high = np.quantile(draws, [0.025, 0.975]) if draws else (float("nan"),) * 2
    return point, float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="configs/pilot1000.yaml",
                        help="config whose escalation outcomes are being predicted")
    parser.add_argument("--source", default="configs/smol256.yaml",
                        help="config whose activations do the predicting")
    parser.add_argument("--cache", default="results/activations_source.npz")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    target = load_config(args.target)
    source = load_config(args.source)

    records = rescore_records(
        deduplicate_records(read_records(target.paths.records)), ScoringPolicy()
    )
    grouped: dict[str, dict] = defaultdict(dict)
    for record in records:
        grouped[record.example_id][record.config_id] = record

    manifest = {e.example_id: e for e in read_manifest(target.paths.pilot_manifest)}
    probe_id = target.router.feature_config_id
    # The label is the *target* model's outcome; only the activations change.
    usable = [
        e for e in manifest
        if probe_id in grouped[e] and "full" in grouped[e] and not grouped[e][probe_id].correct
    ][: args.limit]
    if len(usable) < 50:
        print(f"only {len(usable)} failed queries available; nothing to fit")
        return

    cache = Path(args.cache)
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        activations, cached_ids = stored["activations"], list(stored["example_ids"])
        usable = [e for e in usable if e in set(cached_ids)]
        order = {e: i for i, e in enumerate(cached_ids)}
        activations = activations[[order[e] for e in usable]]
        print(f"loaded cached source activations for {len(usable)} examples")
    else:
        from gwel.modeling.smolvlm import SmolVlmEngine

        engine = SmolVlmEngine(source.model)
        engine.ensure_loaded()
        size = target.runner.lowres_sizes[0]
        rows = []
        for index, example_id in enumerate(usable):
            example = manifest[example_id]
            with Image.open(example.image_path) as raw:
                image = downscale(raw.convert("RGB"), size)
            rows.append(engine.extract_activations([image], example.question))
            if (index + 1) % 100 == 0:
                print(f"  {index + 1}/{len(usable)}")
        activations = np.stack(rows)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, activations=activations, example_ids=usable)

    labels = np.array([float(grouped[e]["full"].correct) for e in usable])
    split = make_split(
        usable,
        [grouped[e][probe_id].dataset for e in usable],
        val_fraction=target.router.val_fraction,
        test_fraction=target.router.test_fraction,
        seed=target.router.seed,
    )
    position = {e: i for i, e in enumerate(usable)}
    train = np.array([position[e] for e in split.train])
    test = np.array([position[e] for e in split.test])

    print(f"\nsource activations: {source.model.model_id}")
    print(f"target outcomes:    {target.model.model_id}")
    print(f"{len(usable)} failed queries, train={len(train)} test={len(test)}, "
          f"positives={int(labels[test].sum())}\n")

    print(f"  {'layer':>6}{'AUROC':>9}{'95% CI':>20}")
    best = (None, 0.0)
    for layer in range(activations.shape[1]):
        probe = fit_layer_probe(activations[train, layer, :], labels[train], layer)
        point, low, high = bootstrap_auroc(
            probe.score(activations[test, layer, :]), labels[test]
        )
        if layer % 4 == 0 or point > best[1]:
            print(f"  {layer:>6}{point:>9.3f}   [{low:.3f}, {high:.3f}]")
        if point > best[1]:
            best = (layer, point)
    print(f"\nbest cross-model layer {best[0]}: AUROC {best[1]:.3f}")
    print("Compare against the same-model probe reported by scripts/probe_activations.py.")


if __name__ == "__main__":
    main()
