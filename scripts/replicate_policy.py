"""Replicate the full policy comparison on a different serving model.

Every routing result in this work rests on one model deciding for itself. This
runs the whole chain --- activation extraction, probe fitting on the joint
target, escalation-rate sweep, Pareto analysis --- against whichever serving
model a config names, so the main claim can be checked rather than assumed to
generalise.

Usage: python scripts/replicate_policy.py --config configs/serve256.yaml
"""

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from gwel.config import load_config
from gwel.data.loaders import read_manifest
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.modeling.imaging import downscale
from gwel.modeling.signals import ConfidenceSignals
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.evaluate import auroc, bootstrap_interval, pareto_front
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split

RATES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.70)


def load_activations(config, examples, manifest, cache: Path, layer_count: int) -> np.ndarray:
    """Residual states at the last prompt token, extracted or loaded."""
    if cache.exists():
        try:
            stored = np.load(cache, allow_pickle=True)
            order = {e: i for i, e in enumerate(list(stored["example_ids"]))}
            return stored["activations"][[order[e] for e in examples]]
        except (zipfile.BadZipFile, KeyError, ValueError, EOFError) as error:
            # A run killed mid-write leaves a truncated archive. Re-extract
            # rather than fail: the cache is an optimisation, not a source.
            print(f"  cache at {cache} is unusable ({type(error).__name__}); re-extracting")
            cache.unlink()

    from gwel.modeling.smolvlm import SmolVlmEngine

    engine = SmolVlmEngine(config.model)
    engine.ensure_loaded()
    size = config.runner.lowres_sizes[0]
    rows = []
    for index, example_id in enumerate(examples):
        with Image.open(manifest[example_id].image_path) as raw:
            image = downscale(raw.convert("RGB"), size)
        rows.append(engine.extract_activations([image], manifest[example_id].question))
        if (index + 1) % 200 == 0:
            print(f"  extracted {index + 1}/{len(examples)}")
    activations = np.stack(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, activations=activations, example_ids=examples)
    del layer_count
    return activations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serve256.yaml")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--costs", default=None, help="component_latency json for this model")
    args = parser.parse_args()

    config = load_config(args.config)
    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    grouped: dict[str, dict] = defaultdict(dict)
    for record in records:
        grouped[record.example_id][record.config_id] = record

    probe_id = config.router.feature_config_id
    manifest = {e.example_id: e for e in read_manifest(config.paths.pilot_manifest)}
    usable = [
        e for e in manifest
        if probe_id in grouped[e] and "full" in grouped[e] and grouped[e][probe_id].signals
    ]
    if len(usable) < 100:
        print(f"only {len(usable)} usable examples; run the oracle first")
        return

    name = Path(config.paths.records).stem.replace("_records", "")
    activations = load_activations(
        config, usable, manifest, Path(f"results/activations_{name}.npz"), args.layer
    )

    cheap_ok = np.array([grouped[e][probe_id].correct for e in usable])
    full_ok = np.array([grouped[e]["full"].correct for e in usable])
    entropy = np.array(
        [ConfidenceSignals.from_dict(grouped[e][probe_id].signals).mean_entropy
         for e in usable]
    )
    gain = ((~cheap_ok) & full_ok).astype(float)

    split = make_split(
        usable, [grouped[e][probe_id].dataset for e in usable],
        val_fraction=config.router.val_fraction,
        test_fraction=config.router.test_fraction, seed=config.router.seed,
    )
    order = {e: i for i, e in enumerate(usable)}
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])

    probe = fit_layer_probe(activations[train, args.layer, :], gain[train], args.layer)
    probe_score = probe.score(activations[:, args.layer, :])

    # Latency: measured per model where available, otherwise from the records.
    if args.costs and Path(args.costs).exists():
        rows = {r["config"]: r for r in json.loads(Path(args.costs).read_text())}
        cheap_ms = rows["longest_384"]["total_ms"]
        full_ms = rows["longest_1536"]["total_ms"]
        probe_ms = (
            rows["longest_384"]["vision_encoder_ms"] + rows["longest_384"]["projector_ms"]
            + rows["longest_384"]["prefill_ms"] * args.layer / 32
        )
    else:
        cheap_ms = float(np.median([grouped[e][probe_id].latency_ms for e in usable]))
        full_ms = float(np.median([grouped[e]["full"].latency_ms for e in usable]))
        probe_ms = cheap_ms * args.layer / 32  # conservative: ignores encoder split

    print(f"\nserving model: {config.model.model_id}")
    print(f"{len(usable)} examples, {len(test)} held out")
    print(f"cheap {cheap_ms:.1f} ms | probe {probe_ms:.1f} ms | full {full_ms:.1f} ms")
    print(f"\nAUROC on the joint escalation-gain target:")
    print(f"  entropy {auroc(entropy[test].tolist(), [bool(v) for v in gain[test]]):.3f}")
    print(f"  probe   {auroc(probe_score[test].tolist(), [bool(v) for v in gain[test]]):.3f}")

    points, labels = [], []
    print(f"\n{'policy':<16}{'accuracy':>22}{'latency':>10}")
    base = np.where(False, full_ok[test], cheap_ok[test])
    print(f"{'always cheap':<16}{str(bootstrap_interval(base.astype(float).tolist())):>22}"
          f"{cheap_ms:>9.1f}")
    points.append((cheap_ms, float(base.mean()))); labels.append("always cheap")

    for rate in RATES:
        for tag, score, read in (("entropy", entropy, "full"), ("probe", probe_score, "probe")):
            cut = np.quantile(score[train], 1.0 - rate)
            escalates = score[test] >= cut
            accuracy = np.where(escalates, full_ok[test], cheap_ok[test])
            cost = float(
                (cheap_ms + escalates * full_ms).mean() if read == "full"
                else np.where(escalates, probe_ms + full_ms, cheap_ms).mean()
            )
            points.append((cost, float(accuracy.mean())))
            labels.append(f"{tag} @{rate:.0%}")
            print(f"{labels[-1]:<16}"
                  f"{str(bootstrap_interval(accuracy.astype(float).tolist())):>22}{cost:>9.1f}")

    top = full_ok[test]
    print(f"{'always full':<16}{str(bootstrap_interval(top.astype(float).tolist())):>22}"
          f"{full_ms:>9.1f}")
    points.append((full_ms, float(top.mean()))); labels.append("always full")

    front = set(pareto_front([p[0] for p in points], [p[1] for p in points]))
    on_front = [labels[i] for i in sorted(front, key=lambda i: points[i][0])]
    print(f"\nPareto front: {', '.join(on_front)}")
    entropy_on_front = [n for n in on_front if n.startswith("entropy")]
    probe_on_front = [n for n in on_front if n.startswith("probe")]
    print(f"  probe points on front: {len(probe_on_front)}, "
          f"entropy points on front: {len(entropy_on_front)}")
    if probe_on_front and not entropy_on_front:
        print("  REPLICATED: every entropy operating point is dominated.")
    else:
        print("  NOT replicated on this model.")


if __name__ == "__main__":
    main()
