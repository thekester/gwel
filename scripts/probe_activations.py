"""Can pre-generation activations predict what output confidence cannot?

Three targets over the same activations, taken at the last prompt token of the
low-resolution pass before anything is generated:

- **correct**: will this cheap pass answer correctly? (the field's target)
- **helps**: will escalating to full resolution flip wrong to right?
- **answerable**: can any routable action answer this at all?

The first has a strong output-side baseline in our data (mean-entropy AUROC
0.758). The second is what actually spends budget and has never been decoded
from internal states. The third separates the two kinds of unanswerable.

Usage: python scripts/probe_activations.py --config configs/pilot1000.yaml
"""

import argparse
import json
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
from gwel.router.evaluate import auroc
from gwel.router.probes import sweep_layers
from gwel.router.splits import make_split


def build_targets(config, escalate_to: str) -> dict[str, dict[str, bool]]:
    """Per-example labels for the three probe targets, from cached records."""
    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    grouped: dict[str, dict] = defaultdict(dict)
    for record in records:
        grouped[record.example_id][record.config_id] = record

    probe_id = config.router.feature_config_id
    targets: dict[str, dict[str, bool]] = {}
    for example_id, configs in grouped.items():
        if probe_id not in configs or escalate_to not in configs:
            continue
        cheap = configs[probe_id].correct
        targets[example_id] = {
            "correct": cheap,
            "helps": (not cheap) and configs[escalate_to].correct,
            "answerable": any(
                r.correct
                for cid, r in configs.items()
                if cid.startswith(("lowres_", "crop_", "ocr_"))
            ),
            "entropy": float(
                ConfidenceSignals.from_dict(configs[probe_id].signals).mean_entropy
            )
            if configs[probe_id].signals
            else float("nan"),
        }
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--escalate-to", default="full")
    parser.add_argument("--limit", type=int, default=400, help="examples to extract")
    parser.add_argument("--cache", default="results/activations.npz")
    args = parser.parse_args()

    config = load_config(args.config)
    targets = build_targets(config, args.escalate_to)
    manifest = {e.example_id: e for e in read_manifest(config.paths.pilot_manifest)}
    example_ids = [e for e in manifest if e in targets][: args.limit]
    if not example_ids:
        print("no examples carry both the probe and escalation configs")
        return

    cache = Path(args.cache)
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        activations = stored["activations"]
        example_ids = list(stored["example_ids"])
        print(f"loaded cached activations for {len(example_ids)} examples")
    else:
        from gwel.modeling.smolvlm import SmolVlmEngine

        engine = SmolVlmEngine(config.model)
        engine.ensure_loaded()
        size = config.runner.lowres_sizes[0]
        rows = []
        for index, example_id in enumerate(example_ids):
            example = manifest[example_id]
            with Image.open(example.image_path) as raw:
                image = downscale(raw.convert("RGB"), size)
            rows.append(engine.extract_activations([image], example.question))
            if (index + 1) % 50 == 0:
                print(f"  extracted {index + 1}/{len(example_ids)}")
        activations = np.stack(rows)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, activations=activations, example_ids=example_ids)
        print(f"cached to {cache}")

    print(f"activations: {activations.shape} (examples, layers, hidden)\n")

    datasets = [manifest[e].dataset for e in example_ids]
    split = make_split(
        example_ids,
        datasets,
        val_fraction=config.router.val_fraction,
        test_fraction=config.router.test_fraction,
        seed=config.router.seed,
    )
    position = {e: i for i, e in enumerate(example_ids)}
    train_index = np.array([position[e] for e in split.train if e in position])
    test_index = np.array([position[e] for e in split.test if e in position])

    summary: dict[str, object] = {}
    for target in ("correct", "helps", "answerable"):
        if target == "helps":
            # Only meaningful where the cheap pass failed.
            subset = [e for e in example_ids if not targets[e]["correct"]]
            keep = np.array([position[e] for e in subset])
            train = np.array([i for i in train_index if i in set(keep)])
            test = np.array([i for i in test_index if i in set(keep)])
        else:
            train, test = train_index, test_index

        labels = np.array([float(targets[e][target]) for e in example_ids])
        if len(train) < 10 or len(test) < 5 or len(set(labels[test].tolist())) < 2:
            print(f"== {target}: skipped (train={len(train)}, test={len(test)}) ==\n")
            continue

        rows = sweep_layers(activations, labels, train, test)
        best_layer, best_auroc, _ = max(rows, key=lambda row: row[1])

        entropy_scores = [-targets[e]["entropy"] for e in example_ids]
        baseline = auroc(
            [entropy_scores[i] for i in test],
            [bool(labels[i]) for i in test],
        )

        print(f"== target: {target} ==")
        print(f"  train={len(train)}  test={len(test)}  positive rate={labels[test].mean():.0%}")
        print(f"  {'layer':>6}{'AUROC':>9}{'Fisher':>10}")
        for layer, area, fisher in rows:
            marker = "  <- best" if layer == best_layer else ""
            print(f"  {layer:>6}{area:>9.3f}{fisher:>10.3f}{marker}")
        print(f"  best layer {best_layer}: AUROC {best_auroc:.3f}")
        print(f"  output-entropy baseline on the same test set: {baseline:.3f}")
        verdict = "probe wins" if best_auroc > baseline + 0.02 else (
            "baseline wins" if baseline > best_auroc + 0.02 else "tie"
        )
        print(f"  verdict: {verdict}\n")
        summary[target] = {
            "best_layer": best_layer,
            "best_auroc": best_auroc,
            "entropy_baseline": baseline,
            "per_layer": [{"layer": l, "auroc": a, "fisher": f} for l, a, f in rows],
        }

    out = Path("results/activation_probes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
