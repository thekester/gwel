"""Can a border perturbation inflate escalation, and does it move the probe?

Liu et al. (arXiv 2606.15308) force multimodal cascades to defer by learning a
border trigger that flattens the weak model's output distribution. Their attack
targets a post-generation confidence signal. This measures the unoptimised
version of that attack against two allocation signals side by side:

- output entropy, which every escalation method uses;
- a pre-generation activation probe, which reads the residual stream the
  attacker's objective never mentions.

An allocation signal that a border band can inflate is one an adversary can use
to shift compute cost onto the provider without making any answer wrong.

Usage: python scripts/test_deferral_attack.py --config configs/pilot1000.yaml
"""

import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

from gwel.config import load_config
from gwel.data.loaders import read_manifest
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.modeling.imaging import downscale
from gwel.modeling.perturbations import BorderPerturbation
from gwel.modeling.signals import ConfidenceSignals
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--activations", default="results/activations_full.npz")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--band", type=float, default=0.1)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=("noise", "flat", "both"),
        default="both",
        help="noise probes forced deferral, flat probes suppressed deferral",
    )
    args = parser.parse_args()

    from gwel.modeling.smolvlm import SmolVlmEngine

    config = load_config(args.config)
    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    grouped: dict[str, dict] = defaultdict(dict)
    for record in records:
        grouped[record.example_id][record.config_id] = record

    # Fit the probe on training examples only, so the attack faces a probe that
    # never saw it — the same discipline an attacker would face in deployment.
    stored = np.load(args.activations, allow_pickle=True)
    activations, ids = stored["activations"], list(stored["example_ids"])
    position = {e: i for i, e in enumerate(ids)}
    usable = [e for e in ids if "lowres_384" in grouped[e] and grouped[e]["lowres_384"].signals]
    split = make_split(
        usable,
        [grouped[e]["lowres_384"].dataset for e in usable],
        val_fraction=config.router.val_fraction,
        test_fraction=config.router.test_fraction,
        seed=config.router.seed,
    )
    train_rows = np.array([position[e] for e in split.train if e in position])
    labels = np.array([float(grouped[e]["lowres_384"].correct) for e in ids])
    layer = 23
    probe = fit_layer_probe(activations[train_rows, layer, :], labels[train_rows], layer)

    engine = SmolVlmEngine(config.model)
    engine.ensure_loaded()
    manifest = {e.example_id: e for e in read_manifest(config.paths.pilot_manifest)}
    targets = [e for e in split.test if e in manifest][: args.limit]
    modes = ("noise", "flat") if args.mode == "both" else (args.mode,)
    size = config.runner.lowres_sizes[0]

    def measure(images):
        entropies, scores = [], []
        for example_id, image in images:
            question = manifest[example_id].question
            output = engine.generate([image], question)
            entropies.append(float(output.signals.mean_entropy))
            states = engine.extract_activations([image], question)
            scores.append(float(probe.score(states[layer][None, :])[0]))
        return np.array(entropies), np.array(scores)

    bases = []
    for example_id in targets:
        with Image.open(manifest[example_id].image_path) as raw:
            bases.append((example_id, downscale(raw.convert("RGB"), size)))
    clean_entropy, clean_probe = measure(bases)

    print(f"{len(targets)} held-out images, border band {args.band:.0%} of the shorter side\n")

    def rate(clean: np.ndarray, values: np.ndarray, higher_defers: bool) -> float:
        # Escalation rate at the median clean threshold: the operating point a
        # provider would have calibrated before the attack existed.
        threshold = float(np.median(clean))
        return float(
            (values >= threshold).mean() if higher_defers else (values <= threshold).mean()
        )

    print(f"{'attack':<8}{'signal':<20}{'mean shift':>12}{'escalation rate':>20}")
    for mode in modes:
        attack = BorderPerturbation(
            band_fraction=args.band, strength=args.strength, mode=mode
        )
        dirty = [(eid, attack.apply(img)) for eid, img in bases]
        dirty_entropy, dirty_probe = measure(dirty)
        for name, clean, moved, higher in (
            ("output entropy", clean_entropy, dirty_entropy, True),
            ("activation probe", clean_probe, dirty_probe, False),
        ):
            pooled = np.std(np.concatenate([clean, moved])) or 1.0
            before, after = rate(clean, clean, higher), rate(clean, moved, higher)
            print(
                f"{mode:<8}{name:<20}{(moved - clean).mean() / pooled:>+11.2f} SD"
                f"{before:>12.0%} -> {after:<4.0%}  ({after - before:+.0%})"
            )

    print(
        "\nnoise probes forced deferral (inflates the provider's cost);"
        "\nflat probes suppressed deferral (keeps a weak answer that should escalate)."
    )


if __name__ == "__main__":
    main()
