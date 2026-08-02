"""A training-free stand-in for the VisionThink escalation mechanism.

VisionThink teaches a model to emit a resize request with GRPO and an LLM
judge. We cannot train that at sub-1B scale, which is the premise of this work,
but its *inference-time mechanism* is a self-report: the model, shown a
downsampled image, states whether it needs the full-resolution one.

This implements that mechanism without the training. It is deliberately
labelled an approximation: a trained policy would do better, and the gap
between this baseline and a trained one is exactly what we cannot measure. What
it does establish is whether asking the model directly beats reading its
uncertainty --- the comparison the escalation literature never makes, because
it trains the self-report and never tries the untrained version.

Usage: python scripts/baseline_self_report.py --config configs/pilot1000.yaml
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
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.splits import make_split

#: Phrasings tried, since a single prompt can be unlucky. Each must elicit a
#: binary judgement whose first token we can read.
PROMPTS = {
    "direct": (
        "Can you answer the question from this image, or do you need a "
        "higher-resolution view? Reply with one word: ANSWER or ZOOM."
    ),
    "sufficiency": (
        "Is this image detailed enough to answer the question? "
        "Reply with one word: YES or NO."
    ),
}

#: Word signalling "I need the high-resolution image", per prompt.
ESCALATE_WORD = {"direct": "zoom", "sufficiency": "no"}


def says_escalate(answer: str, prompt: str) -> bool:
    """Whether a free-form reply asks for more resolution.

    Replies arrive punctuated ("zoom.", "no."), so compare on stripped
    alphabetic tokens rather than raw whitespace splitting.
    """
    words = [w.strip(".,!?;:'\"").lower() for w in answer.split()]
    return ESCALATE_WORD[prompt] in words[:2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--cache", default="results/self_report.json")
    args = parser.parse_args()

    config = load_config(args.config)
    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    grouped: dict[str, dict] = defaultdict(dict)
    for record in records:
        grouped[record.example_id][record.config_id] = record

    manifest = {e.example_id: e for e in read_manifest(config.paths.pilot_manifest)}
    usable = [
        e for e in manifest if "lowres_384" in grouped[e] and "full" in grouped[e]
    ][: args.limit]

    cache = Path(args.cache)
    if cache.exists():
        scores = json.loads(cache.read_text(encoding="utf-8"))
        usable = [e for e in usable if e in scores]
        print(f"loaded cached self-reports for {len(usable)} examples")
    else:
        from gwel.modeling.smolvlm import SmolVlmEngine

        engine = SmolVlmEngine(config.model)
        engine.ensure_loaded()
        size = config.runner.lowres_sizes[0]
        scores = {}
        for index, example_id in enumerate(usable):
            example = manifest[example_id]
            with Image.open(example.image_path) as raw:
                image = downscale(raw.convert("RGB"), size)
            per_prompt = {}
            for name, instruction in PROMPTS.items():
                output = engine.generate(
                    [image], f"{example.question}\n{instruction}"
                )
                answer = output.answer.strip().lower()
                # Score the escalation decision, plus the model's own
                # uncertainty about it, which is a strictly richer signal than
                # the hard token a trained policy would emit.
                per_prompt[name] = {
                    "answer": answer,
                    "entropy": float(output.signals.mean_entropy),
                }
            scores[example_id] = per_prompt
            if (index + 1) % 100 == 0:
                print(f"  {index + 1}/{len(usable)}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    cheap_ok = np.array([grouped[e]["lowres_384"].correct for e in usable])
    full_ok = np.array([grouped[e]["full"].correct for e in usable])
    gain = ((~cheap_ok) & full_ok).astype(float)

    split = make_split(
        usable, [grouped[e]["lowres_384"].dataset for e in usable],
        val_fraction=config.router.val_fraction,
        test_fraction=config.router.test_fraction, seed=config.router.seed,
    )
    order = {e: i for i, e in enumerate(usable)}
    test = np.array([order[e] for e in split.test])

    rows = json.loads(Path("results/component_latency.json").read_text())
    components = {r["config"]: r for r in rows}
    cheap = components["longest_384"]["total_ms"]
    full = components["longest_1536"]["total_ms"]

    print(f"\n{len(usable)} examples, {len(test)} held out")
    print(f"{'prompt':<14}{'escalates':>11}{'accuracy':>11}{'latency':>10}{'AUROC on G':>13}")

    from gwel.router.evaluate import auroc

    for name in PROMPTS:
        escalates = np.array(
            [says_escalate(scores[e][name]["answer"], name) for e in usable]
        )[test]
        accuracy = np.where(escalates, full_ok[test], cheap_ok[test]).mean()
        # A self-report costs a full cheap pass, exactly like reading entropy.
        latency = float((cheap + escalates * full).mean())
        # Soft version: rank by the model's uncertainty about its own report.
        soft = np.array([scores[e][name]["entropy"] for e in usable])[test]
        area = auroc(soft.tolist(), [bool(v) for v in gain[test]])
        print(f"{name:<14}{escalates.mean():>11.0%}{accuracy:>11.3f}"
              f"{latency:>10.1f}{area:>13.3f}")

    print(f"\n{'always cheap':<14}{0:>11.0%}{cheap_ok[test].mean():>11.3f}{cheap:>10.1f}")
    print(f"{'always full':<14}{1:>11.0%}{full_ok[test].mean():>11.3f}{full:>10.1f}")


if __name__ == "__main__":
    main()
