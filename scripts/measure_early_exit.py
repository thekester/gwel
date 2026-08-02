"""Is a mid-prefill abort actually faster, or only faster on paper?

The probe reads layer 6 of 32, so a policy that escalates can in principle
abandon the cheap pass there and start the full-resolution one. Our cost model
assumes the abandoned remainder is genuinely not paid. That assumption is worth
testing rather than asserting: truncating a transformer stack changes memory
access patterns, and the saving could be eaten by overheads the arithmetic does
not see.

We measure it by physically truncating the decoder layer list, which is a real
early exit rather than a simulated one, and comparing the two cascade paths
end to end.

Usage: python scripts/measure_early_exit.py --config configs/pilot1000.yaml
"""

import argparse
import time
from contextlib import contextmanager

import numpy as np
from PIL import Image

from gwel.config import load_config
from gwel.data.loaders import read_manifest
from gwel.modeling.imaging import downscale
from gwel.profiling.stats import summarize_repeats


def decoder_layers(model):
    """The decoder's layer list, across the checkpoint variants we support."""
    inner = getattr(model, "model", model)
    for holder in (getattr(inner, "text_model", None), inner, model):
        layers = getattr(holder, "layers", None)
        if layers is not None:
            return holder, layers
    raise RuntimeError("could not locate the decoder layer list")


@contextmanager
def truncated_to(model, depth: int):
    """Temporarily run only the first ``depth`` decoder layers."""
    holder, original = decoder_layers(model)
    try:
        holder.layers = original[:depth]
        yield
    finally:
        holder.layers = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    import torch

    from gwel.modeling.smolvlm import SmolVlmEngine

    config = load_config(args.config)
    engine = SmolVlmEngine(config.model)
    engine.ensure_loaded()
    model = engine._model
    _, layers = decoder_layers(model)
    depth = len(layers)

    examples = read_manifest(config.paths.pilot_manifest)[: args.examples]
    small = config.runner.lowres_sizes[0]
    large = config.runner.full_longest_side

    def sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed(fn) -> float:
        sync()
        start = time.perf_counter()
        fn()
        sync()
        return (time.perf_counter() - start) * 1000.0

    abort_path, complete_path, probe_only = [], [], []
    for example in examples:
        with Image.open(example.image_path) as raw:
            base = raw.convert("RGB")
            cheap_image = downscale(base, small)
            full_image = downscale(base, large)

        cheap_inputs = engine._prepare_inputs(
            engine._build_prompt(1, example.question, None), [cheap_image]
        )

        def truncated_prefill() -> None:
            with truncated_to(model, args.layer), torch.inference_mode():
                model(**cheap_inputs)

        def full_cheap_answer() -> None:
            engine.generate([cheap_image], example.question)

        def escalated_answer() -> None:
            engine.generate([full_image], example.question)

        for index in range(args.warmup + args.repeats):
            probe = timed(truncated_prefill)
            cheap = timed(full_cheap_answer)
            escalated = timed(escalated_answer)
            if index < args.warmup:
                continue
            probe_only.append(probe)
            abort_path.append(probe + escalated)
            complete_path.append(cheap + escalated)

    print(f"{depth}-layer decoder, abort at layer {args.layer} "
          f"({args.layer / depth:.0%} of the stack)")
    print(f"{args.examples} images, {args.repeats} repeats, {args.warmup} warmup\n")

    for label, samples in (
        ("truncated prefill alone", probe_only),
        ("abort path: probe + full", abort_path),
        ("complete path: cheap + full", complete_path),
    ):
        stats = summarize_repeats(samples)
        print(f"  {label:<30}{stats.median:>8.1f} ms  (IQR {stats.iqr:.1f})")

    complete = summarize_repeats(complete_path).median
    abort = summarize_repeats(abort_path).median
    saving = complete - abort
    print(f"\n  measured saving on an escalated query: {saving:.1f} ms "
          f"({saving / complete:.0%})")
    print("  the cost model assumes the abandoned remainder is not paid;")
    print("  a positive saving here is that assumption holding on real hardware.")

    import json
    from pathlib import Path

    out = Path("results/early_exit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model_id": config.model.model_id,
                "decoder_layers": depth,
                "exit_layer": args.layer,
                "truncated_prefill_ms": summarize_repeats(probe_only).median,
                "abort_path_ms": abort,
                "complete_path_ms": complete,
                "saving_ms": saving,
                "saving_fraction": saving / complete,
                "examples": args.examples,
                "repeats": args.repeats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
