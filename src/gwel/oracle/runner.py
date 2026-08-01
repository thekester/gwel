"""Multi-configuration oracle runner.

For each pilot example the runner executes every visual configuration:

- ``no_image``            blind baseline (diagnostic, not routable)
- ``lowres_{S}``          low-res preview at each configured size (ANSWER_LOW)
- ``full``                capped full-resolution pass (diagnostic)
- ``crop_r{R}c{C}``       preview + one high-res crop per grid cell (CROP)
- ``ocr_{source}``        preview + OCR transcript (OCR)

Each pass logs the answer, correctness, confidence signals, and hardware
measurements as one :class:`~gwel.oracle.records.RunRecord`, appended
incrementally so interrupted runs resume where they stopped.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from PIL import Image

from ..actions import Action
from ..config import GwelConfig
from ..data.loaders import PilotExample
from ..data.vqa_metrics import exact_match, vqa_accuracy
from ..modeling.imaging import crop_grid, downscale, extract_crop
from ..modeling.ocr import LazyOcrEngine
from ..profiling.energy import EnergyMeter, build_energy_meter
from ..profiling.memory import track_memory
from ..profiling.stats import summarize_repeats
from .records import RunRecord, append_records, load_done_keys

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedOp:
    """One ready-to-run visual configuration for one example."""

    config_id: str
    action: Action | None
    images: tuple[Image.Image, ...]
    context_text: str | None = None
    tool_ms: float = 0.0  # non-model overhead (e.g. OCR) added to latency
    meta: dict[str, object] = field(default_factory=dict)


def prepare_ops(
    image: Image.Image,
    config: GwelConfig,
    *,
    ocr_engine: LazyOcrEngine | None = None,
) -> list[PreparedOp]:
    """Build every configuration the oracle evaluates for one image."""
    runner = config.runner
    base_meta: dict[str, object] = {"orig_width": image.width, "orig_height": image.height}
    ops: list[PreparedOp] = []

    if runner.include_no_image:
        ops.append(PreparedOp(config_id="no_image", action=None, images=(), meta=dict(base_meta)))

    for size in runner.lowres_sizes:
        ops.append(
            PreparedOp(
                config_id=f"lowres_{size}",
                action=Action.ANSWER_LOW,
                images=(downscale(image, size),),
                meta={**base_meta, "longest_side": size},
            )
        )

    if runner.include_full:
        full = downscale(image, runner.full_longest_side)
        ops.append(
            PreparedOp(
                config_id="full",
                action=None,
                images=(full,),
                meta={**base_meta, "longest_side": runner.full_longest_side},
            )
        )

    crop_cfg = runner.crop
    preview = downscale(image, crop_cfg.preview_size)
    boxes = crop_grid(
        image.width, image.height,
        rows=crop_cfg.rows, cols=crop_cfg.cols, overlap=crop_cfg.overlap,
    )
    for index, box in enumerate(boxes):
        row, col = divmod(index, crop_cfg.cols)
        ops.append(
            PreparedOp(
                config_id=f"crop_r{row}c{col}",
                action=Action.CROP,
                images=(preview, extract_crop(image, box, longest_side=crop_cfg.longest_side)),
                meta={**base_meta, "box": box.to_dict(), "preview_size": crop_cfg.preview_size},
            )
        )

    if ocr_engine is not None:
        ocr_cfg = runner.ocr
        source = image if ocr_cfg.source == "full" else downscale(image, max(runner.lowres_sizes))
        result = ocr_engine.extract(source)
        ops.append(
            PreparedOp(
                config_id=f"ocr_{ocr_cfg.source}",
                action=Action.OCR,
                images=(downscale(image, ocr_cfg.preview_size),),
                context_text=result.text or "(no text found)",
                tool_ms=result.ocr_ms + (result.load_ms or 0.0),
                meta={
                    **base_meta,
                    "ocr_backend": ocr_engine.backend,
                    "ocr_ms": result.ocr_ms,
                    "ocr_load_ms": result.load_ms,
                    "ocr_chars": len(result.text),
                    "preview_size": ocr_cfg.preview_size,
                },
            )
        )
    return ops


def planned_config_ids(config: GwelConfig) -> tuple[str, ...]:
    """Return operation IDs without loading an image or invoking any tool."""
    runner = config.runner
    ids: list[str] = []
    if runner.include_no_image:
        ids.append("no_image")
    ids.extend(f"lowres_{size}" for size in runner.lowres_sizes)
    if runner.include_full:
        ids.append("full")
    ids.extend(
        f"crop_r{row}c{col}"
        for row in range(runner.crop.rows)
        for col in range(runner.crop.cols)
    )
    ids.append(f"ocr_{runner.ocr.source}")
    return tuple(ids)


class OracleRunner:
    """Execute all configurations for a set of examples and log records."""

    def __init__(
        self,
        config: GwelConfig,
        *,
        engine=None,
        ocr_engine: LazyOcrEngine | None = None,
        energy_meter: EnergyMeter | None = None,
    ) -> None:
        self.config = config
        if engine is None:
            from ..modeling.smolvlm import SmolVlmEngine

            engine = SmolVlmEngine(config.model)
        self.engine = engine
        self.ocr_engine = ocr_engine or LazyOcrEngine(config.runner.ocr.backend)
        self.energy_meter = energy_meter or build_energy_meter(
            config.profiling.energy_backends,
            sample_interval_ms=config.profiling.sample_interval_ms,
            nvml_device_index=config.profiling.nvml_device_index,
        )

    def run_op(self, example: PilotExample, op: PreparedOp) -> RunRecord:
        """Run one configuration with full instrumentation."""
        repeats = max(self.config.runner.repeats, 1)
        warmup = self.config.runner.warmup

        outputs = []
        latencies: list[float] = []
        with track_memory(sample_interval_ms=self.config.profiling.sample_interval_ms) as tracker:
            self.energy_meter.start()
            for index in range(warmup + repeats):
                output = self.engine.generate(
                    op.images or None, example.question, context_text=op.context_text
                )
                if index >= warmup:
                    outputs.append(output)
                    latencies.append(output.generate_ms)
            energy = self.energy_meter.stop()

        # Energy was integrated over all generations; report the per-call mean.
        calls = warmup + repeats
        energy = {name: (value / calls if value is not None else None) for name, value in energy.items()}

        output = outputs[-1]
        stats = summarize_repeats(latencies)
        match = exact_match(output.answer, example.answers)
        score = vqa_accuracy(output.answer, example.answers)
        report = tracker.report

        return RunRecord(
            example_id=example.example_id,
            dataset=example.dataset,
            question=example.question,
            gold_answers=example.answers,
            config_id=op.config_id,
            action=op.action,
            answer=output.answer,
            exact_match=match,
            vqa_score=score,
            correct=match or score >= 0.5,
            latency_ms=stats.median + op.tool_ms,
            latency_stats=stats.to_dict() if repeats > 1 else None,
            ttft_ms=output.ttft_ms,
            ram_peak_mb=report.ram_peak_mb if report else None,
            vram_peak_mb=report.vram_peak_mb if report else None,
            energy_mj=energy,
            visual_tokens=output.visual_tokens,
            prompt_tokens=output.prompt_tokens,
            generated_tokens=output.generated_tokens,
            signals=output.signals.to_dict(),
            meta={**op.meta, "tool_ms": op.tool_ms, "repeats": repeats, "warmup": warmup},
        )

    def run(self, examples: Sequence[PilotExample], out_path: str) -> dict[str, int]:
        """Run every configuration for every example, resuming from ``out_path``.

        Returns counters: examples seen, records written, records skipped
        (already present), and failures.
        """
        done = load_done_keys(out_path)
        counters = {"examples": 0, "written": 0, "skipped": 0, "failed": 0}
        config_ids = planned_config_ids(self.config)
        ensure_loaded = getattr(self.engine, "ensure_loaded", None)
        model_prepared = False

        for example in examples:
            counters["examples"] += 1
            if all((example.example_id, config_id) in done for config_id in config_ids):
                counters["skipped"] += len(config_ids)
                continue
            try:
                image = Image.open(example.image_path).convert("RGB")
            except OSError as error:
                logger.warning("cannot open %s: %s", example.image_path, error)
                counters["failed"] += 1
                continue

            ops = prepare_ops(image, self.config, ocr_engine=self.ocr_engine)
            for op in ops:
                if (example.example_id, op.config_id) in done:
                    counters["skipped"] += 1
                    continue
                try:
                    # Startup is reported separately and must not contaminate
                    # the first action's steady-state hardware measurements.
                    if not model_prepared and callable(ensure_loaded):
                        ensure_loaded()
                        model_prepared = True
                    record = self.run_op(example, op)
                except Exception:
                    logger.exception(
                        "failed %s / %s", example.example_id, op.config_id
                    )
                    counters["failed"] += 1
                    continue
                append_records(out_path, [record])
                counters["written"] += 1
        return counters
