from dataclasses import replace

from PIL import Image

from gwel.config import GwelConfig, OcrConfig
from gwel.modeling.ocr import LazyOcrEngine
from gwel.oracle.runner import prepare_ops


def _ocr_op(config: GwelConfig):
    """The whole-page OCR op, ignoring any per-region OCR ops."""
    image = Image.new("RGB", (2000, 1500))
    ops = prepare_ops(image, config, ocr_engine=LazyOcrEngine("pytesseract"))
    return next(op for op in ops if op.config_id == f"ocr_{config.runner.ocr.source}")


def test_source_longest_side_caps_the_ocr_input() -> None:
    config = GwelConfig()
    config = replace(config, runner=replace(config.runner, ocr=OcrConfig(source_longest_side=768)))
    assert max(_ocr_op(config).ocr_image.size) == 768


def test_uncapped_source_full_keeps_original_resolution() -> None:
    config = GwelConfig()
    config = replace(
        config, runner=replace(config.runner, ocr=OcrConfig(source="full", source_longest_side=None))
    )
    assert max(_ocr_op(config).ocr_image.size) == 2000


def test_ocr_source_size_is_recorded_in_meta() -> None:
    config = GwelConfig()
    config = replace(config, runner=replace(config.runner, ocr=OcrConfig(source_longest_side=512)))
    assert _ocr_op(config).meta["ocr_source_size"] == 512


def test_preview_stays_independent_of_the_ocr_source() -> None:
    config = GwelConfig()
    config = replace(
        config,
        runner=replace(config.runner, ocr=OcrConfig(source_longest_side=768, preview_size=256)),
    )
    op = _ocr_op(config)
    assert max(op.images[0].size) == 256
