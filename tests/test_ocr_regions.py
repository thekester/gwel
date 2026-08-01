from dataclasses import replace

from PIL import Image

from gwel.actions import Action
from gwel.config import CropConfig, GwelConfig, OcrConfig
from gwel.modeling.ocr import LazyOcrEngine
from gwel.oracle.runner import planned_config_ids, prepare_ops


def _config(**ocr_overrides) -> GwelConfig:
    config = GwelConfig()
    return replace(
        config,
        runner=replace(
            config.runner,
            crop=CropConfig(rows=2, cols=2),
            ocr=OcrConfig(**ocr_overrides),
        ),
    )


def _ops(config: GwelConfig):
    image = Image.new("RGB", (1600, 1200))
    return prepare_ops(image, config, ocr_engine=LazyOcrEngine("pytesseract"))


def test_region_ocr_produces_one_op_per_grid_cell() -> None:
    ids = [op.config_id for op in _ops(_config(regions=True))]
    assert {"ocr_r0c0", "ocr_r0c1", "ocr_r1c0", "ocr_r1c1"} <= set(ids)
    assert "ocr_full" in ids  # whole-page OCR stays as the comparison point


def test_regions_can_be_disabled() -> None:
    ids = [op.config_id for op in _ops(_config(regions=False))]
    assert not any(i.startswith("ocr_r") for i in ids)
    assert "ocr_full" in ids


def test_region_ocr_spends_preview_tokens_not_crop_tokens() -> None:
    ops = {op.config_id: op for op in _ops(_config(regions=True, preview_size=256))}
    region, crop = ops["ocr_r0c0"], ops["crop_r0c0"]
    assert len(region.images) == 1 and max(region.images[0].size) == 256
    assert len(crop.images) == 2  # preview + high-res crop
    assert region.action is Action.OCR


def test_region_ocr_reads_the_cell_at_its_own_resolution() -> None:
    ops = {op.config_id: op for op in _ops(_config(regions=True, region_longest_side=512))}
    assert max(ops["ocr_r0c0"].ocr_image.size) == 512
    assert ops["ocr_r0c0"].meta["box"] != ops["ocr_r1c1"].meta["box"]


def test_planned_ids_match_prepared_ops() -> None:
    config = _config(regions=True)
    assert set(planned_config_ids(config)) == {op.config_id for op in _ops(config)}
