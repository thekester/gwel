from PIL import Image

from gwel.actions import Action
from gwel.config import GwelConfig
from gwel.oracle.runner import planned_config_ids, prepare_ops
from gwel.router.budget import ActionProfile, BudgetRouter


def test_prepare_ops_covers_all_configurations() -> None:
    image = Image.new("RGB", (800, 600))
    ops = prepare_ops(image, GwelConfig(), ocr_engine=None)
    ids = [op.config_id for op in ops]

    assert ids[0] == "no_image"
    assert "lowres_384" in ids and "lowres_768" in ids
    assert "full" in ids
    assert sum(1 for i in ids if i.startswith("crop_")) == 4  # 2x2 grid
    assert len(ids) == len(set(ids))


def test_planned_ids_include_tool_ops_without_running_them() -> None:
    ids = planned_config_ids(GwelConfig())

    assert ids == (
        "no_image",
        "lowres_384",
        "lowres_768",
        "full",
        "crop_r0c0",
        "crop_r0c1",
        "crop_r1c0",
        "crop_r1c1",
        "ocr_r0c0",
        "ocr_r0c1",
        "ocr_r1c0",
        "ocr_r1c1",
        "ocr_full",
    )


def test_prepare_ops_actions_and_images() -> None:
    image = Image.new("RGB", (800, 600))
    by_id = {op.config_id: op for op in prepare_ops(image, GwelConfig(), ocr_engine=None)}

    assert by_id["no_image"].action is None and by_id["no_image"].images == ()
    assert by_id["full"].action is None
    assert by_id["lowres_384"].action is Action.ANSWER_LOW
    assert max(by_id["lowres_384"].images[0].size) == 384
    crop = by_id["crop_r0c0"]
    assert crop.action is Action.CROP
    assert len(crop.images) == 2  # preview + crop
    assert crop.meta["orig_width"] == 800


def test_budget_router_picks_best_fitting_action() -> None:
    router = BudgetRouter(
        [
            ActionProfile(Action.ANSWER_LOW, utility=0.4, latency_ms=20, memory_mb=300, energy_mj=5),
            ActionProfile(Action.CROP, utility=0.8, latency_ms=50, memory_mb=500, energy_mj=12),
            ActionProfile(Action.OCR, utility=0.9, latency_ms=90, memory_mb=700, energy_mj=20),
        ]
    )
    decision = router.route(latency_ms=60, memory_mb=600, energy_mj=15)
    assert decision.action is Action.CROP


def _resolution_only() -> GwelConfig:
    """A config that studies the resolution ladder and nothing else."""
    from dataclasses import replace

    base = GwelConfig()
    return replace(
        base, runner=replace(base.runner, include_crop=False, include_ocr=False)
    )


def test_disabling_crop_and_ocr_leaves_only_the_resolution_ladder() -> None:
    image = Image.new("RGB", (800, 600))
    ids = [op.config_id for op in prepare_ops(image, _resolution_only(), ocr_engine=None)]

    assert ids == ["no_image", "lowres_384", "lowres_768", "full"]


def test_planned_ids_agree_with_prepared_ops_when_tools_are_disabled() -> None:
    """The planner must not promise configurations the runner will not build."""
    image = Image.new("RGB", (800, 600))
    config = _resolution_only()
    prepared = tuple(op.config_id for op in prepare_ops(image, config, ocr_engine=None))

    assert planned_config_ids(config) == prepared


def test_disabling_ocr_alone_keeps_the_crops() -> None:
    from dataclasses import replace

    base = GwelConfig()
    config = replace(base, runner=replace(base.runner, include_ocr=False))
    ids = planned_config_ids(config)

    assert sum(1 for i in ids if i.startswith("crop_")) == 4
    assert not any(i.startswith("ocr_") for i in ids)


def test_tools_are_enabled_by_default_so_existing_configs_are_unchanged() -> None:
    base = GwelConfig()
    assert base.runner.include_crop and base.runner.include_ocr
    assert len(planned_config_ids(base)) == 13
