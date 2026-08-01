"""Parts of the SmolVLM engine that do not require loading weights."""

import numpy as np
import pytest

pytest.importorskip("torch")

from gwel.config import ModelConfig  # noqa: E402
from gwel.modeling.smolvlm import (  # noqa: E402
    ComponentTiming,
    GenerationOutput,
    LoadReport,
    SmolVlmEngine,
)
from gwel.modeling.signals import signals_from_scores  # noqa: E402


def test_engine_defers_loading() -> None:
    engine = SmolVlmEngine(ModelConfig())
    assert not engine.is_loaded
    assert engine.load_report is None


def test_device_resolution_honours_an_explicit_choice() -> None:
    assert SmolVlmEngine(ModelConfig(device="cpu"))._resolve_device() == "cpu"
    assert SmolVlmEngine(ModelConfig(device="cuda"))._resolve_device() == "cuda"


def test_auto_device_resolves_to_a_real_device() -> None:
    assert SmolVlmEngine(ModelConfig(device="auto"))._resolve_device() in ("cpu", "cuda")


def test_transformers_version_gate_accepts_the_installed_version() -> None:
    # SmolVLM needs >= 4.46; the environment must satisfy it for runs to work.
    SmolVlmEngine(ModelConfig())._check_transformers_version()


def test_component_timing_totals_and_share() -> None:
    timing = ComponentTiming(
        vision_encoder_ms=50.0,
        projector_ms=10.0,
        prefill_ms=90.0,
        decode_ms=50.0,
        visual_tokens=320,
    )
    assert timing.total_ms == 200.0
    assert timing.vision_share == pytest.approx(0.30)


def test_component_timing_share_is_zero_for_an_empty_pass() -> None:
    empty = ComponentTiming(0.0, 0.0, 0.0, 0.0, 0)
    assert empty.vision_share == 0.0


def test_component_timing_serialises_every_field() -> None:
    payload = ComponentTiming(1.0, 2.0, 3.0, 4.0, 64).to_dict()
    assert set(payload) == {
        "vision_encoder_ms", "projector_ms", "prefill_ms",
        "decode_ms", "visual_tokens", "vision_share",
    }


def test_generation_output_carries_the_fields_the_runner_logs() -> None:
    signals = signals_from_scores([np.array([2.0, 1.0, 0.0])], chosen_ids=[0])
    output = GenerationOutput(
        answer="yes",
        signals=signals,
        visual_tokens=64,
        prompt_tokens=80,
        generated_tokens=1,
        ttft_ms=12.0,
        generate_ms=100.0,
    )
    assert output.answer == "yes"
    assert output.signals.num_tokens == 1
    assert output.visual_tokens == 64


def test_load_report_fields() -> None:
    report = LoadReport(load_ms=3800.0, ram_delta_mb=130.0)
    assert report.load_ms > 0 and report.ram_delta_mb > 0
