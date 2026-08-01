"""Typed configuration loaded from a YAML file.

Plain dataclasses keep the dependency surface small; every experiment knob
lives in ``configs/*.yaml`` so no path or hyperparameter is hard-coded.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    device: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 32
    answer_prompt: str = "Answer with a single word or short phrase."


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "data"
    pilot_dir: str = "data/processed/pilot"
    pilot_manifest: str = "data/processed/pilot/manifest.jsonl"
    records: str = "results/runs/pilot_records.jsonl"
    labels: str = "results/runs/pilot_labels.jsonl"
    router_dir: str = "results/router"


@dataclass(frozen=True)
class CropConfig:
    rows: int = 2
    cols: int = 2
    overlap: float = 0.2
    longest_side: int = 512
    preview_size: int = 256


@dataclass(frozen=True)
class OcrConfig:
    backend: str = "pytesseract"
    source: str = "full"
    preview_size: int = 256


@dataclass(frozen=True)
class RunnerConfig:
    lowres_sizes: tuple[int, ...] = (256, 384)
    full_longest_side: int = 1536
    crop: CropConfig = field(default_factory=CropConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    repeats: int = 1
    warmup: int = 0
    include_no_image: bool = True
    include_full: bool = True


@dataclass(frozen=True)
class ProfilingConfig:
    energy_backends: str | tuple[str, ...] = "auto"
    sample_interval_ms: int = 20
    nvml_device_index: int = 0
    hardware_repeats: int = 5
    hardware_warmup: int = 2


@dataclass(frozen=True)
class CostConfig:
    error_weight: float = 1.0
    lambda_latency_per_ms: float = 0.0005
    lambda_energy_per_mj: float = 0.0002
    lambda_memory_per_mb: float = 0.0
    lambda_visual_tokens: float = 0.002


@dataclass(frozen=True)
class RouterConfig:
    feature_config_id: str = "lowres_256"
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 60
    batch_size: int = 64
    val_fraction: float = 0.2
    seed: int = 1234


@dataclass(frozen=True)
class DatasetsConfig:
    seed: int = 1234
    shuffle_buffer_size: int = 10_000
    image_dir: str = "data/processed/pilot/images"
    per_dataset: dict[str, int] = field(
        default_factory=lambda: {"vqav2": 250, "textvqa": 250, "docvqa": 250, "vstar": 100}
    )


@dataclass(frozen=True)
class GwelConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)


def _build(cls: type, payload: Any) -> Any:
    """Recursively build a dataclass from a nested mapping."""
    if payload is None:
        return cls()
    if not isinstance(payload, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(payload).__name__}")

    kwargs: dict[str, Any] = {}
    valid = {f.name for f in fields(cls)}
    for key, value in payload.items():
        if key not in valid:
            raise KeyError(f"unknown config key {key!r} for {cls.__name__}")
        target = _field_dataclass(cls, key)
        if target is not None:
            kwargs[key] = _build(target, value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _field_dataclass(cls: type, name: str) -> type | None:
    """Return the dataclass type of a field, if it is one."""
    for f in fields(cls):
        if f.name == name:
            default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
            if is_dataclass(default):
                return type(default)
    return None


def load_config(path: str | Path) -> GwelConfig:
    """Load a :class:`GwelConfig` from a YAML file."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _build(GwelConfig, payload)
