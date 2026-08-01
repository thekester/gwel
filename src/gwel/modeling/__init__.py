"""Model, OCR, and image-operation wrappers (heavy deps imported lazily)."""

from .imaging import CropBox, crop_grid, downscale, extract_crop
from .ocr import LazyOcrEngine, OcrResult
from .signals import ConfidenceSignals, entropy_from_logits, signals_from_scores, top2_margin_from_logits

__all__ = [
    "ConfidenceSignals",
    "CropBox",
    "LazyOcrEngine",
    "OcrResult",
    "crop_grid",
    "downscale",
    "entropy_from_logits",
    "extract_crop",
    "signals_from_scores",
    "top2_margin_from_logits",
]


def __getattr__(name: str):
    """Lazily expose the SmolVLM engine so importing gwel never pulls torch."""
    if name in ("SmolVlmEngine", "GenerationOutput", "LoadReport"):
        from . import smolvlm

        return getattr(smolvlm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
