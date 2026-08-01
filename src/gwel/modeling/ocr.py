"""Lazily initialised OCR engine with load-time accounting.

The OCR backend is only imported and initialised on first use so the pipeline
can measure the real cold-start cost of escalating to the OCR action.
"""

import time
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .imaging import CropBox


@dataclass(frozen=True)
class OcrResult:
    """Transcript extracted from an image region."""

    text: str
    ocr_ms: float
    load_ms: float | None  # set only on the call that initialised the backend


class LazyOcrEngine:
    """OCR wrapper deferring backend import/initialisation to first use."""

    def __init__(self, backend: str = "pytesseract") -> None:
        if backend not in ("pytesseract", "easyocr"):
            raise ValueError(f"unsupported OCR backend {backend!r}")
        self.backend = backend
        self._reader: object | None = None
        self.load_ms: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._reader is not None

    def _ensure_loaded(self) -> float | None:
        """Initialise the backend if needed; return the load time when it ran."""
        if self._reader is not None:
            return None
        start = time.perf_counter()
        if self.backend == "pytesseract":
            import pytesseract

            if shutil.which("tesseract") is None:
                windows_install = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
                if windows_install.exists():
                    pytesseract.pytesseract.tesseract_cmd = str(windows_install)

            self._reader = pytesseract
        else:
            import easyocr

            self._reader = easyocr.Reader(["en"], verbose=False)
        self.load_ms = (time.perf_counter() - start) * 1000.0
        return self.load_ms

    def extract(self, image: Image.Image, box: CropBox | None = None) -> OcrResult:
        """Run OCR on ``image`` (optionally restricted to ``box``)."""
        load_ms = self._ensure_loaded()
        if box is not None:
            image = image.crop((box.left, box.top, box.right, box.bottom))

        start = time.perf_counter()
        if self.backend == "pytesseract":
            text = self._reader.image_to_string(image)  # type: ignore[union-attr]
        else:
            fragments = self._reader.readtext(np.array(image), detail=0)  # type: ignore[union-attr]
            text = "\n".join(fragments)
        ocr_ms = (time.perf_counter() - start) * 1000.0
        return OcrResult(text=text.strip(), ocr_ms=ocr_ms, load_ms=load_ms)
