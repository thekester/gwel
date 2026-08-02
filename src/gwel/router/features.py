"""Feature extraction for the supervised router.

Features come from the cheap ANSWER_LOW pass only, the router must decide
before any expensive operation runs. They combine the model's confidence
signals, simple question statistics, image geometry, and the live hardware
state. All features are deterministic given a record.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from ..modeling.signals import ConfidenceSignals
from ..oracle.records import RunRecord

_WH_WORDS = ("what", "where", "who", "how", "why", "which", "when")

#: Question words hinting that reading text is required (OCR-shaped queries).
_TEXT_HINTS = (
    "text", "say", "says", "written", "word", "words", "read", "sign", "label",
    "number", "price", "title", "date", "name", "brand", "letter", "page",
)

SIGNAL_FEATURES: tuple[str, ...] = (
    "mean_logprob",
    "min_logprob",
    "mean_entropy",
    "max_entropy",
    "first_entropy",
    "mean_margin",
    "min_margin",
    "num_tokens",
)

QUESTION_FEATURES: tuple[str, ...] = (
    "question_chars",
    "question_words",
    "question_has_number",
    "question_text_hint",
    *(f"question_wh_{word}" for word in _WH_WORDS),
)

IMAGE_FEATURES: tuple[str, ...] = (
    "image_width",
    "image_height",
    "image_aspect",
    "image_megapixels",
)

RUN_FEATURES: tuple[str, ...] = (
    "visual_tokens",
    "prompt_tokens",
    "generated_tokens",
)

HARDWARE_FEATURES: tuple[str, ...] = (
    "ram_available_mb",
    "ram_used_fraction",
    "cpu_load_fraction",
    "vram_free_mb",
    "vram_present",
)

FEATURE_NAMES: tuple[str, ...] = (
    *SIGNAL_FEATURES,
    *QUESTION_FEATURES,
    *IMAGE_FEATURES,
    *RUN_FEATURES,
    *HARDWARE_FEATURES,
)


def question_features(question: str) -> list[float]:
    """Cheap lexical statistics of the question string."""
    lowered = question.lower()
    words = lowered.split()
    values = [
        float(len(question)),
        float(len(words)),
        float(any(char.isdigit() for char in lowered)),
        float(any(hint in words for hint in _TEXT_HINTS)),
    ]
    first_word = words[0] if words else ""
    values.extend(float(first_word == wh) for wh in _WH_WORDS)
    return values


def build_features(
    record: RunRecord,
    *,
    hardware_state: Mapping[str, float | None] | None = None,
) -> np.ndarray:
    """Build the router input vector from one ANSWER_LOW record.

    ``hardware_state`` uses the schema of
    :func:`gwel.profiling.memory.current_hardware_state`; when omitted the
    hardware features are zero (useful for offline training on cached runs).
    """
    if record.signals is None:
        raise ValueError(f"record {record.example_id}/{record.config_id} has no signals")
    signals = ConfidenceSignals.from_dict(record.signals)
    values: list[float] = [float(getattr(signals, name)) for name in SIGNAL_FEATURES]

    values.extend(question_features(record.question))

    width = float(record.meta.get("orig_width", 0.0))  # type: ignore[arg-type]
    height = float(record.meta.get("orig_height", 0.0))  # type: ignore[arg-type]
    aspect = width / height if height > 0 else 0.0
    values.extend([width, height, aspect, width * height / 1e6])

    values.extend(
        [float(record.visual_tokens), float(record.prompt_tokens), float(record.generated_tokens)]
    )

    state = hardware_state or {}
    vram_free = state.get("vram_free_mb")
    values.extend(
        [
            float(state.get("ram_available_mb") or 0.0),
            float(state.get("ram_used_fraction") or 0.0),
            float(state.get("cpu_load_fraction") or 0.0),
            float(vram_free or 0.0),
            float(vram_free is not None),
        ]
    )

    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(FEATURE_NAMES),):
        raise AssertionError("feature vector length drifted from FEATURE_NAMES")
    return vector


def build_feature_matrix(
    records: Sequence[RunRecord],
    *,
    hardware_state: Mapping[str, float | None] | None = None,
) -> np.ndarray:
    """Stack feature vectors for many records into an (N, D) matrix."""
    return np.stack([build_features(r, hardware_state=hardware_state) for r in records])
