"""Pilot dataset construction: a small, seeded, reproducible mixture.

The pilot covers four question regimes: general VQA (VQAv2), scene text
(TextVQA), documents (DocVQA), and fine-grained detail (V*Bench). Images are
materialised to local files and the mixture is described by a JSONL manifest
so the oracle runner never depends on ``datasets`` at run time.
"""

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..config import DatasetsConfig

#: HuggingFace sources for each pilot subset. All are validation/test splits
#: with public answers, streamed so only the sampled rows are downloaded.
DATASET_SOURCES: dict[str, dict[str, str]] = {
    "vqav2": {"path": "HuggingFaceM4/VQAv2", "split": "validation"},
    "textvqa": {"path": "lmms-lab/textvqa", "split": "validation"},
    "docvqa": {"path": "lmms-lab/DocVQA", "name": "DocVQA", "split": "validation"},
    "vstar": {"path": "craigwu/vstar_bench", "split": "test"},
}


@dataclass(frozen=True)
class PilotExample:
    """One (image, question, answers) triple with a materialised image file."""

    example_id: str
    dataset: str
    image_path: str
    question: str
    answers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "dataset": self.dataset,
            "image_path": self.image_path,
            "question": self.question,
            "answers": list(self.answers),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PilotExample":
        return cls(
            example_id=str(payload["example_id"]),
            dataset=str(payload["dataset"]),
            image_path=str(payload["image_path"]),
            question=str(payload["question"]),
            answers=tuple(str(a) for a in payload["answers"]),  # type: ignore[union-attr]
        )


def write_manifest(path: str | Path, examples: Iterable[PilotExample]) -> None:
    """Write the pilot manifest as UTF-8 JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")


def read_manifest(path: str | Path) -> list[PilotExample]:
    """Read the pilot manifest."""
    examples: list[PilotExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(PilotExample.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid manifest line {line_number} of {path}") from error
    return examples


def _extract_qa(dataset: str, row: dict[str, object]) -> tuple[str, tuple[str, ...]] | None:
    """Pull (question, answers) from a raw row, tolerating schema variants."""
    question = row.get("question") or row.get("text")
    if not question:
        return None
    raw_answers = row.get("answers") or row.get("answer") or row.get("label")
    if raw_answers is None:
        return None
    if isinstance(raw_answers, str):
        answers: tuple[str, ...] = (raw_answers,)
    elif isinstance(raw_answers, dict):  # VQAv2 style: {"text": [...], ...}
        answers = tuple(str(a) for a in raw_answers.get("text", []))
    else:
        answers = tuple(
            str(a["answer"]) if isinstance(a, dict) else str(a) for a in raw_answers
        )
    if not answers:
        return None
    return str(question), answers


def build_pilot(
    config: DatasetsConfig,
    *,
    manifest_path: str | Path,
    max_image_side: int = 2048,
) -> list[PilotExample]:
    """Sample the pilot mixture, materialise images, and write the manifest.

    Streams each source split, reservoir-samples ``per_dataset[name]`` rows
    with the configured seed, saves images as PNG under ``config.image_dir``,
    and writes the manifest. Requires the ``datasets`` extra.
    """
    try:
        import datasets as hf_datasets
    except ImportError as error:
        raise ImportError("pip install 'gwel[data]' to build the pilot dataset") from error

    from PIL import Image

    from ..modeling.imaging import downscale

    image_dir = Path(config.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    examples: list[PilotExample] = []

    for name, count in config.per_dataset.items():
        if count <= 0:
            continue
        if name not in DATASET_SOURCES:
            raise KeyError(f"unknown pilot dataset {name!r}; known: {sorted(DATASET_SOURCES)}")
        source = DATASET_SOURCES[name]
        stream = hf_datasets.load_dataset(
            source["path"],
            name=source.get("name"),
            split=source["split"],
            streaming=True,
        )

        # Reservoir sampling over the stream keeps memory flat and the seed
        # makes the pilot reproducible for a fixed source snapshot.
        reservoir: list[tuple[int, dict[str, object]]] = []
        for index, row in enumerate(stream):
            if len(reservoir) < count:
                reservoir.append((index, dict(row)))
            else:
                slot = rng.randint(0, index)
                if slot < count:
                    reservoir[slot] = (index, dict(row))
        reservoir.sort(key=lambda item: item[0])

        for index, row in reservoir:
            qa = _extract_qa(name, row)
            image = row.get("image")
            if qa is None or not isinstance(image, Image.Image):
                continue
            example_id = f"{name}-{index:06d}"
            image_path = image_dir / f"{example_id}.png"
            if not image_path.exists():
                downscale(image.convert("RGB"), max_image_side).save(image_path)
            examples.append(
                PilotExample(
                    example_id=example_id,
                    dataset=name,
                    image_path=str(image_path),
                    question=qa[0],
                    answers=qa[1],
                )
            )

    write_manifest(manifest_path, examples)
    return examples
