"""Pilot dataset construction: a small, seeded, reproducible mixture.

The pilot covers four question regimes: general VQA (VQAv2), scene text
(TextVQA), documents (DocVQA), and fine-grained detail (V*Bench). Images are
materialised to local files and the mixture is described by a JSONL manifest
so the oracle runner never depends on ``datasets`` at run time.
"""

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..config import DatasetsConfig

#: HuggingFace sources for each pilot subset. All are validation/test splits
#: with public answers, streamed so only the sampled rows are downloaded.
DATASET_SOURCES: dict[str, dict[str, str]] = {
    "vqav2": {"path": "lmms-lab/VQAv2", "split": "validation"},
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


def _vstar_gold_answers(options: list[str]) -> tuple[str, ...]:
    """Gold answers from V*Bench options (index 0 is correct by convention).

    Options are templated sentences ("The color of the flag is white."); the
    short-form answer is the part of option 0 not shared with the others, so
    both the short form and the full sentence are returned for matching.
    """
    correct = options[0].strip()
    if len(options) < 2:
        return (correct,)
    prefix_len = 0
    for index in range(min(len(option) for option in options)):
        if all(option[index] == correct[index] for option in options[1:]):
            prefix_len = index + 1
        else:
            break
    short = correct[prefix_len:].strip().rstrip(".").strip()
    answers = [short, correct] if short else [correct]
    # Positional answers ("right side of the box") should also accept the
    # bare direction, which is what a short-phrase prompt elicits.
    head = short.split()[0].lower() if short else ""
    if head in ("left", "right", "top", "bottom", "above", "below") and head != short.lower():
        answers.append(head)
    return tuple(answers)


def _build_vstar(
    count: int,
    *,
    seed: int,
    image_dir: Path,
    max_image_side: int,
) -> list[PilotExample]:
    """Sample V*Bench examples directly from its raw image+JSON layout.

    The repository is not an Arrow dataset: each example is a ``stem.json``
    metadata file next to ``stem.{jpg,png,webp,...}``. Sampling is seeded over
    the sorted JSON list, balanced across subfolders by interleaving.
    """
    import random

    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image

    from ..modeling.imaging import downscale

    repo = DATASET_SOURCES["vstar"]["path"]
    files = [f.rfilename for f in HfApi().dataset_info(repo).siblings]
    json_files = sorted(f for f in files if f.endswith(".json"))
    stems_to_image = {
        f.rsplit(".", 1)[0]: f for f in files if not f.endswith((".json", ".gitattributes"))
    }

    rng = random.Random(seed)
    candidates = [f for f in json_files if f.rsplit(".", 1)[0] in stems_to_image]
    rng.shuffle(candidates)

    examples: list[PilotExample] = []
    for json_name in candidates:
        if len(examples) >= count:
            break
        stem = json_name.rsplit(".", 1)[0]
        payload = json.loads(
            Path(hf_hub_download(repo, json_name, repo_type="dataset")).read_text(encoding="utf-8")
        )
        question = payload.get("question")
        options = payload.get("options")
        if not question or not isinstance(options, list) or not options:
            continue
        image_file = hf_hub_download(repo, stems_to_image[stem], repo_type="dataset")
        example_id = "vstar-" + stem.replace("/", "-")
        image_path = image_dir / f"{example_id}.png"
        if not image_path.exists():
            with Image.open(image_file) as image:
                downscale(image.convert("RGB"), max_image_side).save(image_path)
        examples.append(
            PilotExample(
                example_id=example_id,
                dataset="vstar",
                image_path=str(image_path),
                question=str(question),
                answers=_vstar_gold_answers([str(o) for o in options]),
            )
        )
    return examples


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
    # On Windows, loading PyArrow before an installed CUDA Torch can make
    # c10.dll initialization fail. Datasets detects Torch anyway, so establish
    # the safe DLL load order explicitly before importing datasets.
    if os.name == "nt":
        try:
            import torch  # noqa: F401
        except ImportError:
            pass

    try:
        import datasets as hf_datasets
    except ImportError as error:
        raise ImportError("pip install 'gwel[data]' to build the pilot dataset") from error

    from PIL import Image

    from ..modeling.imaging import downscale

    image_dir = Path(config.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    examples: list[PilotExample] = []

    for dataset_index, (name, count) in enumerate(config.per_dataset.items()):
        if count <= 0:
            continue
        if name not in DATASET_SOURCES:
            raise KeyError(f"unknown pilot dataset {name!r}; known: {sorted(DATASET_SOURCES)}")
        if name == "vstar":
            examples.extend(
                _build_vstar(
                    count,
                    seed=config.seed + dataset_index,
                    image_dir=image_dir,
                    max_image_side=max_image_side,
                )
            )
            continue
        source = DATASET_SOURCES[name]
        stream = hf_datasets.load_dataset(
            source["path"],
            name=source.get("name"),
            split=source["split"],
            streaming=True,
        )
        stream = stream.shuffle(
            seed=config.seed + dataset_index,
            buffer_size=config.shuffle_buffer_size,
        )

        selected = 0
        for index, raw_row in enumerate(stream):
            row = dict(raw_row)
            qa = _extract_qa(name, row)
            image = row.get("image")
            if qa is None or not isinstance(image, Image.Image):
                continue
            row_id = row.get("question_id") or row.get("id") or index
            example_id = f"{name}-{row_id}"
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
            selected += 1
            if selected >= count:
                break

    write_manifest(manifest_path, examples)
    return examples
