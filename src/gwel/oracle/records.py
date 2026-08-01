"""Serializable per-(example, config) run records and their JSONL/Parquet IO.

One record captures everything observed while answering one question under
one visual configuration: the answer, its correctness, confidence signals,
and all hardware measurements. Records are appended incrementally so a run
can be resumed after interruption.
"""

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..actions import Action


@dataclass(frozen=True)
class RunRecord:
    """Full observation of one visual configuration on one example."""

    example_id: str
    dataset: str
    question: str
    gold_answers: tuple[str, ...]
    config_id: str
    action: Action | None  # None for diagnostic configs (no_image, full)
    answer: str
    exact_match: bool
    vqa_score: float
    correct: bool
    latency_ms: float          # median generation + tool overhead
    latency_stats: dict[str, object] | None
    ttft_ms: float | None
    ram_peak_mb: float | None
    vram_peak_mb: float | None
    energy_mj: dict[str, float | None]
    visual_tokens: int
    prompt_tokens: int
    generated_tokens: int
    signals: dict[str, float | int] | None
    meta: dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "example_id": self.example_id,
            "dataset": self.dataset,
            "question": self.question,
            "gold_answers": list(self.gold_answers),
            "config_id": self.config_id,
            "action": self.action.value if self.action is not None else None,
            "answer": self.answer,
            "exact_match": self.exact_match,
            "vqa_score": self.vqa_score,
            "correct": self.correct,
            "latency_ms": self.latency_ms,
            "latency_stats": self.latency_stats,
            "ttft_ms": self.ttft_ms,
            "ram_peak_mb": self.ram_peak_mb,
            "vram_peak_mb": self.vram_peak_mb,
            "energy_mj": self.energy_mj,
            "visual_tokens": self.visual_tokens,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "signals": self.signals,
            "meta": self.meta,
            "timestamp": self.timestamp,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RunRecord":
        action_raw = payload["action"]
        return cls(
            example_id=str(payload["example_id"]),
            dataset=str(payload["dataset"]),
            question=str(payload["question"]),
            gold_answers=tuple(str(a) for a in payload["gold_answers"]),  # type: ignore[union-attr]
            config_id=str(payload["config_id"]),
            action=Action(action_raw) if action_raw is not None else None,
            answer=str(payload["answer"]),
            exact_match=bool(payload["exact_match"]),
            vqa_score=float(payload["vqa_score"]),  # type: ignore[arg-type]
            correct=bool(payload["correct"]),
            latency_ms=float(payload["latency_ms"]),  # type: ignore[arg-type]
            latency_stats=payload.get("latency_stats"),  # type: ignore[arg-type]
            ttft_ms=None if payload.get("ttft_ms") is None else float(payload["ttft_ms"]),  # type: ignore[arg-type]
            ram_peak_mb=None if payload.get("ram_peak_mb") is None else float(payload["ram_peak_mb"]),  # type: ignore[arg-type]
            vram_peak_mb=None if payload.get("vram_peak_mb") is None else float(payload["vram_peak_mb"]),  # type: ignore[arg-type]
            energy_mj=dict(payload.get("energy_mj") or {}),  # type: ignore[arg-type]
            visual_tokens=int(payload["visual_tokens"]),  # type: ignore[arg-type]
            prompt_tokens=int(payload["prompt_tokens"]),  # type: ignore[arg-type]
            generated_tokens=int(payload["generated_tokens"]),  # type: ignore[arg-type]
            signals=payload.get("signals"),  # type: ignore[arg-type]
            meta=dict(payload.get("meta") or {}),  # type: ignore[arg-type]
            timestamp=float(payload.get("timestamp", 0.0)),  # type: ignore[arg-type]
        )

    @property
    def total_energy_mj(self) -> float | None:
        return self.energy_mj.get("total")

    @property
    def key(self) -> tuple[str, str]:
        """Resume key identifying this record within a run file."""
        return (self.example_id, self.config_id)


def append_records(path: str | Path, records: Iterable[RunRecord]) -> None:
    """Append records as UTF-8 JSONL, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def read_records(path: str | Path) -> list[RunRecord]:
    """Read a JSONL run file, rejecting malformed lines with their number."""
    records: list[RunRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(RunRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid run record at line {line_number} of {path}") from error
    return records


def load_done_keys(path: str | Path) -> set[tuple[str, str]]:
    """Return (example_id, config_id) pairs already present in a run file."""
    path = Path(path)
    if not path.exists():
        return set()
    return {record.key for record in read_records(path)}


def records_to_parquet(records: Sequence[RunRecord], path: str | Path) -> None:
    """Flatten records into a Parquet table (requires pandas + pyarrow)."""
    import pandas as pd

    rows = []
    for record in records:
        row: dict[str, object] = record.to_dict()
        signals = row.pop("signals") or {}
        for name, value in signals.items():  # type: ignore[union-attr]
            row[f"signal_{name}"] = value
        energy = row.pop("energy_mj") or {}
        for name, value in energy.items():  # type: ignore[union-attr]
            row[f"energy_{name}_mj"] = value
        row["gold_answers"] = json.dumps(row["gold_answers"])
        row["meta"] = json.dumps(row["meta"], sort_keys=True)
        row.pop("latency_stats", None)
        rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
