"""Portable JSONL format for raw active-perception measurements."""

import json
from dataclasses import dataclass
from pathlib import Path

from .oracle import ActionMeasurement
from .router import Action


@dataclass(frozen=True)
class BenchmarkExample:
    """One question and all actions evaluated for it."""

    example_id: str
    question: str
    category: str
    measurements: tuple[ActionMeasurement, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "question": self.question,
            "category": self.category,
            "measurements": [
                {
                    "action": measurement.action.value,
                    "correct": measurement.correct,
                    "latency_ms": measurement.latency_ms,
                    "memory_mb": measurement.memory_mb,
                    "energy_mj": measurement.energy_mj,
                    "visual_tokens": measurement.visual_tokens,
                }
                for measurement in self.measurements
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BenchmarkExample":
        raw_measurements = payload.get("measurements")
        if not isinstance(raw_measurements, list):
            raise ValueError("measurements must be a list")

        measurements = tuple(
            ActionMeasurement(
                action=Action(item["action"]),
                correct=bool(item["correct"]),
                latency_ms=float(item["latency_ms"]),
                memory_mb=float(item["memory_mb"]),
                energy_mj=float(item["energy_mj"]),
                visual_tokens=int(item["visual_tokens"]),
            )
            for item in raw_measurements
        )
        return cls(
            example_id=str(payload["example_id"]),
            question=str(payload["question"]),
            category=str(payload["category"]),
            measurements=measurements,
        )


def write_jsonl(path: Path, examples: list[BenchmarkExample]) -> None:
    """Write benchmark examples as UTF-8 JSONL, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[BenchmarkExample]:
    """Read benchmark examples and reject blank or malformed records."""
    examples: list[BenchmarkExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank record at line {line_number}")
            try:
                payload = json.loads(line)
                examples.append(BenchmarkExample.from_dict(payload))
            except (AttributeError, TypeError, KeyError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid benchmark record at line {line_number}") from error
    return examples
