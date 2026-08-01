"""Oracle labeling: cheapest routable action that answered correctly.

The oracle is computed offline from run records. For each example it takes
the argmin of the cost J over configurations whose answer was correct,
restricted to routable actions (diagnostic configs are excluded). Ties break
deterministically on (cost, action lightness, config_id), so an equally cheap
lighter action wins.
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..actions import Action
from .cost import CostWeights, compute_cost
from .records import RunRecord


@dataclass(frozen=True)
class OracleLabel:
    """Minimal sufficient configuration for one example."""

    example_id: str
    dataset: str
    config_id: str | None      # None when no routable action was correct
    action: Action | None
    cost: float | None
    config_costs: dict[str, float]
    any_correct: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "dataset": self.dataset,
            "config_id": self.config_id,
            "action": self.action.value if self.action is not None else None,
            "cost": self.cost,
            "config_costs": self.config_costs,
            "any_correct": self.any_correct,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OracleLabel":
        action_raw = payload["action"]
        return cls(
            example_id=str(payload["example_id"]),
            dataset=str(payload["dataset"]),
            config_id=None if payload["config_id"] is None else str(payload["config_id"]),
            action=Action(action_raw) if action_raw is not None else None,
            cost=None if payload["cost"] is None else float(payload["cost"]),  # type: ignore[arg-type]
            config_costs={str(k): float(v) for k, v in (payload["config_costs"] or {}).items()},  # type: ignore[union-attr]
            any_correct=bool(payload["any_correct"]),
        )


def record_cost(record: RunRecord, weights: CostWeights = CostWeights()) -> float:
    """Evaluate J on one record's measured outcome.

    Energy uses the idle-baseline-subtracted value when available, so the
    cost reflects the marginal draw of the action rather than the whole board.
    """
    return compute_cost(
        correct=record.correct,
        latency_ms=record.latency_ms,
        energy_mj=record.net_energy_mj,
        memory_mb=record.ram_peak_mb,
        visual_tokens=record.visual_tokens,
        weights=weights,
    )


def label_example(
    records: Sequence[RunRecord],
    *,
    weights: CostWeights = CostWeights(),
) -> OracleLabel:
    """Label one example from all its measured configurations."""
    if not records:
        raise ValueError("at least one record is required")
    example_id = records[0].example_id
    if any(record.example_id != example_id for record in records):
        raise ValueError("all records must belong to the same example")

    config_costs = {record.config_id: record_cost(record, weights) for record in records}
    routable = [record for record in records if record.action is not None]
    correct = [record for record in routable if record.correct]

    action_rank = {action: rank for rank, action in enumerate(Action.ordered())}
    selected: RunRecord | None = None
    if correct:
        selected = min(
            correct,
            key=lambda record: (
                config_costs[record.config_id],
                action_rank[record.action],
                record.config_id,
            ),
        )

    return OracleLabel(
        example_id=example_id,
        dataset=records[0].dataset,
        config_id=selected.config_id if selected else None,
        action=selected.action if selected else None,
        cost=config_costs[selected.config_id] if selected else None,
        config_costs=config_costs,
        any_correct=bool(correct),
    )


def derive_labels(
    records: Iterable[RunRecord],
    *,
    weights: CostWeights = CostWeights(),
) -> list[OracleLabel]:
    """Group records by example and label each one; ordered by first appearance."""
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    order: list[str] = []
    for record in records:
        if record.example_id not in grouped:
            order.append(record.example_id)
        grouped[record.example_id].append(record)
    return [label_example(grouped[example_id], weights=weights) for example_id in order]


def write_labels(path: str | Path, labels: Iterable[OracleLabel]) -> None:
    """Write oracle labels as UTF-8 JSONL, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(json.dumps(label.to_dict(), sort_keys=True) + "\n")


def read_labels(path: str | Path) -> list[OracleLabel]:
    """Read oracle labels from JSONL."""
    labels: list[OracleLabel] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                labels.append(OracleLabel.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid oracle label at line {line_number} of {path}") from error
    return labels
