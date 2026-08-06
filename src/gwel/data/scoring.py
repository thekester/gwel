"""Replayable correctness scoring, decoupled from the measurement run.

Records store the raw answer and the gold answers, so the correctness policy
can be changed and re-applied offline without re-running the model. Each
dataset gets the metric its benchmark defines: VQA accuracy for VQAv2
and TextVQA, ANLS for DocVQA, exact match for V*Bench multiple choice.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from ..oracle.records import RunRecord
from .vqa_metrics import anls, exact_match, relaxed_accuracy, vqa_accuracy

#: Metric name per dataset; datasets not listed fall back to ``default_metric``.
DATASET_METRICS: dict[str, str] = {
    "vqav2": "vqa",
    "textvqa": "vqa",
    "docvqa": "anls",
    "infographicvqa": "anls",
    "chartqa": "relaxed",
    "vstar": "exact",
}


@dataclass(frozen=True)
class ScoringPolicy:
    """How a prediction is turned into a correctness verdict."""

    dataset_metrics: dict[str, str] | None = None
    default_metric: str = "vqa"
    correct_threshold: float = 0.5
    anls_threshold: float = 0.5

    def metric_for(self, dataset: str) -> str:
        metrics = self.dataset_metrics if self.dataset_metrics is not None else DATASET_METRICS
        return metrics.get(dataset, self.default_metric)

    def score(self, dataset: str, prediction: str, gold_answers: Sequence[str]) -> float:
        """Return the graded score in [0, 1] for one prediction."""
        metric = self.metric_for(dataset)
        if metric == "vqa":
            return vqa_accuracy(prediction, list(gold_answers))
        if metric == "anls":
            return anls(prediction, list(gold_answers), threshold=self.anls_threshold)
        if metric == "relaxed":
            return relaxed_accuracy(prediction, list(gold_answers))
        if metric == "exact":
            return 1.0 if exact_match(prediction, list(gold_answers)) else 0.0
        raise ValueError(f"unknown metric {metric!r} for dataset {dataset!r}")

    def is_correct(self, dataset: str, prediction: str, gold_answers: Sequence[str]) -> bool:
        return self.score(dataset, prediction, gold_answers) >= self.correct_threshold


def rescore_records(
    records: Iterable[RunRecord],
    policy: ScoringPolicy = ScoringPolicy(),
) -> list[RunRecord]:
    """Re-derive ``vqa_score``, ``exact_match`` and ``correct`` under ``policy``.

    Hardware measurements are untouched; only the verdict changes, which keeps
    a single expensive run reusable across scoring choices.
    """
    rescored: list[RunRecord] = []
    for record in records:
        score = policy.score(record.dataset, record.answer, record.gold_answers)
        rescored.append(
            replace(
                record,
                vqa_score=score,
                exact_match=exact_match(record.answer, list(record.gold_answers)),
                correct=score >= policy.correct_threshold,
                meta={**record.meta, "metric": policy.metric_for(record.dataset)},
            )
        )
    return rescored
