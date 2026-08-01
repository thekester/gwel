"""Shared factories for unit tests (no model, no GPU, no network)."""

import pytest

from gwel.actions import Action
from gwel.oracle.records import RunRecord


@pytest.fixture
def make_record():
    """Factory for RunRecords with sensible defaults, overridable per test."""

    def factory(
        example_id: str = "ex-1",
        config_id: str = "lowres_256",
        action: Action | None = Action.ANSWER_LOW,
        correct: bool = True,
        latency_ms: float = 100.0,
        energy_total_mj: float | None = 50.0,
        visual_tokens: int = 64,
        **overrides,
    ) -> RunRecord:
        defaults = dict(
            example_id=example_id,
            dataset="textvqa",
            question="What does the sign say?",
            gold_answers=("stop",),
            config_id=config_id,
            action=action,
            answer="stop" if correct else "go",
            exact_match=correct,
            vqa_score=1.0 if correct else 0.0,
            correct=correct,
            latency_ms=latency_ms,
            latency_stats=None,
            ttft_ms=30.0,
            ram_peak_mb=1500.0,
            vram_peak_mb=900.0,
            energy_mj={"nvml": energy_total_mj, "total": energy_total_mj},
            visual_tokens=visual_tokens,
            prompt_tokens=visual_tokens + 20,
            generated_tokens=3,
            signals={
                "mean_logprob": -0.4,
                "min_logprob": -1.1,
                "mean_entropy": 0.8,
                "max_entropy": 1.5,
                "first_entropy": 0.9,
                "mean_margin": 0.55,
                "min_margin": 0.2,
                "num_tokens": 3,
            },
            meta={"orig_width": 1024, "orig_height": 768},
        )
        defaults.update(overrides)
        return RunRecord(**defaults)

    return factory
