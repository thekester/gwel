"""Confidence signals derived from per-step generation scores.

All functions operate on numpy arrays so they stay testable without torch.
Scores are raw logits over the vocabulary, one row per generated token.
Entropies are in nats; margins are probability gaps between the top-1 and
top-2 tokens.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class ConfidenceSignals:
    """Scalar summaries of the model's certainty over one generated answer."""

    mean_logprob: float
    min_logprob: float
    mean_entropy: float
    max_entropy: float
    first_entropy: float
    mean_margin: float
    min_margin: float
    num_tokens: int

    def to_dict(self) -> dict[str, float | int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ConfidenceSignals":
        kwargs = {f.name: payload[f.name] for f in fields(cls)}
        kwargs["num_tokens"] = int(kwargs["num_tokens"])  # type: ignore[arg-type]
        return cls(**{k: (v if k == "num_tokens" else float(v)) for k, v in kwargs.items()})  # type: ignore[arg-type]


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def entropy_from_logits(logits: np.ndarray) -> float:
    """Shannon entropy (nats) of the categorical distribution given logits."""
    logprobs = log_softmax(np.asarray(logits, dtype=np.float64))
    probs = np.exp(logprobs)
    return float(-(probs * logprobs).sum())


def top2_margin_from_logits(logits: np.ndarray) -> float:
    """Probability gap between the two most likely tokens."""
    logprobs = log_softmax(np.asarray(logits, dtype=np.float64))
    top2 = np.sort(logprobs)[-2:]
    return float(np.exp(top2[1]) - np.exp(top2[0]))


def signals_from_scores(
    step_logits: Sequence[np.ndarray],
    chosen_ids: Sequence[int],
) -> ConfidenceSignals:
    """Summarize per-step logits and the sampled token ids into signals.

    ``step_logits[i]`` is the vocabulary logit row that produced token
    ``chosen_ids[i]``. Special tokens (e.g. EOS) may be included; they carry
    confidence information like any other step.
    """
    if len(step_logits) == 0:
        raise ValueError("at least one generation step is required")
    if len(step_logits) != len(chosen_ids):
        raise ValueError("step_logits and chosen_ids must have the same length")

    logprobs: list[float] = []
    entropies: list[float] = []
    margins: list[float] = []
    for logits, token_id in zip(step_logits, chosen_ids):
        row = log_softmax(np.asarray(logits, dtype=np.float64).reshape(-1))
        probs = np.exp(row)
        logprobs.append(float(row[token_id]))
        entropies.append(float(-(probs * row).sum()))
        top2 = np.sort(row)[-2:]
        margins.append(float(np.exp(top2[1]) - np.exp(top2[0])))

    return ConfidenceSignals(
        mean_logprob=float(np.mean(logprobs)),
        min_logprob=float(np.min(logprobs)),
        mean_entropy=float(np.mean(entropies)),
        max_entropy=float(np.max(entropies)),
        first_entropy=entropies[0],
        mean_margin=float(np.mean(margins)),
        min_margin=float(np.min(margins)),
        num_tokens=len(chosen_ids),
    )
