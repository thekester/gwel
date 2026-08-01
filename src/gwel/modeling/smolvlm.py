"""SmolVLM inference engine with confidence and cost instrumentation.

Torch and transformers are imported lazily so the rest of the package (oracle
labeling, router training on cached records, tests) works without them.
SmolVLM requires ``transformers >= 4.46`` (Idefics3 architecture).
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..config import ModelConfig
from ..profiling.timers import FirstTokenTimer
from .signals import ConfidenceSignals, signals_from_scores

_MIN_TRANSFORMERS = (4, 46)


@dataclass(frozen=True)
class GenerationOutput:
    """Answer plus every signal and count the runner logs per pass."""

    answer: str
    signals: ConfidenceSignals
    visual_tokens: int
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float | None
    generate_ms: float


@dataclass(frozen=True)
class LoadReport:
    """Wall-clock and memory cost of loading the model into this process."""

    load_ms: float
    ram_delta_mb: float


class SmolVlmEngine:
    """Lazily loaded SmolVLM wrapper producing instrumented generations."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None
        self._image_token_id: int | None = None
        self.load_report: LoadReport | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _check_transformers_version(self) -> None:
        import transformers

        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        if version < _MIN_TRANSFORMERS:
            raise RuntimeError(
                f"SmolVLM needs transformers >= {'.'.join(map(str, _MIN_TRANSFORMERS))}, "
                f"found {transformers.__version__}; pip install -U 'transformers>=4.49'"
            )

    def _resolve_device(self) -> str:
        import torch

        if self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_loaded(self) -> None:
        """Load processor and weights once, recording the warm-process cost."""
        if self._model is not None:
            return
        self._check_transformers_version()

        import psutil
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        rss_before = psutil.Process().memory_info().rss
        start = time.perf_counter()

        dtype = getattr(torch, self.config.dtype)
        device = self._resolve_device()
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_id, torch_dtype=dtype
        ).to(device)
        self._model.eval()

        image_token = getattr(self._processor, "image_token", None) or "<image>"
        if not isinstance(image_token, str):
            image_token = str(image_token)
        self._image_token_id = self._processor.tokenizer.convert_tokens_to_ids(image_token)

        self.load_report = LoadReport(
            load_ms=(time.perf_counter() - start) * 1000.0,
            ram_delta_mb=(psutil.Process().memory_info().rss - rss_before) / 1e6,
        )

    def _build_prompt(
        self,
        num_images: int,
        question: str,
        context_text: str | None,
    ) -> str:
        """Assemble the chat-template prompt for ``num_images`` images."""
        content: list[dict[str, str]] = [{"type": "image"} for _ in range(num_images)]
        parts = []
        if context_text:
            parts.append(f"Extracted text from the image:\n{context_text}")
        parts.append(question)
        parts.append(self.config.answer_prompt)
        content.append({"type": "text", "text": "\n".join(parts)})
        messages = [{"role": "user", "content": content}]
        return self._processor.apply_chat_template(messages, add_generation_prompt=True)

    def generate(
        self,
        images: Sequence[Image.Image] | None,
        question: str,
        *,
        context_text: str | None = None,
    ) -> GenerationOutput:
        """Answer ``question`` from ``images`` (or blind when ``None``).

        Greedy decoding with per-step scores; returns the decoded answer with
        confidence signals, token counts, time-to-first-token, and total
        generation wall time.
        """
        self.ensure_loaded()

        import torch

        image_list = list(images) if images else []
        prompt = self._build_prompt(len(image_list), question, context_text)
        inputs = self._processor(
            text=prompt,
            images=image_list or None,
            return_tensors="pt",
        ).to(self._model.device)

        input_ids = inputs["input_ids"]
        prompt_tokens = int(input_ids.shape[1])
        visual_tokens = 0
        if self._image_token_id is not None and self._image_token_id >= 0:
            visual_tokens = int((input_ids == self._image_token_id).sum().item())

        ttft = FirstTokenTimer()
        ttft.arm()
        start = time.perf_counter()
        with torch.inference_mode():
            result = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                streamer=ttft,
            )
        generate_ms = (time.perf_counter() - start) * 1000.0

        sequences = result.sequences[0]
        generated_ids = sequences[prompt_tokens:]
        answer = self._processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        step_logits = [score[0].float().cpu().numpy().astype(np.float64) for score in result.scores]
        chosen_ids = [int(token) for token in generated_ids[: len(step_logits)]]
        signals = signals_from_scores(step_logits, chosen_ids)

        return GenerationOutput(
            answer=answer,
            signals=signals,
            visual_tokens=visual_tokens,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(chosen_ids),
            ttft_ms=ttft.ttft_ms,
            generate_ms=generate_ms,
        )
