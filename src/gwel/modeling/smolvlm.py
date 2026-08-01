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


@dataclass(frozen=True)
class ComponentTiming:
    """Latency attributed to each stage of one forward pass.

    Following the profiling decomposition of Shin et al. (arXiv 2607.08029),
    who separate vision encoder, projector and LLM cost with explicit CUDA
    synchronisation. Subtracting a text-only pass from an image pass — which is
    how we previously estimated the visual share — conflates all three, and
    they scale differently: encoder with pixels, projector with patches, LLM
    attention with sequence length.
    """

    vision_encoder_ms: float
    projector_ms: float
    prefill_ms: float
    decode_ms: float
    visual_tokens: int

    @property
    def total_ms(self) -> float:
        return self.vision_encoder_ms + self.projector_ms + self.prefill_ms + self.decode_ms

    @property
    def vision_share(self) -> float:
        """Fraction of the pass spent producing and projecting visual tokens."""
        total = self.total_ms
        return (self.vision_encoder_ms + self.projector_ms) / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "vision_encoder_ms": self.vision_encoder_ms,
            "projector_ms": self.projector_ms,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "visual_tokens": self.visual_tokens,
            "vision_share": self.vision_share,
        }


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
        import transformers
        from transformers import AutoProcessor

        rss_before = psutil.Process().memory_info().rss
        start = time.perf_counter()

        dtype = getattr(torch, self.config.dtype)
        device = self._resolve_device()
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)

        # AutoModelForVision2Seq is deprecated and does not recognise newer
        # configs such as SmolVLM2's; prefer the current class where available.
        loader = getattr(transformers, "AutoModelForImageTextToText", None)
        if loader is None:
            loader = transformers.AutoModelForVision2Seq
        self._model = loader.from_pretrained(self.config.model_id, dtype=dtype).to(device)
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

    def _prepare_inputs(self, prompt: str, image_list: list[Image.Image]):
        """Tokenise text and images, keeping visual tokens tied to resolution.

        The Idefics3 image processor upscales any input to its configured
        longest edge, which would make a 256 px preview cost as many visual
        tokens as a 1536 px pass. Capping the target at the largest input side
        keeps the token count proportional to the chosen resolution.
        """
        image_processor = self._processor.image_processor
        default_size = dict(image_processor.size)
        if image_list and "longest_edge" in default_size:
            largest_input = max(max(image.size) for image in image_list)
            image_processor.size = {
                "longest_edge": min(default_size["longest_edge"], largest_input)
            }
        try:
            return self._processor(
                text=prompt,
                images=image_list or None,
                return_tensors="pt",
            ).to(self._model.device)
        finally:
            image_processor.size = default_size

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
        inputs = self._prepare_inputs(prompt, image_list)

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

    def extract_activations(
        self,
        images: Sequence[Image.Image] | None,
        question: str,
        *,
        context_text: str | None = None,
        position: int = -1,
    ) -> np.ndarray:
        """Residual-stream activations at one prompt position, per layer.

        Returns an ``(num_layers + 1, hidden_size)`` array taken from a single
        forward pass over the prompt, **before any token is generated**. The
        default position is the last prompt token, following Moreno Cencerrado
        et al. (arXiv 2509.10625) and Lugoloobi et al. (arXiv 2602.09924), who
        both read the post-instruction position where "the model's internal
        assessment of the prompt crystallizes".

        This costs one prefill and no decoding, which is what makes a probe on
        it cheaper than any signal read from generated tokens.
        """
        self.ensure_loaded()

        import torch

        image_list = list(images) if images else []
        prompt = self._build_prompt(len(image_list), question, context_text)
        inputs = self._prepare_inputs(prompt, image_list)

        with torch.inference_mode():
            outputs = self._model(**inputs, output_hidden_states=True)

        return np.stack(
            [state[0, position, :].float().cpu().numpy() for state in outputs.hidden_states]
        )

    def profile_components(
        self,
        images: Sequence[Image.Image] | None,
        question: str,
        *,
        context_text: str | None = None,
        repeats: int = 5,
        warmup: int = 2,
    ) -> ComponentTiming:
        """Time the vision encoder, projector, prefill and decode separately.

        Each stage is measured with an explicit CUDA synchronisation so the
        timings are not distorted by asynchronous kernel launches. Medians over
        ``repeats`` are reported after discarding ``warmup`` passes.
        """
        self.ensure_loaded()

        import torch

        image_list = list(images) if images else []
        prompt = self._build_prompt(len(image_list), question, context_text)
        inputs = self._prepare_inputs(prompt, image_list)
        pixel_values = inputs.get("pixel_values")
        input_ids = inputs["input_ids"]
        visual_tokens = 0
        if self._image_token_id is not None and self._image_token_id >= 0:
            visual_tokens = int((input_ids == self._image_token_id).sum().item())

        model = self._model
        # Attribute names differ between Idefics3 and SmolVLM2 checkpoints.
        inner = getattr(model, "model", model)
        vision_model = getattr(inner, "vision_model", None) or getattr(
            inner, "vision_tower", None
        )
        connector = getattr(inner, "connector", None) or getattr(
            inner, "multi_modal_projector", None
        )

        # Idefics3 carries pixel_values as (batch, num_images, C, H, W); the
        # vision tower takes a plain 4D batch, so fold the image axis in. The
        # processor emits float32 while the tower runs in the model's dtype.
        if pixel_values is not None:
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.flatten(0, 1)
            pixel_values = pixel_values.to(dtype=model.dtype)

        def sync() -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        encoder_ms: list[float] = []
        projector_ms: list[float] = []
        prefill_ms: list[float] = []
        decode_ms: list[float] = []

        with torch.inference_mode():
            for index in range(warmup + repeats):
                sync()
                start = time.perf_counter()
                features = (
                    vision_model(pixel_values=pixel_values).last_hidden_state
                    if vision_model is not None and pixel_values is not None
                    else None
                )
                sync()
                after_encoder = time.perf_counter()

                if connector is not None and features is not None:
                    connector(features)
                sync()
                after_projector = time.perf_counter()

                model(**inputs)
                sync()
                after_prefill = time.perf_counter()

                model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
                sync()
                after_decode = time.perf_counter()

                if index < warmup:
                    continue
                encoder_ms.append((after_encoder - start) * 1000.0)
                projector_ms.append((after_projector - after_encoder) * 1000.0)
                prefill_ms.append((after_prefill - after_projector) * 1000.0)
                # generate() repeats the prefill, so subtract it out.
                decode_ms.append(
                    max((after_decode - after_prefill) * 1000.0 - prefill_ms[-1], 0.0)
                )

        median = lambda values: float(np.median(values)) if values else 0.0  # noqa: E731
        return ComponentTiming(
            vision_encoder_ms=median(encoder_ms),
            projector_ms=median(projector_ms),
            prefill_ms=median(prefill_ms),
            decode_ms=median(decode_ms),
            visual_tokens=visual_tokens,
        )
