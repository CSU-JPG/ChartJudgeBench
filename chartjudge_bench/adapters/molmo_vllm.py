from __future__ import annotations

from typing import Any

from ._multimodal import close_images, flatten_messages
from .base import ModelAdapter
from ..protocols import EvaluationRequest


class MolmoVLLMAdapter(ModelAdapter):
    """vLLM adapter for Molmo-7B-D-0924 paper inference."""

    thread_safe = False

    def __init__(
        self,
        *,
        model_path: str,
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        generation: dict[str, Any] | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("Install the model-compatible vLLM environment") from exc
        kwargs = dict(engine_kwargs or {})
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("dtype", dtype)
        kwargs.setdefault("gpu_memory_utilization", gpu_memory_utilization)
        kwargs.setdefault("max_model_len", max_model_len)
        kwargs.setdefault("limit_mm_per_prompt", {"image": 2})
        self.engine = LLM(model=model_path, **kwargs)
        self.sampling = SamplingParams(**dict(generation or {}))

    def generate(self, request: EvaluationRequest) -> str:
        prompt, images = flatten_messages(request.messages, image_marker="")
        try:
            outputs = self.engine.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": images}}],
                sampling_params=self.sampling,
                use_tqdm=False,
            )
            return outputs[0].outputs[0].text.strip()
        finally:
            close_images(images)
