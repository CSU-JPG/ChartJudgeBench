from __future__ import annotations

from typing import Any

from .base import ModelAdapter
from ..protocols import EvaluationRequest


class QwenTransformersAdapter(ModelAdapter):
    """Local Transformers adapter for Qwen-family vision-language models."""

    thread_safe = False
    model_class_name = "AutoModelForImageTextToText"

    def __init__(
        self,
        *,
        model_path: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        max_pixels: int | None = 1024 * 1024,
        min_pixels: int | None = None,
        generation: dict[str, Any] | None = None,
    ) -> None:
        try:
            import torch
            import transformers
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "Install local Qwen support with: pip install -e '.[qwen]'"
            ) from exc

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if max_pixels is not None and hasattr(image_processor, "max_pixels"):
                image_processor.max_pixels = max_pixels
            if min_pixels is not None and hasattr(image_processor, "min_pixels"):
                image_processor.min_pixels = min_pixels

        dtype = getattr(torch, torch_dtype)
        model_class = getattr(transformers, self.model_class_name)
        self.model = model_class.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        ).eval()
        self.generation = dict(generation or {})

    def generate(self, request: EvaluationRequest) -> str:
        text = self.processor.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(request.messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.generation)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class ThinkLiteAdapter(QwenTransformersAdapter):
    """Paper adapter for ThinkLite-VL-7B."""

    model_class_name = "Qwen2_5_VLForConditionalGeneration"

