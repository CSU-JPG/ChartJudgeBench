from __future__ import annotations

from typing import Any

from .base import ModelAdapter
from ..protocols import EvaluationRequest


class QwenTransformersAdapter(ModelAdapter):
    """Local Transformers adapter for Qwen-family vision-language models."""

    thread_safe = False

    def __init__(
        self,
        *,
        model_path: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        max_pixels: int | None = 1024 * 1024,
        min_pixels: int | None = None,
        model_classes: list[str] | None = None,
        request_suffix: str = "",
        request_suffix_by_task: dict[str, str] | None = None,
        generation: dict[str, Any] | None = None,
        model_load_kwargs: dict[str, Any] | None = None,
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
        self.request_suffix = request_suffix
        self.request_suffix_by_task = dict(request_suffix_by_task or {})
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if max_pixels is not None and hasattr(image_processor, "max_pixels"):
                image_processor.max_pixels = max_pixels
            if min_pixels is not None and hasattr(image_processor, "min_pixels"):
                image_processor.min_pixels = min_pixels

        class_candidates = model_classes or ["AutoModelForImageTextToText"]
        load_kwargs = dict(model_load_kwargs or {})
        load_kwargs.setdefault("torch_dtype", getattr(torch, torch_dtype))
        load_kwargs.setdefault("device_map", device_map)
        load_kwargs.setdefault("trust_remote_code", trust_remote_code)
        errors = []
        for class_name in class_candidates:
            try:
                model_class = getattr(transformers, class_name)
                self.model = model_class.from_pretrained(
                    model_path, **load_kwargs
                ).eval()
                break
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
        else:
            raise RuntimeError(
                "Unable to load model with configured classes: " + " | ".join(errors)
            )
        self.generation = dict(generation or {})

    def generate(self, request: EvaluationRequest) -> str:
        messages = request.messages
        suffix = self.request_suffix_by_task.get(request.task, self.request_suffix)
        if suffix:
            messages = [
                {
                    "role": message["role"],
                    "content": (
                        [dict(block) for block in message["content"]]
                        if isinstance(message["content"], list)
                        else message["content"]
                    ),
                }
                for message in request.messages
            ]
            for message in reversed(messages):
                if isinstance(message["content"], list):
                    for block in reversed(message["content"]):
                        if block.get("type") == "text":
                            block["text"] += suffix
                            break
                    else:
                        continue
                    break
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
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

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_classes", ["Qwen2_5_VLForConditionalGeneration"])
        super().__init__(**kwargs)
