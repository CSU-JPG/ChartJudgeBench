from __future__ import annotations

from typing import Any

from ._multimodal import close_images, messages_with_image_placeholders
from .base import ModelAdapter
from ..protocols import EvaluationRequest


class TransformersChatAdapter(ModelAdapter):
    """Configurable adapter for Transformers models with multimodal chat templates.

    ``image_mode='separate'`` matches Kimi-style processors that receive a text
    prompt plus a separate image list. ``image_mode='inline'`` matches GLM-4V
    and Molmo2 processors that tokenize PIL images embedded in the messages.
    """

    thread_safe = False

    def __init__(
        self,
        *,
        model_path: str,
        model_class: str = "AutoModelForImageTextToText",
        image_mode: str = "separate",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        generation: dict[str, Any] | None = None,
        model_load_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        drop_input_keys: list[str] | None = None,
    ) -> None:
        if image_mode not in {"separate", "inline"}:
            raise ValueError("image_mode must be 'separate' or 'inline'")
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "Install the model's official PyTorch/Transformers environment first"
            ) from exc

        self.torch = torch
        self.image_mode = image_mode
        self.generation = dict(generation or {})
        self.drop_input_keys = list(drop_input_keys or [])
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            **dict(processor_kwargs or {}),
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        try:
            model_type = getattr(transformers, model_class)
        except AttributeError as exc:
            raise RuntimeError(
                f"{model_class} is unavailable in the installed Transformers version"
            ) from exc
        load_kwargs = dict(model_load_kwargs or {})
        load_kwargs.setdefault("torch_dtype", getattr(torch, torch_dtype))
        load_kwargs.setdefault("device_map", device_map)
        load_kwargs.setdefault("trust_remote_code", trust_remote_code)
        self.model = model_type.from_pretrained(model_path, **load_kwargs).eval()

    def generate(self, request: EvaluationRequest) -> str:
        prepared, images = messages_with_image_placeholders(
            request.messages,
            inline_pil=self.image_mode == "inline",
        )
        try:
            if self.image_mode == "inline":
                inputs = self.processor.apply_chat_template(
                    prepared,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self.model.device) for key, value in inputs.items()
                }
            else:
                text = self.processor.apply_chat_template(
                    prepared,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.processor(
                    text=[text],
                    images=images,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)

            for key in self.drop_input_keys:
                inputs.pop(key, None)
            with self.torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **self.generation)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = generated_ids[:, input_length:]
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            return tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        finally:
            close_images(images)
