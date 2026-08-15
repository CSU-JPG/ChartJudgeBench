from __future__ import annotations

from typing import Any

from ._multimodal import close_images, flatten_messages
from .base import ModelAdapter
from ..protocols import EvaluationRequest


class DeepSeekVLAdapter(ModelAdapter):
    """Adapter for the original DeepSeek-VL multimodal chat implementation."""

    thread_safe = False

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        generation: dict[str, Any] | None = None,
    ) -> None:
        try:
            import torch
            from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "Install DeepSeek-VL from its official repository before using this adapter"
            ) from exc

        self.torch = torch
        self.device = device
        self.generation = dict(generation or {})
        self.processor = VLChatProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.model: MultiModalityCausalLM = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=getattr(torch, torch_dtype),
            )
            .to(device)
            .eval()
        )

    def generate(self, request: EvaluationRequest) -> str:
        prompt, images = flatten_messages(
            request.messages, image_marker="<image_placeholder>"
        )
        conversation = [
            {"role": "User", "content": prompt, "images": images},
            {"role": "Assistant", "content": ""},
        ]
        try:
            prepared = self.processor(
                conversations=conversation,
                images=images,
                force_batchify=True,
            ).to(self.model.device)
            with self.torch.inference_mode():
                embeddings = self.model.prepare_inputs_embeds(**prepared)
                outputs = self.model.language_model.generate(
                    inputs_embeds=embeddings,
                    attention_mask=prepared.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    bos_token_id=self.processor.tokenizer.bos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    **self.generation,
                )
            return self.processor.tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )
        finally:
            close_images(images)
