from __future__ import annotations

import math
from typing import Any

from ._multimodal import close_images, flatten_messages
from .base import ModelAdapter
from ..protocols import EvaluationRequest


class InternVLAdapter(ModelAdapter):
    """InternVL2.5/3.5 adapter based on the paper evaluation scripts."""

    thread_safe = False

    def __init__(
        self,
        *,
        model_path: str,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        image_size: int = 448,
        max_tiles: int = 12,
        use_thumbnail: bool = True,
        balanced_device_map: bool = True,
        generation: dict[str, Any] | None = None,
        model_load_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            import torch
            import torchvision.transforms as transforms
            from torchvision.transforms.functional import InterpolationMode
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install InternVL support with PyTorch, torchvision, and Transformers"
            ) from exc

        self.torch = torch
        self.dtype = getattr(torch, torch_dtype)
        self.image_size = image_size
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail
        self.generation = dict(generation or {})
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        load_kwargs = dict(model_load_kwargs or {})
        load_kwargs.setdefault("torch_dtype", getattr(torch, torch_dtype))
        load_kwargs.setdefault("low_cpu_mem_usage", True)
        load_kwargs.setdefault("use_flash_attn", False)
        load_kwargs.setdefault("trust_remote_code", trust_remote_code)
        if balanced_device_map:
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
            load_kwargs.setdefault("device_map", self._balanced_device_map(config))
        else:
            load_kwargs.setdefault("device_map", "auto")
        self.model = AutoModel.from_pretrained(model_path, **load_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )

    def _balanced_device_map(self, config: Any) -> dict[str, int]:
        world_size = self.torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError(
                "InternVL paper inference requires at least one CUDA device"
            )
        layers = config.llm_config.num_hidden_layers
        per_gpu = math.ceil(layers / (world_size - 0.5))
        allocation = [per_gpu] * world_size
        allocation[0] = math.ceil(allocation[0] * 0.5)
        device_map: dict[str, int] = {}
        layer_index = 0
        for device, count in enumerate(allocation):
            for _ in range(count):
                if layer_index >= layers:
                    break
                device_map[f"language_model.model.layers.{layer_index}"] = device
                layer_index += 1
        for name in (
            "vision_model",
            "mlp1",
            "language_model.model.tok_embeddings",
            "language_model.model.embed_tokens",
            "language_model.output",
            "language_model.model.norm",
            "language_model.model.rotary_emb",
            "language_model.lm_head",
        ):
            device_map[name] = 0
        device_map[f"language_model.model.layers.{layers - 1}"] = 0
        return device_map

    def _tiles(self, image: Any) -> list[Any]:
        width, height = image.size
        aspect = width / height
        ratios = sorted(
            {
                (columns, rows)
                for count in range(1, self.max_tiles + 1)
                for columns in range(1, count + 1)
                for rows in range(1, count + 1)
                if 1 <= columns * rows <= self.max_tiles
            },
            key=lambda pair: pair[0] * pair[1],
        )
        best_difference = float("inf")
        ratio = (1, 1)
        area = width * height
        for candidate in ratios:
            difference = abs(aspect - candidate[0] / candidate[1])
            if difference < best_difference:
                best_difference = difference
                ratio = candidate
            elif difference == best_difference:
                threshold = 0.5 * self.image_size**2 * candidate[0] * candidate[1]
                if area > threshold:
                    ratio = candidate
        target_width = self.image_size * ratio[0]
        target_height = self.image_size * ratio[1]
        resized = image.resize((target_width, target_height))
        tiles = []
        for index in range(ratio[0] * ratio[1]):
            left = (index % ratio[0]) * self.image_size
            top = (index // ratio[0]) * self.image_size
            tiles.append(
                resized.crop((left, top, left + self.image_size, top + self.image_size))
            )
        resized.close()
        if self.use_thumbnail and len(tiles) != 1:
            tiles.append(image.resize((self.image_size, self.image_size)))
        return tiles

    def generate(self, request: EvaluationRequest) -> str:
        prompt, images = flatten_messages(request.messages, image_marker="<image>")
        try:
            batches = []
            patch_counts = []
            for image in images:
                tiles = self._tiles(image)
                patch_counts.append(len(tiles))
                batches.append(
                    self.torch.stack([self.transform(tile) for tile in tiles])
                )
                close_images(tiles)
            pixel_values = self.torch.cat(batches).to(device="cuda", dtype=self.dtype)
            response, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                self.generation,
                num_patches_list=patch_counts,
                history=None,
                return_history=True,
            )
            return response
        finally:
            close_images(images)
