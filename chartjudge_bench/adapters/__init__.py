from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .base import ModelAdapter


def load_model_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_adapter(config: dict[str, Any]) -> ModelAdapter:
    adapter_type = config.get("adapter")
    kwargs = dict(config.get("adapter_kwargs", {}))
    if adapter_type == "openai_compatible":
        from .openai_compatible import OpenAICompatibleAdapter

        return OpenAICompatibleAdapter(**kwargs)
    if adapter_type == "qwen_transformers":
        from .transformers_qwen import QwenTransformersAdapter

        return QwenTransformersAdapter(**kwargs)
    if adapter_type == "thinklite_transformers":
        from .transformers_qwen import ThinkLiteAdapter

        return ThinkLiteAdapter(**kwargs)
    if adapter_type == "custom":
        class_path = config.get("class_path")
        if not class_path or ":" not in class_path:
            raise ValueError("Custom configs require class_path='module:ClassName'")
        module_name, class_name = class_path.split(":", 1)
        adapter_class = getattr(importlib.import_module(module_name), class_name)
        adapter = adapter_class(**kwargs)
        if not isinstance(adapter, ModelAdapter):
            raise TypeError(f"{class_path} must inherit ModelAdapter")
        return adapter
    raise ValueError(f"Unsupported adapter type: {adapter_type!r}")


__all__ = ["ModelAdapter", "build_adapter", "load_model_config"]

