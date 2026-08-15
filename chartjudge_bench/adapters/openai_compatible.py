from __future__ import annotations

import base64
import io
import mimetypes
import os
from pathlib import Path
from typing import Any

from .base import ModelAdapter
from ..protocols import EvaluationRequest


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for OpenAI-compatible multimodal chat-completion APIs."""

    thread_safe = True

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_API_URL",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        generation: dict[str, Any] | None = None,
    ) -> None:
        try:
            from openai import OpenAI
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "Install API support with: pip install -e '.[api]'"
            ) from exc

        load_dotenv()
        resolved_key = api_key or os.getenv(api_key_env)
        resolved_url = base_url or os.getenv(base_url_env)
        if not resolved_key:
            raise ValueError(f"API key is missing; set {api_key_env}")
        self.model = model
        self.generation = dict(generation or {})
        client_kwargs: dict[str, Any] = {"api_key": resolved_key, "timeout": timeout}
        if resolved_url:
            client_kwargs["base_url"] = resolved_url
        self.client = OpenAI(**client_kwargs)

    def generate(self, request: EvaluationRequest) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(request.messages),
            **self.generation,
        )
        return response.choices[0].message.content or ""

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()

    @classmethod
    def _convert_messages(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                converted.append({"role": message["role"], "content": content})
                continue
            blocks = []
            for block in content:
                if block["type"] == "text":
                    blocks.append({"type": "text", "text": block["text"]})
                elif block["type"] == "image":
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": cls._image_to_data_url(block["image"])
                            },
                        }
                    )
                else:
                    raise ValueError(f"Unsupported message block: {block['type']}")
            converted.append({"role": message["role"], "content": blocks})
        return converted

    @staticmethod
    def _image_to_data_url(image: Any) -> str:
        if isinstance(image, str) and image.startswith("data:"):
            return image

        mime_type = "image/png"
        payload: bytes
        if isinstance(image, (str, Path)):
            path = Path(image)
            payload = path.read_bytes()
            mime_type = mimetypes.guess_type(path.name)[0] or mime_type
        elif isinstance(image, bytes):
            payload = image
        elif isinstance(image, dict) and image.get("bytes") is not None:
            payload = image["bytes"]
        elif isinstance(image, dict) and image.get("path"):
            path = Path(image["path"])
            payload = path.read_bytes()
            mime_type = mimetypes.guess_type(path.name)[0] or mime_type
        elif hasattr(image, "save"):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
        else:
            raise TypeError(f"Unsupported image value: {type(image)!r}")
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
