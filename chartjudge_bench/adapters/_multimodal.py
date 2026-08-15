from __future__ import annotations

import io
from pathlib import Path
from typing import Any


def to_pil_image(value: Any):
    """Return a detached RGB PIL image for a dataset/path image value."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a core dependency.
        raise RuntimeError("Pillow is required for local multimodal inference") from exc

    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB").copy()
    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as image:
            return image.convert("RGB").copy()
    if isinstance(value, dict) and value.get("bytes") is not None:
        with Image.open(io.BytesIO(value["bytes"])) as image:
            return image.convert("RGB").copy()
    if isinstance(value, dict) and value.get("path"):
        with Image.open(value["path"]) as image:
            return image.convert("RGB").copy()
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def collect_images(messages: list[dict[str, Any]]) -> list[Any]:
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "image":
                images.append(block["image"])
    return images


def messages_with_image_placeholders(
    messages: list[dict[str, Any]],
    *,
    inline_pil: bool,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Prepare chat-template messages and return their PIL images in order."""
    images = [to_pil_image(value) for value in collect_images(messages)]
    image_index = 0
    prepared: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            prepared.append({"role": message["role"], "content": content})
            continue
        blocks: list[dict[str, Any]] = []
        for block in content:
            if block["type"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                image_block: dict[str, Any] = {"type": "image"}
                if inline_pil:
                    image_block["image"] = images[image_index]
                blocks.append(image_block)
                image_index += 1
            else:
                raise ValueError(f"Unsupported message block: {block['type']}")
        prepared.append({"role": message["role"], "content": blocks})
    return prepared, images


def flatten_messages(
    messages: list[dict[str, Any]],
    *,
    image_marker: str,
) -> tuple[str, list[Any]]:
    """Flatten benchmark messages while preserving text and image order."""
    images: list[Any] = []
    sections: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            sections.append(content)
            continue
        parts: list[str] = []
        for block in content:
            if block["type"] == "text":
                parts.append(block["text"])
            elif block["type"] == "image":
                images.append(to_pil_image(block["image"]))
                parts.append(image_marker)
            else:
                raise ValueError(f"Unsupported message block: {block['type']}")
        sections.append("".join(parts))
    return "\n\n".join(section for section in sections if section), images


def close_images(images: list[Any]) -> None:
    for image in images:
        close = getattr(image, "close", None)
        if close:
            close()
