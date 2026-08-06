from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent / "prompt_assets" / "paper_v1"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load and verify an immutable paper-v1 prompt."""
    manifest = json.loads((ASSET_DIR / "manifest.json").read_text(encoding="utf-8"))
    try:
        entry = manifest["prompts"][name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt asset: {name}") from exc

    text = (ASSET_DIR / entry["file"]).read_text(encoding="utf-8")
    if entry.get("remove_file_terminal_newline", False):
        if not text.endswith("\n"):
            raise RuntimeError(f"Prompt asset {name} has an unexpected file ending")
        text = text[:-1]

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(
            f"Prompt asset {name} was modified: expected {entry['sha256']}, got {digest}"
        )
    return text


def verify_all_prompts() -> dict[str, str]:
    manifest = json.loads((ASSET_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {
        name: hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()
        for name in manifest["prompts"]
    }

