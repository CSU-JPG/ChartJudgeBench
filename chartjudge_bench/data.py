from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .constants import DATASET_REPO_ID, TASK_SPECS, canonical_task_name


def load_chartjudge(
    task: str,
    *,
    split: str = "train",
    repo_id: str = DATASET_REPO_ID,
    cache_dir: str | None = None,
    token: str | bool | None = None,
):
    """Load one ChartJudgeBench task from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the package dependencies before loading data."
        ) from exc

    task = canonical_task_name(task)
    spec = TASK_SPECS[task]
    return load_dataset(
        repo_id,
        spec.config,
        split=split,
        cache_dir=cache_dir,
        token=token,
    )


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"None of the expected fields were found: {', '.join(keys)}")


def normalize_truth(task: str, row: Mapping[str, Any]) -> dict[str, Any]:
    task = canonical_task_name(task)
    spec = TASK_SPECS[task]
    sample_id = str(_first(row, spec.id_field, "sample_id", "case_id"))
    base = {
        "task": task,
        "sample_id": sample_id,
        "chart_type": str(row.get("chart_type", "unknown")),
    }
    if task == "CPA":
        base.update(
            {
                "category": str(_first(row, "category", "main_category")),
                "subcategory": str(_first(row, "subcategory", "sub_category")),
            }
        )
    else:
        base["ground_truth"] = str(_first(row, "decision", "ground_truth")).title()
    return base


def normalize_sample(task: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a decoded Hugging Face row for inference."""
    task = canonical_task_name(task)
    sample = normalize_truth(task, row)
    if task == "CPA":
        sample["better_image"] = _first(row, "better_image", "better")
        sample["worse_image"] = _first(row, "worse_image", "worse")
    elif task == "ChartEditing":
        sample["reference_image"] = _first(row, "reference", "reference_image")
        sample["edited_image"] = _first(row, "edited", "edited_image")
        sample["instruction"] = str(_first(row, "instruction", "query"))
    else:
        sample["generated_image"] = _first(row, "input", "generated_image")
        sample["target_image"] = _first(row, "target", "target_image")
    return sample


def iter_truth(
    task: str, rows: Iterable[Mapping[str, Any]]
) -> Iterator[dict[str, Any]]:
    for row in rows:
        yield normalize_truth(task, row)
