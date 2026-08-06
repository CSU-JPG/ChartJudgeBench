from __future__ import annotations

from dataclasses import dataclass


DATASET_REPO_ID = "Lijian9/ChartJudgeBench"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    config: str
    expected_size: int
    id_field: str


TASK_SPECS = {
    "CPA": TaskSpec("CPA", "CPA", 1003, "pair_id"),
    "ChartEditing": TaskSpec("ChartEditing", "ChartEditing", 335, "example_id"),
    "ChartReproduction": TaskSpec(
        "ChartReproduction", "ChartReproduction", 315, "example_id"
    ),
}

TASK_ALIASES = {
    "cpa": "CPA",
    "chartediting": "ChartEditing",
    "chart_editing": "ChartEditing",
    "editing": "ChartEditing",
    "edit": "ChartEditing",
    "chartreproduction": "ChartReproduction",
    "chart_reproduction": "ChartReproduction",
    "reproduction": "ChartReproduction",
    "repro": "ChartReproduction",
}

CPA_DIMENSIONS = (
    ("Visual_Effects", "Color_Consistency"),
    ("Visual_Effects", "Visual_Style"),
    ("Visual_Effects", "Aspect_Ratio"),
    ("Data_Fidelity", "Axis_Scaling"),
    ("Data_Fidelity", "Data_Encoding"),
    ("Layout_Structure", "Component_Positioning"),
    ("Layout_Structure", "Text_Occlusion"),
)


def canonical_task_name(task: str) -> str:
    if task in TASK_SPECS:
        return task
    normalized = task.strip().replace("-", "_").lower()
    try:
        return TASK_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(TASK_SPECS)
        raise ValueError(f"Unknown task {task!r}. Choose one of: {choices}") from exc

