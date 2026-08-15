from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import canonical_task_name
from .parsing import cpa_status, parse_accept_reject, parse_cpa_choice
from .prompts import load_prompt


@dataclass(frozen=True)
class EvaluationRequest:
    task: str
    sample_id: str
    pass_name: str
    messages: list[dict[str, Any]]


class TaskProtocol:
    task: str

    def build_requests(self, sample: dict[str, Any]) -> list[EvaluationRequest]:
        raise NotImplementedError

    def build_prediction(
        self,
        sample: dict[str, Any],
        raw_responses: list[str | None],
        errors: list[str | None] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class CPAProtocol(TaskProtocol):
    task = "CPA"

    def build_requests(self, sample: dict[str, Any]) -> list[EvaluationRequest]:
        system_instruction = load_prompt("cpa_system_instruction")
        configs = (
            ("Forward", sample["better_image"], sample["worse_image"]),
            ("Reverse", sample["worse_image"], sample["better_image"]),
        )
        requests = []
        for pass_name, image_a, image_b in configs:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_instruction + "\n\nImage A:\n",
                        },
                        {"type": "image", "image": image_a},
                        {"type": "text", "text": "\nImage B:\n"},
                        {"type": "image", "image": image_b},
                        {
                            "type": "text",
                            "text": "\nWhich image is better? Remember to put the choice in \\boxed{}.",
                        },
                    ],
                }
            ]
            requests.append(
                EvaluationRequest(self.task, sample["sample_id"], pass_name, messages)
            )
        return requests

    def build_prediction(
        self,
        sample: dict[str, Any],
        raw_responses: list[str | None],
        errors: list[str | None] | None = None,
    ) -> dict[str, Any]:
        padded = (raw_responses + [None, None])[:2]
        r1, _ = parse_cpa_choice(padded[0])
        r2, _ = parse_cpa_choice(padded[1])
        status = cpa_status(r1, r2)
        return {
            "task": self.task,
            "sample_id": sample["sample_id"],
            "case_id": sample["sample_id"],
            "chart_type": sample["chart_type"],
            "main_category": sample["category"],
            "sub_category": sample["subcategory"],
            "status": status,
            "is_correct": status == "Consistent_Correct",
            "r1_choice": r1,
            "r2_choice": r2,
            "r1_raw": padded[0],
            "r2_raw": padded[1],
            "errors": errors or [None, None],
        }


class _AcceptRejectProtocol(TaskProtocol):
    def _prediction(
        self,
        sample: dict[str, Any],
        raw_responses: list[str | None],
        errors: list[str | None] | None,
    ) -> dict[str, Any]:
        raw = raw_responses[0] if raw_responses else None
        decision, cleaned = parse_accept_reject(raw)
        expected = sample["ground_truth"]
        return {
            "task": self.task,
            "sample_id": sample["sample_id"],
            "case_id": sample["sample_id"],
            "chart_type": sample["chart_type"],
            "type": "Good_Case" if expected == "Accept" else "Bad_Case",
            "ground_truth": expected,
            "model_decision": decision,
            "is_correct": decision == expected,
            "reason": cleaned[:300].replace("\n", " ") if cleaned else "",
            "raw_response": raw,
            "error": (errors or [None])[0],
        }


class ChartEditingProtocol(_AcceptRejectProtocol):
    task = "ChartEditing"

    def build_requests(self, sample: dict[str, Any]) -> list[EvaluationRequest]:
        system_prompt = load_prompt("chart_editing_system")
        query_text = sample["instruction"]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f'{system_prompt}\n\nHere is the editing task evaluation:\n\n**User Instruction**: "{query_text}"\n\n--- **Image 1 (Original)** ---',
                    },
                    {"type": "image", "image": sample["reference_image"]},
                    {"type": "text", "text": "\n--- **Image 2 (Result)** ---"},
                    {"type": "image", "image": sample["edited_image"]},
                    {
                        "type": "text",
                        "text": "\nCompare Image 1 and Image 2 based on the Instruction. Should we [Accept] or [Reject]?",
                    },
                ],
            }
        ]
        return [EvaluationRequest(self.task, sample["sample_id"], "Single", messages)]

    def build_prediction(self, sample, raw_responses, errors=None):
        return self._prediction(sample, raw_responses, errors)


class ChartReproductionProtocol(_AcceptRejectProtocol):
    task = "ChartReproduction"

    def build_requests(self, sample: dict[str, Any]) -> list[EvaluationRequest]:
        messages = [
            {"role": "system", "content": load_prompt("chart_reproduction_system")},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the comparison:"},
                    {
                        "type": "text",
                        "text": "--- [Ground Truth Image] (Target) ---",
                    },
                    {"type": "image", "image": sample["target_image"]},
                    {
                        "type": "text",
                        "text": "--- [Generated Image] (Model Output) ---",
                    },
                    {"type": "image", "image": sample["generated_image"]},
                    {
                        "type": "text",
                        "text": "Based on the User Query and Criteria, should we [Accept] or [Reject] this reproduction?",
                    },
                ],
            },
        ]
        return [EvaluationRequest(self.task, sample["sample_id"], "Single", messages)]

    def build_prediction(self, sample, raw_responses, errors=None):
        prediction = self._prediction(sample, raw_responses, errors)
        prediction["path_gen"] = None
        return prediction


def get_protocol(task: str) -> TaskProtocol:
    task = canonical_task_name(task)
    return {
        "CPA": CPAProtocol,
        "ChartEditing": ChartEditingProtocol,
        "ChartReproduction": ChartReproductionProtocol,
    }[task]()
