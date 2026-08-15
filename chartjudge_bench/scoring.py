from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .constants import CPA_DIMENSIONS, TASK_SPECS, canonical_task_name
from .data import normalize_truth
from .parsing import cpa_status


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}"
                ) from exc
    return records


def _prediction_id(record: Mapping[str, Any]) -> str:
    for key in ("sample_id", "case_id", "pair_id", "example_id"):
        if record.get(key) is not None:
            return str(record[key])
    raise ValueError("Prediction record is missing sample_id/case_id")


def _index_unique(records: Iterable[Mapping[str, Any]], kind: str):
    indexed: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        sample_id = _prediction_id(record)
        if sample_id in indexed:
            duplicates.append(sample_id)
        indexed[sample_id] = record
    if duplicates:
        values = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate {kind} IDs: {values}")
    return indexed


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _normalize_cpa_status(prediction: Mapping[str, Any]) -> str:
    status = str(prediction.get("status", ""))
    for canonical in (
        "Consistent_Correct",
        "Consistent_Wrong",
        "Bias_Position_A",
        "Bias_Position_B",
        "Parse_Fail",
    ):
        if canonical in status:
            return canonical
    return cpa_status(prediction.get("r1_choice"), prediction.get("r2_choice"))


def _score_cpa(
    truth: dict[str, Mapping[str, Any]],
    predictions: dict[str, Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(truth)
    status_counts = defaultdict(int)
    dimension_totals = {sub: 0 for _, sub in CPA_DIMENSIONS}
    dimension_correct = {sub: 0 for _, sub in CPA_DIMENSIONS}
    missing = 0

    for sample_id, item in truth.items():
        subcategory = str(item["subcategory"])
        dimension_totals.setdefault(subcategory, 0)
        dimension_correct.setdefault(subcategory, 0)
        dimension_totals[subcategory] += 1
        prediction = predictions.get(sample_id)
        if prediction is None:
            missing += 1
            continue
        status = _normalize_cpa_status(prediction)
        status_counts[status] += 1
        if status == "Consistent_Correct":
            dimension_correct[subcategory] += 1

    correct = status_counts["Consistent_Correct"]
    dimensions = {
        subcategory: {
            "correct": dimension_correct[subcategory],
            "total": dimension_totals[subcategory],
            "accuracy": _safe_div(
                dimension_correct[subcategory], dimension_totals[subcategory]
            ),
        }
        for _, subcategory in CPA_DIMENSIONS
    }
    return {
        "task": "CPA",
        "ranking_metric": "overall_accuracy",
        "overall_accuracy": _safe_div(correct, total),
        "correct": correct,
        "total": total,
        "dimension_accuracy": dimensions,
        "diagnostics": {
            "parse_fail_rate": _safe_div(status_counts["Parse_Fail"], total),
            "position_a_bias_rate": _safe_div(status_counts["Bias_Position_A"], total),
            "position_b_bias_rate": _safe_div(status_counts["Bias_Position_B"], total),
            "consistent_wrong_rate": _safe_div(
                status_counts["Consistent_Wrong"], total
            ),
            "coverage": _safe_div(total - missing, total),
            "missing_predictions": missing,
            "status_counts": dict(status_counts),
        },
    }


def _normalize_decision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    decision = value.strip().title()
    return decision if decision in {"Accept", "Reject"} else None


def _score_accept_reject(
    task: str,
    truth: dict[str, Mapping[str, Any]],
    predictions: dict[str, Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(truth)
    correct = parsed = parse_fail = missing = 0
    tp = tn = fp = fn = 0
    for sample_id, item in truth.items():
        prediction = predictions.get(sample_id)
        if prediction is None:
            missing += 1
            continue
        decision = _normalize_decision(prediction.get("model_decision"))
        if decision is None:
            parse_fail += 1
            continue
        parsed += 1
        expected = item["ground_truth"]
        if decision == expected:
            correct += 1
        if expected == "Accept":
            if decision == "Accept":
                tp += 1
            else:
                fn += 1
        else:
            if decision == "Reject":
                tn += 1
            else:
                fp += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "task": task,
        "ranking_metric": "overall_accuracy",
        "overall_accuracy": _safe_div(correct, total),
        "correct": correct,
        "total": total,
        "diagnostics": {
            "valid_precision": precision,
            "valid_recall": recall,
            "valid_specificity": specificity,
            "valid_f1": f1,
            "parse_fail_rate": _safe_div(parse_fail, total),
            "coverage": _safe_div(total - missing, total),
            "parsed_predictions": parsed,
            "parse_failures": parse_fail,
            "missing_predictions": missing,
            "valid_confusion_matrix": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        },
    }


def score_predictions(
    task: str,
    predictions: str | Path | Iterable[Mapping[str, Any]],
    truth_rows: Iterable[Mapping[str, Any]],
    *,
    require_official_size: bool = True,
    strict_extra_ids: bool = True,
) -> dict[str, Any]:
    """Score predictions against dataset-owned labels and IDs."""
    task = canonical_task_name(task)
    prediction_records = (
        read_jsonl(predictions) if isinstance(predictions, (str, Path)) else predictions
    )
    truth = _index_unique((normalize_truth(task, row) for row in truth_rows), "truth")
    indexed_predictions = _index_unique(prediction_records, "prediction")

    if require_official_size and len(truth) != TASK_SPECS[task].expected_size:
        raise ValueError(
            f"{task} requires {TASK_SPECS[task].expected_size} truth rows; got {len(truth)}"
        )
    extra_ids = sorted(set(indexed_predictions) - set(truth))
    if strict_extra_ids and extra_ids:
        raise ValueError(
            f"Predictions contain unknown IDs: {', '.join(extra_ids[:10])}"
        )

    metrics = (
        _score_cpa(truth, indexed_predictions)
        if task == "CPA"
        else _score_accept_reject(task, truth, indexed_predictions)
    )
    metrics["diagnostics"]["extra_prediction_ids"] = extra_ids
    return metrics
