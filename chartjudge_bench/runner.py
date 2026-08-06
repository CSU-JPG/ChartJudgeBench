from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .adapters.base import ModelAdapter
from .data import normalize_sample
from .protocols import TaskProtocol, get_protocol


def _prediction_id(record: Mapping[str, Any]) -> str:
    for key in ("sample_id", "case_id", "pair_id", "example_id"):
        if key in record:
            return str(record[key])
    raise ValueError("Prediction record is missing a sample ID")


def load_completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                completed.add(_prediction_id(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid resume file at line {line_number}: {path}") from exc
    return completed


def evaluate_sample(
    protocol: TaskProtocol,
    adapter: ModelAdapter,
    sample: dict[str, Any],
) -> dict[str, Any]:
    raw_responses: list[str | None] = []
    errors: list[str | None] = []
    for request in protocol.build_requests(sample):
        try:
            raw_responses.append(adapter.generate(request))
            errors.append(None)
        except Exception as exc:  # A failed model call becomes an incorrect prediction.
            raw_responses.append(None)
            errors.append(f"{type(exc).__name__}: {exc}")
    return protocol.build_prediction(sample, raw_responses, errors)


def _iter_normalized(
    task: str,
    rows: Iterable[Mapping[str, Any]],
    completed: set[str],
    limit: int | None,
):
    selected = 0
    for row in rows:
        if limit is not None and selected >= limit:
            break
        selected += 1
        sample = normalize_sample(task, row)
        if sample["sample_id"] not in completed:
            yield sample


def run_evaluation(
    *,
    task: str,
    adapter: ModelAdapter,
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    workers: int = 1,
    resume: bool = True,
    limit: int | None = None,
) -> dict[str, int]:
    """Run one task and append one JSON object per dataset sample."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers > 1 and not adapter.thread_safe:
        raise ValueError("This adapter is not thread-safe; use workers=1")

    protocol = get_protocol(task)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(output_path) if resume else set()
    mode = "a" if resume and output_path.exists() else "w"
    samples = _iter_normalized(protocol.task, rows, completed, limit)
    written = 0

    with output_path.open(mode, encoding="utf-8") as output:
        if workers == 1:
            for sample in tqdm(samples, desc=f"Evaluating {protocol.task}"):
                prediction = evaluate_sample(protocol, adapter, sample)
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output.flush()
                written += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending: dict[Future, str] = {}
                exhausted = False
                progress = tqdm(desc=f"Evaluating {protocol.task}")
                while pending or not exhausted:
                    while not exhausted and len(pending) < workers * 2:
                        try:
                            sample = next(samples)
                        except StopIteration:
                            exhausted = True
                            break
                        future = executor.submit(evaluate_sample, protocol, adapter, sample)
                        pending[future] = sample["sample_id"]
                    if not pending:
                        continue
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        prediction = future.result()
                        output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                        output.flush()
                        written += 1
                        progress.update(1)
                progress.close()

    return {"already_completed": len(completed), "newly_written": written}

