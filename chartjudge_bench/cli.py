from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import build_adapter, load_model_config
from .constants import canonical_task_name
from .data import load_chartjudge
from .prompts import verify_all_prompts
from .runner import run_evaluation
from .scoring import score_predictions


def _add_task(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task", required=True, help="CPA, ChartEditing, or ChartReproduction"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chartjudge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a model on one benchmark task")
    _add_task(run_parser)
    run_parser.add_argument("--config", required=True, help="Model JSON config")
    run_parser.add_argument(
        "--model-path",
        help="Override adapter_kwargs.model_path without editing a bundled local config",
    )
    run_parser.add_argument(
        "--model-name",
        help="Override adapter_kwargs.model for an API configuration",
    )
    run_parser.add_argument("--output", required=True, help="Prediction JSONL path")
    run_parser.add_argument("--split", default="train")
    run_parser.add_argument("--cache-dir")
    run_parser.add_argument("--workers", type=int)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--no-resume", action="store_true")

    score_parser = subparsers.add_parser("score", help="Score an existing JSONL file")
    _add_task(score_parser)
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument("--output", help="Optional metrics JSON path")
    score_parser.add_argument("--split", default="train")
    score_parser.add_argument("--cache-dir")
    score_parser.add_argument("--allow-nonofficial-size", action="store_true")
    score_parser.add_argument("--allow-extra-ids", action="store_true")

    subparsers.add_parser("verify-prompts", help="Verify immutable paper-v1 prompts")
    return parser


def _truth_only(task: str, dataset):
    if not hasattr(dataset, "select_columns"):
        return dataset
    fields = {
        "CPA": ["pair_id", "chart_type", "category", "subcategory"],
        "ChartEditing": ["example_id", "chart_type", "decision"],
        "ChartReproduction": ["example_id", "chart_type", "decision"],
    }[task]
    available = [field for field in fields if field in dataset.column_names]
    return dataset.select_columns(available)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "verify-prompts":
        print(json.dumps(verify_all_prompts(), indent=2))
        return

    task = canonical_task_name(args.task)
    dataset = load_chartjudge(task, split=args.split, cache_dir=args.cache_dir)
    if args.command == "run":
        config = load_model_config(args.config)
        adapter_kwargs = config.setdefault("adapter_kwargs", {})
        if args.model_path:
            if "model_path" not in adapter_kwargs:
                raise ValueError("This adapter does not accept --model-path")
            adapter_kwargs["model_path"] = args.model_path
        if args.model_name:
            if "model" not in adapter_kwargs:
                raise ValueError("This adapter does not accept --model-name")
            adapter_kwargs["model"] = args.model_name
        configured_task = config.get("task")
        if configured_task and canonical_task_name(configured_task) != task:
            raise ValueError(f"Config is for {configured_task}, not {task}")
        adapter = build_adapter(config)
        workers = (
            args.workers
            if args.workers is not None
            else int(config.get("runner", {}).get("workers", 1))
        )
        try:
            summary = run_evaluation(
                task=task,
                adapter=adapter,
                rows=dataset,
                output_path=args.output,
                workers=workers,
                resume=not args.no_resume,
                limit=args.limit,
            )
        finally:
            adapter.close()
        print(json.dumps(summary, indent=2))
        return

    metrics = score_predictions(
        task,
        args.predictions,
        _truth_only(task, dataset),
        require_official_size=not args.allow_nonofficial_size,
        strict_extra_ids=not args.allow_extra_ids,
    )
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
