from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

from mt_bench_eval import DEFAULT_FASTCHAT_QUESTION_URL, load_mtbench_questions, run_mtbench_judging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge existing MT-Bench answer files without rerunning training or generation."
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--question-source", choices=["auto", "fastchat", "scalebio_test"], default="auto")
    parser.add_argument("--question-file", default="")
    parser.add_argument("--question-url", default="")
    parser.add_argument("--question-limit", type=int, default=-1)
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--judge-timeout-seconds", type=float, default=-1.0)
    parser.add_argument("--judge-max-retries", type=int, default=-1)
    parser.add_argument("--judge-retry-base-seconds", type=float, default=-1.0)
    parser.add_argument(
        "--use-reference",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    with path.open() as file:
        return json.load(file)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def _resolve_string(cli_value: str, config_value: object, default: str) -> str:
    if cli_value:
        return cli_value
    if isinstance(config_value, str) and config_value:
        return config_value
    return default


def _resolve_int(cli_value: int, config_value: object, default: int) -> int:
    if cli_value >= 0:
        return cli_value
    try:
        return int(config_value)
    except (TypeError, ValueError):
        return default


def _resolve_float(cli_value: float, config_value: object, default: float) -> float:
    if cli_value >= 0:
        return cli_value
    try:
        return float(config_value)
    except (TypeError, ValueError):
        return default


def _resolve_bool(cli_value: bool | None, config_value: object, default: bool) -> bool:
    if cli_value is not None:
        return cli_value
    if isinstance(config_value, bool):
        return config_value
    return default


def _iter_answer_files(runs_root: Path) -> List[Path]:
    return sorted(path for path in runs_root.rglob("mtbench_eval/mtbench_model_answer.jsonl") if path.is_file())


def _load_questions_for_run(
    *,
    run_dir: Path,
    mtbench_dir: Path,
    config: Dict[str, object],
    args: argparse.Namespace,
) -> tuple[Sequence[object], Dict[str, object]]:
    source = args.question_source
    if source == "auto":
        source = str(config.get("mtbench_question_source") or "fastchat")

    question_file = _resolve_string(args.question_file, config.get("mtbench_question_file"), "")
    question_url = _resolve_string(
        args.question_url,
        config.get("mtbench_question_url"),
        DEFAULT_FASTCHAT_QUESTION_URL,
    )
    question_limit = _resolve_int(args.question_limit, config.get("mtbench_question_limit"), 0)

    scale_bio_test_records: Sequence[Dict[str, object]] = []
    if source == "scalebio_test":
        from real_world_instruction_data import load_instruction_test_records

        cache_path = str(config.get("cache_path") or "")
        scale_bio_test_records = load_instruction_test_records(cache_path=cache_path)

    questions, question_meta = load_mtbench_questions(
        source=source,
        question_file=question_file,
        question_url=question_url,
        question_limit=question_limit,
        output_dir=mtbench_dir,
        scale_bio_test_records=scale_bio_test_records,
    )
    question_meta = dict(question_meta)
    question_meta["resolved_from"] = "followup_judge"
    question_meta["run_dir"] = str(run_dir)
    return questions, question_meta


def _build_judge_args(config: Dict[str, object], args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        mtbench_api_key_env=_resolve_string(args.api_key_env, config.get("mtbench_api_key_env"), "OPENAI_API_KEY"),
        mtbench_judge_model=_resolve_string(args.judge_model, config.get("mtbench_judge_model"), "gpt-4o"),
        mtbench_api_base=_resolve_string(
            args.api_base,
            config.get("mtbench_api_base"),
            os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        ),
        mtbench_judge_timeout_seconds=_resolve_float(
            args.judge_timeout_seconds,
            config.get("mtbench_judge_timeout_seconds"),
            120.0,
        ),
        mtbench_judge_max_retries=_resolve_int(
            args.judge_max_retries,
            config.get("mtbench_judge_max_retries"),
            6,
        ),
        mtbench_judge_retry_base_seconds=_resolve_float(
            args.judge_retry_base_seconds,
            config.get("mtbench_judge_retry_base_seconds"),
            2.0,
        ),
        mtbench_use_reference=_resolve_bool(args.use_reference, config.get("mtbench_use_reference"), True),
    )


def _update_metrics(
    *,
    run_dir: Path,
    question_meta: Dict[str, object],
    judging_summary: Dict[str, object],
    answer_file: Path,
) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics: Dict[str, object] = {}
    if metrics_path.exists():
        metrics = _load_json(metrics_path)

    mtbench_payload = metrics.get("mtbench")
    if not isinstance(mtbench_payload, dict):
        mtbench_payload = {}

    generation_payload = mtbench_payload.get("generation")
    if not isinstance(generation_payload, dict):
        generation_payload = {}
    generation_payload.setdefault("answer_file", str(answer_file))

    mtbench_payload["question_source"] = question_meta
    mtbench_payload["generation"] = generation_payload
    mtbench_payload["judging"] = judging_summary
    metrics["mtbench"] = mtbench_payload
    _write_json(metrics_path, metrics)


def _process_run(answer_file: Path, args: argparse.Namespace) -> Dict[str, object]:
    mtbench_dir = answer_file.parent
    run_dir = mtbench_dir.parent
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {
            "run_dir": str(run_dir),
            "status": "skipped_missing_config",
            "answer_file": str(answer_file),
        }

    config = _load_json(config_path)
    questions, question_meta = _load_questions_for_run(
        run_dir=run_dir,
        mtbench_dir=mtbench_dir,
        config=config,
        args=args,
    )

    summary_path = mtbench_dir / "mtbench_summary.json"
    judgments_path = mtbench_dir / "mtbench_judgment.jsonl"
    if args.skip_existing and summary_path.exists():
        judging_summary = _load_json(summary_path)
        _update_metrics(
            run_dir=run_dir,
            question_meta=question_meta,
            judging_summary=judging_summary,
            answer_file=answer_file,
        )
        return {
            "run_dir": str(run_dir),
            "status": "skipped_existing",
            "answer_file": str(answer_file),
            "summary_file": str(summary_path),
        }

    if args.dry_run:
        return {
            "run_dir": str(run_dir),
            "status": "dry_run",
            "answer_file": str(answer_file),
            "question_count": len(questions),
            "question_source": question_meta,
        }

    if not args.skip_existing:
        judgments_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    judging_summary = run_mtbench_judging(
        args=_build_judge_args(config, args),
        output_dir=mtbench_dir,
        questions=questions,
        answer_file=str(answer_file),
    )
    _update_metrics(
        run_dir=run_dir,
        question_meta=question_meta,
        judging_summary=judging_summary,
        answer_file=answer_file,
    )
    return {
        "run_dir": str(run_dir),
        "status": str(judging_summary.get("status") or "completed"),
        "answer_file": str(answer_file),
        "summary_file": str(judging_summary.get("summary_file") or summary_path),
        "judgment_file": str(judging_summary.get("judgment_file") or judgments_path),
        "judge_model": str(judging_summary.get("judge_model") or ""),
    }


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    answer_files = _iter_answer_files(runs_root)
    if args.max_runs > 0:
        answer_files = answer_files[: args.max_runs]

    print(
        json.dumps(
            {
                "stage": "mtbench_followup_scan",
                "runs_root": str(runs_root),
                "answer_file_count": len(answer_files),
                "skip_existing": args.skip_existing,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    results: List[Dict[str, object]] = []
    for answer_file in answer_files:
        result = _process_run(answer_file, args)
        results.append(result)
        print(json.dumps(result, indent=2))

    status_counts: Dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    print(
        json.dumps(
            {
                "stage": "mtbench_followup_done",
                "runs_root": str(runs_root),
                "processed_runs": len(results),
                "status_counts": status_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
