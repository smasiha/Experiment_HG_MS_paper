from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch

from mt_bench_eval import (
    DEFAULT_FASTCHAT_QUESTION_URL,
    generate_mtbench_answers,
    load_mtbench_questions,
    run_mtbench_judging,
)
from real_world_instruction_data import load_instruction_test_records


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as file:
        return json.load(file)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def _value(config: Dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


def _build_eval_args(config: Dict[str, Any], cli_args: argparse.Namespace) -> SimpleNamespace:
    question_limit = (
        cli_args.question_limit
        if cli_args.question_limit is not None
        else int(_value(config, "mtbench_question_limit", 0))
    )
    judge_model = cli_args.judge_model or str(_value(config, "mtbench_judge_model", "gpt-4o"))
    api_base = cli_args.api_base or str(
        _value(config, "mtbench_api_base", os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"))
    )
    return SimpleNamespace(
        model_name=str(_value(config, "model_name", "Qwen/Qwen2-7B")),
        bf16=bool(_value(config, "bf16", True)),
        gradient_checkpointing=bool(_value(config, "gradient_checkpointing", False)),
        lora_r=int(_value(config, "lora_r", 16)),
        lora_alpha=int(_value(config, "lora_alpha", 32)),
        lora_dropout=float(_value(config, "lora_dropout", 0.05)),
        target_modules=list(_value(config, "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        mtbench_question_source=str(_value(config, "mtbench_question_source", "fastchat")),
        mtbench_question_file=cli_args.question_file or str(_value(config, "mtbench_question_file", "")),
        mtbench_question_url=str(_value(config, "mtbench_question_url", DEFAULT_FASTCHAT_QUESTION_URL)),
        mtbench_question_limit=question_limit,
        mtbench_max_input_length=int(_value(config, "mtbench_max_input_length", 1024)),
        mtbench_max_new_tokens=int(_value(config, "mtbench_max_new_tokens", 512)),
        mtbench_generation_batch_size=int(_value(config, "mtbench_generation_batch_size", 4)),
        mtbench_prompt_style=str(_value(config, "mtbench_prompt_style", "training_text")),
        mtbench_model_id=cli_args.model_id or str(_value(config, "mtbench_model_id", "")),
        mtbench_api_key_env=cli_args.api_key_env or str(_value(config, "mtbench_api_key_env", "OPENAI_API_KEY")),
        mtbench_judge_model=judge_model,
        mtbench_api_base=api_base,
        mtbench_judge_timeout_seconds=float(_value(config, "mtbench_judge_timeout_seconds", 120.0)),
        mtbench_judge_max_retries=int(_value(config, "mtbench_judge_max_retries", 6)),
        mtbench_judge_retry_base_seconds=float(_value(config, "mtbench_judge_retry_base_seconds", 2.0)),
        mtbench_use_reference=bool(_value(config, "mtbench_use_reference", True)),
    )


def _load_adapter_state(path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected adapter state dict in {path}, found {type(state)!r}.")
    return state


def _question_records(args: SimpleNamespace, config: Dict[str, Any]) -> list[dict]:
    if args.mtbench_question_source != "scalebio_test":
        return []
    cache_path = str(_value(config, "cache_path", ""))
    return load_instruction_test_records(cache_path=cache_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MT-Bench generation/judging for a completed instruction run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--adapter-file", default="final_ft_best_adapter_state.pt")
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--question-file", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-generate", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    run_dir = Path(cli_args.run_dir).resolve()
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    adapter_path = run_dir / cli_args.adapter_file
    mtbench_dir = run_dir / "mtbench_eval"
    mtbench_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter: {adapter_path}")

    config = _load_json(config_path)
    eval_args = _build_eval_args(config, cli_args)
    questions, question_meta = load_mtbench_questions(
        source=eval_args.mtbench_question_source,
        question_file=eval_args.mtbench_question_file,
        question_url=eval_args.mtbench_question_url,
        question_limit=eval_args.mtbench_question_limit,
        output_dir=mtbench_dir,
        scale_bio_test_records=_question_records(eval_args, config),
    )

    answer_file = mtbench_dir / "mtbench_model_answer.jsonl"
    if cli_args.force_generate or not answer_file.exists():
        if cli_args.force_generate:
            for stale_path in [mtbench_dir / "mtbench_judgment.jsonl", mtbench_dir / "mtbench_summary.json"]:
                stale_path.unlink(missing_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generation_summary = generate_mtbench_answers(
            args=eval_args,
            adapter_state=_load_adapter_state(adapter_path),
            output_dir=mtbench_dir,
            questions=questions,
            device=device,
        )
        generation_status = "generated"
    else:
        generation_summary = {
            "answer_file": str(answer_file),
            "question_count": len(questions),
            "turn_count": sum(len(question.turns) for question in questions),
            "model_id": eval_args.mtbench_model_id or Path(eval_args.model_name).name,
        }
        generation_status = "reused_existing"

    judging_summary: Dict[str, Any] | None = None
    if cli_args.judge:
        judging_summary = run_mtbench_judging(
            args=eval_args,
            output_dir=mtbench_dir,
            questions=questions,
            answer_file=str(answer_file),
        )

    metrics = _load_json(metrics_path)
    mtbench_payload = metrics.get("mtbench")
    if not isinstance(mtbench_payload, dict):
        mtbench_payload = {}
    mtbench_payload.update(
        {
            "status": generation_status,
            "question_source": question_meta,
            "generation": generation_summary,
        }
    )
    if judging_summary is not None:
        mtbench_payload["judging"] = judging_summary
    metrics["mtbench"] = mtbench_payload
    _write_json(metrics_path, metrics)

    summary = {
        "run_dir": str(run_dir),
        "adapter_file": str(adapter_path),
        "mtbench_dir": str(mtbench_dir),
        "generation": generation_summary,
        "judging": judging_summary,
        "api_key_set": bool(os.environ.get(eval_args.mtbench_api_key_env, "").strip()),
    }
    _write_json(mtbench_dir / "mtbench_followup_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
