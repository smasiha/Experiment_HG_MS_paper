#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from exact_math_reasoning import (
    evaluate_task,
    prepare_eval_records,
    seed_everything,
    subsample_examples,
)
from exact_math_reasoning_data import load_eval_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs-root")
    group.add_argument("--run-dir")
    parser.add_argument("--math-eval-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_tokenizer(model_name: str):
    local_files_only = os.environ.get("LOCAL_MODEL_ONLY", "0") == "1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except OSError:
        if local_files_only:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False,
            force_download=True,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def has_finite_math_accuracy(metrics_path: Path) -> bool:
    if not metrics_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return False
    value = metrics.get("math_accuracy")
    return isinstance(value, (int, float)) and math.isfinite(value)


def evaluate_run(run_dir: Path, args: argparse.Namespace, math_raw_examples) -> dict[str, object]:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    state_path = run_dir / "final_ft_best_adapter_state.pt"

    config = json.loads(config_path.read_text())
    run_args = argparse.Namespace(**config)
    run_args.eval_math = True
    run_args.math_eval_size = args.math_eval_size
    if args.eval_batch_size > 0:
        run_args.eval_batch_size = args.eval_batch_size

    seed_everything(int(run_args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    tokenizer = load_tokenizer(run_args.model_name)
    math_eval_raw = subsample_examples(
        math_raw_examples,
        limit=run_args.math_eval_size,
        seed=int(run_args.seed) + 5050,
    )
    math_records = prepare_eval_records(tokenizer, math_eval_raw, max_length=int(run_args.max_length))
    del tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    adapter_state = torch.load(state_path, map_location="cpu", weights_only=False)
    eval_start = time.perf_counter()
    math_metrics = evaluate_task(
        args=run_args,
        adapter_state=adapter_state,
        dataset_name="math",
        examples=math_records,
        device=device,
    )
    eval_elapsed = time.perf_counter() - eval_start

    with open(run_dir / "math_predictions.jsonl", "w") as handle:
        for row in math_metrics["predictions"]:
            handle.write(json.dumps(row) + "\n")

    metrics = json.loads(metrics_path.read_text())
    metrics["math_accuracy"] = math_metrics["accuracy"]
    metrics["math_eval_elapsed_sec"] = eval_elapsed
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "run_dir": str(run_dir),
        "method": metrics.get("method"),
        "seed": metrics.get("seed"),
        "math_accuracy": math_metrics["accuracy"],
        "math_eval_elapsed_sec": eval_elapsed,
        "example_count": len(math_records),
    }


def main() -> int:
    args = parse_args()
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        runs_root = Path(args.runs_root)
        run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    math_raw_examples = load_eval_examples(include_math=True)["math"]

    summary = []
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        state_path = run_dir / "final_ft_best_adapter_state.pt"
        if not config_path.exists() or not metrics_path.exists() or not state_path.exists():
            continue
        if not args.force and has_finite_math_accuracy(metrics_path):
            continue
        result = evaluate_run(run_dir, args, math_raw_examples)
        print(json.dumps({"stage": "math_eval_done", **result}))
        summary.append(result)

    print(json.dumps({"stage": "math_eval_summary", "completed_runs": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
