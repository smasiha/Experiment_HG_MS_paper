#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from exact_math_reasoning_data import load_eval_examples, normalize_math_answer
from faithful_uniform_math_fsdp import generate_batches, load_tokenizer, local_model_only, prepare_eval_records


def infer_method(run_dir: Path, run_args: argparse.Namespace) -> str:
    if getattr(run_args, "method", None):
        return str(run_args.method)
    name = run_dir.name.lower()
    if "uniform" in name:
        return "uniform"
    if "scalebio" in name:
        return "scalebio"
    return run_dir.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs-root")
    group.add_argument("--run-dir")
    parser.add_argument("--math-eval-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def has_finite_math_accuracy(metrics_path: Path) -> bool:
    if not metrics_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return False
    value = metrics.get("math_accuracy")
    return isinstance(value, (int, float)) and math.isfinite(value)


def load_existing_metrics(metrics_path: Path, run_dir: Path, run_args: argparse.Namespace) -> dict[str, object]:
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "method": infer_method(run_dir, run_args),
        "model_name": getattr(run_args, "model_name", None),
        "seed": getattr(run_args, "seed", None),
    }


def subsample_examples(raw_examples, *, limit: int, seed: int):
    if limit <= 0 or limit >= len(raw_examples):
        return list(raw_examples)
    rng = random.Random(seed)
    indices = list(range(len(raw_examples)))
    rng.shuffle(indices)
    picked = sorted(indices[:limit])
    return [raw_examples[idx] for idx in picked]


def load_model_for_eval(run_args: argparse.Namespace, state_path: Path, device: torch.device):
    dtype = torch.bfloat16 if getattr(run_args, "bf16", False) and device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        run_args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=getattr(run_args, "attn_implementation", "sdpa"),
        local_files_only=local_model_only(),
        low_cpu_mem_usage=True,
    ).to(device)
    full_state_dict = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(full_state_dict, strict=True)
    model.eval()
    model.config.use_cache = True
    return model


def evaluate_run(run_dir: Path, args: argparse.Namespace, math_raw_examples) -> dict[str, object]:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    state_path = run_dir / "final_full_state.pt"

    config = json.loads(config_path.read_text())
    run_args = argparse.Namespace(**config)
    if args.eval_batch_size > 0:
        run_args.eval_batch_size = args.eval_batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    tokenizer = load_tokenizer(run_args.model_name)
    math_eval_raw = subsample_examples(
        math_raw_examples,
        limit=args.math_eval_size,
        seed=int(run_args.seed) + 5050,
    )
    math_records = prepare_eval_records(tokenizer, math_eval_raw, max_length=int(run_args.max_length))
    model = load_model_for_eval(run_args, state_path, device)
    eval_start = time.perf_counter()
    decoded = generate_batches(
        model,
        tokenizer,
        [item["prompt"] for item in math_records],
        max_input_length=int(run_args.max_length),
        max_new_tokens=int(run_args.eval_max_new_tokens),
        batch_size=int(run_args.eval_batch_size),
        device=device,
    )
    eval_elapsed = time.perf_counter() - eval_start

    correct = 0
    predictions = []
    for item, prediction_text in zip(math_records, decoded):
        pred_answer = normalize_math_answer(prediction_text)
        target_answer = item["target_answer"] or ""
        ok = pred_answer == target_answer
        correct += int(ok)
        predictions.append(
            {
                "prompt": item["prompt"],
                "target_answer": target_answer,
                "prediction_text": prediction_text,
                "prediction_answer": pred_answer,
                "correct": bool(ok),
            }
        )

    with open(run_dir / "math_predictions.jsonl", "w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")

    metrics = load_existing_metrics(metrics_path, run_dir, run_args)
    metrics["math_accuracy"] = correct / max(1, len(math_records))
    metrics["math_eval_elapsed_sec"] = eval_elapsed
    metrics["math_eval_examples"] = len(math_records)
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    del model
    del tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "run_dir": str(run_dir),
        "method": metrics.get("method"),
        "seed": metrics.get("seed"),
        "math_accuracy": metrics["math_accuracy"],
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
        state_path = run_dir / "final_full_state.pt"
        if not config_path.exists() or not state_path.exists():
            continue
        if not args.force and has_finite_math_accuracy(metrics_path):
            continue
        result = evaluate_run(run_dir, args, math_raw_examples)
        print(json.dumps({"stage": "faithful_math_eval_done", **result}))
        summary.append(result)

    print(json.dumps({"stage": "faithful_math_eval_summary", "completed_runs": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
