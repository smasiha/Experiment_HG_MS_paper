#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exact_math_reasoning import generate_batches, prepare_eval_records, seed_everything, subsample_examples
from exact_math_reasoning_data import extract_gsm8k_answer, load_eval_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2424)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--gsm8k-eval-size", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, *, device: torch.device, bf16: bool):
    local_files_only = os.environ.get("LOCAL_MODEL_ONLY", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if bf16 and device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=local_files_only,
    )
    model.config.use_cache = True
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2)

    model, tokenizer = load_model_and_tokenizer(args.model_name, device=device, bf16=args.bf16)
    gsm8k_raw = subsample_examples(
        load_eval_examples(include_math=False)["gsm8k"],
        limit=args.gsm8k_eval_size,
        seed=args.seed + 101,
    )
    gsm8k_records = prepare_eval_records(tokenizer, gsm8k_raw, max_length=args.max_length)

    eval_start = time.perf_counter()
    decoded = generate_batches(
        model,
        tokenizer,
        [item["prompt"] for item in gsm8k_records],
        max_input_length=args.max_length,
        max_new_tokens=args.eval_max_new_tokens,
        batch_size=args.eval_batch_size,
        device=device,
    )
    elapsed = time.perf_counter() - eval_start

    correct = 0
    predictions = []
    for item, prediction_text in zip(gsm8k_records, decoded):
        pred_answer = extract_gsm8k_answer(prediction_text)
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

    accuracy = correct / max(1, len(gsm8k_records))
    with open(output_dir / "gsm8k_predictions.jsonl", "w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(
            {
                "model_name": args.model_name,
                "seed": args.seed,
                "gsm8k_accuracy": accuracy,
                "eval_example_count": len(gsm8k_records),
                "eval_elapsed_sec": elapsed,
            },
            handle,
            indent=2,
        )

    print(json.dumps({"stage": "zero_shot_gsm8k_done", "accuracy": accuracy, "count": len(gsm8k_records)}, indent=2))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
