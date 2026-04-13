from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from exact_math_reasoning_data import (
    collate_batch,
    extract_gsm8k_answer,
    load_eval_examples,
    load_or_prepare_exact_splits,
    sample_final_train_examples,
    tokenize_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2424)
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--reweight-train-per-source", type=int, default=3000)
    parser.add_argument("--train-raw-oversample", type=float, default=4.0)
    parser.add_argument("--val-per-source", type=int, default=280)
    parser.add_argument("--val-raw-oversample", type=float, default=1.2)
    parser.add_argument("--final-train-size", type=int, default=20000)
    parser.add_argument("--final-ft-lr", type=float, default=8e-6)
    parser.add_argument("--final-ft-epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--gsm8k-eval-size", type=int, default=0)
    parser.add_argument("--fsdp-min-num-params", type=int, default=10_000_000)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    return local_rank, rank, world_size, torch.device("cuda", local_rank)


def is_main_process(rank: int) -> bool:
    return rank == 0


def dist_barrier(local_rank: int) -> None:
    dist.barrier(device_ids=[local_rank])


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def maybe_make_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def equal_source_counts(num_sources: int, total_size: int) -> List[int]:
    expected = [total_size / num_sources for _ in range(num_sources)]
    counts = [int(math.floor(value)) for value in expected]
    remainder = total_size - sum(counts)
    for idx in range(remainder):
        counts[idx] += 1
    return counts


def top_up_tokenized_examples_with_replacement(
    *,
    examples: List[Dict[str, object]],
    target_size: int,
    seed: int,
) -> List[Dict[str, object]]:
    if len(examples) >= target_size:
        return examples[:target_size]
    if not examples:
        raise RuntimeError("Cannot top up an empty tokenized training set.")
    rng = random.Random(seed + 808_081)
    topped_up = list(examples)
    topped_up.extend(dict(rng.choice(examples)) for _ in range(target_size - len(examples)))
    rng.shuffle(topped_up)
    return topped_up


def local_model_only() -> bool:
    return os.environ.get("LOCAL_MODEL_ONLY", "0") == "1"


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_model_only(),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def infer_transformer_layer_classes(model) -> tuple[type, ...]:
    target_names = {"LlamaDecoderLayer", "Qwen2DecoderLayer", "Gemma2DecoderLayer"}
    classes = []
    for module in model.modules():
        cls = type(module)
        if cls.__name__ in target_names and cls not in classes:
            classes.append(cls)
    return tuple(classes)


def build_fsdp_model(args: argparse.Namespace, device: torch.device):
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        local_files_only=local_model_only(),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    transformer_layer_cls = infer_transformer_layer_classes(model)
    if transformer_layer_cls:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        )
    else:
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=args.fsdp_min_num_params,
        )

    mixed_precision = None
    if args.bf16:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def cpu_collate(batch: Sequence[Dict[str, object]], pad_token_id: int) -> Dict[str, object]:
    return collate_batch(batch, device=torch.device("cpu"), pad_token_id=pad_token_id)


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def prepare_eval_records(tokenizer, raw_examples: Sequence[object], max_length: int) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for example in raw_examples:
        prompt_ids = tokenizer(
            example.prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length + 1,
        )["input_ids"]
        if len(prompt_ids) > max_length:
            continue
        records.append({"prompt": example.prompt, "target_answer": example.target_answer})
    return records


def generate_batches(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_input_length: int,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
) -> List[str]:
    tokenizer.padding_side = "left"
    outputs: List[str] = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if tokenizer_limit is None or tokenizer_limit <= 0 or tokenizer_limit > 1_000_000:
        tokenizer_limit = max_input_length
    else:
        tokenizer_limit = min(int(tokenizer_limit), max_input_length)
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = list(prompts[start : start + batch_size])
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer_limit,
            ).to(device)
            generated = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            outputs.extend(decoded)
    tokenizer.padding_side = "right"
    return outputs


def evaluate_gsm8k(
    *,
    args: argparse.Namespace,
    full_state_dict: Dict[str, torch.Tensor],
    tokenizer,
    output_dir: Path,
) -> Dict[str, float]:
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=getattr(args, "attn_implementation", "sdpa"),
        local_files_only=local_model_only(),
        low_cpu_mem_usage=True,
    ).to(device)
    model.load_state_dict(full_state_dict, strict=True)
    model.eval()
    model.config.use_cache = True

    eval_examples = load_eval_examples(include_math=False)
    gsm8k_raw = eval_examples["gsm8k"]
    if args.gsm8k_eval_size > 0:
        rng = random.Random(args.seed + 4040)
        indices = list(range(len(gsm8k_raw)))
        rng.shuffle(indices)
        picked = sorted(indices[: args.gsm8k_eval_size])
        gsm8k_raw = [gsm8k_raw[idx] for idx in picked]
    gsm8k_records = prepare_eval_records(tokenizer, gsm8k_raw, max_length=args.max_length)
    prompts = [item["prompt"] for item in gsm8k_records]
    decoded = generate_batches(
        model,
        tokenizer,
        prompts,
        max_input_length=args.max_length,
        max_new_tokens=args.eval_max_new_tokens,
        batch_size=args.eval_batch_size,
        device=device,
    )
    correct = 0
    predictions = []
    for item, prediction_text in zip(gsm8k_records, decoded):
        prediction_answer = extract_gsm8k_answer(prediction_text)
        target_answer = item["target_answer"] or ""
        is_correct = prediction_answer == target_answer
        correct += int(is_correct)
        predictions.append(
            {
                "prompt": item["prompt"],
                "target_answer": target_answer,
                "prediction_text": prediction_text,
                "prediction_answer": prediction_answer,
                "correct": bool(is_correct),
            }
        )
    accuracy = correct / max(1, len(gsm8k_records))
    with open(output_dir / "gsm8k_predictions.jsonl", "w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    del model
    torch.cuda.empty_cache()
    return {"gsm8k_accuracy": accuracy, "gsm8k_eval_examples": len(gsm8k_records)}


def evaluate_tune_loss(
    *,
    args: argparse.Namespace,
    full_state_dict: Dict[str, torch.Tensor],
    tokenizer,
    tune_examples_tokenized: Sequence[Dict[str, object]],
) -> float:
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=getattr(args, "attn_implementation", "sdpa"),
        local_files_only=local_model_only(),
        low_cpu_mem_usage=True,
    ).to(device)
    model.load_state_dict(full_state_dict, strict=True)
    model.eval()

    loader = DataLoader(
        list(tune_examples_tokenized),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=partial(cpu_collate, pad_token_id=tokenizer.pad_token_id),
    )
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with autocast_context(device, args.bf16):
                loss = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                ).loss
            total_loss += float(loss.detach().cpu())
            batch_count += 1
    del model
    torch.cuda.empty_cache()
    return total_loss / max(1, batch_count)


def main() -> None:
    args = parse_args()
    local_rank, rank, world_size, device = init_distributed()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = maybe_make_output_dir(args.output_dir)
    if is_main_process(rank):
        with open(output_dir / "config.json", "w") as handle:
            json.dump(vars(args), handle, indent=2, default=str)

    cache_path = args.cache_path or str(
        output_dir
        / (
            f"faithful_uniform_seed{args.seed}_train{args.reweight_train_per_source}"
            f"_val{args.val_per_source}_tov{args.train_raw_oversample}_vov{args.val_raw_oversample}.json"
        )
    )

    if is_main_process(rank):
        load_or_prepare_exact_splits(
            cache_path=cache_path,
            reweight_train_per_source=args.reweight_train_per_source,
            train_raw_oversample=args.train_raw_oversample,
            val_per_source=args.val_per_source,
            val_raw_oversample=args.val_raw_oversample,
            seed=args.seed,
        )
    dist_barrier(local_rank)

    tokenizer = load_tokenizer(args.model_name)
    splits = load_or_prepare_exact_splits(
        cache_path=cache_path,
        reweight_train_per_source=args.reweight_train_per_source,
        train_raw_oversample=args.train_raw_oversample,
        val_per_source=args.val_per_source,
        val_raw_oversample=args.val_raw_oversample,
        seed=args.seed,
    )

    source_counts = equal_source_counts(len(splits["source_names"]), args.final_train_size)
    oversample_schedule = [
        args.train_raw_oversample,
        max(args.train_raw_oversample, 2.5),
        max(args.train_raw_oversample, 4.0),
    ]
    final_train_tokenized: List[Dict[str, object]] = []
    for oversample in oversample_schedule:
        final_examples = sample_final_train_examples(
            source_counts=source_counts,
            train_raw_oversample=oversample,
            seed=args.seed,
        )
        final_train_tokenized = tokenize_examples(tokenizer, final_examples, max_length=args.max_length)
        if len(final_train_tokenized) >= args.final_train_size:
            break
    if len(final_train_tokenized) < args.final_train_size:
        final_train_tokenized = top_up_tokenized_examples_with_replacement(
            examples=final_train_tokenized,
            target_size=args.final_train_size,
            seed=args.seed,
        )
    else:
        final_train_tokenized = final_train_tokenized[: args.final_train_size]
    tune_examples_tokenized = tokenize_examples(tokenizer, splits["val_tune"], max_length=args.max_length)
    if not tune_examples_tokenized:
        raise RuntimeError("Validation tune set became empty after tokenization.")

    model = build_fsdp_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.final_ft_lr)

    train_sampler = DistributedSampler(
        final_train_tokenized,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    train_loader = DataLoader(
        final_train_tokenized,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=partial(cpu_collate, pad_token_id=tokenizer.pad_token_id),
    )

    train_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for epoch in range(args.final_ft_epochs):
        train_sampler.set_epoch(args.seed + epoch)
        model.train()
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            with autocast_context(device, args.bf16):
                loss = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                ).loss
            (loss / args.grad_accum).backward()
            if ((step + 1) % args.grad_accum) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
    if step % args.grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    train_elapsed = time.perf_counter() - train_start

    full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
        full_state_dict = model.state_dict()

    dist_barrier(local_rank)
    del model
    torch.cuda.empty_cache()
    dist.destroy_process_group()

    if not is_main_process(rank):
        return

    final_state_path = output_dir / "final_full_state.pt"
    torch.save(full_state_dict, final_state_path)

    tune_loss = evaluate_tune_loss(
        args=args,
        full_state_dict=full_state_dict,
        tokenizer=tokenizer,
        tune_examples_tokenized=tune_examples_tokenized,
    )
    gsm8k_metrics = evaluate_gsm8k(
        args=args,
        full_state_dict=full_state_dict,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )

    metrics = {
        "method": "uniform",
        "model_name": args.model_name,
        "seed": args.seed,
        "source_names": splits["source_names"],
        "source_counts": source_counts,
        "final_ft_tune_loss": tune_loss,
        "final_train_examples": len(final_train_tokenized),
        "train_elapsed_sec": train_elapsed,
        "world_size": world_size,
        "final_full_state_path": str(final_state_path),
        **gsm8k_metrics,
    }
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
