from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from exact_math_reasoning_data import load_or_prepare_exact_splits, sample_final_train_examples, tokenize_examples
from faithful_uniform_math_fsdp import (
    autocast_context,
    build_fsdp_model,
    cpu_collate,
    dist_barrier,
    evaluate_gsm8k,
    evaluate_tune_loss,
    init_distributed,
    is_main_process,
    load_tokenizer,
    maybe_make_output_dir,
    move_batch_to_device,
    seed_everything,
    top_up_tokenized_examples_with_replacement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--reweight-metrics-path", type=Path, required=True)
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
    parser.add_argument(
        "--use-best-reweight-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def source_counts_from_lambda(lambdas: Sequence[float], total_size: int) -> List[int]:
    expected = [total_size * weight for weight in lambdas]
    counts = [int(math.floor(value)) for value in expected]
    remainder = total_size - sum(counts)
    order = sorted(
        range(len(lambdas)),
        key=lambda idx: expected[idx] - counts[idx],
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def load_reweight_payload(
    *,
    metrics_path: Path,
    use_best_reweight_checkpoint: bool,
) -> tuple[List[float], List[str], Dict[str, object]]:
    payload = json.loads(metrics_path.read_text())
    source_names = [str(item) for item in payload["source_names"]]
    key = "best_lambda_reweight" if use_best_reweight_checkpoint else "learned_lambda"
    if key not in payload:
        raise RuntimeError(f"{metrics_path} does not contain {key}.")
    lambdas = [float(value) for value in payload[key]]
    if len(lambdas) != len(source_names):
        raise RuntimeError(f"Lambda/source-name size mismatch in {metrics_path}.")
    return lambdas, source_names, payload


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
            f"faithful_{args.method}_seed{args.seed}_train{args.reweight_train_per_source}"
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

    learned_lambda, lambda_source_names, reweight_payload = load_reweight_payload(
        metrics_path=args.reweight_metrics_path,
        use_best_reweight_checkpoint=args.use_best_reweight_checkpoint,
    )
    if lambda_source_names != splits["source_names"]:
        raise RuntimeError("Source names in reweight metrics do not match the faithful data split order.")
    source_counts = source_counts_from_lambda(learned_lambda, args.final_train_size)

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
    torch.distributed.destroy_process_group()

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
        "method": args.method,
        "model_name": args.model_name,
        "seed": args.seed,
        "source_names": splits["source_names"],
        "learned_lambda": learned_lambda,
        "source_counts": source_counts,
        "reweight_metrics_path": str(args.reweight_metrics_path),
        "reweight_payload_summary": {
            "method": reweight_payload.get("method"),
            "reference_model_name": reweight_payload.get("baseline_details", {}).get("reference_model_name"),
        },
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
