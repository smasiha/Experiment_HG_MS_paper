from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from exact_math_reasoning import (
    apply_theta_update,
    per_example_losses_from_logits,
    source_counts_from_lambda,
    theta_diagnostics,
    tokenized_by_source_or_fail,
)
from exact_math_reasoning_data import (
    load_or_prepare_exact_splits,
    sample_final_train_examples,
    tokenize_examples,
)
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


@dataclass
class LoaderState:
    loader: DataLoader
    sampler: DistributedSampler
    epoch: int
    iterator: Iterator | None = None


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
    parser.add_argument("--allow-train-shortfall", action="store_true")
    parser.add_argument("--reweight-epochs", type=int, default=3)
    parser.add_argument("--per-source-batch", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--reweight-model-lr", type=float, default=8e-6)
    parser.add_argument("--reweight-weight-decay", type=float, default=0.0)
    parser.add_argument("--reweight-outer-lr", type=float, default=1e-2)
    parser.add_argument("--theta-optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--theta-adam-beta1", type=float, default=0.9)
    parser.add_argument("--theta-adam-beta2", type=float, default=0.999)
    parser.add_argument("--theta-adam-eps", type=float, default=1e-8)
    parser.add_argument("--theta-grad-clip", type=float, default=0.0)
    parser.add_argument("--scalebio-alpha", type=float, default=100.0)
    parser.add_argument("--tune-every", type=int, default=50)
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


def dist_average_scalar(value: torch.Tensor) -> float:
    reduced = value.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return float(reduced.cpu())


def make_batch_loader_state(
    items: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
    drop_last: bool,
) -> LoaderState:
    sampler = DistributedSampler(
        list(items),
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=drop_last,
    )
    loader = DataLoader(
        list(items),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=list,
    )
    return LoaderState(loader=loader, sampler=sampler, epoch=0, iterator=None)


def next_items(state: LoaderState) -> List[Dict[str, object]]:
    if state.iterator is None:
        state.sampler.set_epoch(state.epoch)
        state.iterator = iter(state.loader)
    try:
        return next(state.iterator)
    except StopIteration:
        state.epoch += 1
        state.sampler.set_epoch(state.epoch)
        state.iterator = iter(state.loader)
        return next(state.iterator)


def weighted_loss_from_batch(
    *,
    model,
    batch: Dict[str, object],
    theta: torch.Tensor,
    use_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with autocast_context(batch["input_ids"].device, use_bf16):
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).logits
    per_example = per_example_losses_from_logits(logits, batch["labels"])
    unweighted_loss = per_example.mean()
    lambdas = torch.softmax(theta, dim=0)
    weighted = torch.zeros((), device=per_example.device, dtype=per_example.dtype)
    for source_id in range(theta.numel()):
        mask = batch["source_ids"] == source_id
        if mask.any():
            weighted = weighted + lambdas[source_id] * per_example[mask].mean()
    return weighted, unweighted_loss


def backward_weighted_source_batches(
    *,
    model,
    source_batches: Sequence[tuple[int, Dict[str, object]]],
    lambdas: torch.Tensor,
    use_bf16: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted = torch.zeros((), device=lambdas.device, dtype=torch.float32)
    source_losses: List[torch.Tensor] = []
    for source_id, batch in source_batches:
        source_loss = mean_loss_from_batch(model=model, batch=batch, use_bf16=use_bf16)
        detached_loss = source_loss.detach().float()
        source_losses.append(detached_loss)
        weighted = weighted + lambdas[source_id].float() * detached_loss
        (scale * lambdas[source_id] * source_loss).backward()
        del source_loss
    return weighted, torch.stack(source_losses)


def mean_loss_from_batch(
    *,
    model,
    batch: Dict[str, object],
    use_bf16: bool,
) -> torch.Tensor:
    with autocast_context(batch["input_ids"].device, use_bf16):
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).logits
    return per_example_losses_from_logits(logits, batch["labels"]).mean()


def evaluate_distributed_mean_loss(
    *,
    model,
    items: Sequence[Dict[str, object]],
    batch_size: int,
    tokenizer,
    rank: int,
    world_size: int,
    seed: int,
    device: torch.device,
    use_bf16: bool,
) -> float:
    sampler = DistributedSampler(
        list(items),
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed,
        drop_last=False,
    )
    loader = DataLoader(
        list(items),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=partial(cpu_collate, pad_token_id=tokenizer.pad_token_id),
    )
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_batches = torch.zeros((), device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss = mean_loss_from_batch(model=model, batch=batch, use_bf16=use_bf16)
            total_loss += loss.detach().float()
            total_batches += 1.0
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)
    model.train()
    return float((total_loss / total_batches.clamp_min(1.0)).cpu())


def train_final_model(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    train_items: Sequence[Dict[str, object]],
    rank: int,
    world_size: int,
    device: torch.device,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.final_ft_lr)
    sampler = DistributedSampler(
        list(train_items),
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        list(train_items),
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=partial(cpu_collate, pad_token_id=tokenizer.pad_token_id),
    )
    train_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for epoch in range(args.final_ft_epochs):
        sampler.set_epoch(args.seed + epoch)
        model.train()
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss = mean_loss_from_batch(model=model, batch=batch, use_bf16=args.bf16)
            (loss / args.grad_accum).backward()
            if ((step + 1) % args.grad_accum) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
    if step % args.grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return time.perf_counter() - train_start


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
            f"faithful_scalebio_seed{args.seed}_train{args.reweight_train_per_source}"
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

    tokenized_train_by_source, effective_train_per_source = tokenized_by_source_or_fail(
        tokenizer=tokenizer,
        raw_by_source=splits["reweight_train"],
        max_length=args.max_length,
        needed_per_source=args.reweight_train_per_source,
        allow_shortfall=args.allow_train_shortfall,
    )
    tokenized_val_opt = tokenize_examples(tokenizer, splits["val_opt"], max_length=args.max_length)
    tokenized_val_tune = tokenize_examples(tokenizer, splits["val_tune"], max_length=args.max_length)
    if not tokenized_val_opt or not tokenized_val_tune:
        raise RuntimeError("Validation splits became empty after tokenization.")

    train_loader_states = [
        make_batch_loader_state(
            source_examples,
            batch_size=args.per_source_batch,
            rank=rank,
            world_size=world_size,
            seed=args.seed + 101 * source_id,
            drop_last=True,
        )
        for source_id, source_examples in enumerate(tokenized_train_by_source)
    ]
    val_loader_state = make_batch_loader_state(
        tokenized_val_opt,
        batch_size=args.val_batch_size,
        rank=rank,
        world_size=world_size,
        seed=args.seed + 20_000,
        drop_last=False,
    )

    model_w = build_fsdp_model(args, device)
    model_u = build_fsdp_model(args, device)
    optimizer_w = torch.optim.AdamW(
        model_w.parameters(),
        lr=args.reweight_model_lr,
        weight_decay=args.reweight_weight_decay,
    )
    optimizer_u = torch.optim.AdamW(
        model_u.parameters(),
        lr=args.reweight_model_lr,
        weight_decay=args.reweight_weight_decay,
    )

    theta = torch.zeros(len(splits["source_names"]), dtype=torch.float32, device=device, requires_grad=True)
    theta_optimizer_state: Dict[str, torch.Tensor | int] | None = None
    best_theta = theta.detach().clone()
    best_tune_loss = float("inf")
    history: List[Dict[str, object]] = []

    steps_per_epoch = len(train_loader_states[0].loader)
    total_steps = args.reweight_epochs * steps_per_epoch
    reweight_start = time.perf_counter()
    for outer_step in range(total_steps):
        train_source_batches: List[tuple[int, Dict[str, object]]] = []
        for source_id, source_state in enumerate(train_loader_states):
            source_items = next_items(source_state)
            source_batch = move_batch_to_device(cpu_collate(source_items, tokenizer.pad_token_id), device)
            train_source_batches.append((source_id, source_batch))

        local_val_items = next_items(val_loader_state)
        val_batch = move_batch_to_device(cpu_collate(local_val_items, tokenizer.pad_token_id), device)

        optimizer_w.zero_grad(set_to_none=True)
        optimizer_u.zero_grad(set_to_none=True)
        if theta.grad is not None:
            theta.grad.zero_()

        lambdas_detached = torch.softmax(theta.detach(), dim=0)
        train_loss_u, source_losses_u = backward_weighted_source_batches(
            model=model_u,
            source_batches=train_source_batches,
            lambdas=lambdas_detached,
            use_bf16=args.bf16,
            scale=args.scalebio_alpha,
        )
        optimizer_u.step()

        train_loss_w, source_losses_w = backward_weighted_source_batches(
            model=model_w,
            source_batches=train_source_batches,
            lambdas=lambdas_detached,
            use_bf16=args.bf16,
            scale=args.scalebio_alpha,
        )

        theta_objective = args.scalebio_alpha * (
            (torch.softmax(theta, dim=0) * source_losses_w).sum()
            - (torch.softmax(theta, dim=0) * source_losses_u).sum()
        )
        grad_theta = torch.autograd.grad(
            theta_objective,
            theta,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]
        grad_theta = torch.zeros_like(theta) if grad_theta is None else grad_theta.detach()
        dist.all_reduce(grad_theta, op=dist.ReduceOp.SUM)
        grad_theta /= world_size

        val_loss = mean_loss_from_batch(model=model_w, batch=val_batch, use_bf16=args.bf16)
        val_loss.backward()
        optimizer_w.step()

        grad_stats = theta_diagnostics(theta, grad_theta)
        theta, theta_optimizer_state = apply_theta_update(
            theta=theta,
            grad_theta=grad_theta,
            args=args,
            optimizer_state=theta_optimizer_state,
        )

        train_loss_w_scalar = dist_average_scalar(train_loss_w)
        train_loss_u_scalar = dist_average_scalar(train_loss_u)
        val_loss_scalar = dist_average_scalar(val_loss)

        if outer_step == 0 or (outer_step + 1) % args.tune_every == 0 or outer_step + 1 == total_steps:
            tune_loss = evaluate_distributed_mean_loss(
                model=model_w,
                items=tokenized_val_tune,
                batch_size=args.eval_batch_size,
                tokenizer=tokenizer,
                rank=rank,
                world_size=world_size,
                seed=args.seed + 30_000,
                device=device,
                use_bf16=args.bf16,
            )
            if tune_loss < best_tune_loss:
                best_tune_loss = tune_loss
                best_theta = theta.detach().clone()
        else:
            tune_loss = math.nan

        record = {
            "outer_step": outer_step,
            "train_loss_weighted": train_loss_w_scalar,
            "train_loss_auxiliary": train_loss_u_scalar,
            "val_loss": val_loss_scalar,
            "lambda": torch.softmax(theta.detach(), dim=0).cpu().tolist(),
            "tune_loss": tune_loss,
            "best_tune_loss": best_tune_loss,
            "elapsed_reweight_sec": time.perf_counter() - reweight_start,
            **grad_stats,
        }
        if is_main_process(rank):
            history.append(record)
            if outer_step == 0 or (outer_step + 1) % 50 == 0 or outer_step + 1 == total_steps:
                print(json.dumps({"stage": "reweight_progress", **record}))

    best_lambda_reweight = torch.softmax(best_theta, dim=0).cpu().tolist()
    final_lambda_reweight = torch.softmax(theta.detach(), dim=0).cpu().tolist()
    reweight_elapsed = time.perf_counter() - reweight_start
    if is_main_process(rank):
        with open(output_dir / "reweight_history.jsonl", "w") as handle:
            for row in history:
                handle.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    "stage": "reweight_done",
                    "best_lambda_reweight": best_lambda_reweight,
                    "final_lambda_reweight": final_lambda_reweight,
                    "best_tune_loss": best_tune_loss,
                    "reweight_elapsed_sec": reweight_elapsed,
                    "reweight_steps_completed": total_steps,
                },
                indent=2,
            )
        )

    source_counts = source_counts_from_lambda(best_lambda_reweight, args.final_train_size)
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

    del model_u
    gc.collect()
    torch.cuda.empty_cache()

    final_train_elapsed = train_final_model(
        args=args,
        model=model_w,
        tokenizer=tokenizer,
        train_items=final_train_tokenized,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model_w, StateDictType.FULL_STATE_DICT, full_state_config):
        full_state_dict = model_w.state_dict()

    dist_barrier(local_rank)
    del model_w
    torch.cuda.empty_cache()
    dist.destroy_process_group()

    if not is_main_process(rank):
        return

    final_state_path = output_dir / "final_full_state.pt"
    torch.save(full_state_dict, final_state_path)

    final_tune_loss = evaluate_tune_loss(
        args=args,
        full_state_dict=full_state_dict,
        tokenizer=tokenizer,
        tune_examples_tokenized=tokenized_val_tune,
    )
    gsm8k_metrics = evaluate_gsm8k(
        args=args,
        full_state_dict=full_state_dict,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )
    metrics = {
        "method": "scalebio",
        "model_name": args.model_name,
        "seed": args.seed,
        "source_names": splits["source_names"],
        "best_lambda_reweight": best_lambda_reweight,
        "final_lambda_reweight": final_lambda_reweight,
        "source_counts": source_counts,
        "best_tune_loss_reweight": best_tune_loss,
        "reweight_steps_completed": total_steps,
        "reweight_elapsed_sec": reweight_elapsed,
        "final_ft_tune_loss": final_tune_loss,
        "final_train_examples": len(final_train_tokenized),
        "final_train_elapsed_sec": final_train_elapsed,
        "world_size": world_size,
        "final_full_state_path": str(final_state_path),
        **gsm8k_metrics,
    }
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
