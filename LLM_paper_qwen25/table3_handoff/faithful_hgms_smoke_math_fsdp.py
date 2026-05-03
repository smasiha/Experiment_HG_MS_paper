from __future__ import annotations

import argparse
import gc
import json
import math
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from exact_math_reasoning import apply_theta_update, theta_diagnostics, tokenized_by_source_or_fail
from exact_math_reasoning_data import load_or_prepare_exact_splits, tokenize_examples
from faithful_scalebio_math_fsdp import (
    backward_weighted_source_batches,
    dist_average_scalar,
    evaluate_distributed_mean_loss,
    make_batch_loader_state,
    mean_loss_from_batch,
    next_items,
)
from faithful_uniform_math_fsdp import (
    build_fsdp_model,
    cpu_collate,
    dist_barrier,
    init_distributed,
    is_main_process,
    load_tokenizer,
    maybe_make_output_dir,
    move_batch_to_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2424)
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--reweight-train-per-source", type=int, default=500)
    parser.add_argument("--train-raw-oversample", type=float, default=4.0)
    parser.add_argument("--val-per-source", type=int, default=80)
    parser.add_argument("--val-raw-oversample", type=float, default=1.2)
    parser.add_argument("--allow-train-shortfall", action="store_true")
    parser.add_argument("--reweight-steps", type=int, default=10)
    parser.add_argument("--m-branches", type=int, default=2)
    parser.add_argument("--per-source-batch", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--reweight-model-lr", type=float, default=8e-6)
    parser.add_argument("--reweight-weight-decay", type=float, default=0.0)
    parser.add_argument("--reweight-outer-lr", type=float, default=1e-2)
    parser.add_argument("--theta-optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--theta-adam-beta1", type=float, default=0.9)
    parser.add_argument("--theta-adam-beta2", type=float, default=0.999)
    parser.add_argument("--theta-adam-eps", type=float, default=1e-8)
    parser.add_argument("--theta-grad-clip", type=float, default=0.0)
    parser.add_argument("--tune-every", type=int, default=5)
    parser.add_argument("--min-lambda-movement", type=float, default=0.0)
    parser.add_argument("--min-centered-grad-norm", type=float, default=0.0)
    parser.add_argument("--fsdp-min-num-params", type=int, default=10_000_000)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


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
            f"faithful_hgms_smoke_seed{args.seed}_train{args.reweight_train_per_source}"
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

    branch_train_loader_states = [
        [
            make_batch_loader_state(
                source_examples,
                batch_size=args.per_source_batch,
                rank=rank,
                world_size=world_size,
                seed=args.seed + 10_000 * branch_id + 101 * source_id,
                drop_last=False,
            )
            for source_id, source_examples in enumerate(tokenized_train_by_source)
        ]
        for branch_id in range(args.m_branches)
    ]
    val_loader_state = make_batch_loader_state(
        tokenized_val_opt,
        batch_size=args.val_batch_size,
        rank=rank,
        world_size=world_size,
        seed=args.seed + 90_000,
        drop_last=False,
    )

    if is_main_process(rank):
        print(
            json.dumps(
                {
                    "stage": "hgms_fsdp_smoke_start",
                    "diagnostic_note": (
                        "Memory-safe FSDP branch-selection diagnostic. "
                        "This is not the exact differentiable HG-MS hypergradient path used by the smaller LLM runs."
                    ),
                    "model_name": args.model_name,
                    "branches": args.m_branches,
                    "world_size": world_size,
                    "effective_train_per_source": effective_train_per_source,
                    "val_opt_size": len(tokenized_val_opt),
                    "val_tune_size": len(tokenized_val_tune),
                }
            )
        )

    models = [build_fsdp_model(args, device) for _ in range(args.m_branches)]
    # Keep the diagnostic memory-bound on model/activation storage rather than
    # Adam moment buffers; this smoke only checks branch selection and lambda motion.
    optimizers = [
        torch.optim.SGD(
            model.parameters(),
            lr=args.reweight_model_lr,
            weight_decay=args.reweight_weight_decay,
        )
        for model in models
    ]

    theta = torch.zeros(len(splits["source_names"]), dtype=torch.float32, device=device, requires_grad=True)
    theta_optimizer_state: Dict[str, torch.Tensor | int] | None = None
    best_theta = theta.detach().clone()
    best_tune_loss = float("inf")
    history: List[Dict[str, object]] = []
    max_centered_grad_norm = 0.0
    reweight_start = time.perf_counter()

    for outer_step in range(args.reweight_steps):
        lambdas_detached = torch.softmax(theta.detach(), dim=0)
        local_val_items = next_items(val_loader_state)
        val_batch = move_batch_to_device(cpu_collate(local_val_items, tokenizer.pad_token_id), device)
        branch_records: List[Dict[str, object]] = []
        branch_source_losses: List[torch.Tensor] = []

        for branch_id, (model, optimizer) in enumerate(zip(models, optimizers)):
            model.train()
            train_source_batches = []
            for source_id, source_state in enumerate(branch_train_loader_states[branch_id]):
                source_items = next_items(source_state)
                source_batch = move_batch_to_device(cpu_collate(source_items, tokenizer.pad_token_id), device)
                train_source_batches.append((source_id, source_batch))

            optimizer.zero_grad(set_to_none=True)
            train_loss, source_losses = backward_weighted_source_batches(
                model=model,
                source_batches=train_source_batches,
                lambdas=lambdas_detached,
                use_bf16=args.bf16,
                scale=1.0,
            )
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = mean_loss_from_batch(model=model, batch=val_batch, use_bf16=args.bf16)
            val_loss_scalar = dist_average_scalar(val_loss)
            train_loss_scalar = dist_average_scalar(train_loss)
            branch_source_losses.append(source_losses.detach())
            branch_records.append(
                {
                    "branch_index": branch_id,
                    "train_loss": train_loss_scalar,
                    "selection_val_loss": val_loss_scalar,
                }
            )
            del train_source_batches, train_loss, source_losses, val_loss
            torch.cuda.empty_cache()

        selected_branch = min(branch_records, key=lambda row: float(row["selection_val_loss"]))
        selected_index = int(selected_branch["branch_index"])
        selected_source_losses = branch_source_losses[selected_index]
        theta_objective = (torch.softmax(theta, dim=0) * selected_source_losses).sum()
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

        grad_stats = theta_diagnostics(theta, grad_theta)
        max_centered_grad_norm = max(max_centered_grad_norm, float(grad_stats["theta_grad_centered_norm"]))
        theta, theta_optimizer_state = apply_theta_update(
            theta=theta,
            grad_theta=grad_theta,
            args=args,
            optimizer_state=theta_optimizer_state,
        )

        if outer_step == 0 or (outer_step + 1) % args.tune_every == 0 or outer_step + 1 == args.reweight_steps:
            tune_loss = evaluate_distributed_mean_loss(
                model=models[selected_index],
                items=tokenized_val_tune,
                batch_size=args.val_batch_size,
                tokenizer=tokenizer,
                rank=rank,
                world_size=world_size,
                seed=args.seed + 100_000,
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
            "lambda": torch.softmax(theta.detach(), dim=0).cpu().tolist(),
            "selected_branch": selected_branch,
            "branch_summaries": branch_records,
            "tune_loss": tune_loss,
            "best_tune_loss": best_tune_loss,
            "elapsed_reweight_sec": time.perf_counter() - reweight_start,
            **grad_stats,
        }
        if is_main_process(rank):
            history.append(record)
            print(json.dumps({"stage": "hgms_fsdp_smoke_progress", **record}))

        del branch_source_losses, val_batch, selected_source_losses, theta_objective, grad_theta
        gc.collect()
        torch.cuda.empty_cache()

    best_lambda_reweight = torch.softmax(best_theta, dim=0).cpu().tolist()
    final_lambda_reweight = torch.softmax(theta.detach(), dim=0).cpu().tolist()
    uniform = [1.0 / len(final_lambda_reweight)] * len(final_lambda_reweight)
    final_lambda_l1_from_uniform = float(sum(abs(a - b) for a, b in zip(final_lambda_reweight, uniform)))
    reweight_elapsed = time.perf_counter() - reweight_start

    if is_main_process(rank):
        with open(output_dir / "reweight_history.jsonl", "w") as handle:
            for row in history:
                handle.write(json.dumps(row) + "\n")
        metrics = {
            "method": "hg_ms_fsdp_smoke",
            "diagnostic_note": (
                "Memory-safe FSDP branch-selection diagnostic. "
                "It verifies 32B branch selection and nonuniform lambda movement, "
                "but it is not an exact differentiable HG-MS hypergradient result."
            ),
            "model_name": args.model_name,
            "seed": args.seed,
            "source_names": splits["source_names"],
            "best_lambda_reweight": best_lambda_reweight,
            "final_lambda_reweight": final_lambda_reweight,
            "final_lambda_l1_from_uniform": final_lambda_l1_from_uniform,
            "max_centered_grad_norm": max_centered_grad_norm,
            "best_tune_loss_reweight": best_tune_loss,
            "reweight_steps_completed": args.reweight_steps,
            "reweight_elapsed_sec": reweight_elapsed,
            "world_size": world_size,
            "final_eval_status": "skipped_stop_after_reweight",
        }
        with open(output_dir / "metrics.json", "w") as handle:
            json.dump(metrics, handle, indent=2)
        print(json.dumps({"stage": "hgms_fsdp_smoke_done", **metrics}, indent=2))

        if args.min_lambda_movement > 0 and final_lambda_l1_from_uniform < args.min_lambda_movement:
            raise RuntimeError(
                f"Lambda movement smoke failed: L1={final_lambda_l1_from_uniform:.3e} "
                f"< threshold={args.min_lambda_movement:.3e}."
            )
        if args.min_centered_grad_norm > 0 and max_centered_grad_norm < args.min_centered_grad_norm:
            raise RuntimeError(
                f"Gradient smoke failed: max centered grad={max_centered_grad_norm:.3e} "
                f"< threshold={args.min_centered_grad_norm:.3e}."
            )

    dist_barrier(local_rank)
    del optimizers
    del models
    gc.collect()
    torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
