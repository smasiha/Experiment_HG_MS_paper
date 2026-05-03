from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM

from exact_math_reasoning import BranchOutcome, apply_theta_update, pick_branch_index, theta_diagnostics, tokenized_by_source_or_fail
from exact_math_reasoning_data import load_or_prepare_exact_splits, tokenize_examples
from faithful_scalebio_math_fsdp import (
    dist_average_scalar,
    evaluate_distributed_mean_loss,
    make_batch_loader_state,
    mean_loss_from_batch,
    next_items,
)
from faithful_uniform_math_fsdp import (
    cpu_collate,
    dist_barrier,
    infer_transformer_layer_classes,
    init_distributed,
    is_main_process,
    load_tokenizer,
    local_model_only,
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
    parser.add_argument("--m-branches", type=int, default=4)
    parser.add_argument("--per-source-batch", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--branch-noise-std", type=float, default=5e-4)
    parser.add_argument("--selection-train-tolerance", type=float, default=0.05)
    parser.add_argument("--selection-train-penalty", type=float, default=0.0)
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
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    return parser.parse_args()


def build_lora_fsdp_model(args: argparse.Namespace, device: torch.device):
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        local_files_only=local_model_only(),
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    if args.bf16:
        # PEFT initializes LoRA adapters in fp32 by default. FSDP cannot flatten
        # bf16 base weights together with fp32 adapter weights inside the same
        # wrapped transformer block, so keep adapter dtype aligned with the base.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.bfloat16)
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

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


def trainable_named_parameters(model) -> List[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def get_trainable_state(model) -> Dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in trainable_named_parameters(model)}


def set_trainable_state(model, state: Dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            params[name].copy_(tensor.to(device=params[name].device, dtype=params[name].dtype))


def add_trainable_noise(model, *, noise_std: float, seed: int) -> None:
    if noise_std <= 0:
        return
    generator = torch.Generator(device=next(model.parameters()).device)
    generator.manual_seed(seed)
    with torch.no_grad():
        for _, param in trainable_named_parameters(model):
            param.add_(torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=generator) * noise_std)


def safe_grads(params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor | None]) -> List[torch.Tensor]:
    return [torch.zeros_like(param) if grad is None else grad for param, grad in zip(params, grads)]


def distributed_grad_norm(grads: Sequence[torch.Tensor]) -> float:
    total = torch.zeros((), device=grads[0].device if grads else "cuda", dtype=torch.float32)
    for grad in grads:
        total = total + grad.detach().float().square().sum()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float(torch.sqrt(total).cpu())


def backward_grads(model, loss: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    # FSDP with activation checkpointing is compatible with backward(), while
    # torch.autograd.grad can silently return disconnected original handles in
    # this PEFT+FSDP setup. Collect first-order LoRA gradients from .grad.
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = [
        torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
        for param in params
    ]
    model.zero_grad(set_to_none=True)
    return grads


def apply_sgd_update(
    params: Sequence[torch.nn.Parameter],
    grads: Sequence[torch.Tensor],
    *,
    lr: float,
    weight_decay: float,
) -> None:
    with torch.no_grad():
        for param, grad in zip(params, grads):
            update = grad
            if weight_decay > 0:
                update = update + weight_decay * param
            param.add_(update, alpha=-lr)


def source_losses_from_batches(
    *,
    model,
    source_batches: Sequence[tuple[int, Dict[str, object]]],
    use_bf16: bool,
) -> List[torch.Tensor]:
    losses: List[torch.Tensor] = []
    for _, batch in source_batches:
        losses.append(mean_loss_from_batch(model=model, batch=batch, use_bf16=use_bf16))
    return losses


def weighted_loss_from_source_losses(theta_or_lambdas: torch.Tensor, source_losses: Sequence[torch.Tensor]) -> torch.Tensor:
    if theta_or_lambdas.requires_grad:
        weights = torch.softmax(theta_or_lambdas, dim=0)
    else:
        weights = theta_or_lambdas
    total = torch.zeros((), device=weights.device, dtype=source_losses[0].dtype)
    for source_id, source_loss in enumerate(source_losses):
        total = total + weights[source_id].to(source_loss.dtype) * source_loss
    return total


def exact_one_step_theta_grad(
    *,
    model,
    initial_state: Dict[str, torch.Tensor],
    train_source_items: Sequence[tuple[int, Sequence[Dict[str, object]]]],
    val_items: Sequence[Dict[str, object]],
    tokenizer,
    theta: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    set_trainable_state(model, initial_state)
    params = [param for _, param in trainable_named_parameters(model)]
    if not params:
        raise RuntimeError("No trainable LoRA parameters found for exact HG-MS FSDP hypergradient.")

    per_source_grads: List[List[torch.Tensor]] = []
    per_source_grad_norms: List[float] = []
    source_loss_scalars: List[float] = []
    weighted_grads = [torch.zeros_like(param) for param in params]
    lambdas_detached = torch.softmax(theta.detach(), dim=0)

    for source_id, source_items in train_source_items:
        batch = move_batch_to_device(cpu_collate(source_items, tokenizer.pad_token_id), device)
        loss = mean_loss_from_batch(model=model, batch=batch, use_bf16=args.bf16)
        grads = backward_grads(model, loss, params)
        detached_grads = [grad.detach() for grad in grads]
        per_source_grads.append(detached_grads)
        per_source_grad_norms.append(distributed_grad_norm(detached_grads))
        source_loss_scalars.append(dist_average_scalar(loss.detach()))
        for idx, grad in enumerate(detached_grads):
            weighted_grads[idx] = weighted_grads[idx] + lambdas_detached[source_id].to(grad.dtype) * grad
        del batch, loss, grads

    apply_sgd_update(
        params,
        weighted_grads,
        lr=args.reweight_model_lr,
        weight_decay=args.reweight_weight_decay,
    )

    val_batch = move_batch_to_device(cpu_collate(val_items, tokenizer.pad_token_id), device)
    val_loss = mean_loss_from_batch(model=model, batch=val_batch, use_bf16=args.bf16)
    val_grads = backward_grads(model, val_loss, params)
    val_loss_scalar = dist_average_scalar(val_loss.detach())
    val_grad_norm = distributed_grad_norm(val_grads)

    dot_values: List[torch.Tensor] = []
    for source_grads in per_source_grads:
        local_dot = torch.zeros((), device=device, dtype=torch.float32)
        for grad_s, grad_v in zip(source_grads, val_grads):
            local_dot = local_dot + (grad_s.float() * grad_v.detach().float()).sum()
        dist.all_reduce(local_dot, op=dist.ReduceOp.SUM)
        dot_values.append(local_dot.detach())

    dot_tensor = torch.stack(dot_values).to(device=theta.device, dtype=theta.dtype)
    theta_objective = -args.reweight_model_lr * (torch.softmax(theta, dim=0) * dot_tensor).sum()
    grad_theta = torch.autograd.grad(theta_objective, theta, retain_graph=False, create_graph=False)[0].detach()
    dot_values_cpu = [float(value.cpu()) for value in dot_values]
    dot_min = min(dot_values_cpu) if dot_values_cpu else 0.0
    dot_max = max(dot_values_cpu) if dot_values_cpu else 0.0

    del val_batch, val_loss, val_grads, per_source_grads, weighted_grads, dot_tensor, theta_objective
    gc.collect()
    torch.cuda.empty_cache()
    return grad_theta, {
        "hypergrad_val_loss": val_loss_scalar,
        "selected_source_loss_mean": float(sum(source_loss_scalars) / max(1, len(source_loss_scalars))),
        "selected_source_losses": source_loss_scalars,
        "source_grad_norms": per_source_grad_norms,
        "val_grad_norm": val_grad_norm,
        "source_val_dot_values": dot_values_cpu,
        "source_val_dot_min": dot_min,
        "source_val_dot_max": dot_max,
        "source_val_dot_range": dot_max - dot_min,
    }


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
            f"faithful_hgms_direct_seed{args.seed}_train{args.reweight_train_per_source}"
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
                    "stage": "hgms_direct_fsdp_start",
                    "method_note": (
                        "Direct HG-MS FSDP reweighting with LoRA trainable parameters. "
                        "Branch selection is followed by an exact one-step source-logit hypergradient "
                        "for the selected branch."
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

    models = [build_lora_fsdp_model(args, device) for _ in range(args.m_branches)]
    optimizers = [
        torch.optim.SGD(
            (param for _, param in trainable_named_parameters(model)),
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
    current_branch_index = 0
    reweight_start = time.perf_counter()

    for outer_step in range(args.reweight_steps):
        current_state = get_trainable_state(models[current_branch_index])
        branch_initial_states: List[Dict[str, torch.Tensor]] = []
        branch_train_items: List[List[tuple[int, Sequence[Dict[str, object]]]]] = []
        branch_records: List[Dict[str, object]] = []
        local_val_items = next_items(val_loader_state)
        val_batch = move_batch_to_device(cpu_collate(local_val_items, tokenizer.pad_token_id), device)
        lambdas_detached = torch.softmax(theta.detach(), dim=0)

        for branch_id, (model, optimizer) in enumerate(zip(models, optimizers)):
            set_trainable_state(model, current_state)
            add_trainable_noise(model, noise_std=args.branch_noise_std if branch_id > 0 else 0.0, seed=args.seed + outer_step * 1009 + branch_id)
            initial_state = get_trainable_state(model)
            branch_initial_states.append(initial_state)

            model.train()
            train_source_batches = []
            train_source_items = []
            for source_id, source_state in enumerate(branch_train_loader_states[branch_id]):
                source_items = next_items(source_state)
                source_batch = move_batch_to_device(cpu_collate(source_items, tokenizer.pad_token_id), device)
                train_source_items.append((source_id, source_items))
                train_source_batches.append((source_id, source_batch))
            branch_train_items.append(train_source_items)

            optimizer.zero_grad(set_to_none=True)
            source_losses = source_losses_from_batches(model=model, source_batches=train_source_batches, use_bf16=args.bf16)
            train_loss = weighted_loss_from_source_losses(lambdas_detached, source_losses)
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = mean_loss_from_batch(model=model, batch=val_batch, use_bf16=args.bf16)
            branch_records.append(
                {
                    "branch_index": branch_id,
                    "train_loss": dist_average_scalar(train_loss.detach()),
                    "selection_val_loss": dist_average_scalar(val_loss.detach()),
                }
            )
            del train_source_batches, source_losses, train_loss, val_loss
            torch.cuda.empty_cache()

        selected_index = pick_branch_index(
            "hg_ms",
            [
                BranchOutcome(
                    branch_index=int(record["branch_index"]),
                    train_loss_scalar=float(record["train_loss"]),
                    selection_val_scalar=float(record["selection_val_loss"]),
                    params={},
                )
                for record in branch_records
            ],
            random.Random(args.seed + outer_step),
            train_tolerance=args.selection_train_tolerance,
            train_penalty=args.selection_train_penalty,
        )
        selected_branch = branch_records[selected_index]
        grad_theta, hypergrad_info = exact_one_step_theta_grad(
            model=models[selected_index],
            initial_state=branch_initial_states[selected_index],
            train_source_items=branch_train_items[selected_index],
            val_items=local_val_items,
            tokenizer=tokenizer,
            theta=theta,
            args=args,
            device=device,
        )
        grad_stats = theta_diagnostics(theta, grad_theta)
        max_centered_grad_norm = max(max_centered_grad_norm, float(grad_stats["theta_grad_centered_norm"]))
        theta, theta_optimizer_state = apply_theta_update(
            theta=theta,
            grad_theta=grad_theta,
            args=args,
            optimizer_state=theta_optimizer_state,
        )
        current_branch_index = selected_index

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
            "selected_branch": {
                **selected_branch,
                **hypergrad_info,
            },
            "branch_summaries": branch_records,
            "tune_loss": tune_loss,
            "best_tune_loss": best_tune_loss,
            "elapsed_reweight_sec": time.perf_counter() - reweight_start,
            **grad_stats,
        }
        if is_main_process(rank):
            history.append(record)
            print(json.dumps({"stage": "hgms_direct_fsdp_progress", **record}))

        del branch_initial_states, branch_train_items, val_batch, grad_theta
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
            "method": "hg_ms_direct",
            "method_note": (
                "Direct HG-MS FSDP source reweighting with LoRA trainable parameters and exact "
                "one-step selected-branch source-logit hypergradients."
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
        print(json.dumps({"stage": "hgms_direct_fsdp_done", **metrics}, indent=2))

        if args.min_lambda_movement > 0 and final_lambda_l1_from_uniform < args.min_lambda_movement:
            raise RuntimeError(
                f"Lambda movement failed: L1={final_lambda_l1_from_uniform:.3e} "
                f"< threshold={args.min_lambda_movement:.3e}."
            )
        if args.min_centered_grad_norm > 0 and max_centered_grad_norm < args.min_centered_grad_norm:
            raise RuntimeError(
                f"Gradient failed: max centered grad={max_centered_grad_norm:.3e} "
                f"< threshold={args.min_centered_grad_norm:.3e}."
            )

    dist_barrier(local_rank)
    del optimizers, models
    gc.collect()
    torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
