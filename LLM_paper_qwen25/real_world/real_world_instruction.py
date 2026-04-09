from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from exact_math_reasoning import (
    autocast_context,
    batch_mean_loss,
    branch_selection_loss_inplace_from_batches,
    clipped_theta_grad,
    clone_trainable_state,
    ensure_theta_grad_connected,
    fine_tune_model,
    get_trainable_state,
    load_model_and_tokenizer,
    load_model_and_tokenizer_with_name,
    maybe_make_output_dir,
    maybe_tune_loss_inplace,
    model_batch_hidden_mean,
    model_batch_mean_loss,
    prepare_val_batches,
    safe_grad_tensor,
    safe_grads,
    seed_everything,
    sgd_step,
    source_counts_from_lambda,
    theta_grad_from_batches,
    theta_diagnostics,
    weighted_source_loss,
)
from exact_math_reasoning_data import collate_batch, iter_batches, sample_balanced_batch, sample_batch
from real_world_instruction_data import (
    INSTRUCTION_SOURCE_SPECS,
    iter_source_records,
    load_or_prepare_instruction_splits,
    sample_final_train_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["uniform", "scalebio", "hg_single", "hg_random", "hg_ms", "less", "rho_loss"],
        required=True,
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2-7B")
    parser.add_argument("--reference-model-name", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2424)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--reweight-train-per-source", type=int, default=3000)
    parser.add_argument("--reweight-train-raw-oversample", type=float, default=1.5)
    parser.add_argument("--reweight-epochs", type=int, default=3)
    parser.add_argument("--reweight-steps", type=int, default=0)
    parser.add_argument("--reweight-wall-clock-seconds", type=float, default=0.0)
    parser.add_argument("--per-source-batch", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--selection-val-batch-size", type=int, default=16)
    parser.add_argument("--selection-val-batches", type=int, default=1)
    parser.add_argument("--hypergrad-val-batches", type=int, default=1)
    parser.add_argument(
        "--reuse-selection-batches-for-hypergrad",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--selection-train-tolerance", type=float, default=0.05)
    parser.add_argument("--selection-train-penalty", type=float, default=0.0)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--m-branches", type=int, default=2)
    parser.add_argument("--branch-noise-std", type=float, default=5e-4)
    parser.add_argument("--reweight-model-lr", type=float, default=1e-5)
    parser.add_argument("--reweight-outer-lr", type=float, default=1e-2)
    parser.add_argument("--theta-optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--theta-adam-beta1", type=float, default=0.9)
    parser.add_argument("--theta-adam-beta2", type=float, default=0.999)
    parser.add_argument("--theta-adam-eps", type=float, default=1e-8)
    parser.add_argument("--reweight-weight-decay", type=float, default=0.0)
    parser.add_argument("--theta-grad-clip", type=float, default=0.0)
    parser.add_argument("--scalebio-alpha", type=float, default=100.0)
    parser.add_argument("--tune-every", type=int, default=100)
    parser.add_argument("--final-train-size", type=int, default=10000)
    parser.add_argument("--final-ft-lr", type=float, default=8e-6)
    parser.add_argument("--final-ft-epochs", type=int, default=1)
    parser.add_argument("--final-ft-micro-batch", type=int, default=1)
    parser.add_argument("--final-ft-grad-accum", type=int, default=64)
    parser.add_argument("--final-train-raw-oversample", type=float, default=1.1)
    parser.add_argument("--reference-ft-steps", type=int, default=200)
    parser.add_argument("--reference-batch-size", type=int, default=8)
    parser.add_argument("--reference-ft-lr", type=float, default=2e-4)
    parser.add_argument("--baseline-score-samples-per-source", type=int, default=256)
    parser.add_argument("--stop-after-reweight", action="store_true")
    parser.add_argument(
        "--use-best-reweight-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--allow-train-shortfall", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def encode_dialog_record(tokenizer, record: Dict[str, object], max_length: int) -> Dict[str, object] | None:
    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if eos_token_id is None:
        return None
    input_ids: List[int] = []
    labels: List[int] = []
    for item in record["items"]:
        role = item["role"]
        prefix = "User: " if role == "user" else "Assistant: "
        text = prefix + item["content"].strip() + "\n\n"
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if role == "assistant":
            token_labels = list(token_ids)
        else:
            token_labels = [-100] * len(token_ids)
        input_ids.extend(token_ids)
        labels.extend(token_labels)
    if len(input_ids) + 1 > max_length:
        return None
    input_ids.append(eos_token_id)
    labels.append(eos_token_id)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "source_id": int(record.get("source_id", -1)),
        "category": record.get("category"),
    }


def tokenize_dialog_records(tokenizer, records: Sequence[Dict[str, object]], max_length: int) -> List[Dict[str, object]]:
    tokenized: List[Dict[str, object]] = []
    for record in records:
        encoded = encode_dialog_record(tokenizer, record, max_length=max_length)
        if encoded is not None:
            tokenized.append(encoded)
    return tokenized


def tokenized_by_source_or_fail(
    *,
    tokenizer,
    raw_by_source: Sequence[Sequence[Dict[str, object]]],
    max_length: int,
    needed_per_source: int,
    seed: int,
) -> tuple[List[List[Dict[str, object]]], int]:
    tokenized_by_source: List[List[Dict[str, object]]] = []
    available_counts: List[int] = []
    allow_shortfall = getattr(tokenizer, "_allow_train_shortfall", False)
    for source_index, source_records in enumerate(raw_by_source):
        tokenized = tokenize_dialog_records(tokenizer, source_records, max_length=max_length)
        available_counts.append(len(tokenized))
        if allow_shortfall:
            tokenized_by_source.append(tokenized)
            continue
        if len(tokenized) < needed_per_source:
            seen_signatures = {
                json.dumps(record["items"], sort_keys=True, ensure_ascii=False)
                for record in source_records
            }
            max_extra_seen = max(needed_per_source * 200, needed_per_source)
            stream = iter_source_records(source_id=source_index)
            skip = (seed + 7919 * (source_index + 1)) % max(1, needed_per_source * 5)
            extra_seen = 0
            for record in stream:
                signature = json.dumps(record["items"], sort_keys=True, ensure_ascii=False)
                if signature in seen_signatures:
                    continue
                if skip > 0:
                    skip -= 1
                    continue
                encoded = encode_dialog_record(tokenizer, record, max_length=max_length)
                extra_seen += 1
                if encoded is not None:
                    tokenized.append(encoded)
                    seen_signatures.add(signature)
                    if len(tokenized) >= needed_per_source:
                        break
                if extra_seen >= max_extra_seen:
                    break
        if len(tokenized) == 0:
            raise RuntimeError(
                f"No tokenized examples available for {INSTRUCTION_SOURCE_SPECS[source_index]['name']} after streaming top-up."
            )
        kept = min(len(tokenized), needed_per_source)
        if kept < needed_per_source:
            print(
                json.dumps(
                    {
                        "stage": "tokenized_shortfall",
                        "source": INSTRUCTION_SOURCE_SPECS[source_index]["name"],
                        "available": len(tokenized),
                        "requested": needed_per_source,
                    }
                )
            )
        tokenized_by_source.append(tokenized[:kept])
    if not allow_shortfall:
        return tokenized_by_source, needed_per_source
    min_available = min(available_counts)
    effective_per_source = min(needed_per_source, min_available)
    if effective_per_source < needed_per_source:
        print(
            json.dumps(
                {
                    "stage": "tokenized_train_cap",
                    "requested_per_source": needed_per_source,
                    "effective_per_source": effective_per_source,
                    "available_per_source": {
                        INSTRUCTION_SOURCE_SPECS[source_index]["name"]: count
                        for source_index, count in enumerate(available_counts)
                    },
                }
            )
        )
    return [tokenized[:effective_per_source] for tokenized in tokenized_by_source], effective_per_source


def pick_branch_index(
    method: str,
    branch_outcomes: Sequence[Dict[str, object]],
    rng: random.Random,
    *,
    train_tolerance: float,
    train_penalty: float,
) -> int:
    if method == "hg_single":
        return 0
    if method == "hg_random":
        return rng.randrange(len(branch_outcomes))
    if method == "hg_ms":
        min_train_loss = min(outcome["train_loss_scalar"] for outcome in branch_outcomes)
        eligible = [
            idx
            for idx, outcome in enumerate(branch_outcomes)
            if outcome["train_loss_scalar"] <= min_train_loss + train_tolerance
        ]
        if not eligible:
            eligible = list(range(len(branch_outcomes)))
        return min(
            eligible,
            key=lambda idx: branch_outcomes[idx]["selection_val_scalar"]
            + train_penalty * (branch_outcomes[idx]["train_loss_scalar"] - min_train_loss),
        )
    raise ValueError(f"Unsupported method {method}")


def apply_theta_update(
    *,
    theta: torch.Tensor,
    grad_theta: torch.Tensor | None,
    args: argparse.Namespace,
    optimizer_state: Dict[str, torch.Tensor | int] | None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor | int] | None]:
    grad = clipped_theta_grad(theta, grad_theta, args.theta_grad_clip).detach()
    if args.theta_optimizer == "adam":
        if optimizer_state is None:
            optimizer_state = {
                "m": torch.zeros_like(theta),
                "v": torch.zeros_like(theta),
                "step": 0,
            }
        optimizer_state["step"] = int(optimizer_state["step"]) + 1
        beta1 = args.theta_adam_beta1
        beta2 = args.theta_adam_beta2
        optimizer_state["m"] = beta1 * optimizer_state["m"] + (1.0 - beta1) * grad
        optimizer_state["v"] = beta2 * optimizer_state["v"] + (1.0 - beta2) * grad.square()
        step = int(optimizer_state["step"])
        m_hat = optimizer_state["m"] / (1.0 - beta1**step)
        v_hat = optimizer_state["v"] / (1.0 - beta2**step)
        update = m_hat / (v_hat.sqrt() + args.theta_adam_eps)
    else:
        update = grad
    with torch.no_grad():
        theta = theta - args.reweight_outer_lr * update
    return theta.detach().requires_grad_(True), optimizer_state


def run_branch(
    *,
    model,
    initial_state: Dict[str, torch.Tensor],
    theta: torch.Tensor,
    tokenized_train_by_source: Sequence[Sequence[Dict[str, object]]],
    selection_batches: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    branch_index: int,
    seed: int,
    device: torch.device,
    pad_token_id: int,
    create_graph: bool,
    compute_selection_loss: bool = True,
) -> Dict[str, object]:
    rng = random.Random(seed + 100_000 * (branch_index + 1))
    state = clone_trainable_state(initial_state, noise_std=args.branch_noise_std if branch_index > 0 else 0.0)
    train_loss_scalar = math.nan
    for inner_step in range(args.inner_steps):
        torch.manual_seed(seed + branch_index * 1000 + inner_step)
        train_batch = sample_balanced_batch(
            tokenized_train_by_source,
            per_source_batch=args.per_source_batch,
            rng=rng,
            device=device,
            pad_token_id=pad_token_id,
        )
        train_loss, _ = weighted_source_loss(model, state, theta, train_batch, use_bf16=args.bf16)
        grads = torch.autograd.grad(
            train_loss,
            tuple(state.values()),
            create_graph=create_graph,
            allow_unused=True,
        )
        state = sgd_step(
            state,
            safe_grads(state, grads),
            lr=args.reweight_model_lr,
            weight_decay=args.reweight_weight_decay,
            create_graph=create_graph,
        )
        train_loss_scalar = float(train_loss.detach().cpu())
    selection_val = math.nan
    if compute_selection_loss:
        selection_val = branch_selection_loss_inplace_from_batches(
            model,
            state,
            selection_batches,
            use_bf16=args.bf16,
        )
    return {
        "branch_index": branch_index,
        "train_loss_scalar": train_loss_scalar,
        "selection_val_scalar": float(selection_val),
        "params": state,
    }


def compute_reference_baseline_lambda(
    *,
    args: argparse.Namespace,
    method: str,
    splits: Dict[str, object],
    device: torch.device,
) -> tuple[List[float], Dict[str, object]]:
    reference_model, reference_tokenizer = load_model_and_tokenizer_with_name(args, args.reference_model_name, device)
    setattr(reference_tokenizer, "_allow_train_shortfall", args.allow_train_shortfall)
    tokenized_train_by_source, effective_train_per_source = tokenized_by_source_or_fail(
        tokenizer=reference_tokenizer,
        raw_by_source=splits["reweight_train"],
        max_length=args.max_length,
        needed_per_source=args.reweight_train_per_source,
        seed=args.seed,
    )
    tokenized_val_opt = tokenize_dialog_records(reference_tokenizer, splits["val_opt"], max_length=args.max_length)
    if len(tokenized_val_opt) == 0:
        raise RuntimeError("Validation set became empty after tokenization.")

    score_count = min(args.baseline_score_samples_per_source, effective_train_per_source)
    base_losses: List[float] = []
    reference_model.eval()
    with torch.no_grad():
        for source_examples in tokenized_train_by_source:
            losses = []
            for batch in iter_batches(
                source_examples[:score_count],
                batch_size=args.reference_batch_size,
                device=device,
                pad_token_id=reference_tokenizer.pad_token_id,
            ):
                losses.append(float(model_batch_mean_loss(reference_model, batch, use_bf16=args.bf16).detach().cpu()))
            base_losses.append(sum(losses) / max(1, len(losses)))

    optimizer = torch.optim.AdamW(
        (parameter for parameter in reference_model.parameters() if parameter.requires_grad),
        lr=args.reference_ft_lr,
    )
    reference_model.train()
    tune_rng = random.Random(args.seed + 515151)
    val_pool = list(tokenized_val_opt)
    for _ in range(args.reference_ft_steps):
        sample = tune_rng.sample(val_pool, k=min(args.reference_batch_size, len(val_pool)))
        batch = collate_batch(sample, device=device, pad_token_id=reference_tokenizer.pad_token_id)
        with autocast_context(device, args.bf16):
            logits = reference_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).logits
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                batch["labels"][:, 1:].contiguous().view(-1),
                reduction="none",
                ignore_index=-100,
            )
            loss = loss.view(batch["labels"].size(0), -1)
            mask = (batch["labels"][:, 1:] != -100).float()
            loss = ((loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    reference_model.eval()
    reference_vector = None
    if method == "less":
        val_vectors = []
        with torch.no_grad():
            for batch in iter_batches(
                val_pool[:score_count],
                batch_size=args.reference_batch_size,
                device=device,
                pad_token_id=reference_tokenizer.pad_token_id,
            ):
                val_vectors.append(model_batch_hidden_mean(reference_model, batch, use_bf16=args.bf16).mean(dim=0))
        reference_vector = torch.stack(val_vectors, dim=0).mean(dim=0)

    scores: List[float] = []
    tuned_losses: List[float] = []
    with torch.no_grad():
        for source_examples, base_loss in zip(tokenized_train_by_source, base_losses):
            subset = source_examples[:score_count]
            losses = []
            vectors = []
            for batch in iter_batches(
                subset,
                batch_size=args.reference_batch_size,
                device=device,
                pad_token_id=reference_tokenizer.pad_token_id,
            ):
                losses.append(float(model_batch_mean_loss(reference_model, batch, use_bf16=args.bf16).detach().cpu()))
                if method == "less":
                    vectors.append(model_batch_hidden_mean(reference_model, batch, use_bf16=args.bf16).mean(dim=0))
            tuned_loss = sum(losses) / max(1, len(losses))
            tuned_losses.append(tuned_loss)
            if method == "rho_loss":
                score = max(0.0, base_loss - tuned_loss)
            else:
                assert reference_vector is not None
                source_vector = torch.stack(vectors, dim=0).mean(dim=0)
                cosine = F.cosine_similarity(source_vector.unsqueeze(0), reference_vector.unsqueeze(0)).item()
                score = max(0.0, cosine)
            scores.append(score)

    if sum(scores) <= 0:
        lambdas = [1.0 / len(scores)] * len(scores)
    else:
        total = sum(scores)
        lambdas = [score / total for score in scores]

    details = {
        "reference_model_name": args.reference_model_name,
        "score_samples_per_source": score_count,
        "base_losses": base_losses,
        "tuned_losses": tuned_losses,
        "scores": scores,
    }
    del reference_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return lambdas, details


def evaluate_token_loss(
    *,
    args: argparse.Namespace,
    adapter_state: Dict[str, torch.Tensor],
    tokenized_examples: Sequence[Dict[str, object]],
    device: torch.device,
) -> float:
    model, tokenizer = load_model_and_tokenizer(args, device)
    with torch.no_grad():
        named_parameters = dict(model.named_parameters())
        for name, tensor in adapter_state.items():
            named_parameters[name].copy_(tensor.to(named_parameters[name].device, dtype=named_parameters[name].dtype))
    loss = maybe_tune_loss_inplace(
        model,
        tokenized_examples,
        batch_size=args.eval_batch_size,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
        use_bf16=args.bf16,
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss


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
    rng = random.Random(seed + 909_091)
    topped_up = list(examples)
    topped_up.extend(copy.deepcopy(rng.choice(examples)) for _ in range(target_size - len(examples)))
    rng.shuffle(topped_up)
    return topped_up


def main() -> None:
    args = parse_args()
    overall_start_time = time.perf_counter()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    output_dir = maybe_make_output_dir(args.output_dir)
    with open(output_dir / "config.json", "w") as file:
        json.dump(vars(args), file, indent=2)

    cache_path = args.cache_path or str(
        output_dir / f"instruction_splits_seed{args.seed}_train{args.reweight_train_per_source}.json"
    )
    splits = load_or_prepare_instruction_splits(
        cache_path=cache_path,
        reweight_train_per_source=args.reweight_train_per_source,
        reweight_train_raw_oversample=args.reweight_train_raw_oversample,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "stage": "splits_ready",
                "cache_path": cache_path,
                "source_count": len(splits["source_names"]),
                "val_opt_size": len(splits["val_opt"]),
                "val_tune_size": len(splits["val_tune"]),
                "test_size": len(splits["test"]),
            },
            indent=2,
        )
    )

    source_names = splits["source_names"]
    baseline_details = None
    history: List[Dict[str, object]] = []
    best_tune_loss = math.nan
    reweight_stage_start_time = time.perf_counter()
    reweight_budget_seconds = float(max(0.0, args.reweight_wall_clock_seconds))
    reweight_deadline = (
        reweight_stage_start_time + reweight_budget_seconds
        if reweight_budget_seconds > 0
        else None
    )
    reweight_budget_hit = False

    if args.method == "uniform":
        learned_lambda = [1.0 / len(source_names)] * len(source_names)
        best_lambda_reweight = list(learned_lambda)
        final_lambda_reweight = list(learned_lambda)
    elif args.method in {"less", "rho_loss"}:
        learned_lambda, baseline_details = compute_reference_baseline_lambda(
            args=args,
            method=args.method,
            splits=splits,
            device=device,
        )
        best_lambda_reweight = list(learned_lambda)
        final_lambda_reweight = list(learned_lambda)
        print(json.dumps({"stage": "baseline_done", "learned_lambda": learned_lambda, "details": baseline_details}, indent=2))
    else:
        model, tokenizer = load_model_and_tokenizer(args, device)
        setattr(tokenizer, "_allow_train_shortfall", args.allow_train_shortfall)
        tokenized_train_by_source, effective_train_per_source = tokenized_by_source_or_fail(
            tokenizer=tokenizer,
            raw_by_source=splits["reweight_train"],
            max_length=args.max_length,
            needed_per_source=args.reweight_train_per_source,
            seed=args.seed,
        )
        tokenized_val_opt = tokenize_dialog_records(tokenizer, splits["val_opt"], max_length=args.max_length)
        tokenized_val_tune = tokenize_dialog_records(tokenizer, splits["val_tune"], max_length=args.max_length)
        if len(tokenized_val_opt) == 0 or len(tokenized_val_tune) == 0:
            raise RuntimeError("Validation sets became empty after tokenization.")

        trainable_state = get_trainable_state(model)
        auxiliary_state = clone_trainable_state(trainable_state, noise_std=0.0)
        theta = torch.zeros(len(source_names), dtype=torch.float32, device=device, requires_grad=True)
        theta_optimizer_state: Dict[str, torch.Tensor | int] | None = None
        total_steps = (
            args.reweight_steps
            if args.reweight_steps > 0
            else args.reweight_epochs * math.ceil(effective_train_per_source / args.per_source_batch)
        )
        best_tune_loss = float("inf")
        best_theta = theta.detach().clone()

        for outer_step in range(total_steps):
            if reweight_deadline is not None and outer_step > 0 and time.perf_counter() >= reweight_deadline:
                reweight_budget_hit = True
                print(
                    json.dumps(
                        {
                            "stage": "reweight_budget_hit",
                            "outer_step": outer_step,
                            "budget_seconds": reweight_budget_seconds,
                            "elapsed_seconds": time.perf_counter() - reweight_stage_start_time,
                        }
                    )
                )
                break
            step_seed = args.seed + 7_919 * (outer_step + 1)
            if args.method == "scalebio":
                rng = random.Random(step_seed)
                train_batch = sample_balanced_batch(
                    tokenized_train_by_source,
                    per_source_batch=args.per_source_batch,
                    rng=rng,
                    device=device,
                    pad_token_id=tokenizer.pad_token_id,
                )
                val_batch = sample_batch(
                    tokenized_val_opt,
                    batch_size=min(args.val_batch_size, len(tokenized_val_opt)),
                    rng=rng,
                    device=device,
                    pad_token_id=tokenizer.pad_token_id,
                )
                train_loss_w, _ = weighted_source_loss(model, trainable_state, theta, train_batch, use_bf16=args.bf16)
                train_loss_u, _ = weighted_source_loss(model, auxiliary_state, theta, train_batch, use_bf16=args.bf16)
                val_loss = batch_mean_loss(model, trainable_state, val_batch, use_bf16=args.bf16)
                alpha = args.scalebio_alpha
                grad_w = torch.autograd.grad(
                    val_loss + alpha * train_loss_w,
                    tuple(trainable_state.values()),
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_theta = torch.autograd.grad(
                    alpha * (train_loss_w - train_loss_u),
                    theta,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                ensure_theta_grad_connected(args.method, grad_theta)
                grad_u = torch.autograd.grad(
                    alpha * train_loss_u,
                    tuple(auxiliary_state.values()),
                    create_graph=False,
                    allow_unused=True,
                )
                trainable_state = sgd_step(
                    trainable_state,
                    safe_grads(trainable_state, grad_w),
                    lr=args.reweight_model_lr,
                    weight_decay=args.reweight_weight_decay,
                    create_graph=False,
                )
                auxiliary_state = sgd_step(
                    auxiliary_state,
                    safe_grads(auxiliary_state, grad_u),
                    lr=args.reweight_model_lr,
                    weight_decay=args.reweight_weight_decay,
                    create_graph=False,
                )
                grad_stats = theta_diagnostics(theta, grad_theta)
                theta, theta_optimizer_state = apply_theta_update(
                    theta=theta,
                    grad_theta=grad_theta,
                    args=args,
                    optimizer_state=theta_optimizer_state,
                )
                selected_summary = {
                    "branch_index": 0,
                    "train_loss": float(train_loss_w.detach().cpu()),
                    "val_loss": float(val_loss.detach().cpu()),
                }
            else:
                branch_count = 1 if args.method == "hg_single" else args.m_branches
                selection_batches = prepare_val_batches(
                    tokenized_val_opt,
                    batch_size=args.selection_val_batch_size,
                    batch_count=args.selection_val_batches,
                    rng=random.Random(step_seed + 17_003),
                    device=device,
                    pad_token_id=tokenizer.pad_token_id,
                )
                if args.reuse_selection_batches_for_hypergrad:
                    hypergrad_batches = selection_batches
                else:
                    hypergrad_batches = prepare_val_batches(
                        tokenized_val_opt,
                        batch_size=args.selection_val_batch_size,
                        batch_count=args.hypergrad_val_batches,
                        rng=random.Random(step_seed + 909_091),
                        device=device,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                branch_outcomes = []
                for branch_index in range(branch_count):
                    branch_outcomes.append(
                        run_branch(
                            model=model,
                            initial_state=trainable_state,
                            theta=theta,
                            tokenized_train_by_source=tokenized_train_by_source,
                            selection_batches=selection_batches,
                            args=args,
                            branch_index=branch_index,
                            seed=step_seed,
                            device=device,
                            pad_token_id=tokenizer.pad_token_id,
                            create_graph=False,
                        )
                    )
                picked_index = pick_branch_index(
                    args.method,
                    branch_outcomes,
                    random.Random(step_seed),
                    train_tolerance=args.selection_train_tolerance,
                    train_penalty=args.selection_train_penalty,
                )
                branch_summaries = [
                    {
                        "branch_index": int(outcome["branch_index"]),
                        "train_loss": float(outcome["train_loss_scalar"]),
                        "selection_val_loss": float(outcome["selection_val_scalar"]),
                    }
                    for outcome in branch_outcomes
                ]
                selected = branch_outcomes[picked_index]
                selected_branch_index = int(selected["branch_index"])
                selected_train_loss = float(selected["train_loss_scalar"])
                selected_selection_val = float(selected["selection_val_scalar"])
                del branch_outcomes
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                selected = run_branch(
                    model=model,
                    initial_state=trainable_state,
                    theta=theta,
                    tokenized_train_by_source=tokenized_train_by_source,
                    selection_batches=selection_batches,
                    args=args,
                    branch_index=selected_branch_index,
                    seed=step_seed,
                    device=device,
                    pad_token_id=tokenizer.pad_token_id,
                    create_graph=True,
                    compute_selection_loss=False,
                )
                selected_val_scalar, grad_theta = theta_grad_from_batches(
                    model,
                    selected["params"],
                    theta,
                    hypergrad_batches,
                    use_bf16=args.bf16,
                )
                ensure_theta_grad_connected(args.method, grad_theta)
                grad_stats = theta_diagnostics(theta, grad_theta)
                theta, theta_optimizer_state = apply_theta_update(
                    theta=theta,
                    grad_theta=grad_theta,
                    args=args,
                    optimizer_state=theta_optimizer_state,
                )
                trainable_state = {
                    name: tensor.detach().clone().requires_grad_(True)
                    for name, tensor in selected["params"].items()
                }
                selected_summary = {
                    "branch_index": selected_branch_index,
                    "train_loss": selected_train_loss,
                    "selection_val_loss": selected_selection_val,
                    "hypergrad_val_loss": float(selected_val_scalar),
                }

            if outer_step == 0 or (outer_step + 1) % args.tune_every == 0 or outer_step + 1 == total_steps:
                named_parameters = dict(model.named_parameters())
                with torch.no_grad():
                    for name, tensor in trainable_state.items():
                        named_parameters[name].copy_(tensor.to(named_parameters[name].device, dtype=named_parameters[name].dtype))
                tune_loss = maybe_tune_loss_inplace(
                    model,
                    tokenized_val_tune,
                    batch_size=args.val_batch_size,
                    device=device,
                    pad_token_id=tokenizer.pad_token_id,
                    use_bf16=args.bf16,
                )
                if tune_loss < best_tune_loss:
                    best_tune_loss = tune_loss
                    best_theta = theta.detach().clone()
            else:
                tune_loss = math.nan

            record = {
                "outer_step": outer_step,
                "selected": selected_summary,
                "lambda": torch.softmax(theta.detach(), dim=0).cpu().tolist(),
                "tune_loss": tune_loss,
                "best_tune_loss": best_tune_loss,
                "elapsed_reweight_sec": time.perf_counter() - reweight_stage_start_time,
                **grad_stats,
            }
            if args.method in {"hg_single", "hg_random", "hg_ms"}:
                record["candidates"] = branch_summaries
            history.append(record)
            if outer_step == 0 or (outer_step + 1) % 200 == 0 or outer_step + 1 == total_steps:
                selected_val_loss = selected_summary["val_loss"] if args.method == "scalebio" else selected_summary["hypergrad_val_loss"]
                print(
                    json.dumps(
                        {
                            "stage": "reweight_progress",
                            "outer_step": outer_step + 1,
                            "selected_branch": selected_summary["branch_index"],
                            "selected_val_loss": selected_val_loss,
                            "best_tune_loss": best_tune_loss,
                            "lambda": record["lambda"],
                            "theta_grad_norm": grad_stats["theta_grad_norm"],
                            "theta_grad_centered_norm": grad_stats["theta_grad_centered_norm"],
                        }
                    )
                )

        best_lambda_reweight = torch.softmax(best_theta, dim=0).cpu().tolist()
        final_lambda_reweight = torch.softmax(theta.detach(), dim=0).cpu().tolist()
        with open(output_dir / "reweight_history.jsonl", "w") as file:
            for row in history:
                file.write(json.dumps(row) + "\n")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reweight_elapsed_sec = time.perf_counter() - reweight_stage_start_time

    if args.stop_after_reweight:
        metrics = {
            "method": args.method,
            "model_name": args.model_name,
            "seed": args.seed,
            "learned_lambda": final_lambda_reweight,
            "best_lambda_reweight": best_lambda_reweight,
            "source_names": source_names,
            "best_tune_loss_reweight": best_tune_loss,
            "baseline_details": baseline_details,
            "stopped_after_reweight": True,
            "reweight_elapsed_sec": reweight_elapsed_sec,
            "reweight_budget_seconds": reweight_budget_seconds,
            "reweight_budget_hit": reweight_budget_hit,
            "reweight_steps_completed": len(history),
            "total_elapsed_sec": time.perf_counter() - overall_start_time,
        }
        with open(output_dir / "metrics.json", "w") as file:
            json.dump(metrics, file, indent=2)
        print(json.dumps(metrics, indent=2))
        return

    final_train_lambda = best_lambda_reweight if args.use_best_reweight_checkpoint else final_lambda_reweight
    source_counts = source_counts_from_lambda(final_train_lambda, args.final_train_size)
    print(json.dumps({"stage": "sample_final_train", "source_counts": source_counts}, indent=2))
    model, tokenizer = load_model_and_tokenizer(args, device)
    final_train_tokenized: List[Dict[str, object]] = []
    for attempt, raw_oversample in enumerate(
        [
            args.final_train_raw_oversample,
            max(args.final_train_raw_oversample, 1.25),
            max(args.final_train_raw_oversample, 1.5),
        ]
    ):
        final_records = sample_final_train_records(
            source_counts=source_counts,
            raw_by_source=splits["reweight_train"],
            raw_oversample=raw_oversample,
            seed=args.seed + 1_000 * attempt,
        )
        final_train_tokenized = tokenize_dialog_records(tokenizer, final_records, max_length=args.max_length)
        if len(final_train_tokenized) >= args.final_train_size:
            break
        print(
            json.dumps(
                {
                    "stage": "final_train_topup_retry",
                    "attempt": attempt + 1,
                    "raw_oversample": raw_oversample,
                    "available_after_tokenization": len(final_train_tokenized),
                    "required": args.final_train_size,
                }
            )
        )
    val_tune_tokenized = tokenize_dialog_records(tokenizer, splits["val_tune"], max_length=args.max_length)
    test_tokenized = tokenize_dialog_records(tokenizer, splits["test"], max_length=args.max_length)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if len(final_train_tokenized) < args.final_train_size:
        if not args.allow_train_shortfall:
            raise RuntimeError(
                f"Only {len(final_train_tokenized)} final train examples remained after tokenization; "
                f"need {args.final_train_size}."
            )
        print(
            json.dumps(
                {
                    "stage": "final_train_with_replacement_topup",
                    "kept_examples": len(final_train_tokenized),
                    "target_examples": args.final_train_size,
                }
            )
        )
        final_train_tokenized = top_up_tokenized_examples_with_replacement(
            examples=final_train_tokenized,
            target_size=args.final_train_size,
            seed=args.seed,
        )
    else:
        final_train_tokenized = final_train_tokenized[: args.final_train_size]

    final_adapter_state, final_tune_loss = fine_tune_model(
        args=args,
        train_examples_tokenized=final_train_tokenized,
        tune_examples_tokenized=val_tune_tokenized,
        output_dir=output_dir,
        device=device,
    )
    test_loss = evaluate_token_loss(
        args=args,
        adapter_state=final_adapter_state,
        tokenized_examples=test_tokenized,
        device=device,
    )

    metrics = {
        "method": args.method,
        "model_name": args.model_name,
        "seed": args.seed,
        "learned_lambda": final_lambda_reweight,
        "best_lambda_reweight": best_lambda_reweight,
        "final_train_lambda": final_train_lambda,
        "final_train_lambda_source": "best_reweight" if args.use_best_reweight_checkpoint else "final_reweight",
        "source_names": source_names,
        "source_counts": source_counts,
        "best_tune_loss_reweight": best_tune_loss,
        "baseline_details": baseline_details,
        "reweight_elapsed_sec": reweight_elapsed_sec,
        "reweight_budget_seconds": reweight_budget_seconds,
        "reweight_budget_hit": reweight_budget_hit,
        "reweight_steps_completed": len(history),
        "final_ft_tune_loss": final_tune_loss,
        "test_loss": test_loss,
        "val_opt_size": len(splits["val_opt"]),
        "val_tune_size": len(splits["val_tune"]),
        "test_size": len(splits["test"]),
        "total_elapsed_sec": time.perf_counter() - overall_start_time,
    }
    with open(output_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
