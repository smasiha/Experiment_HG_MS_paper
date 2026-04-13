from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

from exact_math_reasoning_data import (
    TRAIN_SOURCE_SPECS,
    collate_batch,
    extract_gsm8k_answer,
    iter_batches,
    load_eval_examples,
    load_or_prepare_exact_splits,
    normalize_math_answer,
    sample_balanced_batch,
    sample_batch,
    sample_final_train_examples,
    tokenize_examples,
)


@dataclass
class BranchOutcome:
    branch_index: int
    train_loss_scalar: float
    selection_val_scalar: float
    params: Dict[str, torch.Tensor]


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
    parser.add_argument("--val-per-source", type=int, default=280)
    parser.add_argument("--train-raw-oversample", type=float, default=4.0)
    parser.add_argument("--val-raw-oversample", type=float, default=1.2)
    parser.add_argument("--val-kinds", nargs="*", default=[])
    parser.add_argument("--reweight-epochs", type=int, default=3)
    parser.add_argument("--reweight-steps", type=int, default=0)
    parser.add_argument("--reweight-wall-clock-seconds", type=float, default=0.0)
    parser.add_argument("--per-source-batch", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--selection-val-batch-size", type=int, default=64)
    parser.add_argument("--selection-val-batches", type=int, default=1)
    parser.add_argument("--hypergrad-val-batches", type=int, default=1)
    parser.add_argument(
        "--reuse-selection-batches-for-hypergrad",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--selection-train-tolerance", type=float, default=0.05)
    parser.add_argument("--selection-train-penalty", type=float, default=0.0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--selection-inner-steps", type=int, default=0)
    parser.add_argument("--hypergrad-inner-steps", type=int, default=0)
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
    parser.add_argument("--scalebio-alpha", type=float, default=10.0)
    parser.add_argument("--tune-every", type=int, default=50)
    parser.add_argument("--final-train-size", type=int, default=20000)
    parser.add_argument("--final-ft-lr", type=float, default=8e-6)
    parser.add_argument("--final-ft-epochs", type=int, default=1)
    parser.add_argument("--final-ft-micro-batch", type=int, default=4)
    parser.add_argument("--final-ft-grad-accum", type=int, default=16)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--gsm8k-eval-size", type=int, default=0)
    parser.add_argument("--math-eval-size", type=int, default=0)
    parser.add_argument("--eval-math", action="store_true")
    parser.add_argument("--stop-after-reweight", action="store_true")
    parser.add_argument(
        "--use-best-reweight-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--reference-ft-steps", type=int, default=200)
    parser.add_argument("--reference-ft-epochs", type=int, default=1)
    parser.add_argument("--reference-batch-size", type=int, default=8)
    parser.add_argument("--reference-grad-accum", type=int, default=1)
    parser.add_argument("--reference-ft-lr", type=float, default=2e-4)
    parser.add_argument("--baseline-score-samples-per-source", type=int, default=256)
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--allow-train-shortfall", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def maybe_make_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_model_and_tokenizer_with_name(args: argparse.Namespace, model_name: str, device: torch.device):
    local_files_only = os.environ.get("LOCAL_MODEL_ONLY", "0") == "1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
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
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=local_files_only,
        )
    except OSError:
        if local_files_only:
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=False,
            force_download=True,
        )
    model.config.use_cache = False
    use_gradient_checkpointing = bool(getattr(args, "gradient_checkpointing", False))
    if use_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "base_model") and hasattr(model.base_model, "gradient_checkpointing_disable"):
            model.base_model.gradient_checkpointing_disable()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()
    return model, tokenizer


def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
    return load_model_and_tokenizer_with_name(args, args.model_name, device)


def get_trainable_state(model) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().clone().requires_grad_(True)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def assign_trainable_state(model, state: Dict[str, torch.Tensor]) -> None:
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            named_params[name].copy_(tensor.to(named_params[name].device, dtype=named_params[name].dtype))


def clone_trainable_state(state: Dict[str, torch.Tensor], noise_std: float = 0.0) -> Dict[str, torch.Tensor]:
    cloned = {}
    for name, tensor in state.items():
        out = tensor.detach().clone()
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std
        cloned[name] = out.requires_grad_(True)
    return cloned


def functional_logits(model, params: Dict[str, torch.Tensor], batch: Dict[str, object], use_bf16: bool):
    inputs = {key: batch[key] for key in ["input_ids", "attention_mask", "labels"]}
    with autocast_context(inputs["input_ids"].device, use_bf16):
        outputs = functional_call(model, params, (), inputs, strict=False)
    return outputs.logits


def model_logits(model, batch: Dict[str, object], use_bf16: bool):
    inputs = {key: batch[key] for key in ["input_ids", "attention_mask", "labels"]}
    with autocast_context(inputs["input_ids"].device, use_bf16):
        outputs = model(**inputs)
    return outputs.logits


def per_example_losses_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())
    mask = (shift_labels != -100).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (losses * mask).sum(dim=1) / denom


def batch_mean_loss(model, params: Dict[str, torch.Tensor], batch: Dict[str, object], use_bf16: bool) -> torch.Tensor:
    logits = functional_logits(model, params, batch, use_bf16=use_bf16)
    return per_example_losses_from_logits(logits, batch["labels"]).mean()


def model_batch_mean_loss(model, batch: Dict[str, object], use_bf16: bool) -> torch.Tensor:
    logits = model_logits(model, batch, use_bf16=use_bf16)
    return per_example_losses_from_logits(logits, batch["labels"]).mean()


def model_batch_hidden_mean(model, batch: Dict[str, object], use_bf16: bool) -> torch.Tensor:
    inputs = {key: batch[key] for key in ["input_ids", "attention_mask"]}
    with autocast_context(inputs["input_ids"].device, use_bf16):
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    mask = batch["attention_mask"].unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden.float() * mask).sum(dim=1) / denom


def safe_param_grads(parameters: Sequence[torch.Tensor], grads: Sequence[torch.Tensor | None]) -> List[torch.Tensor]:
    fixed: List[torch.Tensor] = []
    for param, grad in zip(parameters, grads):
        fixed.append(torch.zeros_like(param) if grad is None else grad)
    return fixed


def train_model_on_examples(
    *,
    model,
    examples: Sequence[Dict[str, object]],
    batch_size: int,
    grad_accum: int,
    epochs: int,
    lr: float,
    schedule: str,
    seed: int,
    device: torch.device,
    pad_token_id: int,
    use_bf16: bool,
) -> None:
    train_items = list(examples)
    if not train_items or epochs <= 0:
        return
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=lr,
    )
    num_batches = math.ceil(len(train_items) / max(1, batch_size))
    total_updates = max(1, math.ceil(num_batches * epochs / max(1, grad_accum)))
    if schedule == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1e-3,
            total_iters=total_updates,
        )
    elif schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_updates,
        )
    else:
        scheduler = None

    rng = random.Random(seed)
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        rng.shuffle(train_items)
        for start in range(0, len(train_items), batch_size):
            batch_items = train_items[start : start + batch_size]
            batch = collate_batch(batch_items, device=device, pad_token_id=pad_token_id)
            with autocast_context(device, use_bf16):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                ).logits
                loss = per_example_losses_from_logits(logits, batch["labels"]).mean()
            (loss / max(1, grad_accum)).backward()
            if ((micro_step + 1) % max(1, grad_accum)) == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            micro_step += 1
    if micro_step % max(1, grad_accum) != 0:
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)


def mean_source_loss_inplace(
    model,
    examples: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
    pad_token_id: int,
    use_bf16: bool,
) -> float:
    if not examples:
        return math.nan
    losses = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in iter_batches(
            examples,
            batch_size=max(1, batch_size),
            device=device,
            pad_token_id=pad_token_id,
        ):
            losses.append(float(model_batch_mean_loss(model, batch, use_bf16=use_bf16).detach().cpu()))
    if was_training:
        model.train()
    return sum(losses) / max(1, len(losses))


def mean_trainable_grad_vector(
    model,
    examples: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
    pad_token_id: int,
    use_bf16: bool,
) -> tuple[torch.Tensor, float]:
    if not examples:
        raise RuntimeError("Cannot compute gradient vector for an empty example set.")
    params = tuple(param for param in model.parameters() if param.requires_grad)
    total_vector = None
    total_loss = 0.0
    batch_count = 0
    was_training = model.training
    model.eval()
    for batch in iter_batches(
        examples,
        batch_size=max(1, batch_size),
        device=device,
        pad_token_id=pad_token_id,
    ):
        loss = model_batch_mean_loss(model, batch, use_bf16=use_bf16)
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        flat = torch.cat(
            [grad.detach().float().reshape(-1).cpu() for grad in safe_param_grads(params, grads)],
            dim=0,
        )
        total_vector = flat if total_vector is None else total_vector + flat
        total_loss += float(loss.detach().cpu())
        batch_count += 1
    if was_training:
        model.train()
    assert total_vector is not None
    return total_vector / max(1, batch_count), total_loss / max(1, batch_count)


def weighted_source_loss(
    model,
    params: Dict[str, torch.Tensor],
    theta: torch.Tensor,
    batch: Dict[str, object],
    *,
    use_bf16: bool,
) -> tuple[torch.Tensor, Dict[str, float]]:
    logits = functional_logits(model, params, batch, use_bf16=use_bf16)
    per_example = per_example_losses_from_logits(logits, batch["labels"])
    source_ids = batch["source_ids"]
    lambdas = torch.softmax(theta, dim=0)
    total = torch.zeros((), device=per_example.device, dtype=per_example.dtype)
    components: Dict[str, float] = {}
    for source_id in range(theta.numel()):
        mask = source_ids == source_id
        if mask.any():
            source_loss = per_example[mask].mean()
            total = total + lambdas[source_id] * source_loss
            components[str(source_id)] = float(source_loss.detach().cpu())
    return total, components


def maybe_tune_loss(
    model,
    params: Dict[str, torch.Tensor],
    tune_set: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
    pad_token_id: int,
    use_bf16: bool,
) -> float:
    losses = []
    with torch.no_grad():
        for batch in iter_batches(tune_set, batch_size=batch_size, device=device, pad_token_id=pad_token_id):
            losses.append(float(batch_mean_loss(model, params, batch, use_bf16=use_bf16).detach().cpu()))
    return sum(losses) / max(1, len(losses))


def maybe_tune_loss_inplace(
    model,
    tune_set: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
    pad_token_id: int,
    use_bf16: bool,
) -> float:
    losses = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in iter_batches(tune_set, batch_size=batch_size, device=device, pad_token_id=pad_token_id):
            losses.append(float(model_batch_mean_loss(model, batch, use_bf16=use_bf16).detach().cpu()))
    if was_training:
        model.train()
    return sum(losses) / max(1, len(losses))


def branch_val_loss(
    model,
    params: Dict[str, torch.Tensor],
    val_examples: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    batch_count: int,
    rng: random.Random,
    device: torch.device,
    pad_token_id: int,
    use_bf16: bool,
    track_grad: bool,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    context = nullcontext() if track_grad else torch.no_grad()
    with context:
        for _ in range(max(1, batch_count)):
            batch = sample_batch(
                val_examples,
                batch_size=min(batch_size, len(val_examples)),
                rng=rng,
                device=device,
                pad_token_id=pad_token_id,
            )
            losses.append(batch_mean_loss(model, params, batch, use_bf16=use_bf16))
    return torch.stack(losses, dim=0).mean()


def prepare_val_batches(
    val_examples: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    batch_count: int,
    rng: random.Random,
    device: torch.device,
    pad_token_id: int,
) -> List[Dict[str, object]]:
    return [
        sample_batch(
            val_examples,
            batch_size=min(batch_size, len(val_examples)),
            rng=rng,
            device=device,
            pad_token_id=pad_token_id,
        )
        for _ in range(max(1, batch_count))
    ]


def branch_val_loss_from_batches(
    model,
    params: Dict[str, torch.Tensor],
    batches: Sequence[Dict[str, object]],
    *,
    use_bf16: bool,
    track_grad: bool,
) -> torch.Tensor:
    if not batches:
        reference = next(iter(params.values()))
        return torch.zeros((), device=reference.device, dtype=torch.float32)
    losses: List[torch.Tensor] = []
    context = nullcontext() if track_grad else torch.no_grad()
    with context:
        for batch in batches:
            losses.append(batch_mean_loss(model, params, batch, use_bf16=use_bf16))
    return torch.stack(losses, dim=0).mean()


def theta_grad_from_batches(
    model,
    params: Dict[str, torch.Tensor],
    theta: torch.Tensor,
    batches: Sequence[Dict[str, object]],
    *,
    use_bf16: bool,
) -> tuple[float, torch.Tensor]:
    if not batches:
        return 0.0, torch.zeros_like(theta)
    total_scalar = 0.0
    total_grad = torch.zeros_like(theta)
    for batch_index, batch in enumerate(batches):
        batch_loss = batch_mean_loss(model, params, batch, use_bf16=use_bf16)
        grad_theta = torch.autograd.grad(
            batch_loss,
            theta,
            retain_graph=batch_index + 1 < len(batches),
            create_graph=False,
            allow_unused=True,
        )[0]
        total_scalar += float(batch_loss.detach().cpu())
        total_grad += safe_grad_tensor(theta, grad_theta).detach()
    return total_scalar / len(batches), total_grad / len(batches)


def branch_selection_loss_inplace_from_batches(
    model,
    state: Dict[str, torch.Tensor],
    batches: Sequence[Dict[str, object]],
    *,
    use_bf16: bool,
) -> float:
    assign_trainable_state(
        model,
        {
            name: tensor.detach().clone()
            for name, tensor in state.items()
        },
    )
    losses = []
    with torch.no_grad():
        for batch in batches:
            losses.append(float(model_batch_mean_loss(model, batch, use_bf16=use_bf16).detach().cpu()))
    return sum(losses) / max(1, len(losses))


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
    topped_up.extend(copy.deepcopy(rng.choice(examples)) for _ in range(target_size - len(examples)))
    rng.shuffle(topped_up)
    return topped_up


def subsample_examples(
    examples: Sequence[Dict[str, object]] | Sequence,
    *,
    limit: int,
    seed: int,
):
    if limit <= 0 or len(examples) <= limit:
        return list(examples)
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    picked = sorted(indices[:limit])
    return [examples[idx] for idx in picked]


def pick_branch_index(
    method: str,
    outcomes: Sequence[BranchOutcome],
    rng: random.Random,
    *,
    train_tolerance: float,
    train_penalty: float,
) -> int:
    if method == "hg_single":
        return 0
    if method == "hg_random":
        return rng.randrange(len(outcomes))
    if method == "hg_ms":
        min_train_loss = min(outcome.train_loss_scalar for outcome in outcomes)
        eligible = [
            idx
            for idx, outcome in enumerate(outcomes)
            if outcome.train_loss_scalar <= min_train_loss + train_tolerance
        ]
        if not eligible:
            eligible = list(range(len(outcomes)))
        return min(
            eligible,
            key=lambda idx: outcomes[idx].selection_val_scalar
            + train_penalty * (outcomes[idx].train_loss_scalar - min_train_loss),
        )
    raise ValueError(f"Unsupported method {method}")


def sgd_step(
    state: Dict[str, torch.Tensor],
    grads: Sequence[torch.Tensor],
    *,
    lr: float,
    weight_decay: float,
    create_graph: bool,
) -> Dict[str, torch.Tensor]:
    updated = {}
    for (name, param), grad in zip(state.items(), grads):
        grad_term = grad + weight_decay * param if weight_decay > 0 else grad
        next_param = param - lr * grad_term
        updated[name] = next_param if create_graph else next_param.detach().requires_grad_(True)
    return updated


def safe_grads(parameters: Dict[str, torch.Tensor], grads: Sequence[torch.Tensor | None]) -> List[torch.Tensor]:
    fixed: List[torch.Tensor] = []
    for param, grad in zip(parameters.values(), grads):
        fixed.append(torch.zeros_like(param) if grad is None else grad)
    return fixed


def safe_grad_tensor(reference: torch.Tensor, grad: torch.Tensor | None) -> torch.Tensor:
    return torch.zeros_like(reference) if grad is None else grad


def clipped_theta_grad(theta: torch.Tensor, grad_theta: torch.Tensor | None, clip_norm: float) -> torch.Tensor:
    grad = safe_grad_tensor(theta, grad_theta)
    if clip_norm <= 0:
        return grad
    grad_norm = grad.norm()
    if torch.isfinite(grad_norm) and grad_norm > clip_norm:
        grad = grad * (clip_norm / grad_norm)
    return grad


def theta_diagnostics(theta: torch.Tensor, grad_theta: torch.Tensor | None) -> Dict[str, float | bool]:
    grad = safe_grad_tensor(theta, grad_theta).detach()
    centered_grad = grad - grad.mean()
    lambdas = torch.softmax(theta.detach(), dim=0)
    uniform_weight = 1.0 / theta.numel()
    return {
        "theta_grad_is_none": grad_theta is None,
        "theta_grad_norm": float(grad.norm().cpu()),
        "theta_grad_centered_norm": float(centered_grad.norm().cpu()),
        "theta_grad_max_abs": float(grad.abs().max().cpu()),
        "theta_std": float(theta.detach().std(unbiased=False).cpu()),
        "lambda_max_dev_from_uniform": float((lambdas - uniform_weight).abs().max().cpu()),
    }


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


def ensure_theta_grad_connected(method: str, grad_theta: torch.Tensor | None) -> None:
    if grad_theta is not None:
        return
    raise RuntimeError(
        f"{method} produced a disconnected hypergradient to theta. "
        "In the current PEFT + functional_call setup, the selected validation loss does not depend on theta, "
        "so source reweighting cannot work until the inner parameterization is changed."
    )


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
    inner_steps_override: int | None = None,
) -> BranchOutcome:
    rng = random.Random(seed + 100_000 * (branch_index + 1))
    state = clone_trainable_state(initial_state, noise_std=args.branch_noise_std if branch_index > 0 else 0.0)
    train_loss_scalar = math.nan
    inner_steps = inner_steps_override if inner_steps_override and inner_steps_override > 0 else args.inner_steps
    for inner_step in range(inner_steps):
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
    return BranchOutcome(
        branch_index=branch_index,
        train_loss_scalar=train_loss_scalar,
        selection_val_scalar=float(selection_val),
        params=state,
    )


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


def fine_tune_model(
    *,
    args: argparse.Namespace,
    train_examples_tokenized: Sequence[Dict[str, object]],
    tune_examples_tokenized: Sequence[Dict[str, object]],
    output_dir: Path,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], float]:
    model, tokenizer = load_model_and_tokenizer(args, device)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.final_ft_lr)
    rng = random.Random(args.seed + 900_000)
    train_items = list(train_examples_tokenized)
    total_micro_batch = args.final_ft_micro_batch
    step = 0
    best_tune = float("inf")
    best_state = None
    for epoch in range(args.final_ft_epochs):
        rng.shuffle(train_items)
        for start in range(0, len(train_items), total_micro_batch):
            micro_batch = train_items[start : start + total_micro_batch]
            batch = collate_batch(micro_batch, device=device, pad_token_id=tokenizer.pad_token_id)
            with autocast_context(device, args.bf16):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                ).logits
                loss = per_example_losses_from_logits(logits, batch["labels"]).mean()
            (loss / args.final_ft_grad_accum).backward()
            if ((step + 1) % args.final_ft_grad_accum) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
        tune_loss = maybe_tune_loss_inplace(
            model,
            tune_examples_tokenized,
            batch_size=args.val_batch_size,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
            use_bf16=args.bf16,
        )
        if tune_loss < best_tune:
            best_tune = tune_loss
            best_state = get_trainable_state(model)
    if step % args.final_ft_grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if best_state is None:
        best_state = get_trainable_state(model)
    try:
        torch.save(best_state, output_dir / "final_ft_best_adapter_state.pt")
    except (OSError, RuntimeError) as exc:
        print(
            json.dumps(
                {
                    "stage": "final_ft_save_skipped",
                    "path": str(output_dir / "final_ft_best_adapter_state.pt"),
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, best_tune


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


def evaluate_task(
    *,
    args: argparse.Namespace,
    adapter_state: Dict[str, torch.Tensor],
    dataset_name: str,
    examples: Sequence[Dict[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    model, tokenizer = load_model_and_tokenizer(args, device)
    assign_trainable_state(model, adapter_state)
    model.eval()
    prompts = [item["prompt"] for item in examples]
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
    for item, prediction_text in zip(examples, decoded):
        if dataset_name == "gsm8k":
            pred_answer = extract_gsm8k_answer(prediction_text)
        else:
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
    accuracy = correct / max(1, len(examples))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"accuracy": accuracy, "predictions": predictions}


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


def tokenized_by_source_or_fail(
    *,
    tokenizer,
    raw_by_source: Sequence[Sequence[object]],
    max_length: int,
    needed_per_source: int,
    allow_shortfall: bool = False,
) -> tuple[List[List[Dict[str, object]]], int]:
    tokenized_by_source: List[List[Dict[str, object]]] = []
    available_counts: List[int] = []
    for source_idx, source_examples in enumerate(raw_by_source):
        tokenized = tokenize_examples(tokenizer, source_examples, max_length=max_length)
        tokenized_by_source.append(tokenized)
        available_counts.append(len(tokenized))
    min_available = min(available_counts)
    if min_available < needed_per_source:
        if not allow_shortfall:
            raise RuntimeError(
                f"Only {min_available} tokenized examples available for at least one source but {needed_per_source} are required. "
                "Increase oversampling or enable --allow-train-shortfall."
            )
        print(
            json.dumps(
                {
                    "stage": "tokenized_train_cap",
                    "requested_per_source": needed_per_source,
                    "effective_per_source": min_available,
                    "available_per_source": {
                        TRAIN_SOURCE_SPECS[source_idx]["name"]: count
                        for source_idx, count in enumerate(available_counts)
                    },
                }
            )
        )
    effective_per_source = min(needed_per_source, min_available)
    return [tokenized[:effective_per_source] for tokenized in tokenized_by_source], effective_per_source


def compute_reference_baseline_lambda(
    *,
    args: argparse.Namespace,
    method: str,
    splits: Dict[str, object],
    device: torch.device,
) -> tuple[List[float], Dict[str, object]]:
    reference_model_name = args.model_name if method == "less" else args.reference_model_name
    ref_model, ref_tokenizer = load_model_and_tokenizer_with_name(args, reference_model_name, device)
    tokenized_train_by_source, effective_train_per_source = tokenized_by_source_or_fail(
        tokenizer=ref_tokenizer,
        raw_by_source=splits["reweight_train"],
        max_length=args.max_length,
        needed_per_source=args.reweight_train_per_source,
        allow_shortfall=args.allow_train_shortfall,
    )
    tokenized_val_opt = tokenize_examples(ref_tokenizer, splits["val_opt"], max_length=args.max_length)
    if len(tokenized_val_opt) == 0:
        raise RuntimeError("Reference validation set became empty after tokenization.")

    score_count = min(args.baseline_score_samples_per_source, effective_train_per_source)
    if method == "less":
        warmup_examples = [example for source_examples in tokenized_train_by_source for example in source_examples]
        train_model_on_examples(
            model=ref_model,
            examples=warmup_examples,
            batch_size=args.reference_batch_size,
            grad_accum=args.reference_grad_accum,
            epochs=args.reference_ft_epochs,
            lr=args.reference_ft_lr,
            schedule="linear",
            seed=args.seed + 515151,
            device=device,
            pad_token_id=ref_tokenizer.pad_token_id,
            use_bf16=args.bf16,
        )
        val_subset = tokenized_val_opt[: min(len(tokenized_val_opt), score_count)]
        ref_vector, val_loss = mean_trainable_grad_vector(
            ref_model,
            val_subset,
            batch_size=args.reference_batch_size,
            device=device,
            pad_token_id=ref_tokenizer.pad_token_id,
            use_bf16=args.bf16,
        )
        scores = []
        source_losses = []
        for source_examples in tokenized_train_by_source:
            subset = source_examples[:score_count]
            source_vector, source_loss = mean_trainable_grad_vector(
                ref_model,
                subset,
                batch_size=args.reference_batch_size,
                device=device,
                pad_token_id=ref_tokenizer.pad_token_id,
                use_bf16=args.bf16,
            )
            cosine = F.cosine_similarity(source_vector.unsqueeze(0), ref_vector.unsqueeze(0)).item()
            scores.append(max(0.0, cosine))
            source_losses.append(source_loss)
        details = {
            "reference_model_name": reference_model_name,
            "reference_ft_epochs": args.reference_ft_epochs,
            "reference_ft_lr": args.reference_ft_lr,
            "reference_batch_size": args.reference_batch_size,
            "reference_grad_accum": args.reference_grad_accum,
            "score_samples_per_source": score_count,
            "validation_score_samples": len(val_subset),
            "validation_average_loss": val_loss,
            "source_average_losses": source_losses,
            "scores": scores,
            "gradient_dim": int(ref_vector.numel()),
        }
    else:
        base_losses: List[float] = []
        for source_examples in tokenized_train_by_source:
            base_losses.append(
                mean_source_loss_inplace(
                    ref_model,
                    source_examples[:score_count],
                    batch_size=args.reference_batch_size,
                    device=device,
                    pad_token_id=ref_tokenizer.pad_token_id,
                    use_bf16=args.bf16,
                )
            )

        train_model_on_examples(
            model=ref_model,
            examples=tokenized_val_opt,
            batch_size=args.reference_batch_size,
            grad_accum=args.reference_grad_accum,
            epochs=args.reference_ft_epochs,
            lr=args.reference_ft_lr,
            schedule="cosine",
            seed=args.seed + 515151,
            device=device,
            pad_token_id=ref_tokenizer.pad_token_id,
            use_bf16=args.bf16,
        )

        tuned_losses: List[float] = []
        scores: List[float] = []
        for source_examples, base_loss in zip(tokenized_train_by_source, base_losses):
            tuned_loss = mean_source_loss_inplace(
                ref_model,
                source_examples[:score_count],
                batch_size=args.reference_batch_size,
                device=device,
                pad_token_id=ref_tokenizer.pad_token_id,
                use_bf16=args.bf16,
            )
            tuned_losses.append(tuned_loss)
            scores.append(max(0.0, base_loss - tuned_loss))
        details = {
            "reference_model_name": reference_model_name,
            "reference_ft_epochs": args.reference_ft_epochs,
            "reference_ft_lr": args.reference_ft_lr,
            "reference_batch_size": args.reference_batch_size,
            "reference_grad_accum": args.reference_grad_accum,
            "score_samples_per_source": score_count,
            "base_losses": base_losses,
            "tuned_losses": tuned_losses,
            "scores": scores,
        }

    if sum(scores) <= 0:
        lambdas = [1.0 / len(scores)] * len(scores)
    else:
        total = sum(scores)
        lambdas = [score / total for score in scores]

    del ref_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return lambdas, details


def filter_validation_examples(
    examples: Sequence[object],
    *,
    allowed_kinds: Sequence[str],
) -> list[object]:
    if not allowed_kinds:
        return list(examples)
    allowed = set(allowed_kinds)
    return [example for example in examples if getattr(example, "eval_kind", None) in allowed]


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

    out_dir = maybe_make_output_dir(args.output_dir)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    cache_path = args.cache_path or ""
    if cache_path == "":
        cache_path = str(
            out_dir
            / (
                f"exact_splits_seed{args.seed}_train{args.reweight_train_per_source}_val{args.val_per_source}"
                f"_tov{args.train_raw_oversample}_vov{args.val_raw_oversample}.json"
            )
        )

    splits = load_or_prepare_exact_splits(
        cache_path=cache_path,
        reweight_train_per_source=args.reweight_train_per_source,
        train_raw_oversample=args.train_raw_oversample,
        val_per_source=args.val_per_source,
        val_raw_oversample=args.val_raw_oversample,
        seed=args.seed,
    )
    if args.val_kinds:
        splits = dict(splits)
        splits["val_opt"] = filter_validation_examples(splits["val_opt"], allowed_kinds=args.val_kinds)
        splits["val_tune"] = filter_validation_examples(splits["val_tune"], allowed_kinds=args.val_kinds)
    print(
        json.dumps(
            {
                "stage": "splits_ready",
                "cache_path": cache_path,
                "source_count": len(splits["source_names"]),
                "val_opt_size": len(splits["val_opt"]),
                "val_tune_size": len(splits["val_tune"]),
                "val_kinds": args.val_kinds,
            },
            indent=2,
        )
    )

    source_names = splits["source_names"]
    baseline_details = None
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
        best_tune_loss = math.nan
        history = []
    elif args.method in {"less", "rho_loss"}:
        learned_lambda, baseline_details = compute_reference_baseline_lambda(
            args=args,
            method=args.method,
            splits=splits,
            device=device,
        )
        best_lambda_reweight = list(learned_lambda)
        final_lambda_reweight = list(learned_lambda)
        best_tune_loss = math.nan
        history = []
        print(
            json.dumps(
                {
                    "stage": "baseline_scores_done",
                    "method": args.method,
                    "learned_lambda": learned_lambda,
                    "details": baseline_details,
                },
                indent=2,
            )
        )
    else:
        print(json.dumps({"stage": "reweight_start", "method": args.method}, indent=2))
        model, tokenizer = load_model_and_tokenizer(args, device)
        tokenized_train_by_source, effective_train_per_source = tokenized_by_source_or_fail(
            tokenizer=tokenizer,
            raw_by_source=splits["reweight_train"],
            max_length=args.max_length,
            needed_per_source=args.reweight_train_per_source,
            allow_shortfall=args.allow_train_shortfall,
        )
        tokenized_val_opt = tokenize_examples(tokenizer, splits["val_opt"], max_length=args.max_length)
        tokenized_val_tune = tokenize_examples(tokenizer, splits["val_tune"], max_length=args.max_length)
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
        history = []
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
                selected_branch = {"branch_index": 0, "train_loss": float(train_loss_w.detach().cpu()), "val_loss": float(val_loss.detach().cpu())}
            else:
                branch_count = 1 if args.method == "hg_single" else args.m_branches
                selection_inner_steps = args.selection_inner_steps if args.selection_inner_steps > 0 else args.inner_steps
                hypergrad_inner_steps = args.hypergrad_inner_steps if args.hypergrad_inner_steps > 0 else args.inner_steps
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
                outcomes: List[BranchOutcome] = []
                for branch_index in range(branch_count):
                    outcomes.append(
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
                            inner_steps_override=selection_inner_steps,
                        )
                    )
                picked = pick_branch_index(
                    args.method,
                    outcomes,
                    random.Random(step_seed),
                    train_tolerance=args.selection_train_tolerance,
                    train_penalty=args.selection_train_penalty,
                )
                branch_summaries = [
                    {
                        "branch_index": int(outcome.branch_index),
                        "train_loss": float(outcome.train_loss_scalar),
                        "selection_val_loss": float(outcome.selection_val_scalar),
                    }
                    for outcome in outcomes
                ]
                selected = outcomes[picked]
                selected_branch_index = int(selected.branch_index)
                selected_train_loss = float(selected.train_loss_scalar)
                selected_selection_val = float(selected.selection_val_scalar)
                del outcomes
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                selected_outcome = run_branch(
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
                    inner_steps_override=hypergrad_inner_steps,
                )
                selected_val_scalar, grad_theta = theta_grad_from_batches(
                    model,
                    selected_outcome.params,
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
                    for name, tensor in selected_outcome.params.items()
                }
                selected_branch = {
                    "branch_index": selected_branch_index,
                    "train_loss": selected_train_loss,
                    "selection_val_loss": selected_selection_val,
                    "hypergrad_val_loss": float(selected_val_scalar),
                }

            if outer_step == 0 or (outer_step + 1) % args.tune_every == 0 or outer_step + 1 == total_steps:
                assign_trainable_state(model, trainable_state)
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
                "selected": selected_branch,
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
                selected_val_loss = None
                if args.method in {"hg_single", "hg_random", "hg_ms"}:
                    selected_val_loss = selected_branch["hypergrad_val_loss"]
                print(
                    json.dumps(
                        {
                            "stage": "reweight_progress",
                            "outer_step": outer_step + 1,
                            "selected_branch": selected_branch if args.method in {"hg_single", "hg_random", "hg_ms"} else None,
                            "selected_val_loss": selected_val_loss,
                            "best_tune_loss": best_tune_loss,
                            "lambda": record["lambda"],
                            "theta_grad_norm": grad_stats["theta_grad_norm"],
                            "theta_grad_centered_norm": grad_stats["theta_grad_centered_norm"],
                        }
                    )
                )
            if args.method in {"hg_single", "hg_random", "hg_ms"}:
                selected = None
                selected_outcome = None
                selected_val_scalar = None
                grad_theta = None
                selection_batches = None
                hypergrad_batches = None
            elif args.method == "scalebio":
                train_loss_w = None
                train_loss_u = None
                val_loss = None
                grad_w = None
                grad_u = None
                grad_theta = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        best_lambda_reweight = torch.softmax(best_theta, dim=0).cpu().tolist()
        final_lambda_reweight = torch.softmax(theta.detach(), dim=0).cpu().tolist()
        with open(out_dir / "reweight_history.jsonl", "w") as f:
            for row in history:
                f.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    "stage": "reweight_done",
                    "final_lambda_reweight": final_lambda_reweight,
                    "best_lambda_reweight": best_lambda_reweight,
                    "best_tune_loss": best_tune_loss,
                    "reweight_elapsed_sec": time.perf_counter() - reweight_stage_start_time,
                    "reweight_budget_seconds": reweight_budget_seconds,
                    "reweight_budget_hit": reweight_budget_hit,
                    "reweight_steps_completed": len(history),
                },
                indent=2,
            )
        )
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
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps({"stage": "stopped_after_reweight"}, indent=2))
        print(json.dumps(metrics, indent=2))
        return

    final_train_lambda = best_lambda_reweight if args.use_best_reweight_checkpoint else final_lambda_reweight
    source_counts = source_counts_from_lambda(final_train_lambda, args.final_train_size)
    print(json.dumps({"stage": "sample_final_train", "source_counts": source_counts}, indent=2))
    model, tokenizer = load_model_and_tokenizer(args, device)
    final_train_tokenized = []
    oversample_schedule = [args.train_raw_oversample, max(args.train_raw_oversample, 2.5), max(args.train_raw_oversample, 4.0)]
    for oversample in oversample_schedule:
        final_examples = sample_final_train_examples(
            source_counts=source_counts,
            train_raw_oversample=oversample,
            seed=args.seed,
        )
        final_train_tokenized = tokenize_examples(tokenizer, final_examples, max_length=args.max_length)
        if len(final_train_tokenized) >= args.final_train_size:
            break
        print(
            json.dumps(
                {
                    "stage": "final_train_retry",
                    "oversample": oversample,
                    "kept_examples": len(final_train_tokenized),
                    "target": args.final_train_size,
                },
                indent=2,
            )
        )
    val_tune_tokenized = tokenize_examples(tokenizer, splits["val_tune"], max_length=args.max_length)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(final_train_tokenized) < args.final_train_size:
        if not args.allow_train_shortfall:
            raise RuntimeError(
                f"Only {len(final_train_tokenized)} final train examples remained after tokenization; "
                f"increase oversampling."
            )
        print(
            json.dumps(
                {
                    "stage": "final_train_with_replacement_topup",
                    "kept_examples": len(final_train_tokenized),
                    "target_examples": args.final_train_size,
                },
                indent=2,
            )
        )
        final_train_tokenized = top_up_tokenized_examples_with_replacement(
            examples=final_train_tokenized,
            target_size=args.final_train_size,
            seed=args.seed,
        )
    else:
        final_train_tokenized = final_train_tokenized[: args.final_train_size]
    print(json.dumps({"stage": "final_ft_start", "train_examples": len(final_train_tokenized)}, indent=2))

    final_adapter_state, final_tune_loss = fine_tune_model(
        args=args,
        train_examples_tokenized=final_train_tokenized,
        tune_examples_tokenized=val_tune_tokenized,
        output_dir=out_dir,
        device=device,
    )
    print(json.dumps({"stage": "final_ft_done", "final_ft_tune_loss": final_tune_loss}, indent=2))

    eval_examples = load_eval_examples(include_math=args.eval_math or args.math_eval_size > 0)
    gsm8k_eval_raw = subsample_examples(
        eval_examples["gsm8k"],
        limit=args.gsm8k_eval_size,
        seed=args.seed + 4040,
    )
    math_eval_raw = (
        subsample_examples(
            eval_examples["math"],
            limit=args.math_eval_size,
            seed=args.seed + 5050,
        )
        if args.eval_math
        else []
    )
    gsm8k_records = prepare_eval_records(tokenizer, gsm8k_eval_raw, max_length=args.max_length)
    math_records = prepare_eval_records(tokenizer, math_eval_raw, max_length=args.max_length) if args.eval_math else []

    gsm8k_metrics = evaluate_task(
        args=args,
        adapter_state=final_adapter_state,
        dataset_name="gsm8k",
        examples=gsm8k_records,
        device=device,
    )
    print(json.dumps({"stage": "gsm8k_eval_done", "accuracy": gsm8k_metrics["accuracy"]}, indent=2))
    with open(out_dir / "gsm8k_predictions.jsonl", "w") as f:
        for row in gsm8k_metrics["predictions"]:
            f.write(json.dumps(row) + "\n")

    if args.eval_math:
        math_metrics = evaluate_task(
            args=args,
            adapter_state=final_adapter_state,
            dataset_name="math",
            examples=math_records,
            device=device,
        )
        print(json.dumps({"stage": "math_eval_done", "accuracy": math_metrics["accuracy"]}, indent=2))
        with open(out_dir / "math_predictions.jsonl", "w") as f:
            for row in math_metrics["predictions"]:
                f.write(json.dumps(row) + "\n")
    else:
        math_metrics = {"accuracy": math.nan, "predictions": []}

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
        "gsm8k_accuracy": gsm8k_metrics["accuracy"],
        "math_accuracy": math_metrics["accuracy"],
        "total_elapsed_sec": time.perf_counter() - overall_start_time,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
