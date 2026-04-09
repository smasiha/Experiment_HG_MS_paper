from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

from exact_math_reasoning_data import collate_batch, iter_batches, sample_balanced_batch, sample_batch
from small_scale_data import load_or_prepare_small_scale_splits, tokenize_small_scale_splits


@dataclass
class BranchOutcome:
    branch_index: int
    train_loss_scalar: float
    selection_val_scalar: float
    params: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["denoising", "multilingual"], required=True)
    parser.add_argument("--method", choices=["uniform", "scalebio", "hg_single", "hg_random", "hg_ms"], required=True)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2424)
    parser.add_argument("--fixed-lambda", default="")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--selection-val-batch-size", type=int, default=128)
    parser.add_argument("--selection-val-mode", choices=["random", "target_mixture", "full"], default="target_mixture")
    parser.add_argument("--selection-val-batches", type=int, default=1)
    parser.add_argument("--hypergrad-val-mode", choices=["same", "random", "target_mixture", "full"], default="same")
    parser.add_argument("--hypergrad-val-batches", type=int, default=1)
    parser.add_argument("--reuse-selection-batches-for-hypergrad", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--reweight-epochs", type=int, default=3)
    parser.add_argument("--reweight-steps", type=int, default=0)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--m-branches", type=int, default=2)
    parser.add_argument("--branch-noise-std", type=float, default=1e-3)
    parser.add_argument("--selection-train-tolerance", type=float, default=float("inf"))
    parser.add_argument("--selection-train-penalty", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--reweight-model-lr", type=float, default=1e-5)
    parser.add_argument("--reweight-outer-lr", type=float, default=1e-2)
    parser.add_argument("--reweight-weight-decay", type=float, default=0.01)
    parser.add_argument("--theta-grad-clip", type=float, default=10.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--scalebio-alpha", type=float, default=10.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--tokenized-cache-path", default="")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def parse_fixed_lambda(text: str) -> List[float] | None:
    if not text:
        return None
    normalized = text.replace(":", ",").replace(";", ",")
    values = [float(item.strip()) for item in normalized.split(",") if item.strip()]
    if not values:
        return None
    total = sum(values)
    if total <= 0:
        raise ValueError("--fixed-lambda must have positive sum.")
    return [value / total for value in values]


def theta_from_lambda(lambdas: Sequence[float], device: torch.device) -> torch.Tensor:
    lam = torch.tensor(lambdas, dtype=torch.float32, device=device)
    lam = lam / lam.sum()
    return lam.log()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def module_mode(module, training: bool):
    was_training = module.training
    module.train(training)
    try:
        yield
    finally:
        module.train(was_training)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def maybe_make_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
    local_files_only = os.environ.get("LOCAL_MODEL_ONLY", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    model.to(device)
    model.train()
    return model, tokenizer


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


def init_adamw_state(parameters: Dict[str, torch.Tensor]) -> Dict[str, object]:
    return {
        "step": 0,
        "exp_avg": {name: torch.zeros_like(param, dtype=torch.float32) for name, param in parameters.items()},
        "exp_avg_sq": {name: torch.zeros_like(param, dtype=torch.float32) for name, param in parameters.items()},
    }


def init_optimizer_state(parameters: Dict[str, torch.Tensor], optimizer_name: str) -> Dict[str, object]:
    if optimizer_name == "sgd":
        return {"step": 0}
    if optimizer_name == "adamw":
        return init_adamw_state(parameters)
    raise ValueError(f"Unsupported optimizer {optimizer_name}")


def clone_optimizer_state(optimizer_state: Dict[str, object], optimizer_name: str) -> Dict[str, object]:
    if optimizer_name == "sgd":
        return {"step": int(optimizer_state["step"])}
    return {
        "step": int(optimizer_state["step"]),
        "exp_avg": {name: tensor.detach().clone() for name, tensor in optimizer_state["exp_avg"].items()},
        "exp_avg_sq": {name: tensor.detach().clone() for name, tensor in optimizer_state["exp_avg_sq"].items()},
    }


def functional_logits(model, params: Dict[str, torch.Tensor], batch: Dict[str, object], use_bf16: bool, training: bool):
    inputs = {key: batch[key] for key in ["input_ids", "attention_mask", "labels"]}
    with module_mode(model, training), autocast_context(inputs["input_ids"].device, use_bf16):
        outputs = functional_call(model, params, (), inputs, strict=False)
    return outputs.logits


def model_logits(model, batch: Dict[str, object], use_bf16: bool, training: bool):
    inputs = {key: batch[key] for key in ["input_ids", "attention_mask", "labels"]}
    with module_mode(model, training), autocast_context(inputs["input_ids"].device, use_bf16):
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


def batch_mean_loss(model, params: Dict[str, torch.Tensor], batch: Dict[str, object], use_bf16: bool, training: bool) -> torch.Tensor:
    logits = functional_logits(model, params, batch, use_bf16=use_bf16, training=training)
    return per_example_losses_from_logits(logits, batch["labels"]).mean()


def model_batch_mean_loss(model, batch: Dict[str, object], use_bf16: bool) -> torch.Tensor:
    logits = model_logits(model, batch, use_bf16=use_bf16, training=False)
    return per_example_losses_from_logits(logits, batch["labels"]).mean()


def weighted_source_loss(
    model,
    params: Dict[str, torch.Tensor],
    theta: torch.Tensor,
    batch: Dict[str, object],
    *,
    use_bf16: bool,
) -> tuple[torch.Tensor, Dict[str, float]]:
    logits = functional_logits(model, params, batch, use_bf16=use_bf16, training=True)
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


def safe_grads(parameters: Dict[str, torch.Tensor], grads: Sequence[torch.Tensor | None]) -> List[torch.Tensor]:
    fixed: List[torch.Tensor] = []
    for param, grad in zip(parameters.values(), grads):
        fixed.append(torch.zeros_like(param) if grad is None else grad)
    return fixed


def safe_grad_tensor(reference: torch.Tensor, grad: torch.Tensor | None) -> torch.Tensor:
    return torch.zeros_like(reference) if grad is None else grad


def sanitize_update_tensor(tensor: torch.Tensor, clip_value: float) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clip_value, neginf=-clip_value)
    return tensor.clamp(min=-clip_value, max=clip_value)


def detach_optimizer_state(optimizer_state: Dict[str, object], optimizer_name: str) -> Dict[str, object]:
    if optimizer_name == "sgd":
        return {"step": int(optimizer_state["step"])}
    return {
        "step": int(optimizer_state["step"]),
        "exp_avg": {name: tensor.detach().clone() for name, tensor in optimizer_state["exp_avg"].items()},
        "exp_avg_sq": {name: tensor.detach().clone() for name, tensor in optimizer_state["exp_avg_sq"].items()},
    }


def sgd_step(
    state: Dict[str, torch.Tensor],
    grads: Sequence[torch.Tensor],
    optimizer_state: Dict[str, object],
    *,
    lr: float,
    weight_decay: float,
    create_graph: bool,
) -> tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    updated_params: Dict[str, torch.Tensor] = {}
    for (name, param), grad in zip(state.items(), grads):
        step_direction = grad
        if weight_decay > 0:
            step_direction = step_direction + weight_decay * param
        next_param = param - lr * step_direction
        updated_params[name] = next_param if create_graph else next_param.detach().requires_grad_(True)
    return updated_params, {"step": int(optimizer_state["step"]) + 1}


def adamw_step(
    state: Dict[str, torch.Tensor],
    grads: Sequence[torch.Tensor],
    optimizer_state: Dict[str, object],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    create_graph: bool,
) -> tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    next_step = int(optimizer_state["step"]) + 1
    updated_params: Dict[str, torch.Tensor] = {}
    updated_exp_avg: Dict[str, torch.Tensor] = {}
    updated_exp_avg_sq: Dict[str, torch.Tensor] = {}
    bias_correction1 = 1.0 - beta1**next_step
    bias_correction2 = 1.0 - beta2**next_step

    for (name, param), grad in zip(state.items(), grads):
        grad_f = grad.to(torch.float32)
        exp_avg = optimizer_state["exp_avg"][name]
        exp_avg_sq = optimizer_state["exp_avg_sq"][name]
        next_exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad_f
        next_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad_f.square()
        denom = next_exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + eps
        step_direction = (next_exp_avg / bias_correction1) / denom
        if weight_decay > 0:
            step_direction = step_direction + weight_decay * param.to(torch.float32)
        next_param = param.to(torch.float32) - lr * step_direction
        next_param = next_param.to(param.dtype)
        updated_params[name] = next_param if create_graph else next_param.detach().requires_grad_(True)
        updated_exp_avg[name] = next_exp_avg if create_graph else next_exp_avg.detach()
        updated_exp_avg_sq[name] = next_exp_avg_sq if create_graph else next_exp_avg_sq.detach()

    return updated_params, {
        "step": next_step,
        "exp_avg": updated_exp_avg,
        "exp_avg_sq": updated_exp_avg_sq,
    }


def optimizer_step(
    optimizer_name: str,
    state: Dict[str, torch.Tensor],
    grads: Sequence[torch.Tensor],
    optimizer_state: Dict[str, object],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    create_graph: bool,
) -> tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    if optimizer_name == "sgd":
        return sgd_step(
            state,
            grads,
            optimizer_state,
            lr=lr,
            weight_decay=weight_decay,
            create_graph=create_graph,
        )
    if optimizer_name == "adamw":
        return adamw_step(
            state,
            grads,
            optimizer_state,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            create_graph=create_graph,
        )
    raise ValueError(f"Unsupported optimizer {optimizer_name}")


def evaluate_dataset_loss_inplace(
    model,
    state: Dict[str, torch.Tensor],
    tokenized_examples: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
    pad_token_id: int,
    use_bf16: bool,
) -> float:
    assign_trainable_state(model, state)
    losses = []
    with torch.no_grad():
        for batch in iter_batches(tokenized_examples, batch_size=batch_size, device=device, pad_token_id=pad_token_id):
            losses.append(float(model_batch_mean_loss(model, batch, use_bf16=use_bf16).detach().cpu()))
    return sum(losses) / max(1, len(losses))


def functional_dataset_mean_loss(
    model,
    params: Dict[str, torch.Tensor],
    tokenized_examples: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
    pad_token_id: int,
    use_bf16: bool,
) -> torch.Tensor:
    total_loss = None
    batch_count = 0
    for batch in iter_batches(
        tokenized_examples,
        batch_size=batch_size,
        device=device,
        pad_token_id=pad_token_id,
    ):
        batch_loss = batch_mean_loss(model, params, batch, use_bf16=use_bf16, training=False)
        total_loss = batch_loss if total_loss is None else total_loss + batch_loss
        batch_count += 1
    if total_loss is None:
        return torch.zeros((), device=device, dtype=torch.float32)
    return total_loss / batch_count


def branch_val_loss(
    model,
    params: Dict[str, torch.Tensor],
    val_examples: Sequence[Dict[str, object]],
    val_examples_by_source: Sequence[Sequence[Dict[str, object]]],
    *,
    mode: str,
    batch_size: int,
    estimate_batches: int,
    target_lambda: Sequence[float],
    rng: random.Random,
    device: torch.device,
    pad_token_id: int,
    use_bf16: bool,
) -> torch.Tensor:
    if mode == "full":
        return functional_dataset_mean_loss(
            model,
            params,
            val_examples,
            batch_size=min(batch_size, len(val_examples)),
            device=device,
            pad_token_id=pad_token_id,
            use_bf16=use_bf16,
        )

    total_loss = None
    batch_count = 0
    for _ in range(max(1, estimate_batches)):
        batch = sample_val_batch(
            val_examples,
            val_examples_by_source,
            mode=mode,
            batch_size=min(batch_size, len(val_examples)),
            target_lambda=target_lambda,
            rng=rng,
            device=device,
            pad_token_id=pad_token_id,
        )
        batch_loss = batch_mean_loss(model, params, batch, use_bf16=use_bf16, training=False)
        total_loss = batch_loss if total_loss is None else total_loss + batch_loss
        batch_count += 1
    return total_loss / batch_count


def branch_selection_loss_inplace(
    model,
    state: Dict[str, torch.Tensor],
    val_examples: Sequence[Dict[str, object]],
    val_examples_by_source: Sequence[Sequence[Dict[str, object]]],
    *,
    mode: str,
    batch_size: int,
    estimate_batches: int,
    target_lambda: Sequence[float],
    rng: random.Random,
    device: torch.device,
    pad_token_id: int,
    use_bf16: bool,
) -> float:
    assign_trainable_state(
        model,
        {
            name: tensor.detach().clone()
            for name, tensor in state.items()
        },
    )
    if mode == "full":
        losses = []
        with torch.no_grad():
            for batch in iter_batches(
                val_examples,
                batch_size=min(batch_size, len(val_examples)),
                device=device,
                pad_token_id=pad_token_id,
            ):
                losses.append(float(model_batch_mean_loss(model, batch, use_bf16=use_bf16).detach().cpu()))
        return sum(losses) / max(1, len(losses))

    losses = []
    with torch.no_grad():
        for _ in range(max(1, estimate_batches)):
            batch = sample_val_batch(
                val_examples,
                val_examples_by_source,
                mode=mode,
                batch_size=min(batch_size, len(val_examples)),
                target_lambda=target_lambda,
                rng=rng,
                device=device,
                pad_token_id=pad_token_id,
            )
            losses.append(float(model_batch_mean_loss(model, batch, use_bf16=use_bf16).detach().cpu()))
    return sum(losses) / len(losses)


def prepare_val_batches(
    val_examples: Sequence[Dict[str, object]],
    val_examples_by_source: Sequence[Sequence[Dict[str, object]]],
    *,
    mode: str,
    batch_size: int,
    estimate_batches: int,
    target_lambda: Sequence[float],
    rng: random.Random,
    device: torch.device,
    pad_token_id: int,
) -> List[Dict[str, object]]:
    if mode == "full":
        return list(
            iter_batches(
                val_examples,
                batch_size=min(batch_size, len(val_examples)),
                device=device,
                pad_token_id=pad_token_id,
            )
        )

    return [
        sample_val_batch(
            val_examples,
            val_examples_by_source,
            mode=mode,
            batch_size=min(batch_size, len(val_examples)),
            target_lambda=target_lambda,
            rng=rng,
            device=device,
            pad_token_id=pad_token_id,
        )
        for _ in range(max(1, estimate_batches))
    ]


def branch_val_loss_from_batches(
    model,
    params: Dict[str, torch.Tensor],
    batches: Sequence[Dict[str, object]],
    *,
    use_bf16: bool,
) -> torch.Tensor:
    total_loss = None
    batch_count = 0
    for batch in batches:
        batch_loss = batch_mean_loss(model, params, batch, use_bf16=use_bf16, training=False)
        total_loss = batch_loss if total_loss is None else total_loss + batch_loss
        batch_count += 1
    if total_loss is None:
        reference = next(iter(params.values()))
        return torch.zeros((), device=reference.device, dtype=torch.float32)
    return total_loss / batch_count


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
        batch_loss = batch_mean_loss(model, params, batch, use_bf16=use_bf16, training=False)
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
            key=lambda idx: outcomes[idx].selection_val_scalar + train_penalty * (outcomes[idx].train_loss_scalar - min_train_loss),
        )
    raise ValueError(f"Unsupported method {method}")


def run_branch(
    *,
    model,
    initial_state: Dict[str, torch.Tensor],
    initial_optimizer_state: Dict[str, object],
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
) -> BranchOutcome:
    rng = random.Random(seed + 100_000 * (branch_index + 1))
    state = clone_trainable_state(initial_state, noise_std=args.branch_noise_std if branch_index > 0 else 0.0)
    optimizer_state = clone_optimizer_state(initial_optimizer_state, args.optimizer)
    train_loss_scalar = math.nan
    for _ in range(args.inner_steps):
        train_batch = sample_balanced_batch(
            tokenized_train_by_source,
            per_source_batch=args.batch_size // len(tokenized_train_by_source),
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
        state, optimizer_state = optimizer_step(
            args.optimizer,
            state,
            safe_grads(state, grads),
            optimizer_state,
            lr=args.reweight_model_lr,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            eps=args.adam_eps,
            weight_decay=args.reweight_weight_decay,
            create_graph=create_graph,
        )
        train_loss_scalar = float(train_loss.detach().cpu())
    selection_val_scalar = math.nan
    if compute_selection_loss:
        selection_val_scalar = branch_selection_loss_inplace_from_batches(
            model,
            state,
            selection_batches,
            use_bf16=args.bf16,
        )
    return BranchOutcome(
        branch_index=branch_index,
        train_loss_scalar=train_loss_scalar,
        selection_val_scalar=selection_val_scalar,
        params=state,
        optimizer_state=optimizer_state,
    )


def summarize_target_error(lambdas: Sequence[float], target_lambda: Sequence[float]) -> float:
    return sum(abs(float(a) - float(b)) for a, b in zip(lambdas, target_lambda))


def split_examples_by_source(
    examples: Sequence[Dict[str, object]],
    *,
    source_count: int,
) -> List[List[Dict[str, object]]]:
    grouped: List[List[Dict[str, object]]] = [[] for _ in range(source_count)]
    for item in examples:
        source_id = int(item.get("source_id", -1))
        if 0 <= source_id < source_count:
            grouped[source_id].append(item)
    return grouped


def target_batch_counts(batch_size: int, weights: Sequence[float]) -> List[int]:
    raw_counts = [batch_size * float(weight) for weight in weights]
    counts = [int(math.floor(value)) for value in raw_counts]
    remainder = batch_size - sum(counts)
    order = sorted(range(len(weights)), key=lambda idx: raw_counts[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def sample_val_batch(
    val_examples: Sequence[Dict[str, object]],
    val_examples_by_source: Sequence[Sequence[Dict[str, object]]],
    *,
    mode: str,
    batch_size: int,
    target_lambda: Sequence[float],
    rng: random.Random,
    device: torch.device,
    pad_token_id: int,
) -> Dict[str, object]:
    if mode == "full" or batch_size >= len(val_examples):
        return collate_batch(val_examples, device=device, pad_token_id=pad_token_id)
    if mode == "random":
        return sample_batch(
            val_examples,
            batch_size=batch_size,
            rng=rng,
            device=device,
            pad_token_id=pad_token_id,
        )
    if mode != "target_mixture":
        raise ValueError(f"Unsupported validation batch mode {mode}")

    counts = target_batch_counts(batch_size, target_lambda)
    batch: List[Dict[str, object]] = []
    for source_examples, count in zip(val_examples_by_source, counts):
        if count <= 0 or len(source_examples) == 0:
            continue
        if count <= len(source_examples):
            batch.extend(rng.sample(list(source_examples), count))
        else:
            batch.extend(source_examples)
            batch.extend(source_examples[rng.randrange(len(source_examples))] for _ in range(count - len(source_examples)))
    if len(batch) < batch_size:
        batch.extend(val_examples[rng.randrange(len(val_examples))] for _ in range(batch_size - len(batch)))
    rng.shuffle(batch)
    return collate_batch(batch, device=device, pad_token_id=pad_token_id)


def main() -> None:
    args = parse_args()
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

    raw_cache_path = args.cache_path or str(out_dir / f"{args.task}_splits_seed{args.seed}.json")
    tokenized_cache_path = args.tokenized_cache_path or str(out_dir / f"{args.task}_splits_seed{args.seed}_tok.pt")

    splits = load_or_prepare_small_scale_splits(task=args.task, seed=args.seed, cache_path=raw_cache_path)
    model, tokenizer = load_model_and_tokenizer(args, device)
    tokenized = tokenize_small_scale_splits(
        tokenizer=tokenizer,
        splits=splits,
        max_length=args.max_length,
        tokenized_cache_path=tokenized_cache_path,
    )
    tokenized_train_by_source = tokenized["reweight_train"]
    tokenized_val = tokenized["val"]
    tokenized_test = tokenized["test"]
    tokenized_test_by_source = tokenized["test_by_source"]
    source_names = splits["source_names"]
    target_lambda = splits["target_lambda"]
    tokenized_val_by_source = split_examples_by_source(tokenized_val, source_count=len(source_names))
    if len(tokenized_val) == 0 or len(tokenized_test) == 0:
        raise RuntimeError("Validation or test set became empty after tokenization.")
    if args.batch_size % len(tokenized_train_by_source) != 0:
        raise RuntimeError("batch-size must be divisible by number of sources.")

    total_train_size = sum(len(source_items) for source_items in tokenized_train_by_source)
    total_steps = (
        args.reweight_steps
        if args.reweight_steps > 0
        else args.reweight_epochs * math.ceil(total_train_size / args.batch_size)
    )

    trainable_state = get_trainable_state(model)
    optimizer_state = init_optimizer_state(trainable_state, args.optimizer)
    auxiliary_state = clone_trainable_state(trainable_state, noise_std=0.0)
    auxiliary_optimizer_state = init_optimizer_state(auxiliary_state, args.optimizer)
    fixed_lambda = parse_fixed_lambda(args.fixed_lambda)
    if fixed_lambda is not None and len(fixed_lambda) != len(source_names):
        raise RuntimeError(f"--fixed-lambda must provide {len(source_names)} values for task {args.task}.")
    theta_init = (
        theta_from_lambda(fixed_lambda, device=device)
        if fixed_lambda is not None
        else torch.zeros(len(source_names), dtype=torch.float32, device=device)
    )
    theta = theta_init.detach().clone().requires_grad_(True)

    history = []
    best_val_loss = float("inf")
    best_theta = theta.detach().clone()
    best_state = clone_trainable_state(trainable_state, noise_std=0.0)

    for outer_step in range(total_steps):
        step_seed = args.seed + 7_919 * (outer_step + 1)
        rng = random.Random(step_seed)
        theta_grad_norm = math.nan

        if args.method == "uniform":
            train_batch = sample_balanced_batch(
                tokenized_train_by_source,
                per_source_batch=args.batch_size // len(tokenized_train_by_source),
                rng=rng,
                device=device,
                pad_token_id=tokenizer.pad_token_id,
            )
            train_loss, train_components = weighted_source_loss(model, trainable_state, theta.detach(), train_batch, use_bf16=args.bf16)
            grads = torch.autograd.grad(train_loss, tuple(trainable_state.values()), create_graph=False, allow_unused=True)
            trainable_state, optimizer_state = optimizer_step(
                args.optimizer,
                trainable_state,
                safe_grads(trainable_state, grads),
                optimizer_state,
                lr=args.reweight_model_lr,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                eps=args.adam_eps,
                weight_decay=args.reweight_weight_decay,
                create_graph=False,
            )
            selected_branch = {
                "branch_index": 0,
                "train_loss": float(train_loss.detach().cpu()),
                "val_loss": math.nan,
            }
            branch_summaries = None
        elif args.method == "scalebio":
            train_batch = sample_balanced_batch(
                tokenized_train_by_source,
                per_source_batch=args.batch_size // len(tokenized_train_by_source),
                rng=rng,
                device=device,
                pad_token_id=tokenizer.pad_token_id,
            )
            train_loss_w, train_components = weighted_source_loss(model, trainable_state, theta, train_batch, use_bf16=args.bf16)
            train_loss_u, _ = weighted_source_loss(model, auxiliary_state, theta, train_batch, use_bf16=args.bf16)
            val_loss = branch_val_loss(
                model,
                trainable_state,
                tokenized_val,
                tokenized_val_by_source,
                mode=args.selection_val_mode,
                batch_size=args.selection_val_batch_size,
                estimate_batches=args.selection_val_batches,
                target_lambda=target_lambda,
                rng=rng,
                device=device,
                pad_token_id=tokenizer.pad_token_id,
                use_bf16=args.bf16,
            )
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
            theta_grad_norm = float(safe_grad_tensor(theta, grad_theta).detach().norm().cpu())
            grad_u = torch.autograd.grad(
                alpha * train_loss_u,
                tuple(auxiliary_state.values()),
                create_graph=False,
                allow_unused=True,
            )
            trainable_state, optimizer_state = optimizer_step(
                args.optimizer,
                trainable_state,
                safe_grads(trainable_state, grad_w),
                optimizer_state,
                lr=args.reweight_model_lr,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                eps=args.adam_eps,
                weight_decay=args.reweight_weight_decay,
                create_graph=False,
            )
            auxiliary_state, auxiliary_optimizer_state = optimizer_step(
                args.optimizer,
                auxiliary_state,
                safe_grads(auxiliary_state, grad_u),
                auxiliary_optimizer_state,
                lr=args.reweight_model_lr,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                eps=args.adam_eps,
                weight_decay=args.reweight_weight_decay,
                create_graph=False,
            )
            with torch.no_grad():
                theta -= args.reweight_outer_lr * sanitize_update_tensor(
                    safe_grad_tensor(theta, grad_theta),
                    args.theta_grad_clip,
                )
            theta = theta.detach().requires_grad_(True)
            selected_branch = {
                "branch_index": 0,
                "train_loss": float(train_loss_w.detach().cpu()),
                "val_loss": float(val_loss.detach().cpu()),
            }
            branch_summaries = None
        else:
            branch_count = 1 if args.method == "hg_single" else args.m_branches
            selection_batches = prepare_val_batches(
                tokenized_val,
                tokenized_val_by_source,
                mode=args.selection_val_mode,
                batch_size=args.selection_val_batch_size,
                estimate_batches=args.selection_val_batches,
                target_lambda=target_lambda,
                rng=random.Random(step_seed + 17_003),
                device=device,
                pad_token_id=tokenizer.pad_token_id,
            )
            hypergrad_mode = args.selection_val_mode if args.hypergrad_val_mode == "same" else args.hypergrad_val_mode
            if args.reuse_selection_batches_for_hypergrad:
                hypergrad_batches = selection_batches
            else:
                hypergrad_batches = prepare_val_batches(
                    tokenized_val,
                    tokenized_val_by_source,
                    mode=hypergrad_mode,
                    batch_size=args.selection_val_batch_size,
                    estimate_batches=args.hypergrad_val_batches,
                    target_lambda=target_lambda,
                    rng=random.Random(step_seed + 900_001),
                    device=device,
                    pad_token_id=tokenizer.pad_token_id,
                )
            outcomes: List[BranchOutcome] = []
            for branch_index in range(branch_count):
                outcomes.append(
                    run_branch(
                        model=model,
                        initial_state=trainable_state,
                        initial_optimizer_state=optimizer_state,
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
                    "val_loss": float(outcome.selection_val_scalar),
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
                initial_optimizer_state=optimizer_state,
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
            selected_params = selected_outcome.params
            selected_optimizer_state = selected_outcome.optimizer_state
            selected_val_scalar, grad_theta = theta_grad_from_batches(
                model,
                selected_params,
                theta,
                hypergrad_batches,
                use_bf16=args.bf16,
            )
            theta_grad_norm = float(grad_theta.detach().norm().cpu())
            with torch.no_grad():
                theta -= args.reweight_outer_lr * sanitize_update_tensor(
                    grad_theta,
                    args.theta_grad_clip,
                )
            theta = theta.detach().requires_grad_(True)
            trainable_state = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in selected_params.items()}
            optimizer_state = detach_optimizer_state(selected_optimizer_state, args.optimizer)
            train_components = {}
            selected_branch = {
                "branch_index": selected_branch_index,
                "train_loss": selected_train_loss,
                "val_loss": selected_val_scalar,
                "selection_val_loss": selected_selection_val,
            }

        full_val_loss = math.nan
        if outer_step == 0 or (outer_step + 1) % args.eval_every == 0 or outer_step + 1 == total_steps:
            full_val_loss = evaluate_dataset_loss_inplace(
                model,
                trainable_state,
                tokenized_val,
                batch_size=args.eval_batch_size,
                device=device,
                pad_token_id=tokenizer.pad_token_id,
                use_bf16=args.bf16,
            )
            if full_val_loss < best_val_loss:
                best_val_loss = full_val_loss
                best_theta = theta.detach().clone()
                best_state = clone_trainable_state(trainable_state, noise_std=0.0)

        lambdas = torch.softmax(theta.detach(), dim=0).cpu().tolist()
        record = {
            "outer_step": outer_step,
            "selected": selected_branch,
            "train_components": train_components,
            "lambda": lambdas,
            "theta_grad_norm": theta_grad_norm,
            "target_error_l1": summarize_target_error(lambdas, target_lambda),
            "full_val_loss": full_val_loss,
            "best_val_loss": best_val_loss,
        }
        if branch_summaries is not None:
            record["candidates"] = branch_summaries
        history.append(record)
        if outer_step % max(1, args.eval_every // 2) == 0:
            lambda_summary = ",".join(f"{weight:.4f}" for weight in lambdas)
            print(
                f"step={outer_step} lambda=[{lambda_summary}] "
                f"target_error={record['target_error_l1']:.4f} "
                f"theta_grad_norm={theta_grad_norm:.4f} "
                f"best_val={best_val_loss:.4f}",
                flush=True,
            )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_lambda = torch.softmax(theta.detach(), dim=0).cpu().tolist()
    best_checkpoint_lambda = torch.softmax(best_theta, dim=0).cpu().tolist()
    test_loss = evaluate_dataset_loss_inplace(
        model,
        best_state,
        tokenized_test,
        batch_size=args.eval_batch_size,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
        use_bf16=args.bf16,
    )
    test_losses_by_source = {
        name: evaluate_dataset_loss_inplace(
            model,
            best_state,
            items,
            batch_size=args.eval_batch_size,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
            use_bf16=args.bf16,
        )
        for name, items in tokenized_test_by_source.items()
    }
    metrics = {
        "task": args.task,
        "method": args.method,
        "model_name": args.model_name,
        "seed": args.seed,
        "fixed_lambda": fixed_lambda,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_losses_by_source": test_losses_by_source,
        "learned_lambda": final_lambda,
        "best_checkpoint_lambda": best_checkpoint_lambda,
        "target_lambda": target_lambda,
        "target_error_l1": summarize_target_error(final_lambda, target_lambda),
        "best_checkpoint_target_error_l1": summarize_target_error(best_checkpoint_lambda, target_lambda),
        "total_steps": total_steps,
        "source_names": source_names,
    }
    with open(out_dir / "history.jsonl", "w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
