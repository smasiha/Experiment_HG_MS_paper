"""
Diagnose the connected lower-level manifold hypothesis in the LoRA LLM setup.

This script targets the exact symmetry of the LoRA parameterization. For a LoRA
update DeltaW = B A with rank r, any orthogonal Q in SO(r) gives the same
update:

    DeltaW = B A = (B Q^T) (Q A).

Therefore, if we freeze the outer variable theta and rotate every LoRA pair
along a continuous orthogonal path, we obtain a continuous family of distinct
lower-level parameters that should keep the lower-level objective unchanged.

The diagnostic has two pieces:

1. Connected-path probe:
   build a 1D continuous rotation path and evaluate the lower-level train loss
   along the path on fixed deterministic balanced train batches.

2. Local-dimension probe:
   build several independent infinitesimal rotation generators, evaluate the
   same lower-level loss after each perturbation, and estimate the rank of the
   resulting displacement vectors by SVD.

This script intentionally does not use the upper-level objective.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exact_math_reasoning import (
    functional_logits,
    load_model_and_tokenizer,
    weighted_source_loss,
)
from exact_math_reasoning_data import collate_batch
from real_world_instruction import tokenized_by_source_or_fail
from real_world_instruction_data import load_or_prepare_instruction_splits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA lower-level manifold diagnostic.")
    parser.add_argument("--run-dir", required=True, help="Completed instruction-following run directory.")
    parser.add_argument(
        "--checkpoint",
        choices=["best", "final"],
        default="best",
        help="Which saved reweight-stage adapter state to use.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to <run-dir>/lora_manifold_<checkpoint>.",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=4,
        help="Number of fixed deterministic balanced train batches used to evaluate the lower-level loss.",
    )
    parser.add_argument(
        "--per-source-batch",
        type=int,
        default=0,
        help="Examples per source in each diagnostic batch. Defaults to the run config value.",
    )
    parser.add_argument(
        "--num-path-points",
        type=int,
        default=9,
        help="Number of points on the connected rotation path, including both endpoints.",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=2.0 * math.pi,
        help="Largest rotation angle for the connected path probe.",
    )
    parser.add_argument(
        "--path-plane-count",
        type=int,
        default=1,
        help="How many disjoint rank-2 rotation planes to rotate simultaneously along the path.",
    )
    parser.add_argument(
        "--num-generators",
        type=int,
        default=4,
        help="How many independent local rotation generators to use in the tangent-rank probe.",
    )
    parser.add_argument(
        "--generator-eps",
        type=float,
        default=1e-2,
        help="Small rotation angle for each local generator in radians.",
    )
    parser.add_argument(
        "--eval-bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to keep bf16 evaluation. Default is false so the invariance check runs in fp32.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_output_dir(run_dir: Path, checkpoint: str, output_dir_arg: str) -> Path:
    if output_dir_arg:
        out_dir = Path(output_dir_arg)
    else:
        out_dir = run_dir / f"lora_manifold_{checkpoint}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_run_config(run_dir: Path) -> SimpleNamespace:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}.")
    config = _read_json(config_path)
    args = SimpleNamespace(**config)
    args.output_dir = str(run_dir)
    if not getattr(args, "cache_path", ""):
        args.cache_path = str(run_dir / f"instruction_splits_seed{args.seed}_train{args.reweight_train_per_source}.json")
    return args


def _load_metrics(run_dir: Path) -> Dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    return _read_json(metrics_path)


def _load_theta(run_dir: Path, checkpoint: str, metrics: Dict[str, object], device: torch.device) -> torch.Tensor:
    theta_path = run_dir / f"reweight_{checkpoint}_theta.pt"
    if theta_path.exists():
        theta = torch.load(theta_path, map_location=device)
        if not isinstance(theta, torch.Tensor):
            raise TypeError(f"Unexpected theta payload in {theta_path}.")
        return theta.detach().to(device=device, dtype=torch.float32)

    key = "best_lambda_reweight" if checkpoint == "best" else "learned_lambda"
    lambdas = metrics.get(key)
    if not isinstance(lambdas, list) or not lambdas:
        raise FileNotFoundError(
            f"Missing {theta_path.name} and no {key} entry in metrics.json. "
            "Rerun the instruction job after saving reweight-stage checkpoints."
        )
    lambda_tensor = torch.tensor(lambdas, dtype=torch.float32, device=device).clamp_min(1e-12)
    lambda_tensor = lambda_tensor / lambda_tensor.sum()
    return torch.log(lambda_tensor)


def _load_adapter_state(run_dir: Path, checkpoint: str) -> Dict[str, torch.Tensor]:
    state_path = run_dir / f"reweight_{checkpoint}_adapter_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing {state_path.name}. "
            "Rerun the instruction job after saving reweight-stage checkpoints."
        )
    state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise TypeError(f"Unexpected adapter payload in {state_path}.")
    if not all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in state.items()):
        raise TypeError(f"Unexpected adapter payload in {state_path}.")
    return state


def _move_state_to_model(
    model,
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    named_params = dict(model.named_parameters())
    moved: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name not in named_params:
            raise KeyError(f"State tensor {name} not found in model parameters.")
        ref = named_params[name]
        moved[name] = tensor.detach().to(device=ref.device, dtype=ref.dtype).clone().requires_grad_(True)
    return moved


def _build_deterministic_balanced_batches(
    *,
    tokenized_by_source: Sequence[Sequence[Dict[str, object]]],
    batch_count: int,
    per_source_batch: int,
    device: torch.device,
    pad_token_id: int,
) -> List[Dict[str, object]]:
    batches: List[Dict[str, object]] = []
    for batch_idx in range(max(1, int(batch_count))):
        flat_batch: List[Dict[str, object]] = []
        for source_examples in tokenized_by_source:
            if not source_examples:
                raise RuntimeError("Encountered an empty tokenized source while building diagnostic batches.")
            n = len(source_examples)
            for slot_idx in range(max(1, int(per_source_batch))):
                sample_idx = (batch_idx * max(1, int(per_source_batch)) + slot_idx) % n
                flat_batch.append(source_examples[sample_idx])
        batches.append(collate_batch(flat_batch, device=device, pad_token_id=pad_token_id))
    return batches


def _prepare_instruction_batches(
    *,
    args,
    tokenizer,
    device: torch.device,
    batch_count: int,
    per_source_batch: int,
) -> List[Dict[str, object]]:
    setattr(tokenizer, "_allow_train_shortfall", bool(getattr(args, "allow_train_shortfall", False)))
    splits = load_or_prepare_instruction_splits(
        cache_path=str(args.cache_path),
        reweight_train_per_source=int(args.reweight_train_per_source),
        reweight_train_raw_oversample=float(args.reweight_train_raw_oversample),
        seed=int(args.seed),
    )
    tokenized_by_source, _ = tokenized_by_source_or_fail(
        tokenizer=tokenizer,
        raw_by_source=splits["reweight_train"],
        max_length=int(args.max_length),
        needed_per_source=int(args.reweight_train_per_source),
        seed=int(args.seed),
    )
    return _build_deterministic_balanced_batches(
        tokenized_by_source=tokenized_by_source,
        batch_count=batch_count,
        per_source_batch=per_source_batch,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )


def _find_lora_pairs(state: Dict[str, torch.Tensor]) -> List[Tuple[str, str, str]]:
    suffix_a = "lora_A.default.weight"
    suffix_b = "lora_B.default.weight"
    pairs: List[Tuple[str, str, str]] = []
    for name in sorted(state.keys()):
        if not name.endswith(suffix_a):
            continue
        prefix = name[: -len(suffix_a)]
        paired_name = prefix + suffix_b
        if paired_name not in state:
            raise KeyError(f"Missing paired LoRA tensor {paired_name} for {name}.")
        tensor_a = state[name]
        tensor_b = state[paired_name]
        if tensor_a.ndim != 2 or tensor_b.ndim != 2:
            raise ValueError(f"LoRA tensors must be matrices, got {name} and {paired_name}.")
        rank = int(tensor_a.shape[0])
        if int(tensor_b.shape[1]) != rank:
            raise ValueError(f"Incompatible LoRA pair shapes for {name} and {paired_name}.")
        pairs.append((prefix, name, paired_name))
    if not pairs:
        raise RuntimeError("No LoRA A/B parameter pairs were found in the saved adapter state.")
    return pairs


def _common_lora_rank(
    state: Dict[str, torch.Tensor],
    pairs: Sequence[Tuple[str, str, str]],
) -> int:
    ranks = {int(state[name_a].shape[0]) for _, name_a, _ in pairs}
    if len(ranks) != 1:
        raise ValueError(f"Expected a single LoRA rank, found ranks {sorted(ranks)}.")
    rank = next(iter(ranks))
    if rank < 2:
        raise ValueError(f"LoRA rank must be at least 2, got {rank}.")
    return rank


def _rotation_planes(rank: int, count: int) -> List[Tuple[int, int]]:
    max_planes = rank // 2
    use_count = min(max_planes, max(1, int(count)))
    return [(2 * idx, 2 * idx + 1) for idx in range(use_count)]


def _rotation_matrix(rank: int, angle: float, planes: Sequence[Tuple[int, int]]) -> torch.Tensor:
    q = torch.eye(rank, dtype=torch.float32)
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    for i, j in planes:
        q[i, i] = c
        q[i, j] = -s
        q[j, i] = s
        q[j, j] = c
    return q


def _rotate_lora_state(
    *,
    reference_state: Dict[str, torch.Tensor],
    pairs: Sequence[Tuple[str, str, str]],
    q: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    rotated: Dict[str, torch.Tensor] = {}
    q_t = q.t()
    for name, tensor in reference_state.items():
        rotated[name] = tensor.detach().clone().requires_grad_(True)
    for _, name_a, name_b in pairs:
        a = reference_state[name_a]
        b = reference_state[name_b]
        q_local = q.to(device=a.device, dtype=a.dtype)
        q_local_t = q_t.to(device=b.device, dtype=b.dtype)
        rotated[name_a] = (q_local @ a).detach().clone().requires_grad_(True)
        rotated[name_b] = (b @ q_local_t).detach().clone().requires_grad_(True)
    return rotated


def _parameter_distance(
    left: Dict[str, torch.Tensor],
    right: Dict[str, torch.Tensor],
) -> float:
    with torch.no_grad():
        sq = 0.0
        for name in sorted(left.keys()):
            diff = left[name].detach() - right[name].detach()
            sq += float(diff.pow(2).sum().item())
    return math.sqrt(max(sq, 0.0))


def _flatten_difference(
    left: Dict[str, torch.Tensor],
    right: Dict[str, torch.Tensor],
) -> np.ndarray:
    chunks = []
    for name in sorted(left.keys()):
        diff = (left[name].detach() - right[name].detach()).reshape(-1).float().cpu().numpy()
        chunks.append(diff)
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)


def _evaluate_train_loss_and_logits(
    *,
    model,
    params: Dict[str, torch.Tensor],
    theta: torch.Tensor,
    batches: Sequence[Dict[str, object]],
    use_bf16: bool,
    reference_logits: Sequence[torch.Tensor] | None = None,
) -> Tuple[float, float]:
    losses: List[float] = []
    max_logit_abs_diff = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(batches):
            train_loss, _ = weighted_source_loss(model, params, theta, batch, use_bf16=use_bf16)
            losses.append(float(train_loss.detach().cpu()))
            if reference_logits is not None:
                logits = functional_logits(model, params, batch, use_bf16=use_bf16)
                diff = float((logits.detach() - reference_logits[batch_idx]).abs().max().item())
                max_logit_abs_diff = max(max_logit_abs_diff, diff)
    return float(sum(losses) / max(1, len(losses))), float(max_logit_abs_diff)


def _reference_logits(
    *,
    model,
    params: Dict[str, torch.Tensor],
    batches: Sequence[Dict[str, object]],
    use_bf16: bool,
) -> List[torch.Tensor]:
    logits: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches:
            logits.append(functional_logits(model, params, batch, use_bf16=use_bf16).detach().clone())
    return logits


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    cli_args = _parse_args()
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the LoRA manifold diagnostic.")

    run_dir = Path(cli_args.run_dir).resolve()
    out_dir = _resolve_output_dir(run_dir, cli_args.checkpoint, cli_args.output_dir)
    run_args = _load_run_config(run_dir)
    metrics = _load_metrics(run_dir)

    # Run the diagnostic in eval/fp32 mode by default to minimize numerical noise.
    run_args.bf16 = bool(cli_args.eval_bf16)

    model, tokenizer = load_model_and_tokenizer(run_args, device)
    model.eval()

    theta = _load_theta(run_dir, cli_args.checkpoint, metrics, device)
    state_cpu = _load_adapter_state(run_dir, cli_args.checkpoint)
    reference_state = _move_state_to_model(model, state_cpu)
    pairs = _find_lora_pairs(reference_state)
    rank = _common_lora_rank(reference_state, pairs)

    per_source_batch = int(cli_args.per_source_batch) if int(cli_args.per_source_batch) > 0 else int(run_args.per_source_batch)
    train_batches = _prepare_instruction_batches(
        args=run_args,
        tokenizer=tokenizer,
        device=device,
        batch_count=int(cli_args.batch_count),
        per_source_batch=per_source_batch,
    )

    reference_loss, _ = _evaluate_train_loss_and_logits(
        model=model,
        params=reference_state,
        theta=theta,
        batches=train_batches,
        use_bf16=bool(run_args.bf16),
        reference_logits=None,
    )
    reference_logits = _reference_logits(
        model=model,
        params=reference_state,
        batches=train_batches,
        use_bf16=bool(run_args.bf16),
    )

    path_planes = _rotation_planes(rank, int(cli_args.path_plane_count))
    path_angles = np.linspace(0.0, float(cli_args.max_angle), max(2, int(cli_args.num_path_points)))
    path_rows: List[Dict[str, object]] = []
    path_losses: List[float] = []
    path_logit_diffs: List[float] = []
    path_distances: List[float] = []
    for idx, angle in enumerate(path_angles):
        q = _rotation_matrix(rank, float(angle), path_planes)
        rotated_state = _rotate_lora_state(reference_state=reference_state, pairs=pairs, q=q)
        train_loss, max_logit_abs_diff = _evaluate_train_loss_and_logits(
            model=model,
            params=rotated_state,
            theta=theta,
            batches=train_batches,
            use_bf16=bool(run_args.bf16),
            reference_logits=reference_logits,
        )
        param_distance = _parameter_distance(rotated_state, reference_state)
        path_losses.append(train_loss)
        path_logit_diffs.append(max_logit_abs_diff)
        path_distances.append(param_distance)
        path_rows.append(
            {
                "path_index": idx,
                "angle_rad": float(angle),
                "train_loss": float(train_loss),
                "train_loss_diff_from_reference": float(train_loss - reference_loss),
                "max_logit_abs_diff": float(max_logit_abs_diff),
                "param_distance_from_reference": float(param_distance),
            }
        )
    _write_csv(out_dir / "path_metrics.csv", path_rows)

    generator_planes = _rotation_planes(rank, int(cli_args.num_generators))
    generator_rows: List[Dict[str, object]] = []
    displacement_rows: List[np.ndarray] = []
    for gen_idx, plane in enumerate(generator_planes):
        q = _rotation_matrix(rank, float(cli_args.generator_eps), [plane])
        rotated_state = _rotate_lora_state(reference_state=reference_state, pairs=pairs, q=q)
        train_loss, max_logit_abs_diff = _evaluate_train_loss_and_logits(
            model=model,
            params=rotated_state,
            theta=theta,
            batches=train_batches,
            use_bf16=bool(run_args.bf16),
            reference_logits=reference_logits,
        )
        displacement = _flatten_difference(rotated_state, reference_state)
        displacement_rows.append(displacement)
        generator_rows.append(
            {
                "generator_index": gen_idx,
                "plane_i": int(plane[0]),
                "plane_j": int(plane[1]),
                "angle_rad": float(cli_args.generator_eps),
                "train_loss": float(train_loss),
                "train_loss_diff_from_reference": float(train_loss - reference_loss),
                "max_logit_abs_diff": float(max_logit_abs_diff),
                "param_distance_from_reference": float(np.linalg.norm(displacement)),
            }
        )
    _write_csv(out_dir / "generator_metrics.csv", generator_rows)

    if displacement_rows:
        displacement_matrix = np.stack(displacement_rows, axis=0)
        singular_values = np.linalg.svd(displacement_matrix, compute_uv=False)
    else:
        singular_values = np.zeros((0,), dtype=np.float64)
    if singular_values.size > 0:
        tol = max(1e-12, float(singular_values[0]) * 1e-6)
        estimated_rank = int(np.sum(singular_values > tol))
    else:
        estimated_rank = 0

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(cli_args.checkpoint),
        "diagnostic_type": "lora_connected_lower_level_manifold",
        "reference": {
            "num_lora_pairs": int(len(pairs)),
            "lora_rank": int(rank),
            "batch_count": int(len(train_batches)),
            "per_source_batch": int(per_source_batch),
            "reference_train_loss": float(reference_loss),
        },
        "path_probe": {
            "plane_count": int(len(path_planes)),
            "angles_rad": [float(x) for x in path_angles],
            "train_loss_min": float(min(path_losses)) if path_losses else float("nan"),
            "train_loss_max": float(max(path_losses)) if path_losses else float("nan"),
            "train_loss_range": float(max(path_losses) - min(path_losses)) if path_losses else float("nan"),
            "max_logit_abs_diff": float(max(path_logit_diffs)) if path_logit_diffs else float("nan"),
            "max_param_distance": float(max(path_distances)) if path_distances else float("nan"),
        },
        "local_dimension_probe": {
            "generator_count": int(len(generator_planes)),
            "generator_eps": float(cli_args.generator_eps),
            "estimated_rank": int(estimated_rank),
            "singular_values": [float(x) for x in singular_values.tolist()],
            "max_train_loss_abs_diff": float(max(abs(row["train_loss_diff_from_reference"]) for row in generator_rows)) if generator_rows else float("nan"),
            "max_logit_abs_diff": float(max(row["max_logit_abs_diff"] for row in generator_rows)) if generator_rows else float("nan"),
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
