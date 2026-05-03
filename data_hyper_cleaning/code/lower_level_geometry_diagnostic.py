"""
Probe whether the lower-level solution set in the MNIST data-hyper-cleaning task
looks locally non-singleton, low-dimensional, and selection-sensitive.

The script:
1. Runs a short HG-MS outer optimization to obtain a representative outer weight
   vector x and an anchor lower-level solution y.
2. Freezes x and samples many nearby lower-level solutions by re-optimizing g
   from noisy copies of y.
3. Reports diagnostics on the resulting near-minimizer cloud:
   - non-singletonness and train/validation spread
   - approximate tangent dimension from PCA
   - straight-line low-loss path barriers as a connectivity proxy
   - Hessian Rayleigh quotients of g and f along estimated tangent vs normal
     directions

This does not prove the theorem assumptions. It provides empirical evidence for
or against the local manifold picture in the actual ML task used in the paper.
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ITD_minima_selection import (  # noqa: E402
    DEFAULT_ITD_CONFIG,
    _advance_branches_adamw,
    _clone_leaf_list,
    _init_adamw_state,
    _logspace,
    compute_acc_from_params,
    device,
    f_loss,
    forward_with_params,
    g_loss,
    init_params,
    prepare_datasets,
    run_ITD_Sel,
)


def _flatten_params(params: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params], dim=0)


def _unflatten_like(vec: np.ndarray, template: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    offset = 0
    for p in template:
        n = int(p.numel())
        chunk = vec[offset:offset + n]
        tensor = torch.from_numpy(chunk).to(device=p.device, dtype=p.dtype).view_as(p)
        out.append(tensor)
        offset += n
    if offset != int(vec.shape[0]):
        raise ValueError("Vector length does not match parameter template.")
    return out


def _param_distance(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> float:
    with torch.no_grad():
        return float(
            math.sqrt(
                sum((pa.detach() - pb.detach()).pow(2).sum().item() for pa, pb in zip(a, b))
            )
        )


def _logit_vector(params: Sequence[torch.Tensor], X: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = forward_with_params(X, list(params)).detach().cpu().numpy()
    return logits.reshape(-1)


def _candidate_metrics(
    *,
    idx: int,
    label: str,
    params: Sequence[torch.Tensor],
    anchor_params: Sequence[torch.Tensor],
    x: torch.Tensor,
    tr,
    val,
    test,
    reg_value: float,
    anchor_logits: np.ndarray,
) -> Dict[str, Any]:
    with torch.no_grad():
        w = torch.sigmoid(x)
        g_train = float(g_loss(list(params), w, tr, reg_value).item())
        val_loss = float(f_loss(list(params), val).item())
        train_acc = float(compute_acc_from_params(list(params), tr.data, tr.clean_target))
        test_acc = float(compute_acc_from_params(list(params), test.data, test.clean_target))
        param_dist = _param_distance(params, anchor_params)
        val_logits = _logit_vector(params, val.data)
        val_logit_dist = float(np.linalg.norm(val_logits - anchor_logits))
        param_vec = _flatten_params(params).cpu().numpy()
    return {
        "id": int(idx),
        "label": label,
        "params": [p.detach().clone() for p in params],
        "param_vec": param_vec,
        "val_logits": val_logits,
        "g_train": g_train,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "param_dist_to_anchor": param_dist,
        "val_logit_dist_to_anchor": val_logit_dist,
    }


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _estimate_tangent_rank(eigvals: np.ndarray, max_rank: int, min_ratio: float) -> int:
    if eigvals.size == 0:
        return 0
    total = float(np.sum(eigvals))
    if total <= 0.0:
        return 0
    ratios = eigvals / total
    keep = int(np.sum(ratios >= float(min_ratio)))
    if keep <= 0 and eigvals[0] > 0.0:
        keep = 1
    return int(min(max_rank, keep, eigvals.size))


def _participation_ratio(eigvals: np.ndarray) -> float:
    total = float(np.sum(eigvals))
    sq_total = float(np.sum(eigvals ** 2))
    if total <= 0.0 or sq_total <= 0.0:
        return 0.0
    return (total ** 2) / sq_total


def _pairwise_distances(vecs: np.ndarray) -> np.ndarray:
    diffs = vecs[:, None, :] - vecs[None, :, :]
    return np.sqrt(np.maximum(np.sum(diffs * diffs, axis=-1), 0.0))


def _knn_connected_components(vecs: np.ndarray, k: int) -> int:
    n = int(vecs.shape[0])
    if n <= 1:
        return n
    k = max(1, min(int(k), n - 1))
    dists = _pairwise_distances(vecs)
    adj = [set() for _ in range(n)]
    for i in range(n):
        nbrs = np.argsort(dists[i])[1:k + 1]
        for j in nbrs:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    seen = set()
    components = 0
    for i in range(n):
        if i in seen:
            continue
        components += 1
        stack = [i]
        seen.add(i)
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return int(components)


def _grad_norm(loss_builder, params: Sequence[torch.Tensor]) -> float:
    leaf_params = [p.detach().clone().requires_grad_(True) for p in params]
    loss = loss_builder(leaf_params)
    grads = torch.autograd.grad(loss, leaf_params, create_graph=False, retain_graph=False)
    sq = sum(g.detach().pow(2).sum().item() for g in grads)
    return float(math.sqrt(max(sq, 0.0)))


def _rayleigh(loss_builder, params: Sequence[torch.Tensor], vec: Sequence[torch.Tensor]) -> float:
    leaf_params = [p.detach().clone().requires_grad_(True) for p in params]
    vec_detached = [v.detach() for v in vec]
    loss = loss_builder(leaf_params)
    grad = torch.autograd.grad(loss, leaf_params, create_graph=True, retain_graph=True)
    dot = sum((g * v).sum() for g, v in zip(grad, vec_detached))
    hvp = torch.autograd.grad(dot, leaf_params, create_graph=False, retain_graph=False)
    num = sum((v * h.detach()).sum().item() for v, h in zip(vec_detached, hvp))
    den = sum(v.pow(2).sum().item() for v in vec_detached)
    return float(num / max(den, 1e-12))


def _random_orthonormal_rows(
    basis_rows: np.ndarray,
    *,
    count: int,
    dim: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows: List[np.ndarray] = []
    existing = []
    if basis_rows.size > 0:
        existing.extend([row.copy() for row in basis_rows])
    for _ in range(int(count)):
        vec = rng.standard_normal(dim)
        for row in existing:
            vec -= np.dot(row, vec) * row
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            continue
        vec /= norm
        rows.append(vec.copy())
        existing.append(vec.copy())
    if not rows:
        return np.zeros((0, dim), dtype=np.float64)
    return np.stack(rows, axis=0)


def _linear_path_barrier(
    params_a: Sequence[torch.Tensor],
    params_b: Sequence[torch.Tensor],
    *,
    x: torch.Tensor,
    tr,
    reg_value: float,
    num_points: int,
) -> Dict[str, float]:
    ts = np.linspace(0.0, 1.0, int(num_points))
    losses: List[float] = []
    with torch.no_grad():
        w = torch.sigmoid(x)
        end_a = float(g_loss(list(params_a), w, tr, reg_value).item())
        end_b = float(g_loss(list(params_b), w, tr, reg_value).item())
        for t in ts:
            params_t = [(1.0 - float(t)) * pa.detach() + float(t) * pb.detach() for pa, pb in zip(params_a, params_b)]
            losses.append(float(g_loss(params_t, w, tr, reg_value).item()))
    endpoint = max(end_a, end_b)
    peak = max(losses) if losses else endpoint
    return {
        "endpoint_max": float(endpoint),
        "path_peak": float(peak),
        "barrier": float(peak - endpoint),
    }


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_train_val_scatter(path: str, samples: List[Dict[str, Any]]) -> None:
    plt.figure()
    near = [s for s in samples if s.get("is_near")]
    far = [s for s in samples if not s.get("is_near")]
    if far:
        plt.scatter(
            [s["g_train"] for s in far],
            [s["val_loss"] for s in far],
            marker="o",
            label="all samples",
        )
    if near:
        plt.scatter(
            [s["g_train"] for s in near],
            [s["val_loss"] for s in near],
            marker="x",
            label="near-minimizers",
        )
    selected = [s for s in samples if s.get("is_selected")]
    if selected:
        plt.scatter(
            [selected[0]["g_train"]],
            [selected[0]["val_loss"]],
            marker="*",
            s=180,
            label="selected",
        )
    plt.xlabel("Lower-level train loss g")
    plt.ylabel("Validation loss f")
    plt.title("Near-minimizer selection diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_pca_spectrum(path: str, explained: np.ndarray) -> None:
    plt.figure()
    xs = np.arange(1, int(explained.size) + 1)
    plt.plot(xs, explained, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.title("Near-minimizer PCA spectrum")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_path_barriers(path: str, barriers: List[Dict[str, float]]) -> None:
    if not barriers:
        return
    plt.figure()
    xs = np.arange(1, len(barriers) + 1)
    plt.plot(xs, [b["endpoint_max"] for b in barriers], marker="o", label="endpoint max")
    plt.plot(xs, [b["path_peak"] for b in barriers], marker="x", label="path peak")
    plt.xlabel("Interpolated pair")
    plt.ylabel("Train loss g")
    plt.title("Straight-line path barriers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _sample_candidates(
    *,
    anchor_params: Sequence[torch.Tensor],
    anchor_x: torch.Tensor,
    tr,
    val,
    test,
    args,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    anchor_logits = _logit_vector(anchor_params, val.data)
    samples.append(
        _candidate_metrics(
            idx=0,
            label="anchor",
            params=anchor_params,
            anchor_params=anchor_params,
            x=anchor_x,
            tr=tr,
            val=val,
            test=test,
            reg_value=float(args.reg),
            anchor_logits=anchor_logits,
        )
    )

    lr_mults = _logspace(args.sample_lr_mult_min, args.sample_lr_mult_max, args.num_samples)
    for i in range(int(args.num_samples)):
        sample_seed = int(args.seed) + 10000 + i
        torch.manual_seed(sample_seed)
        branch = _clone_leaf_list(anchor_params, requires_grad=True)
        if float(args.sample_init_noise_std) > 0.0:
            with torch.no_grad():
                for p in branch:
                    p.add_(torch.randn_like(p) * float(args.sample_init_noise_std))
        state = _init_adamw_state(branch)
        new_branches, _ = _advance_branches_adamw(
            [branch],
            [state],
            x=anchor_x,
            tr=tr,
            steps=int(args.sample_steps),
            lr=float(args.sample_lr) * float(lr_mults[i] if i < len(lr_mults) else 1.0),
            reg_value=float(args.reg),
            batch_size=int(args.sample_batch_size),
            beta1=float(args.branch_beta1),
            beta2=float(args.branch_beta2),
            eps=float(args.branch_eps),
            weight_decay=float(args.branch_weight_decay),
        )
        samples.append(
            _candidate_metrics(
                idx=i + 1,
                label=f"sample_{i + 1}",
                params=new_branches[0],
                anchor_params=anchor_params,
                x=anchor_x,
                tr=tr,
                val=val,
                test=test,
                reg_value=float(args.reg),
                anchor_logits=anchor_logits,
            )
        )
    return samples


def _prepare_anchor(args):
    tr, val, test, mask_clean, input_dim = prepare_datasets(args.seed, args.pollute_rate)
    if str(args.anchor_source).lower() == "local":
        anchor_x = torch.zeros(len(tr.data), device=device)
        candidates: List[Dict[str, Any]] = []
        anchor_logits = None
        for i in range(int(args.anchor_local_restarts)):
            params = init_params(
                input_dim,
                seed_val=int(args.seed) + 5000 + i,
                hidden=int(args.hidden),
                num_classes=int(args.num_classes),
                scale=float(args.init_scale),
                device=device,
            )
            state = _init_adamw_state(params)
            new_branches, _ = _advance_branches_adamw(
                [params],
                [state],
                x=anchor_x,
                tr=tr,
                steps=int(args.anchor_local_steps),
                lr=float(args.anchor_local_lr),
                reg_value=float(args.reg),
                batch_size=int(args.anchor_local_batch_size),
                beta1=float(args.branch_beta1),
                beta2=float(args.branch_beta2),
                eps=float(args.branch_eps),
                weight_decay=float(args.branch_weight_decay),
            )
            if anchor_logits is None:
                anchor_logits = _logit_vector(new_branches[0], val.data)
            candidates.append(
                _candidate_metrics(
                    idx=i,
                    label=f"anchor_restart_{i}",
                    params=new_branches[0],
                    anchor_params=new_branches[0],
                    x=anchor_x,
                    tr=tr,
                    val=val,
                    test=test,
                    reg_value=float(args.reg),
                    anchor_logits=anchor_logits,
                )
            )
        anchor = min(candidates, key=lambda s: s["g_train"])
        anchor_metrics = {
            "variant": "local",
            "g_train": float(anchor["g_train"]),
            "val_loss": float(anchor["val_loss"]),
            "train_acc": float(anchor["train_acc"]),
            "test_acc": float(anchor["test_acc"]),
        }
        anchor_params = [p.detach().clone().to(device=device) for p in anchor["params"]]
        return tr, val, test, mask_clean, input_dim, anchor_x, anchor_params, anchor_metrics

    payload = run_ITD_Sel(
        seed=int(args.seed),
        pollute_rate=float(args.pollute_rate),
        outer_itr=int(args.outer_itr),
        print_every=max(1, int(args.outer_itr)),
        D_inner_gd=int(args.outer_D_inner_gd),
        lr_inner_gd=float(args.outer_lr_inner_gd),
        M_candidates=int(args.outer_M_candidates),
        D_inner_sgld=int(args.outer_D_inner_sgld),
        lr_inner_sgld=float(args.outer_lr_inner_sgld),
        sgld_batch_size=int(args.outer_sgld_batch_size),
        persistent_chains=True,
        elitism_chain=True,
        include_base_candidate=True,
        minsel_backend=str(args.outer_minsel_backend),
        branch_init_noise_std=float(args.outer_branch_init_noise_std),
        branch_lr_mult_min=float(args.outer_branch_lr_mult_min),
        branch_lr_mult_max=float(args.outer_branch_lr_mult_max),
        branch_weight_decay=float(args.outer_branch_weight_decay),
        selection_min_improve=float(args.outer_selection_min_improve),
        selection_use_significance=bool(args.outer_selection_use_significance),
        selection_z=float(args.outer_selection_z),
        ms_every=int(args.outer_ms_every),
        lr_x_schedule=str(args.outer_lr_x_schedule),
        lr_x_final=float(args.outer_lr_x_final),
        lr_x_decay_steps=int(args.outer_lr_x_decay_steps),
        lr_x=float(args.outer_lr_x),
        x_clip=float(args.outer_x_clip),
        hyper_clip=float(args.outer_hyper_clip),
        reg=float(args.reg),
        cg_tol=float(args.outer_cg_tol),
        cg_max_it=int(args.outer_cg_max_it),
        cg_ridge=float(args.outer_cg_ridge),
        cg_batch_size=int(args.outer_cg_batch_size),
        hidden=int(args.hidden),
        num_classes=int(args.num_classes),
        init_scale=float(args.init_scale),
        verbose=bool(args.verbose_anchor),
        return_history=False,
        return_state=True,
        time_budget_s=float(args.outer_time_budget_s) if args.outer_time_budget_s else None,
    )
    anchor_state = payload["state"]
    anchor_metrics = payload["final_metrics"]
    anchor_x = anchor_state["x"].detach().clone().to(device=device)
    anchor_params = [p.detach().clone().to(device=device) for p in anchor_state["y"]]
    return tr, val, test, mask_clean, input_dim, anchor_x, anchor_params, anchor_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose lower-level geometry in MNIST data hyper-cleaning.")
    parser.add_argument("--seed", type=int, default=2424)
    parser.add_argument("--pollute-rate", type=float, default=0.8)
    parser.add_argument("--output-dir", type=str, default="paper_experiments/data_hyper_cleaning/geometry_outputs")
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--anchor-source", choices=["hgms", "local"], default="hgms")
    parser.add_argument("--hidden", type=int, default=300)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--init-scale", type=float, default=0.1)
    parser.add_argument("--reg", type=float, default=DEFAULT_ITD_CONFIG["reg"])

    parser.add_argument("--anchor-local-restarts", type=int, default=3)
    parser.add_argument("--anchor-local-steps", type=int, default=20)
    parser.add_argument("--anchor-local-lr", type=float, default=0.02)
    parser.add_argument("--anchor-local-batch-size", type=int, default=256)

    parser.add_argument("--outer-itr", type=int, default=60)
    parser.add_argument("--outer-time-budget-s", type=float, default=0.0)
    parser.add_argument("--outer-D-inner-gd", type=int, default=0)
    parser.add_argument("--outer-lr-inner-gd", type=float, default=0.005)
    parser.add_argument("--outer-M-candidates", type=int, default=8)
    parser.add_argument("--outer-D-inner-sgld", type=int, default=20)
    parser.add_argument("--outer-lr-inner-sgld", type=float, default=0.02)
    parser.add_argument("--outer-sgld-batch-size", type=int, default=256)
    parser.add_argument("--outer-minsel-backend", type=str, default="adamw")
    parser.add_argument("--outer-branch-init-noise-std", type=float, default=0.0)
    parser.add_argument("--outer-branch-lr-mult-min", type=float, default=1.0)
    parser.add_argument("--outer-branch-lr-mult-max", type=float, default=1.0)
    parser.add_argument("--outer-branch-weight-decay", type=float, default=0.0)
    parser.add_argument("--outer-selection-min-improve", type=float, default=0.01)
    parser.add_argument(
        "--outer-selection-use-significance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--outer-selection-z", type=float, default=1.96)
    parser.add_argument("--outer-ms-every", type=int, default=1)
    parser.add_argument("--outer-lr-x-schedule", type=str, default="inv")
    parser.add_argument("--outer-lr-x-final", type=float, default=0.01)
    parser.add_argument("--outer-lr-x-decay-steps", type=int, default=200)
    parser.add_argument("--outer-lr-x", type=float, default=0.1)
    parser.add_argument("--outer-x-clip", type=float, default=5.0)
    parser.add_argument("--outer-hyper-clip", type=float, default=0.0)
    parser.add_argument("--outer-cg-tol", type=float, default=1e-3)
    parser.add_argument("--outer-cg-max-it", type=int, default=30)
    parser.add_argument("--outer-cg-ridge", type=float, default=1e-2)
    parser.add_argument("--outer-cg-batch-size", type=int, default=0)
    parser.add_argument("--verbose-anchor", action="store_true")

    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--sample-steps", type=int, default=80)
    parser.add_argument("--sample-lr", type=float, default=0.02)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--sample-init-noise-std", type=float, default=1e-3)
    parser.add_argument("--sample-lr-mult-min", type=float, default=0.7)
    parser.add_argument("--sample-lr-mult-max", type=float, default=1.3)
    parser.add_argument("--branch-beta1", type=float, default=0.9)
    parser.add_argument("--branch-beta2", type=float, default=0.999)
    parser.add_argument("--branch-eps", type=float, default=1e-8)
    parser.add_argument("--branch-weight-decay", type=float, default=0.0)

    parser.add_argument("--near-tol-abs", type=float, default=5e-4)
    parser.add_argument("--near-tol-rel", type=float, default=1e-3)
    parser.add_argument("--min-near-count", type=int, default=8)
    parser.add_argument("--max-tangent-dim", type=int, default=6)
    parser.add_argument("--pca-min-ratio", type=float, default=0.05)
    parser.add_argument("--interp-points", type=int, default=9)
    parser.add_argument("--max-barrier-paths", type=int, default=6)
    parser.add_argument("--knn-k", type=int, default=3)

    args = parser.parse_args()

    out_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[geometry] anchor_source={args.anchor_source} output_dir={out_dir}")
    tr, val, test, _, _, anchor_x, anchor_params, anchor_metrics = _prepare_anchor(args)
    print("[geometry] anchor ready; sampling lower-level candidates")
    samples = _sample_candidates(
        anchor_params=anchor_params,
        anchor_x=anchor_x,
        tr=tr,
        val=val,
        test=test,
        args=args,
    )
    print("[geometry] candidate sampling complete; computing diagnostics")

    samples_sorted = sorted(samples, key=lambda s: (s["g_train"], s["val_loss"]))
    best_g = float(samples_sorted[0]["g_train"])
    near_tol = float(args.near_tol_abs) + float(args.near_tol_rel) * max(1.0, abs(best_g))
    actual_near_candidates = [s for s in samples_sorted if s["g_train"] <= best_g + near_tol]
    near_candidates = list(actual_near_candidates)
    near_candidates_padded = False
    if len(near_candidates) < int(args.min_near_count):
        near_candidates = samples_sorted[: min(len(samples_sorted), int(args.min_near_count))]
        near_candidates_padded = True

    selected = min(near_candidates, key=lambda s: s["val_loss"])
    second_best_val_gap = float("nan")
    if len(near_candidates) >= 2:
        sorted_by_val = sorted(near_candidates, key=lambda s: s["val_loss"])
        second_best_val_gap = float(sorted_by_val[1]["val_loss"] - sorted_by_val[0]["val_loss"])

    for s in samples:
        s["within_tol"] = bool(any(s["id"] == n["id"] for n in actual_near_candidates))
        s["is_near"] = bool(any(s["id"] == n["id"] for n in near_candidates))
        s["is_selected"] = bool(s["id"] == selected["id"])
        s["param_dist_to_selected"] = float(
            _param_distance(s["params"], selected["params"])
        )
        s["val_logit_dist_to_selected"] = float(
            np.linalg.norm(s["val_logits"] - selected["val_logits"])
        )

    near_vecs = np.stack([s["param_vec"] for s in near_candidates], axis=0)
    selected_vec = selected["param_vec"].copy()
    centered = near_vecs - selected_vec[None, :]
    if centered.shape[0] >= 2 and np.any(np.linalg.norm(centered, axis=1) > 0):
        _, sing_vals, vh = np.linalg.svd(centered, full_matrices=False)
        eigvals = (sing_vals ** 2) / max(1, centered.shape[0] - 1)
    else:
        sing_vals = np.zeros((0,), dtype=np.float64)
        eigvals = np.zeros((0,), dtype=np.float64)
        vh = np.zeros((0, centered.shape[1]), dtype=np.float64)
    explained = eigvals / max(float(np.sum(eigvals)), 1e-12) if eigvals.size > 0 else np.zeros((0,), dtype=np.float64)
    tangent_rank = _estimate_tangent_rank(eigvals, int(args.max_tangent_dim), float(args.pca_min_ratio))
    tangent_basis = vh[:tangent_rank] if tangent_rank > 0 else np.zeros((0, centered.shape[1]), dtype=np.float64)
    normal_basis = _random_orthonormal_rows(
        tangent_basis,
        count=max(1, tangent_rank),
        dim=int(centered.shape[1]),
        seed=int(args.seed) + 7000,
    )

    selected_params = [p.detach().clone() for p in selected["params"]]
    g_builder = lambda params: g_loss(params, torch.sigmoid(anchor_x).detach(), tr, float(args.reg))
    f_builder = lambda params: f_loss(params, val)

    tangent_g_rq: List[float] = []
    tangent_f_rq: List[float] = []
    for row in tangent_basis:
        vec = _unflatten_like(row.astype(np.float32, copy=False), selected_params)
        tangent_g_rq.append(_rayleigh(g_builder, selected_params, vec))
        tangent_f_rq.append(_rayleigh(f_builder, selected_params, vec))

    normal_g_rq: List[float] = []
    normal_f_rq: List[float] = []
    for row in normal_basis:
        vec = _unflatten_like(row.astype(np.float32, copy=False), selected_params)
        normal_g_rq.append(_rayleigh(g_builder, selected_params, vec))
        normal_f_rq.append(_rayleigh(f_builder, selected_params, vec))

    near_dists = _pairwise_distances(near_vecs)
    upper_tri = near_dists[np.triu_indices_from(near_dists, k=1)] if near_dists.size > 1 else np.zeros((0,), dtype=np.float64)
    val_losses_near = np.asarray([s["val_loss"] for s in near_candidates], dtype=np.float64)
    g_losses_near = np.asarray([s["g_train"] for s in near_candidates], dtype=np.float64)

    path_targets = sorted(
        [s for s in near_candidates if s["id"] != selected["id"]],
        key=lambda s: s["param_dist_to_selected"],
        reverse=True,
    )[: int(args.max_barrier_paths)]
    path_barriers: List[Dict[str, float]] = []
    for s in path_targets:
        barrier = _linear_path_barrier(
            selected["params"],
            s["params"],
            x=anchor_x,
            tr=tr,
            reg_value=float(args.reg),
            num_points=int(args.interp_points),
        )
        barrier["candidate_id"] = int(s["id"])
        path_barriers.append(barrier)

    csv_rows: List[Dict[str, Any]] = []
    for s in samples:
        csv_rows.append(
            {
                "id": int(s["id"]),
                "label": s["label"],
                "within_tol": int(bool(s["within_tol"])),
                "is_near": int(bool(s["is_near"])),
                "is_selected": int(bool(s["is_selected"])),
                "g_train": float(s["g_train"]),
                "val_loss": float(s["val_loss"]),
                "train_acc": float(s["train_acc"]),
                "test_acc": float(s["test_acc"]),
                "param_dist_to_anchor": float(s["param_dist_to_anchor"]),
                "param_dist_to_selected": float(s["param_dist_to_selected"]),
                "val_logit_dist_to_anchor": float(s["val_logit_dist_to_anchor"]),
                "val_logit_dist_to_selected": float(s["val_logit_dist_to_selected"]),
            }
        )
    _write_csv(os.path.join(out_dir, "candidate_metrics.csv"), csv_rows)

    _plot_train_val_scatter(os.path.join(out_dir, "train_vs_val_scatter.png"), samples)
    _plot_pca_spectrum(os.path.join(out_dir, "pca_spectrum.png"), explained[: max(1, min(12, explained.size))])
    _plot_path_barriers(os.path.join(out_dir, "path_barriers.png"), path_barriers)

    summary = {
        "config": vars(args),
        "anchor_metrics": anchor_metrics,
        "sampling": {
            "num_total_samples": int(len(samples)),
            "num_within_tolerance": int(len(actual_near_candidates)),
            "num_near_candidates": int(len(near_candidates)),
            "near_candidates_padded": bool(near_candidates_padded),
            "best_train_loss": float(best_g),
            "near_tolerance": float(near_tol),
        },
        "non_singleton_proxy": {
            "pairwise_param_distance_mean": float(np.mean(upper_tri)) if upper_tri.size else 0.0,
            "pairwise_param_distance_max": float(np.max(upper_tri)) if upper_tri.size else 0.0,
            "selected_param_distance_to_anchor": float(selected["param_dist_to_anchor"]),
            "val_loss_range_near": float(np.max(val_losses_near) - np.min(val_losses_near)) if val_losses_near.size else 0.0,
            "train_loss_range_near": float(np.max(g_losses_near) - np.min(g_losses_near)) if g_losses_near.size else 0.0,
            "selected_vs_best_train_gap": float(selected["g_train"] - best_g),
        },
        "selection_proxy": {
            "selected_id": int(selected["id"]),
            "selected_val_loss": float(selected["val_loss"]),
            "selected_train_loss": float(selected["g_train"]),
            "best_vs_second_best_val_gap_near": float(second_best_val_gap),
            "selected_test_acc": float(selected["test_acc"]),
            "val_train_corr_near": float(np.corrcoef(g_losses_near, val_losses_near)[0, 1]) if len(near_candidates) >= 2 and np.std(g_losses_near) > 0 and np.std(val_losses_near) > 0 else float("nan"),
        },
        "pca_proxy": {
            "tangent_rank": int(tangent_rank),
            "participation_ratio": float(_participation_ratio(eigvals)),
            "singular_values": sing_vals[: int(args.max_tangent_dim) + 4],
            "explained_variance_ratio": explained[: int(args.max_tangent_dim) + 4],
        },
        "connectivity_proxy": {
            "knn_components": int(_knn_connected_components(near_vecs, int(args.knn_k))),
            "path_barrier_mean": float(np.mean([b["barrier"] for b in path_barriers])) if path_barriers else 0.0,
            "path_barrier_max": float(np.max([b["barrier"] for b in path_barriers])) if path_barriers else 0.0,
            "path_barriers": path_barriers,
        },
        "curvature_proxy": {
            "selected_g_grad_norm": float(_grad_norm(g_builder, selected_params)),
            "selected_f_grad_norm": float(_grad_norm(f_builder, selected_params)),
            "g_tangent_rayleigh": tangent_g_rq,
            "g_normal_rayleigh": normal_g_rq,
            "f_tangent_rayleigh": tangent_f_rq,
            "f_normal_rayleigh": normal_f_rq,
            "g_tangent_abs_max": float(max((abs(v) for v in tangent_g_rq), default=0.0)),
            "g_normal_min": float(min(normal_g_rq)) if normal_g_rq else float("nan"),
            "f_tangent_min": float(min(tangent_f_rq)) if tangent_f_rq else float("nan"),
        },
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, indent=2)

    print(f"Wrote diagnostic summary to {summary_path}")
    print(
        "Near-minimizers:",
        len(near_candidates),
        "| tangent_rank:",
        tangent_rank,
        "| val gap:",
        second_best_val_gap,
        "| max path barrier:",
        summary["connectivity_proxy"]["path_barrier_max"],
    )


if __name__ == "__main__":
    main()
