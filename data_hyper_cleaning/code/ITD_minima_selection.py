import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import time
import random
import math
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms

from gpu_utils import maybe_print_device_info, resolve_device

# =========================================================
# Default hyperparameters (can be overridden per call)
# =========================================================
DEFAULT_ITD_CONFIG: Dict[str, Any] = {
    "seed": 2424,
    "pollute_rate": 0.8,
    "outer_itr": 1000,
    "print_every": 10,
    # Optional "polish" GD steps on g after selecting a chain state.
    # The HG-MinSel-LMC pseudocode does not require this (set 0 to match it),
    # but a few cheap steps can reduce CG noise if needed.
    "D_inner_gd": 0,
    "lr_inner_gd": 0.005,
    "inner_reg": 0.0,
    # ---- HG-MinSel-LMC: minima selection via persistent parallel LMC chains ----
    # Number of parallel chains (m). More chains explores more basins but costs more.
    "M_candidates": 8,
    # LMC steps per outer iteration (K). Higher improves exploration but costs more.
    "D_inner_sgld": 20,
    # LMC step-size (h). Keep modest for stability.
    "lr_inner_sgld": 0.02,
    # LMC temperature (lambda). Setting 0 disables injected noise.
    # With minibatch gradients (sgld_batch_size < n_train), there is still stochasticity.
    "sgld_temp": 0.0,
    # If < n_train, mini-batching injects stochasticity even when sgld_temp==0.
    # (This matches the common "LMC with minibatch gradient noise" setup.)
    "sgld_batch_size": 256,
    # Persistent chains (X^{(0)}_{t+1,i} <- X_{t,i}).
    "persistent_chains": True,
    # Keep one "elite" chain equal to the selected best state each iteration.
    # This acts like a safe anchor and reduces drift.
    "elitism_chain": True,
    # Always include the current params as a candidate (acts as a safe fallback).
    "include_base_candidate": True,
    # Minima selection backend:
    # - "adamw": persistent parallel AdamW branches optimized on g (recommended; matches imbalance task)
    # - "lmc":   previous persistent LMC/ULA chains (kept for ablations)
    "minsel_backend": "adamw",
    # AdamW branch hyperparameters (used when minsel_backend=="adamw")
    "branch_beta1": 0.9,
    "branch_beta2": 0.999,
    "branch_eps": 1e-8,
    "branch_weight_decay": 0.0,
    "branch_init_noise_std": 0.0,
    "branch_lr_mult_min": 1.0,
    "branch_lr_mult_max": 1.0,
    # Prefer the base candidate unless a SGLD candidate improves val_loss by at least this amount.
    # This helps reduce "winner's curse" overfitting from repeatedly picking the best of M on a small val set.
    "selection_min_improve": 0.01,
    # When val is tiny (here: 50), pure argmin selection can be noisy and sometimes picks a "lucky"
    # candidate that improves val but hurts test. This adds a conservative paired-difference gate:
    # accept a non-base candidate only if its mean val-loss improvement over base is both:
    #   (1) >= selection_min_improve  and  (2) >= selection_z * std(diff)/sqrt(n_val)
    "selection_use_significance": True,
    "selection_z": 1.96,
    # Run the (expensive) minima-selection step every N outer iterations.
    # Set to 1 to do minima selection every iteration (original behavior).
    "ms_every": 1,
    # Outer (hyper) step-size schedule for x.
    # - constant: fixed lr_x
    # - cosine: cosine decay from lr_x to lr_x_final over outer_itr
    # - linear: linear decay from lr_x to lr_x_final over outer_itr
    # - inv: lr_x / (1 + itr / lr_x_decay_steps)
    "lr_x_schedule": "inv",
    "lr_x_final": 0.01,
    "lr_x_decay_steps": 200,
    "lr_x": 0.1,
    "x_clip": 5.0,
    "hyper_clip": 0.0,
    "reg": 1e-4,
    # ---- Implicit hypergradient via CG on HVPs ----
    # We solve: (∇^2_{yy} g + cg_ridge*I) v = ∇_y f
    # using only Hessian-vector products, then:
    #   hypergrad_x = ∇_x f - ∇^2_{xy} g · v
    # In this problem f does not depend directly on x, so hypergrad_x ≈ -∇^2_{xy} g · v.
    "cg_tol": 1e-3,
    "cg_max_it": 30,
    # Ridge (gamma) improves conditioning and makes the linear system closer to SPD.
    "cg_ridge": 1e-2,
    # Batch size for CG/HVP computations on g. 0 => full batch (recommended for stability).
    "cg_batch_size": 0,
    "hidden": 300,
    "num_classes": 10,
    "init_scale": 0.1,
}

def get_default_itd_config() -> Dict[str, Any]:
    return DEFAULT_ITD_CONFIG.copy()

# Device
device = resolve_device()

# =========================================================
# Data utilities
# =========================================================
class Dataset:
    def __init__(self, data, target):
        self.data = data.float() / 255.0
        self.clean_target = target.clone()
        self.dirty_target = None
        self.clean_mask   = torch.ones(len(target), dtype=torch.float32)

    def pollute(self, rho):
        n = len(self.clean_target)
        idxs = list(range(n))
        random.shuffle(idxs)
        self.dirty_target = self.clean_target.clone()
        for i in idxs[:int(rho * n)]:
            choices = list(range(10))
            choices.remove(int(self.clean_target[i]))
            self.dirty_target[i] = random.choice(choices)
            self.clean_mask[i] = 0.0

    def flatten(self):
        self.data = self.data.view(self.data.size(0), -1)

    def to(self, device):
        self.data         = self.data.to(device)
        self.clean_target = self.clean_target.to(device)
        if self.dirty_target is not None:
            self.dirty_target = self.dirty_target.to(device)
        self.clean_mask   = self.clean_mask.to(device)

def split_data(dataset, train_n, val_n, test_n):
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    data, labels = dataset.data, dataset.targets
    return (
        Dataset(data[idxs[:train_n]], labels[idxs[:train_n]]),
        Dataset(data[idxs[train_n:train_n+val_n]], labels[idxs[train_n:train_n+val_n]]),
        Dataset(data[idxs[train_n+val_n:train_n+val_n+test_n]], labels[idxs[train_n+val_n:train_n+val_n+test_n]])
    )

# =========================================================
# Functional model (2-layer MLP with Sigmoid, to match V/G-PBGD and IAPTT-GM)
# =========================================================
def init_params(input_dim, seed_val, hidden=300, num_classes=10, scale=0.1, device='cpu'):
    rng = torch.Generator(device=device).manual_seed(seed_val)
    W1 = torch.randn(hidden, input_dim, generator=rng, device=device) * scale
    b1 = torch.zeros(hidden, device=device)
    W2 = torch.randn(num_classes, hidden, generator=rng, device=device) * scale
    b2 = torch.zeros(num_classes, device=device)
    return [W1.requires_grad_(), b1.requires_grad_(), W2.requires_grad_(), b2.requires_grad_()]

def forward_with_params(X, params):
    W1, b1, W2, b2 = params
    h = torch.sigmoid(X @ W1.t() + b1)
    logits = h @ W2.t() + b2
    return logits

def params_l2(params):
    return sum(p.pow(2).sum() for p in params)

def clone_params(params, requires_grad=None):
    out = []
    for p in params:
        q = p.detach().clone()
        q.requires_grad_(p.requires_grad if requires_grad is None else requires_grad)
        out.append(q)
    return out

def compute_acc_from_params(params, X, y):
    with torch.no_grad():
        logits = forward_with_params(X, params)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item() * 100.0

def val_loss_params(params, X, y):
    logits = forward_with_params(X, params)
    return F.cross_entropy(logits, y)

def val_loss_per_example_params(params, X, y):
    logits = forward_with_params(X, params)
    return F.cross_entropy(logits, y, reduction="none")

# =========================================================
# Losses f and g
# =========================================================
def g_loss(params, x_weights_sigmoid, tr, reg_value):
    logits = forward_with_params(tr.data, params)
    ce = F.cross_entropy(logits, tr.dirty_target, reduction='none')
    return (x_weights_sigmoid * ce).mean() + reg_value * params_l2(params)

def f_loss(params, val):
    return val_loss_params(params, val.data, val.clean_target)

def f_loss_per_example(params, val):
    return val_loss_per_example_params(params, val.data, val.clean_target)

# =========================================================
# Differentiable inner unroll (ITD)
# =========================================================
def inner_unroll_gd(params0, x, tr, D, lr, reg_value, inner_reg_value):
    params = [p for p in params0]
    for _ in range(D):
        w = torch.sigmoid(x)
        loss = g_loss(params, w, tr, reg_value) + inner_reg_value * params_l2(params)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        params = [p - lr * g for p, g in zip(params, grads)]
    return params

# =========================================================
# Selection backends
# =========================================================
def _logspace(min_v: float, max_v: float, n: int) -> List[float]:
    n = int(n)
    if n <= 0:
        return []
    if n == 1:
        return [float(min_v)]
    if min_v <= 0 or max_v <= 0:
        return [1.0 for _ in range(n)]
    a, b = math.log10(float(min_v)), math.log10(float(max_v))
    return [10 ** (a + (b - a) * (i / (n - 1))) for i in range(n)]

def _init_adamw_state(params: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    state: List[Dict[str, torch.Tensor]] = []
    for p in params:
        state.append({
            "step": torch.tensor(0, device=p.device, dtype=torch.int64),
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.zeros_like(p),
        })
    return state

def _adamw_step_(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    state: List[Dict[str, torch.Tensor]],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    lr = float(lr)
    beta1 = float(beta1)
    beta2 = float(beta2)
    eps = float(eps)
    weight_decay = float(weight_decay)
    with torch.no_grad():
        for p, g, st in zip(params, grads, state):
            st["step"].add_(1)
            step_i = int(st["step"].item())
            exp_avg = st["exp_avg"].mul(beta1).add(g, alpha=(1.0 - beta1))
            exp_avg_sq = st["exp_avg_sq"].mul(beta2).addcmul(g, g, value=(1.0 - beta2))
            st["exp_avg"].copy_(exp_avg)
            st["exp_avg_sq"].copy_(exp_avg_sq)

            bias_c1 = 1.0 - (beta1 ** step_i)
            bias_c2 = 1.0 - (beta2 ** step_i)
            step_size = lr / max(bias_c1, 1e-12)
            denom = (exp_avg_sq / max(bias_c2, 1e-12)).sqrt().add_(eps)
            if weight_decay > 0.0:
                p.add_(p, alpha=-lr * weight_decay)
            p.addcdiv_(exp_avg, denom, value=-step_size)

def _advance_branches_adamw(
    branches: List[List[torch.Tensor]],
    branch_states: List[List[Dict[str, torch.Tensor]]],
    *,
    x: torch.Tensor,
    tr: "Dataset",
    steps: int,
    lr: float,
    reg_value: float,
    batch_size: int,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> Tuple[List[List[torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]:
    w_all = torch.sigmoid(x).detach()
    new_branches: List[List[torch.Tensor]] = []
    new_states: List[List[Dict[str, torch.Tensor]]] = []

    for params, state in zip(branches, branch_states):
        params = [p.detach().clone().requires_grad_(True) for p in params]
        state = [
            {
                "step": st["step"].detach().clone(),
                "exp_avg": st["exp_avg"].detach().clone(),
                "exp_avg_sq": st["exp_avg_sq"].detach().clone(),
            }
            for st in state
        ]

        for _ in range(int(steps)):
            loss = _g_loss_minibatch(
                params,
                x_sigmoid=w_all,
                tr=tr,
                reg_value=float(reg_value),
                batch_size=int(batch_size),
            )
            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
            _adamw_step_(
                params,
                list(grads),
                state,
                lr=float(lr),
                beta1=float(beta1),
                beta2=float(beta2),
                eps=float(eps),
                weight_decay=float(weight_decay),
            )

        new_branches.append([p.detach().clone().requires_grad_(True) for p in params])
        new_states.append(state)

    return new_branches, new_states

def sgld_candidates_from(
    params_base,
    x,
    tr,
    M,
    steps,
    lr,
    temp,
    reg_value,
    *,
    batch_size: Optional[int] = None,
    include_base: bool = True,
):
    n_train = int(tr.data.size(0))
    if batch_size is None:
        batch_size = n_train
    else:
        batch_size = int(batch_size)

    use_full_batch = (batch_size <= 0) or (batch_size >= n_train)
    # Avoid redundant work: if updates are deterministic, multiple candidates collapse.
    deterministic_updates = (temp is None or float(temp) <= 0.0) and use_full_batch
    if deterministic_updates:
        M = 1

    w_all = torch.sigmoid(x).detach()

    # NOTE: Candidate generation does NOT need higher-order derivatives.
    # Keep it cheap by updating candidate params in-place (no per-step cloning).
    base = [p.detach().clone() for p in params_base]
    candidates: List[List[torch.Tensor]] = []

    if include_base:
        candidates.append([p.detach().clone().requires_grad_(True) for p in base])

    for _ in range(int(M)):
        params = [p.detach().clone().requires_grad_(True) for p in base]

        for _ in range(int(steps)):
            if use_full_batch:
                loss = g_loss(params, w_all, tr, reg_value)
            else:
                idx = torch.randint(0, n_train, (batch_size,), device=tr.data.device)
                logits = forward_with_params(tr.data[idx], params)
                ce = F.cross_entropy(logits, tr.dirty_target[idx], reduction='none')
                loss = (w_all[idx] * ce).mean() + reg_value * params_l2(params)

            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)

            noise_scale = (2.0 * lr * temp) ** 0.5 if temp and temp > 0 else 0.0
            with torch.no_grad():
                for p, g in zip(params, grads):
                    p.add_(g, alpha=-lr)
                    if temp and temp > 0:
                        p.add_(torch.randn_like(p), alpha=noise_scale)

        candidates.append([p.detach().clone().requires_grad_(True) for p in params])

    return candidates

@torch.no_grad()
def select_by_validation_among(
    params_list,
    val,
    *,
    min_improve_over_base: float = 0.0,
    use_significance: bool = True,
    z_value: float = 1.96,
) -> Tuple[int, List[torch.Tensor], float]:
    if params_list is None or len(params_list) == 0:
        raise RuntimeError("No candidates provided for validation selection.")

    # Candidate 0 is treated as the "base" (safe fallback).
    base_losses = f_loss_per_example(params_list[0], val).detach()
    base_mean = float(base_losses.mean().item())
    best_i = 0
    best_val = base_mean

    n_val = int(base_losses.numel())
    min_improve_over_base = float(min_improve_over_base or 0.0)
    use_significance = bool(use_significance) and (n_val >= 2)
    z_value = float(z_value or 0.0)

    for i in range(1, len(params_list)):
        losses = f_loss_per_example(params_list[i], val).detach()
        cand_mean = float(losses.mean().item())
        improvement = base_mean - cand_mean
        if improvement < min_improve_over_base:
            continue

        if use_significance:
            diffs = (base_losses - losses).detach()
            se = float(diffs.std(unbiased=True).item()) / math.sqrt(float(n_val) + 1e-12)
            if not math.isfinite(se) or improvement < (z_value * se):
                continue

        if cand_mean < best_val:
            best_i = i
            best_val = cand_mean

    best_params = [q.detach().clone().requires_grad_(True) for q in params_list[best_i]]
    return int(best_i), best_params, float(best_val)

# =========================================================
# Utilities
# =========================================================
def clip_grad_norm_(t, max_norm):
    if max_norm is None or max_norm <= 0:
        return
    with torch.no_grad():
        n = t.norm()
        if n > max_norm and torch.isfinite(n):
            t.mul_(max_norm / (n + 1e-12))

def _lr_x_at_iteration(params_cfg: Dict[str, Any], itr: int) -> float:
    lr0 = float(params_cfg.get("lr_x", 0.0) or 0.0)
    schedule = str(params_cfg.get("lr_x_schedule", "constant") or "constant").strip().lower()
    lr_final = float(params_cfg.get("lr_x_final", lr0) or lr0)

    if schedule in {"constant", "const", "fixed"}:
        return lr0

    outer = int(params_cfg.get("outer_itr", 1) or 1)
    denom = max(1, outer - 1)
    t = float(min(max(itr, 0), denom)) / float(denom)

    if schedule == "cosine":
        # Cosine decay from lr0 to lr_final.
        return lr_final + 0.5 * (lr0 - lr_final) * (1.0 + math.cos(math.pi * t))
    if schedule == "linear":
        return lr0 + (lr_final - lr0) * t
    if schedule in {"inv", "inverse", "inverse_time"}:
        decay_steps = float(params_cfg.get("lr_x_decay_steps", 200) or 200)
        decay_steps = max(1.0, decay_steps)
        return lr0 / (1.0 + float(itr) / decay_steps)

    raise ValueError(f"Unknown lr_x_schedule={schedule!r}. Expected one of: constant, cosine, linear, inv.")

def _dot_list(a: List[torch.Tensor], b: List[torch.Tensor]) -> torch.Tensor:
    return sum((x * y).sum() for x, y in zip(a, b))

def _axpy_list_(dst: List[torch.Tensor], a: float, x: List[torch.Tensor]) -> None:
    with torch.no_grad():
        for d, xi in zip(dst, x):
            d.add_(xi, alpha=float(a))

def _scale_list(x: List[torch.Tensor], a: float) -> List[torch.Tensor]:
    return [xi * float(a) for xi in x]

def _clone_leaf_list(x: List[torch.Tensor], *, requires_grad: bool = True) -> List[torch.Tensor]:
    return [t.detach().clone().requires_grad_(requires_grad) for t in x]

def _g_loss_minibatch(
    params: List[torch.Tensor],
    *,
    x_sigmoid: torch.Tensor,
    tr: "Dataset",
    reg_value: float,
    batch_size: int,
) -> torch.Tensor:
    n_train = int(tr.data.size(0))
    bs = int(batch_size)
    if bs <= 0 or bs >= n_train:
        return g_loss(params, x_sigmoid, tr, reg_value)
    idx = torch.randint(0, n_train, (bs,), device=tr.data.device)
    logits = forward_with_params(tr.data[idx], params)
    ce = F.cross_entropy(logits, tr.dirty_target[idx], reduction="none")
    return (x_sigmoid[idx] * ce).mean() + reg_value * params_l2(params)

def _lmc_ula(
    params0: List[torch.Tensor],
    *,
    x: torch.Tensor,
    tr: "Dataset",
    K: int,
    h: float,
    temp: float,
    reg_value: float,
    batch_size: int,
) -> List[torch.Tensor]:
    """
    LMC / ULA update targeting exp(-g/temp) (or deterministic when temp==0).
    Uses minibatch gradients (if batch_size < n_train) for scalability and extra stochasticity.
    """
    params = _clone_leaf_list(params0, requires_grad=True)
    w = torch.sigmoid(x).detach()
    h_f = float(h)
    temp_f = float(temp)
    noise_scale = math.sqrt(max(0.0, 2.0 * h_f * temp_f)) if temp_f > 0.0 else 0.0

    for _ in range(int(K)):
        loss = _g_loss_minibatch(params, x_sigmoid=w, tr=tr, reg_value=reg_value, batch_size=batch_size)
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        with torch.no_grad():
            for p, g in zip(params, grads):
                p.add_(g, alpha=-h_f)
                if noise_scale > 0.0:
                    p.add_(torch.randn_like(p), alpha=noise_scale)
    return _clone_leaf_list(params, requires_grad=True)

def _hvp_g(
    *,
    params: List[torch.Tensor],
    v: List[torch.Tensor],
    x: torch.Tensor,
    tr: "Dataset",
    reg_value: float,
    batch_size: int,
) -> List[torch.Tensor]:
    """
    Hessian-vector product: (∇^2_{yy} g(params, x)) v
    computed via autograd (two backward passes). We detach sigmoid(x) here because
    CG treats x as constant.
    """
    w = torch.sigmoid(x).detach()
    loss = _g_loss_minibatch(params, x_sigmoid=w, tr=tr, reg_value=reg_value, batch_size=batch_size)
    # We need the graph for the second derivative below (HVP), so retain_graph=True here.
    grad1 = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    dot = _dot_list(list(grad1), v)
    hvp = torch.autograd.grad(dot, params, create_graph=False, retain_graph=False)
    return [h.detach() for h in hvp]

def _cg_solve(
    *,
    matvec,
    b: List[torch.Tensor],
    tol: float,
    max_it: int,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Conjugate Gradient solve for A v = b using only matvec(v)=A v.
    Assumes A is (approximately) SPD; use ridge damping to make this more true.
    """
    v = [torch.zeros_like(bi) for bi in b]
    r = [bi.detach().clone() for bi in b]
    p = [ri.detach().clone() for ri in r]

    rr = float(_dot_list(r, r).item())
    bnorm = math.sqrt(max(rr, 0.0))
    if bnorm == 0.0:
        return v, {"cg_iters": 0.0, "cg_rel_res": 0.0}

    tol_abs = float(tol) * bnorm
    it = 0
    while it < int(max_it):
        Ap = matvec(p)
        pAp = float(_dot_list(p, Ap).item())
        if not math.isfinite(pAp) or abs(pAp) < 1e-30:
            break
        alpha = rr / pAp
        _axpy_list_(v, alpha, p)
        _axpy_list_(r, -alpha, Ap)
        rr_new = float(_dot_list(r, r).item())
        rel_res = math.sqrt(max(rr_new, 0.0)) / (bnorm + 1e-12)
        it += 1
        if math.sqrt(max(rr_new, 0.0)) <= tol_abs:
            rr = rr_new
            break
        beta = rr_new / (rr + 1e-30)
        with torch.no_grad():
            for pi, ri in zip(p, r):
                pi.mul_(beta).add_(ri)
        rr = rr_new

    return v, {"cg_iters": float(it), "cg_rel_res": float(math.sqrt(max(rr, 0.0)) / (bnorm + 1e-12))}

def prepare_datasets(seed_val, pollute_rate_val):
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    mnist = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    tr, val, test = split_data(mnist, 5000, 100, 10000)
    tr.pollute(pollute_rate_val)

    for ds in (tr, val, test):
        ds.flatten()
        ds.to(device)

    mask_clean = tr.clean_mask
    input_dim = tr.data.size(1)
    return tr, val, test, mask_clean, input_dim

def _itd_metrics(
    params: List[torch.Tensor],
    x: torch.Tensor,
    tr: Dataset,
    val: Dataset,
    test: Dataset,
    mask_clean: torch.Tensor,
    *,
    reg_value: float,
) -> Dict[str, float]:
    with torch.no_grad():
        val_l = f_loss(params, val).item()
        g_tr = g_loss(params, torch.sigmoid(x), tr, float(reg_value)).item()
        train_acc = compute_acc_from_params(params, tr.data, tr.clean_target)
        test_acc = compute_acc_from_params(params, test.data, test.clean_target)
        weights = torch.sigmoid(x)
        x_bin = (weights >= 0.5).float()
        prec = (x_bin * mask_clean).sum() / (x_bin.sum() + 1e-8)
        rec  = (x_bin * mask_clean).sum() / (mask_clean.sum() + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

    return {
        "f_val": float(val_l),
        "val_loss": float(val_l),
        "g_train": float(g_tr),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }

# =========================================================
# Public runner — ITD with minima selection
# =========================================================
def run_ITD_Sel(config: Optional[Dict[str, Any]] = None, *,
                verbose: bool = True, log_every: Optional[int] = None,
                return_history: bool = False, **overrides):
    params_cfg = DEFAULT_ITD_CONFIG.copy()
    if config:
        params_cfg.update(config)
    if overrides:
        params_cfg.update(overrides)

    if log_every is None:
        log_every = params_cfg["print_every"]

    time_budget_s = params_cfg.get("time_budget_s", None)
    if time_budget_s is not None:
        time_budget_s = float(time_budget_s)
        if time_budget_s <= 0:
            time_budget_s = None

    if verbose:
        maybe_print_device_info(device, tag="ITD_minima_selection")

    tr, val, test, mask_clean, input_dim = prepare_datasets(
        params_cfg["seed"], params_cfg["pollute_rate"]
    )
    y = init_params(
        input_dim,
        seed_val=params_cfg["seed"],
        hidden=params_cfg["hidden"],
        num_classes=params_cfg["num_classes"],
        scale=params_cfg["init_scale"],
        device=device,
    )
    y_best = _clone_leaf_list(y, requires_grad=True)
    x = torch.zeros(len(tr.data), device=device, requires_grad=True)
    opt_x = torch.optim.AdamW([x], lr=float(params_cfg["lr_x"]), weight_decay=0.0)

    history: List[Dict[str, float]] = []
    last_best_idx: Optional[int] = None
    best_test_so_far: float = -float("inf")
    best_train_so_far: float = -float("inf")
    best_val_so_far: float = float("inf")
    best_itr_test: Optional[int] = None
    best_itr_val: Optional[int] = None
    start_t = time.perf_counter()
    last_itr = -1

    ms_every = max(1, int(params_cfg.get("ms_every", 1) or 1))
    m_chains = max(1, int(params_cfg.get("M_candidates", 1) or 1))
    persistent = bool(params_cfg.get("persistent_chains", True))
    elitism = bool(params_cfg.get("elitism_chain", True))
    backend = str(params_cfg.get("minsel_backend", "adamw") or "adamw").strip().lower()

    # Initialize persistent candidate branches.
    branches: List[List[torch.Tensor]] = []
    branch_states: List[List[Dict[str, torch.Tensor]]] = []
    for i in range(m_chains):
        p = init_params(
            input_dim,
            seed_val=int(params_cfg["seed"]) + 1000 + i,
            hidden=params_cfg["hidden"],
            num_classes=params_cfg["num_classes"],
            scale=params_cfg["init_scale"],
            device=device,
        )
        init_noise = float(params_cfg.get("branch_init_noise_std", 0.0) or 0.0)
        if init_noise > 0:
            with torch.no_grad():
                for t in p:
                    t.add_(torch.randn_like(t) * init_noise)
        branches.append([t.detach().clone().requires_grad_(True) for t in p])
        branch_states.append(_init_adamw_state(branches[-1]))

    # Optional branch diversity via optimizer hyperparameters.
    lr_mults = _logspace(
        float(params_cfg.get("branch_lr_mult_min", 1.0) or 1.0),
        float(params_cfg.get("branch_lr_mult_max", 1.0) or 1.0),
        int(m_chains),
    )

    for itr in range(params_cfg["outer_itr"]):
        if time_budget_s is not None and (time.perf_counter() - start_t) >= time_budget_s:
            break
        last_itr = itr

        # Update LR each outer iteration (schedule).
        lr_x_t = _lr_x_at_iteration(params_cfg, itr)
        for g in opt_x.param_groups:
            g["lr"] = lr_x_t

        # ---- Minima selection + candidate advancement ----
        base_candidate = _clone_leaf_list(y_best, requires_grad=True)

        if backend in {"adamw", "adam", "adamw-branches", "branches"}:
            # Advance each branch by a few AdamW steps on g (persistent by default).
            if not persistent:
                branches = []
                branch_states = []
                for i in range(m_chains):
                    p = init_params(
                        input_dim,
                        seed_val=int(params_cfg["seed"]) + 2000 + itr * 100 + i,
                        hidden=params_cfg["hidden"],
                        num_classes=params_cfg["num_classes"],
                        scale=params_cfg["init_scale"],
                        device=device,
                    )
                    branches.append([t.detach().clone().requires_grad_(True) for t in p])
                    branch_states.append(_init_adamw_state(branches[-1]))

            next_branches: List[List[torch.Tensor]] = []
            next_states: List[List[Dict[str, torch.Tensor]]] = []
            for i in range(m_chains):
                b_i, s_i = _advance_branches_adamw(
                    [branches[i]],
                    [branch_states[i]],
                    x=x,
                    tr=tr,
                    steps=int(params_cfg["D_inner_sgld"]),
                    lr=float(params_cfg["lr_inner_sgld"]) * float(lr_mults[i] if i < len(lr_mults) else 1.0),
                    reg_value=float(params_cfg["reg"]),
                    batch_size=int(params_cfg.get("sgld_batch_size", 0) or 0),
                    beta1=float(params_cfg.get("branch_beta1", 0.9) or 0.9),
                    beta2=float(params_cfg.get("branch_beta2", 0.999) or 0.999),
                    eps=float(params_cfg.get("branch_eps", 1e-8) or 1e-8),
                    weight_decay=float(params_cfg.get("branch_weight_decay", 0.0) or 0.0),
                )
                next_branches.append(b_i[0])
                next_states.append(s_i[0])
            branches = next_branches
            branch_states = next_states
            candidates_from_backend = branches
        else:
            # Fallback to legacy LMC/ULA.
            new_chains: List[List[torch.Tensor]] = []
            for i in range(m_chains):
                start_state = branches[i] if persistent else init_params(
                    input_dim,
                    seed_val=int(params_cfg["seed"]) + 2000 + itr * 100 + i,
                    hidden=params_cfg["hidden"],
                    num_classes=params_cfg["num_classes"],
                    scale=params_cfg["init_scale"],
                    device=device,
                )
                new_state = _lmc_ula(
                    start_state,
                    x=x,
                    tr=tr,
                    K=int(params_cfg["D_inner_sgld"]),
                    h=float(params_cfg["lr_inner_sgld"]),
                    temp=float(params_cfg["sgld_temp"]),
                    reg_value=float(params_cfg["reg"]),
                    batch_size=int(params_cfg.get("sgld_batch_size", 0) or 0),
                )
                new_chains.append(new_state)
            branches = new_chains
            candidates_from_backend = branches

        # Optionally skip the expensive selection step every iteration (speed knob).
        do_ms = (ms_every == 1) or (itr % ms_every == 0)
        if do_ms:
            candidates = [base_candidate] + candidates_from_backend if bool(params_cfg.get("include_base_candidate", True)) else candidates_from_backend
            best_idx, best_params, best_val = select_by_validation_among(
                candidates,
                val,
                min_improve_over_base=float(params_cfg.get("selection_min_improve", 0.0) or 0.0),
                use_significance=bool(params_cfg.get("selection_use_significance", True)),
                z_value=float(params_cfg.get("selection_z", 1.96) or 1.96),
            )
        else:
            best_idx = 0
            best_params = base_candidate
            with torch.no_grad():
                best_val = float(f_loss(best_params, val).item())
        last_best_idx = best_idx

        y_tilde = _clone_leaf_list(best_params, requires_grad=True)
        # Keep one chain anchored at the selected best state (helps reduce drift).
        if elitism and len(branches) > 0:
            branches[0] = _clone_leaf_list(y_tilde, requires_grad=True)
            branch_states[0] = _init_adamw_state(branches[0])

        # Optional polish steps on g using AdamW (deterministic refinement) to reduce noise in the Hessian system.
        if int(params_cfg.get("D_inner_gd", 0) or 0) > 0:
            polish_lr = float(params_cfg["lr_inner_gd"])
            beta1 = float(params_cfg.get("branch_beta1", 0.9) or 0.9)
            beta2 = float(params_cfg.get("branch_beta2", 0.999) or 0.999)
            eps = float(params_cfg.get("branch_eps", 1e-8) or 1e-8)
            wd = float(params_cfg.get("branch_weight_decay", 0.0) or 0.0)
            state = _init_adamw_state(y_tilde)
            w_all = torch.sigmoid(x).detach()
            for _ in range(int(params_cfg["D_inner_gd"])):
                loss = _g_loss_minibatch(
                    y_tilde,
                    x_sigmoid=w_all,
                    tr=tr,
                    reg_value=float(params_cfg["reg"]),
                    batch_size=int(params_cfg.get("cg_batch_size", 0) or 0),
                )
                grads = torch.autograd.grad(loss, y_tilde, retain_graph=False, create_graph=False)
                _adamw_step_(y_tilde, list(grads), state, lr=polish_lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=wd)
            y_tilde = _clone_leaf_list(y_tilde, requires_grad=True)

        # ---- Implicit hypergradient via CG/HVPs ----
        # b = ∇_y f(θ, y_tilde)
        f_val = f_loss(y_tilde, val)
        b = list(torch.autograd.grad(f_val, y_tilde, retain_graph=False, create_graph=False))

        cg_batch_size = int(params_cfg.get("cg_batch_size", 0) or 0)
        ridge = float(params_cfg.get("cg_ridge", 0.0) or 0.0)

        def _matvec(p_list: List[torch.Tensor]) -> List[torch.Tensor]:
            hv = _hvp_g(
                params=y_tilde,
                v=[pi.detach() for pi in p_list],
                x=x,
                tr=tr,
                reg_value=float(params_cfg["reg"]),
                batch_size=cg_batch_size,
            )
            if ridge > 0.0:
                hv = [hvi + ridge * pi.detach() for hvi, pi in zip(hv, p_list)]
            return hv

        v_sol, cg_info = _cg_solve(
            matvec=_matvec,
            b=[bi.detach() for bi in b],
            tol=float(params_cfg.get("cg_tol", 1e-3) or 1e-3),
            max_it=int(params_cfg.get("cg_max_it", 30) or 30),
        )

        # Compute ∇^2_{x y} g · v via: ∇_x <∇_y g, v>
        # (v is treated as constant; we do not differentiate through CG).
        w = torch.sigmoid(x)
        g_for_cross = _g_loss_minibatch(
            y_tilde,
            x_sigmoid=w,
            tr=tr,
            reg_value=float(params_cfg["reg"]),
            batch_size=cg_batch_size,
        )
        # We differentiate <∇_y g, v> w.r.t. x below, so keep the graph.
        grad_g_y = torch.autograd.grad(g_for_cross, y_tilde, create_graph=True, retain_graph=True)
        dot_gv = _dot_list(list(grad_g_y), [vi.detach() for vi in v_sol])
        cross = torch.autograd.grad(dot_gv, x, retain_graph=False)[0]

        # hypergrad_x = ∇_x f - ∇^2_{x y} g · v. Here ∇_x f = 0 (f does not use x),
        # so hypergrad_x = -cross.
        opt_x.zero_grad()
        hypergrad_x = -cross
        if params_cfg["hyper_clip"] and params_cfg["hyper_clip"] > 0:
            clip_grad_norm_(hypergrad_x, params_cfg["hyper_clip"])
        x.grad = hypergrad_x.detach()
        opt_x.step()
        with torch.no_grad():
            x.clamp_(-params_cfg["x_clip"], params_cfg["x_clip"])

        y_best = _clone_leaf_list(y_tilde, requires_grad=True)

        should_log = (itr % log_every == 0 or itr == params_cfg["outer_itr"] - 1)
        if should_log:
            elapsed_s = time.perf_counter() - start_t
            metrics = _itd_metrics(y_best, x, tr, val, test, mask_clean, reg_value=float(params_cfg["reg"]))
            metrics["iteration"] = itr
            metrics["elapsed_s"] = float(elapsed_s)
            metrics["variant"] = "HG-MS"
            metrics["lr_x"] = float(lr_x_t)
            metrics["best_idx"] = best_idx
            metrics["best_val"] = float(best_val)
            metrics["cg_iters"] = float(cg_info.get("cg_iters", 0.0))
            metrics["cg_rel_res"] = float(cg_info.get("cg_rel_res", float("nan")))

            # Best-so-far metrics are useful for plotting monotone curves and for checkpointing.
            if float(metrics["test_acc"]) > best_test_so_far:
                best_test_so_far = float(metrics["test_acc"])
                best_itr_test = int(itr)
            if float(metrics["train_acc"]) > best_train_so_far:
                best_train_so_far = float(metrics["train_acc"])
            if float(metrics["val_loss"]) < best_val_so_far:
                best_val_so_far = float(metrics["val_loss"])
                best_itr_val = int(itr)
            metrics["best_test_acc_so_far"] = float(best_test_so_far)
            metrics["best_train_acc_so_far"] = float(best_train_so_far)
            metrics["best_val_loss_so_far"] = float(best_val_so_far)
            metrics["best_test_iteration"] = int(best_itr_test) if best_itr_test is not None else None
            metrics["best_val_iteration"] = int(best_itr_val) if best_itr_val is not None else None
            if return_history:
                history.append(metrics)
            if verbose:
                print(
                    f"Iter {itr:5d} | [HG-MS] Val={metrics['val_loss']:.4f} "
                    f"g={metrics['g_train']:.4f} "
                    f"TrainAcc={metrics['train_acc']:.2f}% TestAcc={metrics['test_acc']:.2f}% "
                    f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                    f"| lr_x={lr_x_t:.4g} best_idx={best_idx} best_val={best_val:.4f} "
                    f"cg={int(metrics['cg_iters'])} rel_res={metrics['cg_rel_res']:.2e}"
                )

    final_metrics = _itd_metrics(y_best, x, tr, val, test, mask_clean, reg_value=float(params_cfg["reg"]))
    final_metrics["variant"] = "HG-MS"
    final_metrics["iterations_ran"] = int(last_itr + 1)
    final_metrics["elapsed_s"] = float(time.perf_counter() - start_t)
    if last_best_idx is not None:
        final_metrics["best_idx"] = last_best_idx
    if return_history:
        return final_metrics, history
    return final_metrics



# ---- Append below this line (end of file) ----

# Keep a handle to the original config-driven runner
_run_ITD_Sel_internal = run_ITD_Sel

def run_ITD_Sel(
    *,
    seed: int = 2424,
    pollute_rate: float = 0.8,
    outer_itr: int = 1000,
    print_every: int = 5,
    # Optional polish GD on g (set 0 to match HG-MinSel-LMC pseudocode).
    D_inner_gd: int = 0,
    lr_inner_gd: float = 0.005,
    inner_reg: float = 0.0,
    # SGLD / minima selection
    M_candidates: int = 8,
    D_inner_sgld: int = 20,
    lr_inner_sgld: float = 0.02,
    sgld_temp: float = 0.0,
    sgld_batch_size: int = 256,
    persistent_chains: bool = True,
    elitism_chain: bool = True,
    include_base_candidate: bool = True,
    # Minima selection backend and branch diversity (parallel AdamW by default).
    minsel_backend: str = "adamw",
    branch_init_noise_std: float = 0.0,
    branch_lr_mult_min: float = 1.0,
    branch_lr_mult_max: float = 1.0,
    branch_weight_decay: float = 0.0,
    selection_min_improve: float = 0.01,
    selection_use_significance: bool = True,
    selection_z: float = 1.96,
    ms_every: int = 1,
    lr_x_schedule: str = "inv",
    lr_x_final: float = 0.01,
    lr_x_decay_steps: int = 200,
    # Hyper / outer
    lr_x: float = 0.1,
    x_clip: float = 5.0,
    hyper_clip: float = 0.0,
    reg: float = 1e-4,
    # Implicit hypergradient (CG/HVP)
    cg_tol: float = 1e-3,
    cg_max_it: int = 30,
    cg_ridge: float = 1e-2,
    cg_batch_size: int = 0,
    hidden: int = 300,
    num_classes: int = 10,
    init_scale: float = 0.1,
    # Runner controls
    verbose: bool = True,
    log_every: Optional[int] = None,
    return_history: bool = False,
    time_budget_s: Optional[float] = None,
):
    """
    Explicit-args wrapper for ITD with minima selection.

    Returns:
        (final_metrics, history) if return_history=True, else final_metrics.
    """
    cfg = {
        "seed": seed,
        "pollute_rate": pollute_rate,
        "outer_itr": outer_itr,
        "print_every": print_every,
        "D_inner_gd": D_inner_gd,
        "lr_inner_gd": lr_inner_gd,
        "inner_reg": inner_reg,
        "M_candidates": M_candidates,
        "D_inner_sgld": D_inner_sgld,
        "lr_inner_sgld": lr_inner_sgld,
        "sgld_temp": sgld_temp,
        "sgld_batch_size": sgld_batch_size,
        "persistent_chains": bool(persistent_chains),
        "elitism_chain": bool(elitism_chain),
        "include_base_candidate": include_base_candidate,
        "minsel_backend": str(minsel_backend),
        "branch_init_noise_std": float(branch_init_noise_std),
        "branch_lr_mult_min": float(branch_lr_mult_min),
        "branch_lr_mult_max": float(branch_lr_mult_max),
        "branch_weight_decay": float(branch_weight_decay),
        "selection_min_improve": selection_min_improve,
        "selection_use_significance": bool(selection_use_significance),
        "selection_z": float(selection_z),
        "ms_every": ms_every,
        "lr_x_schedule": lr_x_schedule,
        "lr_x_final": lr_x_final,
        "lr_x_decay_steps": lr_x_decay_steps,
        "lr_x": lr_x,
        "x_clip": x_clip,
        "hyper_clip": hyper_clip,
        "reg": reg,
        "cg_tol": cg_tol,
        "cg_max_it": cg_max_it,
        "cg_ridge": cg_ridge,
        "cg_batch_size": cg_batch_size,
        "hidden": hidden,
        "num_classes": num_classes,
        "init_scale": init_scale,
        "time_budget_s": time_budget_s,
    }

    return _run_ITD_Sel_internal(
        config=cfg,
        verbose=verbose,
        log_every=log_every,          # if None, internal runner uses print_every
        return_history=return_history,
    )



def main():
    print("Running ITD with minima selection...")
    _run_ITD_Sel_internal()

if __name__ == "__main__":
    main()
