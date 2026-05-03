import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import time
import random
from typing import Dict, List, Optional, Any

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
    "print_every": 5,
    "D_inner_gd": 20,
    "lr_inner_gd": 0.005,
    "inner_reg": 0.0,
    "M_candidates": 5,     # kept for parity (unused here)
    "D_inner_sgld": 20,    # kept for parity (unused here)
    "lr_inner_sgld": 0.02, # kept for parity (unused here)
    "sgld_temp": 0.0,      # kept for parity (unused here)
    "lr_x": 0.1,
    "x_clip": 5.0,
    "hyper_clip": 0.0,
    "reg": 1e-4,
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

# =========================================================
# Differentiable inner unroll (AdamW)
# =========================================================
def inner_unroll_adamw(
    params0,
    x,
    tr,
    steps,
    lr,
    reg_value,
    inner_reg_value,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
):
    params = [p for p in params0]
    exp_avg = [torch.zeros_like(p) for p in params]
    exp_avg_sq = [torch.zeros_like(p) for p in params]
    beta1 = float(beta1)
    beta2 = float(beta2)
    eps = float(eps)
    weight_decay = float(weight_decay)

    for t in range(1, int(steps) + 1):
        w = torch.sigmoid(x)
        loss = g_loss(params, w, tr, reg_value) + float(inner_reg_value) * params_l2(params)
        grads = torch.autograd.grad(loss, params, create_graph=True)

        bias_c1 = 1.0 - (beta1 ** t)
        bias_c2 = 1.0 - (beta2 ** t)
        step_size = float(lr) / max(bias_c1, 1e-12)

        new_params = []
        new_exp_avg = []
        new_exp_avg_sq = []
        for p, g, m, v in zip(params, grads, exp_avg, exp_avg_sq):
            m_new = beta1 * m + (1.0 - beta1) * g
            v_new = beta2 * v + (1.0 - beta2) * (g * g)
            # eps inside sqrt prevents exploding second-order terms when v_hat is tiny
            v_hat = v_new / max(bias_c2, 1e-12)
            v_hat = torch.clamp(v_hat, min=0.0)
            denom = torch.sqrt(v_hat + eps)

            update = m_new / denom
            if weight_decay > 0.0:
                update = update + weight_decay * p
            p_new = p - step_size * update

            new_params.append(p_new)
            new_exp_avg.append(m_new)
            new_exp_avg_sq.append(v_new)

        params = new_params
        exp_avg = new_exp_avg
        exp_avg_sq = new_exp_avg_sq

    return params

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

# =========================================================
# Losses f and g
# =========================================================
def g_loss(params, x_weights_sigmoid, tr, reg_value):
    logits = forward_with_params(tr.data, params)
    ce = F.cross_entropy(logits, tr.dirty_target, reduction='none')
    return (x_weights_sigmoid * ce).mean() + reg_value * params_l2(params)

def f_loss(params, val):
    return val_loss_params(params, val.data, val.clean_target)

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
# Utilities
# =========================================================
def clip_grad_norm_(t, max_norm):
    if max_norm is None or max_norm <= 0:
        return
    with torch.no_grad():
        n = t.norm()
        if n > max_norm and torch.isfinite(n):
            t.mul_(max_norm / (n + 1e-12))

def prepare_datasets(seed_val, pollute_rate_val):
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    mnist_root = "." if os.path.isdir("./MNIST/raw") else "./data"
    download = not os.path.isdir(os.path.join(mnist_root, "MNIST", "raw"))
    mnist = MNIST(root=mnist_root, train=True, download=download, transform=transforms.ToTensor())
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
# Public runner — ITD baseline
# =========================================================
def run_ITD_baseline(config: Optional[Dict[str, Any]] = None, *,
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
        maybe_print_device_info(device, tag="ITD_baseline")

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
    x = torch.zeros(len(tr.data), device=device, requires_grad=True)
    opt_x = torch.optim.AdamW([x], lr=params_cfg["lr_x"], weight_decay=0.0)

    history: List[Dict[str, float]] = []
    start_t = time.perf_counter()
    last_itr = -1

    for itr in range(params_cfg["outer_itr"]):
        if time_budget_s is not None and (time.perf_counter() - start_t) >= time_budget_s:
            break
        last_itr = itr

        y_final = inner_unroll_adamw(
            y,
            x,
            tr,
            params_cfg["D_inner_gd"],
            params_cfg["lr_inner_gd"],
            params_cfg["reg"],
            params_cfg["inner_reg"],
        )

        f_val = f_loss(y_final, val)
        opt_x.zero_grad()
        hypergrad_x = torch.autograd.grad(f_val, x, retain_graph=False)[0]
        if params_cfg["hyper_clip"] and params_cfg["hyper_clip"] > 0:
            clip_grad_norm_(hypergrad_x, params_cfg["hyper_clip"])
        if torch.isfinite(hypergrad_x).all():
            x.grad = hypergrad_x.detach()
            opt_x.step()
        else:
            opt_x.zero_grad(set_to_none=True)
            if verbose:
                print(f"[HG-baseline] Non-finite hypergradient at itr={itr}; skipping x update to avoid NaNs.")
        with torch.no_grad():
            if not torch.isfinite(x).all():
                x.nan_to_num_(nan=0.0, posinf=float(params_cfg["x_clip"]), neginf=-float(params_cfg["x_clip"]))
            x.clamp_(-params_cfg["x_clip"], params_cfg["x_clip"])

        if all(torch.isfinite(p).all() for p in y_final):
            y = [p.detach().clone().requires_grad_(True) for p in y_final]
        else:
            if verbose:
                print(f"[HG-baseline] Non-finite y at itr={itr}; keeping previous y to avoid NaNs.")

        should_log = (itr % log_every == 0 or itr == params_cfg["outer_itr"] - 1)
        if should_log:
            elapsed_s = time.perf_counter() - start_t
            metrics = _itd_metrics(y, x, tr, val, test, mask_clean, reg_value=params_cfg["reg"])
            metrics["iteration"] = itr
            metrics["elapsed_s"] = float(elapsed_s)
            metrics["variant"] = "HG-baseline"
            if return_history:
                history.append(metrics)
            if verbose:
                print(
                    f"Iter {itr:5d} | [HG-baseline] Val={metrics['val_loss']:.4f} "
                    f"g={metrics['g_train']:.4f} "
                    f"TrainAcc={metrics['train_acc']:.2f}% TestAcc={metrics['test_acc']:.2f}% "
                    f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
                )

    final_metrics = _itd_metrics(y, x, tr, val, test, mask_clean, reg_value=params_cfg["reg"])
    final_metrics["variant"] = "HG-baseline"
    final_metrics["iterations_ran"] = int(last_itr + 1)
    final_metrics["elapsed_s"] = float(time.perf_counter() - start_t)
    if return_history:
        return final_metrics, history
    return final_metrics

# ---- Append below this line (end of file) ----

# Keep a handle to the original config-driven runner
_run_ITD_baseline_internal = run_ITD_baseline

def run_ITD_baseline(
    *,
    seed: int = 2424,
    pollute_rate: float = 0.8,
    outer_itr: int = 1000,
    print_every: int = 5,
    D_inner_gd: int = 20,
    lr_inner_gd: float = 0.005,
    inner_reg: float = 0.0,
    # kept for parity with config (unused in baseline)
    M_candidates: int = 5,
    D_inner_sgld: int = 20,
    lr_inner_sgld: float = 0.02,
    sgld_temp: float = 0.0,
    # hyper/outer
    lr_x: float = 0.1,
    x_clip: float = 5.0,
    hyper_clip: float = 0.0,
    reg: float = 1e-4,
    hidden: int = 300,
    num_classes: int = 10,
    init_scale: float = 0.1,
    # runner controls
    verbose: bool = True,
    log_every: Optional[int] = None,
    return_history: bool = False,
    time_budget_s: Optional[float] = None,
):
    """
    Explicit-args wrapper for the ITD baseline.

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
        "lr_x": lr_x,
        "x_clip": x_clip,
        "hyper_clip": hyper_clip,
        "reg": reg,
        "hidden": hidden,
        "num_classes": num_classes,
        "init_scale": init_scale,
        "time_budget_s": time_budget_s,
    }

    return _run_ITD_baseline_internal(
        config=cfg,
        verbose=verbose,
        log_every=log_every,          # if None, internal runner uses print_every
        return_history=return_history,
    )

def main():
    print("Running ITD baseline...")
    _run_ITD_baseline_internal()

if __name__ == "__main__":
    main()
