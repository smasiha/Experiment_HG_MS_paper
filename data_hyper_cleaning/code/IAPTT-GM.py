import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
DEFAULT_IAPTT_GM_CONFIG: Dict[str, Any] = {
    "seed": 2424,
    "pollute_rate": 0.8,
    "outer_itr": 1000,
    "print_every": 5,
    # Upper (x, z) steps
    "lrx": 0.10,
    "lrz": 0.10,
    # Inner (follower) dynamics
    "K_inner": 5,
    "lr_inner": 0.01,
    "reg": 0.0,
    # Model
    "hidden": 300,
    "num_classes": 10,
}

def get_default_iaptt_gm_config() -> Dict[str, Any]:
    return DEFAULT_IAPTT_GM_CONFIG.copy()

# Device
device = resolve_device()

# =========================================================
# Data utilities (match other algorithms in this folder)
# =========================================================
class Dataset:
    def __init__(self, data, target):
        self.data = data.float() / 255.0
        self.clean_target = target.clone()
        self.dirty_target = None
        self.clean_mask = torch.ones(len(target), dtype=torch.float32)

    def pollute(self, rho: float):
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
        self.data = self.data.to(device)
        self.clean_target = self.clean_target.to(device)
        if self.dirty_target is not None:
            self.dirty_target = self.dirty_target.to(device)
        self.clean_mask = self.clean_mask.to(device)

def split_data(dataset, train_n, val_n, test_n):
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    data, labels = dataset.data, dataset.targets
    return (
        Dataset(data[idxs[:train_n]], labels[idxs[:train_n]]),
        Dataset(data[idxs[train_n:train_n+val_n]], labels[idxs[train_n:train_n+val_n]]),
        Dataset(data[idxs[train_n+val_n:train_n+val_n+test_n]],
                labels[idxs[train_n+val_n:train_n+val_n+test_n]]),
    )

def prepare_datasets(seed_val: int, pollute_rate_val: float):
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

# =========================================================
# Functional model (2-layer MLP with Sigmoid like MP_paper)
# =========================================================
def init_params(input_dim: int, seed_val: int, hidden: int = 300, num_classes: int = 10, device=device):
    g = torch.Generator(device=device).manual_seed(seed_val)
    W1 = torch.randn(hidden, input_dim, generator=g, device=device) * 0.1
    b1 = torch.zeros(hidden, device=device)
    W2 = torch.randn(num_classes, hidden, generator=g, device=device) * 0.1
    b2 = torch.zeros(num_classes, device=device)
    return [W1, b1, W2, b2]

def functional_forward(X: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
    W1, b1, W2, b2 = params
    h = torch.sigmoid(X @ W1.t() + b1)
    return h @ W2.t() + b2

def params_l2(params: List[torch.Tensor]) -> torch.Tensor:
    return sum((p ** 2).sum() for p in params)

# =========================================================
# Losses
# =========================================================
def f_ll(x_logits: torch.Tensor, y_params: List[torch.Tensor], tr: Dataset, reg_value: float) -> torch.Tensor:
    w = torch.sigmoid(x_logits)
    logits = functional_forward(tr.data, y_params)
    ce = F.cross_entropy(logits, tr.dirty_target, reduction='none')
    return (w * ce).mean() + reg_value * params_l2(y_params)

def f_ul(y_params: List[torch.Tensor], val: Dataset) -> torch.Tensor:
    logits = functional_forward(val.data, y_params)
    return F.cross_entropy(logits, val.clean_target)

def compute_acc_from_params(params: List[torch.Tensor], X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = functional_forward(X, params)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item() * 100.0

def _metrics(
    y_params: List[torch.Tensor],
    x_logits: torch.Tensor,
    tr: Dataset,
    val: Dataset,
    test: Dataset,
    mask_clean: torch.Tensor,
    *,
    reg_value: float,
) -> Dict[str, float]:
    with torch.no_grad():
        val_l = f_ul(y_params, val).item()
        g_tr = f_ll(x_logits, y_params, tr, float(reg_value)).item()
        train_acc = compute_acc_from_params(y_params, tr.data, tr.clean_target)
        test_acc = compute_acc_from_params(y_params, test.data, test.clean_target)

        w = torch.sigmoid(x_logits)
        x_bin = (w >= 0.5).float()
        prec = (x_bin * mask_clean).sum() / (x_bin.sum() + 1e-8)
        rec = (x_bin * mask_clean).sum() / (mask_clean.sum() + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

    return {
        "f_val": float(val_l),
        "val_loss": float(val_l),
        "g_train": float(g_tr),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

# =========================================================
# Public runner — IAPTT-GM (Pessimistic Trajectory Truncation)
# =========================================================
def run_iaptt_gm(config: Optional[Dict[str, Any]] = None, *,
                verbose: bool = True, log_every: Optional[int] = None,
                return_history: bool = False, **overrides):
    params_cfg = DEFAULT_IAPTT_GM_CONFIG.copy()
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
        maybe_print_device_info(device, tag="IAPTT-GM")

    tr, val, test, mask_clean, input_dim = prepare_datasets(
        params_cfg["seed"], params_cfg["pollute_rate"]
    )

    # Upper variables: x logits and z init for y
    x_logits = torch.nn.Parameter(torch.zeros(len(tr.data), device=device), requires_grad=True)
    z0 = init_params(
        input_dim,
        seed_val=params_cfg["seed"],
        hidden=params_cfg["hidden"],
        num_classes=params_cfg["num_classes"],
        device=device,
    )
    z_params = [torch.nn.Parameter(t.detach().clone(), requires_grad=True) for t in z0]

    opt_x = torch.optim.AdamW([x_logits], lr=params_cfg["lrx"], weight_decay=0.0)
    opt_z = torch.optim.AdamW(z_params, lr=params_cfg["lrz"], weight_decay=0.0)

    # Differentiable inner optimizer state for AdamW unroll (one state per param tensor).
    # We treat this as part of the computation graph so ∇F can flow through the inner dynamics.
    inner_beta1 = float(params_cfg.get("inner_beta1", 0.9) or 0.9)
    inner_beta2 = float(params_cfg.get("inner_beta2", 0.999) or 0.999)
    inner_eps = float(params_cfg.get("inner_eps", 1e-8) or 1e-8)
    inner_weight_decay = float(params_cfg.get("inner_weight_decay", 0.0) or 0.0)

    history: List[Dict[str, float]] = []
    last_y_star: Optional[List[torch.Tensor]] = None
    last_kbar: Optional[int] = None
    start_t = time.perf_counter()
    last_itr = -1

    for itr in range(params_cfg["outer_itr"]):
        if time_budget_s is not None and (time.perf_counter() - start_t) >= time_budget_s:
            break
        last_itr = itr

        # Build follower trajectory y_1..y_K starting from y_0 = z using differentiable AdamW.
        yk = [p for p in z_params]
        m = [torch.zeros_like(p) for p in yk]
        v = [torch.zeros_like(p) for p in yk]
        traj_params: List[List[torch.Tensor]] = []
        traj_Fvals: List[torch.Tensor] = []

        for t_inner in range(1, int(params_cfg["K_inner"]) + 1):
            loss_ll = f_ll(x_logits, yk, tr, params_cfg["reg"])
            grads = torch.autograd.grad(loss_ll, yk, create_graph=True)
            beta1, beta2 = inner_beta1, inner_beta2
            m = [beta1 * mi + (1.0 - beta1) * gi for mi, gi in zip(m, grads)]
            v = [beta2 * vi + (1.0 - beta2) * (gi * gi) for vi, gi in zip(v, grads)]
            bias_c1 = 1.0 - (beta1 ** t_inner)
            bias_c2 = 1.0 - (beta2 ** t_inner)
            step_size = float(params_cfg["lr_inner"]) / max(bias_c1, 1e-12)
            # NOTE: we add eps *inside* sqrt to keep the inner dynamics differentiable and
            # avoid exploding second-order terms when v_hat is very small.
            yk_next: List[torch.Tensor] = []
            for yi, mi, vi in zip(yk, m, v):
                v_hat = vi / max(bias_c2, 1e-12)
                v_hat = torch.clamp(v_hat, min=0.0)
                denom = torch.sqrt(v_hat + inner_eps)
                update = mi / denom
                if inner_weight_decay > 0.0:
                    update = update + inner_weight_decay * yi
                yk_next.append(yi - step_size * update)
            yk = yk_next
            F_val = f_ul(yk, val)
            traj_params.append(yk)
            traj_Fvals.append(F_val)

        # Pessimistic truncation: select worst (max) validation loss along trajectory
        with torch.no_grad():
            vals = torch.stack([v.detach() for v in traj_Fvals])
            kbar = int(torch.argmax(vals).item())

        y_star = traj_params[kbar]
        F_star = traj_Fvals[kbar]
        last_y_star = [p.detach() for p in y_star]
        last_kbar = kbar

        # Upper update of x and z via ∇ F(x, y_kbar(z, x))
        opt_x.zero_grad()
        opt_z.zero_grad()
        F_star.backward()
        grads_finite = True
        if x_logits.grad is None or (not torch.isfinite(x_logits.grad).all()):
            grads_finite = False
        else:
            for p in z_params:
                if p.grad is None or (not torch.isfinite(p.grad).all()):
                    grads_finite = False
                    break

        if grads_finite:
            opt_x.step()
            opt_z.step()
        else:
            opt_x.zero_grad(set_to_none=True)
            opt_z.zero_grad(set_to_none=True)
            if verbose:
                print(f"[IAPTT-GM] Non-finite upper gradients at itr={itr}; skipping update to avoid NaNs.")

        should_log = (itr % log_every == 0 or itr == params_cfg["outer_itr"] - 1)
        if should_log:
            m = _metrics(last_y_star, x_logits.detach(), tr, val, test, mask_clean, reg_value=params_cfg["reg"])
            m["iteration"] = int(itr)
            m["elapsed_s"] = float(time.perf_counter() - start_t)
            m["k_bar"] = int(kbar + 1)
            m["variant"] = "IAPTT-GM"
            if return_history:
                history.append(m)
            if verbose:
                print(
                    f"Iter {itr:5d} | [IAPTT-GM] Val={m['val_loss']:.4f} "
                    f"g={m['g_train']:.4f} "
                    f"TrainAcc={m['train_acc']:.2f}% TestAcc={m['test_acc']:.2f}% "
                    f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                    f"| k_bar={m['k_bar']}/{params_cfg['K_inner']}"
                )

    if last_y_star is None:
        last_y_star = [p.detach() for p in z_params]
    final_metrics = _metrics(last_y_star, x_logits.detach(), tr, val, test, mask_clean, reg_value=params_cfg["reg"])
    final_metrics["variant"] = "IAPTT-GM"
    final_metrics["iterations_ran"] = int(last_itr + 1)
    final_metrics["elapsed_s"] = float(time.perf_counter() - start_t)
    if last_kbar is not None:
        final_metrics["k_bar"] = int(last_kbar + 1)
    if return_history:
        return final_metrics, history
    return final_metrics


# ---- Append below this line (end of file) ----
_run_iaptt_gm_internal = run_iaptt_gm

def run_IAPTT_GM(
    *,
    seed: int = 2424,
    pollute_rate: float = 0.8,
    outer_itr: int = 1000,
    print_every: int = 5,
    lrx: float = 0.10,
    lrz: float = 0.10,
    K_inner: int = 5,
    lr_inner: float = 0.01,
    reg: float = 0.0,
    hidden: int = 300,
    num_classes: int = 10,
    verbose: bool = True,
    log_every: Optional[int] = None,
    return_history: bool = False,
    time_budget_s: Optional[float] = None,
):
    cfg = {
        "seed": seed,
        "pollute_rate": pollute_rate,
        "outer_itr": outer_itr,
        "print_every": print_every,
        "lrx": lrx,
        "lrz": lrz,
        "K_inner": K_inner,
        "lr_inner": lr_inner,
        "reg": reg,
        "hidden": hidden,
        "num_classes": num_classes,
        "time_budget_s": time_budget_s,
    }
    return _run_iaptt_gm_internal(
        config=cfg,
        verbose=verbose,
        log_every=log_every,
        return_history=return_history,
    )

def main():
    print("Running IAPTT-GM...")
    _run_iaptt_gm_internal()

if __name__ == "__main__":
    main()
