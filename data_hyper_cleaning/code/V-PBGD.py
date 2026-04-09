import ssl
# Disable SSL certificate verification for MNIST download
ssl._create_default_https_context = ssl._create_unverified_context

import time
import copy
import random
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

from gpu_utils import maybe_print_device_info, resolve_device

# -----------------------------
# Default hyperparameters
# -----------------------------
DEFAULT_VPBGD_CONFIG: Dict[str, Any] = {
    "seed": 2424,
    "pollute_rate": 0.8,
    "lrx": 0.1,
    "lry": 0.1,
    "lr_inner": 0.001,
    "gamma_init": 0.0,
    "gamma_max": 0.2,
    "gamma_steps": 30000,
    "outer_itr": 10000,
    "inner_itr": 1,
    "reg": 0.0,
    "hidden": 300,
    "log_every": 100,
}


def get_default_vpbgd_config() -> Dict[str, Any]:
    return DEFAULT_VPBGD_CONFIG.copy()
# -----------------------------
# Device setup
# -----------------------------
device = resolve_device()


# -----------------------------
# Custom Dataset class
# -----------------------------
class Dataset:
    def __init__(self, data, target):
        self.data = data.float() / 255.0
        self.clean_target = target.clone()
        self.dirty_target = None
        self.clean_mask = torch.ones(len(target), dtype=torch.float32)

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
        self.data = self.data.to(device)
        self.clean_target = self.clean_target.to(device)
        if self.dirty_target is not None:
            self.dirty_target = self.dirty_target.to(device)
        self.clean_mask = self.clean_mask.to(device)


# -----------------------------
# Utility functions
# -----------------------------
def split_data(dataset, train_n, val_n, test_n):
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    data, labels = dataset.data, dataset.targets
    return (
        Dataset(data[idxs[:train_n]], labels[idxs[:train_n]]),
        Dataset(data[idxs[train_n:train_n+val_n]], labels[idxs[train_n:train_n+val_n]]),
        Dataset(data[idxs[train_n+val_n:train_n+val_n+test_n]], labels[idxs[train_n+val_n:train_n+val_n+test_n]])
    )


def compute_acc(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100


def l2_norm(params):
    return sum(torch.norm(p, 2)**2 for p in params)


def val_loss(model, data, targets):
    logits = model(data)
    return F.cross_entropy(logits, targets)


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


def _vpbgd_metrics(
    net: torch.nn.Module,
    x: torch.Tensor,
    tr: Dataset,
    val: Dataset,
    test: Dataset,
    mask_clean: torch.Tensor,
    *,
    reg_value: float,
) -> Dict[str, float]:
    net.eval()
    with torch.no_grad():
        val_l = val_loss(net, val.data, val.clean_target).item()
        logits_tr = net(tr.data)
        logits_test = net(test.data)
        train_acc = compute_acc(logits_tr, tr.clean_target)
        test_acc = compute_acc(logits_test, test.clean_target)

        w = torch.sigmoid(x)
        ce_tr = F.cross_entropy(logits_tr, tr.dirty_target, reduction="none")
        g_tr = (w * ce_tr).mean() + float(reg_value) * l2_norm(net.parameters())

        x_bin = (w >= 0.5).float()
        prec = (x_bin * mask_clean).sum() / (x_bin.sum() + 1e-8)
        rec  = (x_bin * mask_clean).sum() / (mask_clean.sum() + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
    net.train()
    return {
        "f_val": float(val_l),
        "val_loss": float(val_l),
        "g_train": float(g_tr.item()),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }


# -----------------------------
# Main routine
# -----------------------------
def run_vpbgd(config: Optional[Dict[str, Any]] = None, *,
              verbose: bool = True, log_every: Optional[int] = None,
              return_history: bool = False, **overrides):
    params_cfg = DEFAULT_VPBGD_CONFIG.copy()
    if config:
        params_cfg.update(config)
    if overrides:
        params_cfg.update(overrides)

    if log_every is None:
        log_every = max(1, params_cfg.get("log_every", 1000))

    time_budget_s = params_cfg.get("time_budget_s", None)
    if time_budget_s is not None:
        time_budget_s = float(time_budget_s)
        if time_budget_s <= 0:
            time_budget_s = None

    if verbose:
        maybe_print_device_info(device, tag="V-PBGD")

    tr, val, test, mask_clean, input_dim = prepare_datasets(
        params_cfg["seed"], params_cfg["pollute_rate"]
    )

    hidden = params_cfg["hidden"]
    net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden, 10)
    ).to(device)

    x = torch.zeros(len(tr.data), device=device, requires_grad=True)
    net_inner = copy.deepcopy(net)

    opt_y = torch.optim.AdamW(net.parameters(), lr=params_cfg["lry"], weight_decay=0.0)
    opt_x = torch.optim.AdamW([x], lr=params_cfg["lrx"], weight_decay=0.0)
    opt_inner = torch.optim.AdamW(net_inner.parameters(), lr=params_cfg["lr_inner"], weight_decay=0.0)

    # Gamma schedule:
    # If gamma_steps >> outer_itr (common when reusing long-run defaults in short experiments),
    # gamma barely increases and the penalty term is effectively off. Clamp to outer_itr so we
    # actually reach gamma_max within the run unless the user explicitly sets a smaller value.
    gamma_init = float(params_cfg.get("gamma_init", 0.0) or 0.0)
    gamma_max = float(params_cfg.get("gamma_max", 0.0) or 0.0)
    gamma_steps_raw = int(params_cfg.get("gamma_steps", 0) or 0)
    outer_itr_eff = int(params_cfg.get("outer_itr", 1) or 1)
    gamma_steps_eff = outer_itr_eff if gamma_steps_raw <= 0 else min(gamma_steps_raw, outer_itr_eff)

    gamma = gamma_init
    gamma_step = (gamma_max - gamma_init) / max(1, gamma_steps_eff)

    history: List[Dict[str, float]] = []
    start_t = time.perf_counter()
    last_itr = -1

    for itr in range(params_cfg["outer_itr"]):
        if time_budget_s is not None and (time.perf_counter() - start_t) >= time_budget_s:
            break
        last_itr = itr

        with torch.no_grad():
            weights = torch.sigmoid(x)
        for _ in range(params_cfg["inner_itr"]):
            opt_inner.zero_grad()
            logits_tr_inner = net_inner(tr.data)
            ce_inner = F.cross_entropy(logits_tr_inner, tr.dirty_target, reduction='none')
            loss_inner = (weights * ce_inner).mean() + params_cfg["reg"] * l2_norm(net_inner.parameters())
            loss_inner.backward()
            opt_inner.step()

        opt_y.zero_grad()
        opt_x.zero_grad()

        f_y = val_loss(net, val.data, val.clean_target)

        logits_tr = net(tr.data)
        ce_tr = F.cross_entropy(logits_tr, tr.dirty_target, reduction='none')
        g_xy = (torch.sigmoid(x) * ce_tr).mean() + params_cfg["reg"] * l2_norm(net.parameters())

        logits_v = net_inner(tr.data)
        ce_v = F.cross_entropy(logits_v, tr.dirty_target, reduction='none').detach()
        v_x = (torch.sigmoid(x) * ce_v).mean() + params_cfg["reg"] * l2_norm(net_inner.parameters()).detach()

        penalty = f_y + gamma * (g_xy - v_x)
        decay = min(1.0 / (gamma + 1e-8), 1.0)
        loss = decay * penalty

        loss.backward()
        opt_x.step()
        opt_y.step()

        gamma = min(params_cfg["gamma_max"], gamma + gamma_step)

        should_log = (itr % log_every == 0 or itr == params_cfg["outer_itr"] - 1)
        if should_log:
            elapsed_s = time.perf_counter() - start_t
            metrics = _vpbgd_metrics(net, x, tr, val, test, mask_clean, reg_value=params_cfg["reg"])
            metrics["iteration"] = itr
            metrics["elapsed_s"] = float(elapsed_s)
            metrics["variant"] = "V-PBGD"
            if return_history:
                history.append(metrics)
            if verbose:
                print(
                    f"Iter {itr:5d}: Loss={metrics['val_loss']:.4f}, "
                    f"g={metrics['g_train']:.4f}, "
                    f"TrainAcc={metrics['train_acc']:.2f}%, TestAcc={metrics['test_acc']:.2f}%, "
                    f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
                )

    final_metrics = _vpbgd_metrics(net, x, tr, val, test, mask_clean, reg_value=params_cfg["reg"])
    final_metrics["variant"] = "V-PBGD"
    final_metrics["iterations_ran"] = int(last_itr + 1)
    final_metrics["elapsed_s"] = float(time.perf_counter() - start_t)
    if return_history:
        return final_metrics, history
    return final_metrics


# ---- Append below this line (end of file) ----

# Keep a handle to the original config-driven runner
_run_vpbgd_internal = run_vpbgd

# >>> EDIT THESE to control your run <<<
USER_ARGS = dict(
    seed=2424,
    pollute_rate=0.8,
    lrx=0.1,
    lry=0.1,
    lr_inner=0.001,
    gamma_init=0.0,
    gamma_max=0.2,
    gamma_steps=30000,
    outer_itr=10000,
    inner_itr=1,
    reg=0.0,
    hidden=300,
    log_every=100,       # None -> runner will pick a default
    verbose=True,
    return_history=False,
)

def run_VPBG(**overrides):
    """
    Wrapper that merges USER_ARGS (editable at file end) with any explicit overrides.
    """
    cfg = USER_ARGS.copy()
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    # Pull out flags that are not part of run_vpbgd's config dict
    verbose = cfg.pop("verbose", True)
    return_history = cfg.pop("return_history", False)
    log_every = cfg.get("log_every", None)

    # Optional: quick visibility into what is actually used
    print("[V-PBGD cfg]", {k: cfg[k] for k in [
        "seed","pollute_rate","lrx","lry","lr_inner","gamma_init","gamma_max",
        "gamma_steps","outer_itr","inner_itr","reg","hidden","log_every"
    ]})

    return _run_vpbgd_internal(
        config=cfg,
        verbose=verbose,
        log_every=log_every,
        return_history=return_history,
    )

def main():
    print("Running V-PBGD with USER_ARGS...")
    run_VPBG()  # uses the values in USER_ARGS above

if __name__ == "__main__":
    main()
