import os
import ssl
import math
import time
import random
from copy import deepcopy
from contextlib import nullcontext
from typing import Iterable, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------- SSL convenience (optional) ----------------
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.environ.get("HYPERGRAD_REQUIRE_CUDA", "0") == "1" and device.type != "cuda":
    raise RuntimeError("HYPERGRAD_REQUIRE_CUDA=1 but CUDA is not available.")

def print_device_banner(tag: str = "Imbalance") -> None:
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if device.type == "cuda":
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            alloc = float(torch.cuda.memory_allocated(idx)) / (1024**2)
            reserved = float(torch.cuda.memory_reserved(idx)) / (1024**2)
            print(
                f"[{tag}] device=cuda:{idx} name={name} "
                f"cuda_available={torch.cuda.is_available()} cuda_visible={cuda_visible!r} "
                f"mem_alloc_mb={alloc:.1f} mem_reserved_mb={reserved:.1f}"
            )
        else:
            print(f"[{tag}] device=cpu cuda_available={torch.cuda.is_available()} cuda_visible={cuda_visible!r}")
    except Exception as e:
        print(f"[{tag}] device_banner_failed: {e}")


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def balanced_acc_no_warn(model, loader, x_raw=None, n_classes: int = 10) -> float:
    model.eval()
    ys, ps = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        if x_raw is not None:
            delta, gamma = unpack_x(x_raw)
            logits = logits * gamma.view(1, -1) + delta.view(1, -1)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(labels.numpy())
    model.train()
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    present = np.unique(y_true)
    if present.size == 0:
        return 0.0
    recalls = []
    for c in present:
        mask = y_true == c
        denom = int(mask.sum())
        tp = int((y_pred[mask] == c).sum()) if denom > 0 else 0
        recalls.append(tp / max(denom, 1))
    return float(np.mean(recalls))


class FollowerCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.conv(x))


def _init_kaiming(m) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------- Param loss tuning: x = (Δ_raw, γ_raw) ----------------
DELTA_BOUND = 0.5
GAMMA_SPAN = 0.15


def unpack_x(x_raw) -> Tuple[torch.Tensor, torch.Tensor]:
    delta_raw, gamma_raw = x_raw
    delta = DELTA_BOUND * torch.tanh(delta_raw)
    gamma = 1.0 + GAMMA_SPAN * torch.tanh(gamma_raw)
    return delta, gamma


LAMBDA_DELTA = 1e-2
LAMBDA_GAMMA = 1e-2
# Disabled for fair comparison with PBGD baselines (which use plain cross-entropy).
G_LABEL_SMOOTH = 0.0


def g_loss(model, x_raw, images, labels, reg: float = 0.0) -> torch.Tensor:
    logits = model(images)
    delta, gamma = unpack_x(x_raw)
    adjusted = logits * gamma.view(1, -1) + delta.view(1, -1)
    if G_LABEL_SMOOTH and G_LABEL_SMOOTH > 0:
        loss = F.cross_entropy(adjusted, labels, label_smoothing=float(G_LABEL_SMOOTH))
    else:
        loss = F.cross_entropy(adjusted, labels)
    if reg and reg > 0:
        wdec = torch.tensor(0.0, device=device)
        for p in model.parameters():
            wdec = wdec + (p**2).sum()
        loss = loss + reg * wdec
    return loss


def f_loss(model, images_val, labels_val, u_vec, x_raw=None) -> torch.Tensor:
    logits = model(images_val)
    if x_raw is not None:
        delta, gamma = unpack_x(x_raw)
        logits = logits * gamma.view(1, -1) + delta.view(1, -1)
    loss_vec = F.cross_entropy(logits, labels_val, reduction="none")
    w = u_vec[labels_val]
    return (w * loss_vec).sum() / (w.sum() + 1e-12)


@torch.no_grad()
def eval_avg_g_loss(model, loader, x_raw) -> float:
    model.eval()
    total, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        loss = g_loss(model, x_raw, images, labels)
        bs = int(images.size(0))
        total += float(loss.item()) * bs
        n += bs
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def eval_avg_f_loss(model, loader, u_vec, x_raw) -> float:
    model.eval()
    tot_wloss, tot_w = 0.0, 0.0
    delta, gamma = unpack_x(x_raw)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images) * gamma.view(1, -1) + delta.view(1, -1)
        loss_vec = F.cross_entropy(logits, labels, reduction="none")
        w = u_vec[labels]
        tot_wloss += float((w * loss_vec).sum().item())
        tot_w += float(w.sum().item())
    model.train()
    return tot_wloss / max(tot_w, 1e-12)


def conjugate_gradient_solve(Hv_fn, b_list, num_iter: int = 8, tol: float = 1e-10):
    x = [torch.zeros_like(b) for b in b_list]
    r = [b.clone() for b in b_list]
    p = [ri.clone() for ri in r]
    rs_old = sum((ri * ri).sum() for ri in r)
    for _ in range(num_iter):
        Hp = Hv_fn(p)
        denom = sum((pi * hpi).sum() for pi, hpi in zip(p, Hp)) + 1e-10
        alpha = rs_old / denom
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * hpi for ri, hpi in zip(r, Hp)]
        rs_new = sum((ri * ri).sum() for ri in r)
        if rs_new.sqrt() < tol:
            break
        beta = rs_new / (rs_old + 1e-10)
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rs_old = rs_new
    return x


def _cap_batch(images, labels, max_bs=None):
    if (max_bs is None) or (max_bs <= 0) or (images.size(0) <= max_bs):
        return images, labels
    idx = torch.randperm(images.size(0), device=images.device)[:max_bs]
    return images[idx], labels[idx]


def implicit_hypergrad_x(
    model,
    x_raw,
    train_batch,
    val_batch,
    u_vec,
    *,
    cg_steps: int = 8,
    damping: float = 1e-3,
    use_amp: bool = False,
    hg_train_cap=None,
    hg_val_cap=None,
):
    (images_tr, labels_tr) = train_batch
    (images_va, labels_va) = val_batch
    images_tr, labels_tr = images_tr.to(device), labels_tr.to(device)
    images_va, labels_va = images_va.to(device), labels_va.to(device)

    images_tr, labels_tr = _cap_batch(images_tr, labels_tr, hg_train_cap)
    images_va, labels_va = _cap_batch(images_va, labels_va, hg_val_cap)

    delta_orig, gamma_orig = x_raw
    delta_raw = delta_orig.detach().clone().requires_grad_(True)
    gamma_raw = gamma_orig.detach().clone().requires_grad_(True)
    x_local = (delta_raw, gamma_raw)

    autocast_ctx = torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda")
    )

    with autocast_ctx:
        f_val = f_loss(model, images_va, labels_va, u_vec, x_raw=x_local)
        delta_b, gamma_b = unpack_x(x_local)
        f_val = f_val + LAMBDA_DELTA * (delta_b.pow(2).mean()) + LAMBDA_GAMMA * ((gamma_b - 1.0).pow(2).mean())

        grad_x_f = torch.autograd.grad(f_val, (delta_raw, gamma_raw), retain_graph=True)
        grad_y_f = torch.autograd.grad(f_val, model.parameters(), retain_graph=True, allow_unused=True)
        grad_y_f = [
            g if g is not None else torch.zeros_like(p) for g, p in zip(grad_y_f, model.parameters())
        ]

        g_tr = g_loss(model, x_local, images_tr, labels_tr)
        grad_y_g = torch.autograd.grad(g_tr, model.parameters(), create_graph=True, allow_unused=True)
        grad_y_g = [
            g if g is not None else torch.zeros_like(p) for g, p in zip(grad_y_g, model.parameters())
        ]

    def Hv_fn(v_list):
        with autocast_ctx:
            Hv = torch.autograd.grad(
                grad_y_g,
                model.parameters(),
                grad_outputs=v_list,
                retain_graph=True,
                allow_unused=True,
            )
            Hv = [h if h is not None else torch.zeros_like(p) for h, p in zip(Hv, model.parameters())]
            return [h + damping * v for h, v in zip(Hv, v_list)]

    v_sol = conjugate_gradient_solve(Hv_fn, grad_y_f, num_iter=cg_steps)

    with autocast_ctx:
        scalar = sum((gy * v).sum() for gy, v in zip(grad_y_g, v_sol))
        d_delta_hxy, d_gamma_hxy = torch.autograd.grad(scalar, (delta_raw, gamma_raw), retain_graph=False)

        d_delta = grad_x_f[0] - d_delta_hxy
        d_gamma = grad_x_f[1] - d_gamma_hxy

    return d_delta.detach(), d_gamma.detach()


class OuterAdaptiveLR:
    def __init__(
        self,
        alpha_init: float = 5e-3,
        alpha_min: float = 1e-5,
        alpha_max: float = 1e-2,
        ema: float = 0.8,
        grad_clip=None,
        x_l2: float = 1e-4,
    ):
        self.alpha = float(alpha_init)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.ema = float(ema)
        self.grad_clip = grad_clip
        self.x_l2 = float(x_l2)
        self.prev_param_vec = None
        self.prev_grad_vec = None

    @staticmethod
    def _flat_pair(params_pair):
        a, b = params_pair
        return torch.cat([a.detach().reshape(-1), b.detach().reshape(-1)])

    @staticmethod
    def _flat_grads(grads_pair):
        ga, gb = grads_pair
        return torch.cat([ga.detach().reshape(-1), gb.detach().reshape(-1)])

    def _update_alpha_bb(self, param_vec, grad_vec) -> float:
        if self.prev_param_vec is None:
            self.prev_param_vec = param_vec.clone()
            self.prev_grad_vec = grad_vec.clone()
            return self.alpha
        s = param_vec - self.prev_param_vec
        y = grad_vec - self.prev_grad_vec
        denom = float((y @ y).item()) + 1e-12
        numer = float((s @ y).item())
        if numer > 0:
            alpha_bb2 = max(self.alpha_min, min(self.alpha_max, numer / denom))
            self.alpha = self.ema * self.alpha + (1 - self.ema) * alpha_bb2
        self.prev_param_vec = param_vec.clone()
        self.prev_grad_vec = grad_vec.clone()
        return self.alpha

    def step(self, delta_raw, gamma_raw, d_delta, d_gamma) -> float:
        if self.x_l2 and self.x_l2 > 0:
            d_delta = d_delta + self.x_l2 * delta_raw
            d_gamma = d_gamma + self.x_l2 * gamma_raw
        if self.grad_clip is not None and self.grad_clip > 0:
            d_delta = torch.clamp(d_delta, -self.grad_clip, self.grad_clip)
            d_gamma = torch.clamp(d_gamma, -self.grad_clip, self.grad_clip)
        param_vec = self._flat_pair((delta_raw, gamma_raw))
        grad_vec = self._flat_grads((d_delta, d_gamma))
        alpha = self._update_alpha_bb(param_vec, grad_vec)
        with torch.no_grad():
            delta_raw -= alpha * d_delta
            gamma_raw -= alpha * d_gamma
        return alpha


class OuterAdam:
    def __init__(
        self,
        delta,
        gamma,
        *,
        opt: str = "adamw",
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        x_l2: float = 0.0,
        grad_clip=None,
        clip_mode: str = "value",
        max_norm: float = 1.0,
    ):
        self.params = [delta, gamma]
        opt = str(opt).strip().lower()
        if opt == "adam":
            self.opt = torch.optim.Adam(self.params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        else:
            # Default: AdamW (decoupled weight decay), more stable for this outer update in practice.
            self.opt = torch.optim.AdamW(self.params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.lr = float(lr)
        self.x_l2 = float(x_l2)
        self.grad_clip = grad_clip
        self.clip_mode = str(clip_mode)
        self.max_norm = float(max_norm)

    def step(self, d_delta, d_gamma) -> float:
        g_delta = d_delta + self.x_l2 * self.params[0] if self.x_l2 and self.x_l2 > 0 else d_delta
        g_gamma = d_gamma + self.x_l2 * self.params[1] if self.x_l2 and self.x_l2 > 0 else d_gamma

        if self.grad_clip is not None and self.grad_clip > 0:
            if self.clip_mode == "value":
                g_delta = torch.clamp(g_delta, -self.grad_clip, self.grad_clip)
                g_gamma = torch.clamp(g_gamma, -self.grad_clip, self.grad_clip)
            else:
                total_norm = torch.sqrt(g_delta.detach().float().pow(2).sum() + g_gamma.detach().float().pow(2).sum())
                if total_norm > self.max_norm:
                    scale = self.max_norm / (total_norm + 1e-12)
                    g_delta = g_delta * scale
                    g_gamma = g_gamma * scale

        self.opt.zero_grad(set_to_none=True)
        self.params[0].grad = g_delta.clone()
        self.params[1].grad = g_gamma.clone()
        self.opt.step()
        return self.lr


class OuterSGD:
    """
    Plain outer optimizer for baseline runs: fixed-step SGD on (delta_raw, gamma_raw).
    This avoids AdamW and BB-style adaptive step sizes.
    """

    def __init__(
        self,
        delta,
        gamma,
        *,
        lr: float = 1e-2,
        x_l2: float = 0.0,
        grad_clip=None,
    ):
        self.params = [delta, gamma]
        self.lr = float(lr)
        self.x_l2 = float(x_l2)
        self.grad_clip = grad_clip

    def step(self, d_delta, d_gamma) -> float:
        g_delta = d_delta + self.x_l2 * self.params[0] if self.x_l2 and self.x_l2 > 0 else d_delta
        g_gamma = d_gamma + self.x_l2 * self.params[1] if self.x_l2 and self.x_l2 > 0 else d_gamma
        if self.grad_clip is not None and self.grad_clip > 0:
            g_delta = torch.clamp(g_delta, -self.grad_clip, self.grad_clip)
            g_gamma = torch.clamp(g_gamma, -self.grad_clip, self.grad_clip)
        with torch.no_grad():
            self.params[0].add_(g_delta, alpha=-self.lr)
            self.params[1].add_(g_gamma, alpha=-self.lr)
        return self.lr


class FollowerTrainer:
    def __init__(
        self,
        model,
        *,
        opt: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        nesterov: bool = True,
        eta_min_scale: float = 0.2,
        T0: int = 200,
    ):
        params = [p for p in model.parameters() if p.requires_grad]
        if opt.lower() == "sgd":
            self.opt = torch.optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
            )
        else:
            self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=int(T0), T_mult=1, eta_min=lr * eta_min_scale
        )
        self._step_count = 0

    def step_k(self, model, x_raw, train_loader, *, steps: int = 50, grad_clip: float = 1.0, accum: int = 1) -> None:
        model.train()
        it = iter(train_loader)
        self.opt.zero_grad(set_to_none=True)
        for s in range(int(steps)):
            batch = next(it, None)
            if batch is None:
                it = iter(train_loader)
                batch = next(it)
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            x_detach = (x_raw[0].detach(), x_raw[1].detach())
            loss = g_loss(model, x_detach, images, labels, reg=0.0) / max(int(accum), 1)
            loss.backward()
            if (s + 1) % max(int(accum), 1) == 0:
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.sched.step(self._step_count / max(1, int(steps)))
                self._step_count += 1


def _l2_regularizer(model) -> torch.Tensor:
    reg = torch.tensor(0.0, device=device)
    for p in model.parameters():
        reg = reg + (p**2).sum()
    return reg


def _clone_like(model):
    return deepcopy(model).to(device)


def psgld_sample_adaptive(
    model_proto,
    x_raw,
    train_loader,
    *,
    chains: int = 4,
    burn_in: int = 200,
    steps: int = 50,
    thin: int = 0,
    eps0: float = 1e-4,
    alpha: float = 0.55,
    temperature: float = 0.0,
    weight_decay: float = 1e-5,
    rms_rho: float = 0.99,
    rms_eps: float = 1e-8,
    grad_clip: float = 1.0,
    auto_tune_eps: bool = True,
    target_update_ratio: float = 0.02,
    inc: float = 1.05,
    dec: float = 0.95,
    eps_scale_bounds=(0.1, 10.0),
    seed=None,
):
    if seed is not None:
        set_seed(int(seed))
    chains_models = [_clone_like(model_proto) for _ in range(int(chains))]
    sq_avg = [{id(p): torch.zeros_like(p) for p in m.parameters()} for m in chains_models]
    it = iter(train_loader)
    eps_scale = 1.0
    ema_ratio = None

    def step_one(m, sqmap, t, eps_scale_local) -> float:
        nonlocal it
        batch = next(it, None)
        if batch is None:
            it = iter(train_loader)
            batch = next(it)
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        before = [p.detach().clone() for p in m.parameters()]
        loss = g_loss(m, x_raw, images, labels)
        if weight_decay and weight_decay > 0:
            loss = loss + 0.5 * weight_decay * _l2_regularizer(m)
        for p in m.parameters():
            if p.grad is not None:
                p.grad = None
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
        eps_t = float(eps_scale_local) * float(eps0) / ((1.0 + float(t)) ** float(alpha))
        with torch.no_grad():
            for p in m.parameters():
                if p.grad is None:
                    continue
                sid = id(p)
                sq_avg_val = sqmap[sid].mul_(rms_rho).add_((1 - rms_rho) * (p.grad * p.grad))
                precond = 1.0 / (torch.sqrt(sq_avg_val) + rms_eps)
                if temperature and temperature > 0:
                    noise = (
                        torch.randn_like(p)
                        * torch.sqrt(torch.tensor(2.0 * eps_t * temperature, device=device))
                        * torch.sqrt(precond)
                    )
                else:
                    noise = 0.0
                p.add_(-eps_t * precond * p.grad + noise)
        with torch.no_grad():
            upd_sq, ref_sq = 0.0, 0.0
            for p0, p1 in zip(before, m.parameters()):
                upd_sq += (p1 - p0).pow(2).sum().item()
                ref_sq += (p0).pow(2).sum().item()
            ratio = math.sqrt(upd_sq) / (math.sqrt(ref_sq) + 1e-12)
        return float(ratio)

    t = 0
    for _ in range(int(burn_in)):
        ratios = []
        for m, sqmap in zip(chains_models, sq_avg):
            r = step_one(m, sqmap, t, eps_scale)
            t += 1
            ratios.append(r)
        if auto_tune_eps:
            avg = float(np.mean(ratios))
            ema_ratio = avg if ema_ratio is None else 0.9 * ema_ratio + 0.1 * avg
            if ema_ratio > 1.5 * target_update_ratio:
                eps_scale *= dec
            elif ema_ratio < 0.5 * target_update_ratio:
                eps_scale *= inc
            eps_scale = float(np.clip(eps_scale, eps_scale_bounds[0], eps_scale_bounds[1]))

    samples = [[] for _ in range(int(chains))]
    for s in range(int(steps)):
        ratios = []
        for k, (m, sqmap) in enumerate(zip(chains_models, sq_avg)):
            r = step_one(m, sqmap, t, eps_scale)
            t += 1
            ratios.append(r)
            if thin and thin > 0 and (s % int(thin)) == (int(thin) - 1):
                samples[k].append(_clone_like(m))
        if auto_tune_eps:
            avg = float(np.mean(ratios))
            ema_ratio = avg if ema_ratio is None else 0.9 * ema_ratio + 0.1 * avg
            if ema_ratio > 1.5 * target_update_ratio:
                eps_scale *= dec
            elif ema_ratio < 0.5 * target_update_ratio:
                eps_scale *= inc
            eps_scale = float(np.clip(eps_scale, eps_scale_bounds[0], eps_scale_bounds[1]))

    if thin and thin > 0:
        finals = []
        for k, m in enumerate(chains_models):
            finals.append(samples[k][-1] if len(samples[k]) > 0 else _clone_like(m))
    else:
        finals = [_clone_like(m) for m in chains_models]
    return finals


def psgld_init_state(
    model_proto,
    *,
    chains: int = 4,
    seed: Optional[int] = None,
    init_noise_std: float = 0.0,
):
    """
    Initialize persistent pSGLD chain state (models + RMS stats).
    Use this for HG-MinSel-LMC style persistent chains across outer iterations.
    """
    if seed is not None:
        set_seed(int(seed))
    chains_models = [_clone_like(model_proto) for _ in range(int(chains))]
    init_noise_std = float(init_noise_std or 0.0)
    if init_noise_std > 0:
        # Break symmetry across chains (important when sgld_temperature=0 and minibatches are shared).
        with torch.no_grad():
            for m in chains_models:
                for p in m.parameters():
                    p.add_(torch.randn_like(p) * init_noise_std)
    sq_avg = [{id(p): torch.zeros_like(p) for p in m.parameters()} for m in chains_models]
    return {
        "chains_models": chains_models,
        "sq_avg": sq_avg,
        "t": 0,
        "eps_scale": 1.0,
        "ema_ratio": None,
    }


def psgld_advance_state(
    state: dict,
    *,
    x_raw,
    train_loader,
    steps: int,
    eps0: float = 1e-4,
    alpha: float = 0.55,
    temperature: float = 0.0,
    weight_decay: float = 1e-5,
    rms_rho: float = 0.99,
    rms_eps: float = 1e-8,
    grad_clip: float = 1.0,
    auto_tune_eps: bool = True,
    target_update_ratio: float = 0.02,
    inc: float = 1.05,
    dec: float = 0.95,
    eps_scale_bounds=(0.1, 10.0),
    parallel_streams: bool = True,
    per_chain_minibatch: bool = True,
):
    """
    Advance persistent pSGLD chains in-place for `steps` updates per chain.
    Returns the updated state dict.
    """
    # Treat x as constant during LMC updates (do not backprop into outer variables).
    x_detach = (x_raw[0].detach(), x_raw[1].detach())

    chains_models = state["chains_models"]
    sq_avg = state["sq_avg"]
    t = int(state.get("t", 0))
    eps_scale = float(state.get("eps_scale", 1.0))
    ema_ratio = state.get("ema_ratio", None)
    it = iter(train_loader)

    use_streams = bool(parallel_streams) and device.type == "cuda" and len(chains_models) > 1
    streams = [torch.cuda.Stream() for _ in chains_models] if use_streams else None

    default_stream = torch.cuda.current_stream() if use_streams else None

    for _ in range(int(steps)):
        # IMPORTANT: If sgld_temperature=0, chains only diversify if they see different minibatches
        # (or start from different initializations). By default we feed each chain its own minibatch.
        if bool(per_chain_minibatch) and len(chains_models) > 1:
            batches = []
            for _k in range(len(chains_models)):
                batch = next(it, None)
                if batch is None:
                    it = iter(train_loader)
                    batch = next(it)
                images, labels = batch
                batches.append(
                    (
                        images.to(device, non_blocking=True),
                        labels.to(device, non_blocking=True),
                    )
                )
        else:
            batch = next(it, None)
            if batch is None:
                it = iter(train_loader)
                batch = next(it)
            images, labels = batch
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batches = [(images, labels) for _k in range(len(chains_models))]

        if streams is not None and default_stream is not None:
            for s in streams:
                s.wait_stream(default_stream)

        ratio_tensors = []
        for k, (m, sqmap) in enumerate(zip(chains_models, sq_avg)):
            m.train()
            images, labels = batches[k]
            stream = streams[k] if streams is not None else None
            ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with ctx:
                for p in m.parameters():
                    if p.grad is not None:
                        p.grad = None
                loss = g_loss(m, x_detach, images, labels)
                if weight_decay and weight_decay > 0:
                    loss = loss + 0.5 * float(weight_decay) * _l2_regularizer(m)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)

                eps_t = float(eps_scale) * float(eps0) / ((1.0 + float(t)) ** float(alpha))
                t += 1
                upd_sq = torch.zeros((), device=device)
                ref_sq = torch.zeros((), device=device)
                noise_scale = math.sqrt(max(0.0, 2.0 * eps_t * float(temperature))) if temperature and temperature > 0 else 0.0

                with torch.no_grad():
                    for p in m.parameters():
                        ref_sq = ref_sq + (p.detach() ** 2).sum()
                        if p.grad is None:
                            continue
                        sid = id(p)
                        sq_avg_val = sqmap[sid].mul_(float(rms_rho)).add_((1.0 - float(rms_rho)) * (p.grad * p.grad))
                        precond = 1.0 / (torch.sqrt(sq_avg_val) + float(rms_eps))
                        upd = -float(eps_t) * precond * p.grad
                        if noise_scale > 0.0:
                            upd = upd + torch.randn_like(p) * (float(noise_scale)) * torch.sqrt(precond)
                        upd_sq = upd_sq + (upd.detach() ** 2).sum()
                        p.add_(upd)
                ratio_tensors.append(torch.sqrt(upd_sq) / (torch.sqrt(ref_sq) + 1e-12))

        if streams is not None:
            for s in streams:
                s.synchronize()
        ratios = [float(r.item()) for r in ratio_tensors]
        if auto_tune_eps:
            avg = float(np.mean(ratios)) if ratios else 0.0
            ema_ratio = avg if ema_ratio is None else 0.9 * float(ema_ratio) + 0.1 * avg
            if ema_ratio > 1.5 * target_update_ratio:
                eps_scale *= dec
            elif ema_ratio < 0.5 * target_update_ratio:
                eps_scale *= inc
            eps_scale = float(np.clip(eps_scale, eps_scale_bounds[0], eps_scale_bounds[1]))

    state["t"] = int(t)
    state["eps_scale"] = float(eps_scale)
    state["ema_ratio"] = ema_ratio
    return state


@torch.no_grad()
def select_by_f(candidates: Iterable, x_raw, val_loader, u_vec, *, select_mode: str = "min", eval_batches: int = 0):
    delta, gamma = unpack_x(x_raw)

    def eval_f(m) -> float:
        m.eval()
        tot_wloss, tot_w = 0.0, 0.0
        it = iter(val_loader)
        n_steps = int(eval_batches) if eval_batches and eval_batches > 0 else len(val_loader)
        for _ in range(n_steps):
            batch = next(it, None)
            if batch is None:
                break
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            loss_vec = F.cross_entropy(m(images) * gamma.view(1, -1) + delta.view(1, -1), labels, reduction="none")
            w = u_vec[labels]
            tot_wloss += float((w * loss_vec).sum().item())
            tot_w += float(w.sum().item())
        return tot_wloss / max(tot_w, 1e-12)

    candidates = list(candidates)
    scores = [eval_f(m) for m in candidates]
    idx = int(np.argmin(scores) if select_mode == "min" else np.argmax(scores))
    return candidates[idx], scores


@torch.no_grad()
def select_by_f_conservative(
    candidates: Iterable,
    x_raw,
    val_loader,
    u_vec,
    *,
    select_mode: str = "min",
    eval_batches: int = 0,
    min_improve_over_base: float = 0.0,
    use_significance: bool = True,
    z_value: float = 1.96,
    ema_scores: Optional[list[Optional[float]]] = None,
    ema_beta: float = 0.9,
    rank_by_ema: bool = False,
):
    """
    Conservative variant of selection:
    - Candidate 0 is treated as the base/fallback.
    - Only switches to another candidate if it improves mean f by >= min_improve_over_base.
    - Optionally requires a paired-difference significance gate across val batches.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("No candidates provided.")

    delta, gamma = unpack_x(x_raw)
    n_steps = int(eval_batches) if eval_batches and eval_batches > 0 else len(val_loader)

    def batch_losses(m) -> list[float]:
        was_training = bool(getattr(m, "training", False))
        m.eval()
        try:
            losses = []
            it = iter(val_loader)
            for _ in range(n_steps):
                batch = next(it, None)
                if batch is None:
                    break
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                loss_vec = F.cross_entropy(
                    m(images) * gamma.view(1, -1) + delta.view(1, -1), labels, reduction="none"
                )
                w = u_vec[labels]
                losses.append(float(((w * loss_vec).sum() / (w.sum() + 1e-12)).item()))
            return losses
        finally:
            m.train(was_training)

    per_candidate_losses: list[list[float]] = []
    scores: list[float] = []
    for m in candidates:
        ls = batch_losses(m)
        per_candidate_losses.append(ls)
        scores.append(float(np.mean(ls)) if ls else float("inf"))

    base_losses = per_candidate_losses[0]
    base_mean = float(scores[0])

    # Optional exponential moving average (EMA) over scores to reduce selection noise when val is small.
    if ema_scores is not None:
        if len(ema_scores) != len(scores):
            ema_scores[:] = [None for _ in range(len(scores))]
        beta = float(np.clip(float(ema_beta), 0.0, 0.999))
        for i, s in enumerate(scores):
            if not math.isfinite(float(s)):
                continue
            prev = ema_scores[i]
            if prev is None or not math.isfinite(float(prev)):
                ema_scores[i] = float(s)
            else:
                ema_scores[i] = beta * float(prev) + (1.0 - beta) * float(s)

    rank_scores = scores
    if bool(rank_by_ema) and ema_scores is not None:
        rank_scores = [float(e) if e is not None else float(s) for e, s in zip(ema_scores, scores)]

    best_i = 0
    best_rank = float(rank_scores[0])
    for i in range(1, len(rank_scores)):
        val_i = float(rank_scores[i])
        if select_mode == "min":
            if val_i < best_rank:
                best_rank, best_i = val_i, i
        else:
            if val_i > best_rank:
                best_rank, best_i = val_i, i

    if best_i == 0:
        return candidates[0], scores, 0

    min_improve_over_base = float(min_improve_over_base or 0.0)
    best_mean = float(scores[best_i])
    improvement = (base_mean - best_mean) if select_mode == "min" else (best_mean - base_mean)
    if improvement < min_improve_over_base:
        return candidates[0], scores, 0

    # Paired-difference gate is unreliable with very few batches; prefer EMA/min-improve in that regime.
    if use_significance and len(base_losses) >= 5:
        best_losses = per_candidate_losses[best_i]
        if len(best_losses) == len(base_losses) and len(best_losses) >= 2:
            if select_mode == "min":
                diffs = np.array(base_losses, dtype=float) - np.array(best_losses, dtype=float)
            else:
                diffs = np.array(best_losses, dtype=float) - np.array(base_losses, dtype=float)
            se = float(diffs.std(ddof=1)) / math.sqrt(float(len(diffs)) + 1e-12)
            if not math.isfinite(se) or improvement < float(z_value) * se:
                return candidates[0], scores, 0

    return candidates[best_i], scores, int(best_i)


@torch.no_grad()
def select_by_bal_acc_conservative(
    candidates: Iterable,
    x_raw,
    val_loader,
    u_vec=None,
    *,
    select_mode: str = "max",
    n_classes: int = 10,
    eval_batches: int = 0,
    min_improve_over_base: float = 0.0,
    use_significance: bool = True,
    z_value: float = 1.96,
    ema_scores: Optional[list[Optional[float]]] = None,
    ema_beta: float = 0.9,
    rank_by_ema: bool = False,
):
    """
    Conservative selection using validation balanced accuracy (higher is better).

    Notes:
    - Candidate 0 is treated as the base/fallback.
    - Scores are computed as mean(per-batch balanced accuracy) over `eval_batches`
      (or the full val_loader if eval_batches==0).
    - Optional EMA and paired-difference significance gate mirror the f(val) selector.
    """
    _ = u_vec  # Unused (kept for signature compatibility with f(val) selector).
    select_mode = (select_mode or "max").strip().lower()
    if select_mode not in {"max", "min"}:
        select_mode = "max"
    candidates = list(candidates)
    if not candidates:
        raise ValueError("No candidates provided.")

    delta, gamma = unpack_x(x_raw)
    n_steps = int(eval_batches) if eval_batches and eval_batches > 0 else len(val_loader)

    def batch_bal_accs(m) -> list[float]:
        was_training = bool(getattr(m, "training", False))
        m.eval()
        try:
            accs: list[float] = []
            it = iter(val_loader)
            for _ in range(n_steps):
                batch = next(it, None)
                if batch is None:
                    break
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                logits = m(images) * gamma.view(1, -1) + delta.view(1, -1)
                preds = logits.argmax(1)
                recalls = []
                for c in range(int(n_classes)):
                    mask = labels == c
                    denom = int(mask.sum().item())
                    if denom <= 0:
                        continue
                    tp = int((preds[mask] == c).sum().item())
                    recalls.append(tp / max(denom, 1))
                accs.append(float(np.mean(recalls)) if recalls else 0.0)
            return accs
        finally:
            m.train(was_training)

    per_candidate_accs: list[list[float]] = []
    scores: list[float] = []
    for m in candidates:
        accs = batch_bal_accs(m)
        per_candidate_accs.append(accs)
        scores.append(float(np.mean(accs)) if accs else float("-inf"))

    base_accs = per_candidate_accs[0]
    base_mean = float(scores[0])

    if ema_scores is not None:
        if len(ema_scores) != len(scores):
            ema_scores[:] = [None for _ in range(len(scores))]
        beta = float(np.clip(float(ema_beta), 0.0, 0.999))
        for i, s in enumerate(scores):
            if not math.isfinite(float(s)):
                continue
            prev = ema_scores[i]
            if prev is None or not math.isfinite(float(prev)):
                ema_scores[i] = float(s)
            else:
                ema_scores[i] = beta * float(prev) + (1.0 - beta) * float(s)

    rank_scores = scores
    if bool(rank_by_ema) and ema_scores is not None:
        rank_scores = [float(e) if e is not None else float(s) for e, s in zip(ema_scores, scores)]

    best_i = 0
    best_rank = float(rank_scores[0])
    for i in range(1, len(rank_scores)):
        val_i = float(rank_scores[i])
        if select_mode == "min":
            if val_i < best_rank:
                best_rank, best_i = val_i, i
        else:
            if val_i > best_rank:
                best_rank, best_i = val_i, i

    if best_i == 0:
        return candidates[0], scores, 0

    min_improve_over_base = float(min_improve_over_base or 0.0)
    best_mean = float(scores[best_i])
    improvement = (base_mean - best_mean) if select_mode == "min" else (best_mean - base_mean)
    if improvement < min_improve_over_base:
        return candidates[0], scores, 0

    if use_significance and len(base_accs) >= 5:
        best_accs = per_candidate_accs[best_i]
        if len(best_accs) == len(base_accs) and len(best_accs) >= 2:
            diffs = (
                np.array(base_accs, dtype=float) - np.array(best_accs, dtype=float)
                if select_mode == "min"
                else np.array(best_accs, dtype=float) - np.array(base_accs, dtype=float)
            )
            se = float(diffs.std(ddof=1)) / math.sqrt(float(len(diffs)) + 1e-12)
            if not math.isfinite(se) or improvement < float(z_value) * se:
                return candidates[0], scores, 0

    return candidates[best_i], scores, int(best_i)


MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
FASHION_MEAN, FASHION_STD = (0.2860,), (0.3530,)


def _dataset_spec(name: str):
    name = (name or "mnist").strip().lower()
    if name in {"fashion", "fashionmnist", "fashion-mnist"}:
        return datasets.FashionMNIST, FASHION_MEAN, FASHION_STD, "FashionMNIST"
    return datasets.MNIST, MNIST_MEAN, MNIST_STD, "MNIST"


def _balanced_indices_by_class(labels_numpy, *, per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    idxs = []
    for c in range(10):
        cand = np.where(labels_numpy == c)[0]
        if cand.size == 0:
            continue
        take = rng.choice(cand, size=min(int(per_class), int(cand.size)), replace=False)
        idxs.extend(take.tolist())
    rng.shuffle(idxs)
    return idxs


def _balanced_resample_from_pool(labels_numpy, pool_indices: np.ndarray, *, seed: int) -> list[int]:
    """
    Create a roughly class-balanced subset *of the given pool* with the same total size.
    This keeps the requested train/val split size while avoiding extreme class skew in val.
    """
    pool_indices = np.asarray(pool_indices, dtype=int)
    n_total = int(pool_indices.size)
    if n_total <= 0:
        return []

    rng = np.random.default_rng(int(seed))
    per_class = max(1, n_total // 10)
    out: list[int] = []
    remaining: list[int] = []
    for c in range(10):
        cls = pool_indices[labels_numpy[pool_indices] == c]
        if cls.size == 0:
            continue
        take = min(int(per_class), int(cls.size))
        chosen = rng.choice(cls, size=take, replace=False)
        out.extend(chosen.tolist())
        # leftover candidates for filling remainder
        mask = np.ones(cls.size, dtype=bool)
        if take < cls.size:
            # Mark chosen within cls and keep the rest
            chosen_set = set(int(i) for i in chosen.tolist())
            for j, idx in enumerate(cls.tolist()):
                if int(idx) in chosen_set:
                    mask[j] = False
            remaining.extend(cls[mask].tolist())

    # Fill to exactly n_total (should be close already).
    if len(out) < n_total:
        rng.shuffle(remaining)
        out.extend(remaining[: (n_total - len(out))])
    rng.shuffle(out)
    return out[:n_total]


@torch.no_grad()
def compute_u_from_dataset(ds, n_classes: int = 10) -> torch.Tensor:
    counts = torch.zeros(int(n_classes), device=device)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    for _, labels in loader:
        labels = labels.to(device)
        counts += torch.bincount(labels, minlength=int(n_classes)).to(device)
    counts = torch.clamp(counts, min=1)
    return (1.0 / counts).to(device)


# ---------------- Spurious feature (train-only) ----------------
def _lcg_u32(x: int) -> int:
    # Simple deterministic 32-bit LCG hash (reproducible across processes).
    return (int(x) * 1664525 + 1013904223) & 0xFFFFFFFF


def _u01_from_u32(x: int) -> float:
    return float(int(x) & 0xFFFFFFFF) / float(2**32)


def _spurious_patch_positions(size: int, n_positions: int) -> list[tuple[int, int]]:
    """
    Deterministic list of top-left patch coordinates on a 28x28 image.
    We keep them on the border so the patch acts as a clean "shortcut" feature.
    """
    size = int(size)
    n_positions = int(n_positions)
    if size <= 0:
        raise ValueError("spurious_patch_size must be > 0")
    if n_positions <= 0:
        raise ValueError("spurious_n_positions must be > 0")
    max_start = 28 - size
    max_start = max(0, int(max_start))
    base = [
        (0, 0),
        (0, max_start // 2),
        (0, max_start),
        (max_start // 3, 0),
        (max_start // 3, max_start),
        (2 * max_start // 3, 0),
        (2 * max_start // 3, max_start),
        (max_start, 0),
        (max_start, max_start // 2),
        (max_start, max_start),
    ]
    if n_positions <= len(base):
        return base[:n_positions]
    # If asked for more, repeat the base pattern (keeps behavior deterministic and simple).
    out = list(base)
    while len(out) < n_positions:
        out.extend(base)
    return out[:n_positions]


class SpuriousPatchDataset(torch.utils.data.Dataset):
    """
    Adds a small label-correlated patch to *training* images only.

    Why this is useful:
      - It creates a shortcut feature that can yield low g(train) but poor f(val) if the model relies on it.
      - With multiple LL candidates (minima selection), some branches may learn the real digit features
        while others overfit the shortcut; selecting by f(val) can then matter more.
    """

    def __init__(
        self,
        base_ds,
        *,
        seed: int,
        prob: float = 0.0,
        flip_prob: float = 0.0,
        size: int = 4,
        value: float = 3.0,
        n_positions: int = 10,
    ):
        self.base_ds = base_ds
        self.seed = int(seed)
        self.prob = float(np.clip(float(prob), 0.0, 1.0))
        self.flip_prob = float(np.clip(float(flip_prob), 0.0, 1.0))
        self.size = int(size)
        self.value = float(value)
        self.positions = _spurious_patch_positions(self.size, int(n_positions))

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        x, y = self.base_ds[int(idx)]
        if self.prob <= 0.0:
            return x, y

        # Deterministic per-sample pseudo-randomness (so runs are reproducible even with workers).
        r0 = _lcg_u32(int(idx) ^ (_lcg_u32(self.seed) << 1))
        if self.prob < 1.0 and _u01_from_u32(r0) >= self.prob:
            return x, y

        label = int(y) if not torch.is_tensor(y) else int(y.item())
        patch_label = label
        if self.flip_prob > 0.0:
            r1 = _lcg_u32(r0 ^ 0xA5A5A5A5)
            if _u01_from_u32(r1) < self.flip_prob:
                # Choose a wrong label in {0..9}\{label} deterministically.
                r2 = _lcg_u32(r1 ^ 0x3C3C3C3C)
                patch_label = (label + 1 + (int(r2) % 9)) % 10

        r, c = self.positions[int(patch_label) % len(self.positions)]
        size = self.size

        if torch.is_tensor(x):
            x_out = x.clone()
            if x_out.ndim == 1:
                x_out = x_out.view(1, 28, 28)
                x_out[..., r : r + size, c : c + size] = float(self.value)
                x_out = x_out.view(-1)
            else:
                x_out[..., r : r + size, c : c + size] = float(self.value)
            return x_out, y

        # Fallback for unexpected types
        return x, y


def prepare_data(
    *,
    dataset: str = "mnist",
    imbalance_base: float = 0.4,
    min_keep_prob: float = 0.0,
    bs_train: int = 64,
    bs_val: int = 64,
    seed: int = 1234,
    balanced_val: bool = True,
    match_val_test_imbalance: bool = False,
    val_min_per_class: int = 0,
    val_full_batch: bool = False,
    per_class_val: int = 500,
    val_size: int = 500,
    train_fraction: float = 0.99,
    augment_train: bool = True,
    train_erase_p: float = 0.0,
    spurious_train: bool = False,
    spurious_patch_prob: float = 0.0,
    spurious_patch_flip_prob: float = 0.0,
    spurious_patch_size: int = 4,
    spurious_patch_value: float = 3.0,
    spurious_n_positions: int = 10,
):
    set_seed(seed)
    ds_cls, ds_mean, ds_std, _ = _dataset_spec(dataset)
    min_keep_prob = float(np.clip(float(min_keep_prob), 0.0, 1.0))

    train_erase_p = float(np.clip(float(train_erase_p), 0.0, 1.0))
    _post_tensor = [transforms.Normalize(ds_mean, ds_std)]
    if train_erase_p > 0.0:
        _post_tensor.append(
            transforms.RandomErasing(p=train_erase_p, scale=(0.15, 0.35), ratio=(0.3, 3.3), value=0.0)
        )
    tfm_train = (
        transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                *_post_tensor,
            ]
        )
        if augment_train
        else transforms.Compose([transforms.ToTensor(), *_post_tensor])
    )

    tfm_plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(ds_mean, ds_std)])

    full_train_aug = ds_cls(".", train=True, transform=tfm_train, download=True)
    full_train_plain = ds_cls(".", train=True, transform=tfm_plain, download=True)
    test_set = ds_cls(".", train=False, transform=tfm_plain, download=True)

    # 99/1 split of the original train split (requested).
    n = int(len(full_train_plain))
    perm = torch.randperm(n)
    n_train_pool = int(float(train_fraction) * n)
    n_train_pool = max(1, min(n - 1, n_train_pool))
    train_pool = perm[:n_train_pool].cpu().numpy()
    val_pool = perm[n_train_pool:].cpu().numpy()

    # Apply class-dependent downsampling only on the train pool to create imbalance.
    labels_train = np.array(full_train_aug.targets)[train_pool]
    keep_prob = np.power(float(imbalance_base), labels_train)
    if min_keep_prob > 0.0:
        keep_prob = np.maximum(keep_prob, min_keep_prob)
    keep_mask = np.random.rand(len(labels_train)) < keep_prob
    idx_train = train_pool[keep_mask]
    imbalanced_train_aug = Subset(full_train_aug, idx_train.tolist())
    imbalanced_train_plain = Subset(full_train_plain, idx_train.tolist())

    # Optional: add a spurious, label-correlated patch to the *training* distribution only.
    if bool(spurious_train) and float(spurious_patch_prob) > 0.0:
        imbalanced_train_aug = SpuriousPatchDataset(
            imbalanced_train_aug,
            seed=int(seed),
            prob=float(spurious_patch_prob),
            flip_prob=float(spurious_patch_flip_prob),
            size=int(spurious_patch_size),
            value=float(spurious_patch_value),
            n_positions=int(spurious_n_positions),
        )
        imbalanced_train_plain = SpuriousPatchDataset(
            imbalanced_train_plain,
            seed=int(seed),
            prob=float(spurious_patch_prob),
            flip_prob=float(spurious_patch_flip_prob),
            size=int(spurious_patch_size),
            value=float(spurious_patch_value),
            n_positions=int(spurious_n_positions),
        )

    labels_plain = np.array(full_train_plain.targets)
    test_set_used = test_set
    if bool(match_val_test_imbalance):
        # Apply the same imbalance mechanism to validation and test too.
        # This makes label marginals closer across splits, but can make evaluation noisier.
        rng_val = np.random.default_rng(int(seed) + 12345)
        labels_val = labels_plain[val_pool]
        keep_prob_val = np.power(float(imbalance_base), labels_val)
        if min_keep_prob > 0.0:
            keep_prob_val = np.maximum(keep_prob_val, min_keep_prob)
        keep_mask_val = rng_val.random(len(labels_val)) < keep_prob_val
        # Ensure at least a few examples per class (if available in val_pool) so full-val batches
        # have coverage across classes and f(val) doesn't collapse on a tiny subset.
        min_pc = max(0, int(val_min_per_class))
        if min_pc > 0:
            forced = np.zeros_like(keep_mask_val, dtype=bool)
            for c in range(10):
                cls = np.where(labels_val == c)[0]
                if cls.size <= 0:
                    continue
                take = min(int(min_pc), int(cls.size))
                chosen = rng_val.choice(cls, size=take, replace=False)
                forced[chosen] = True
            keep_mask_val = np.logical_or(keep_mask_val, forced)
        val_idx = val_pool[keep_mask_val]
        if int(val_idx.size) <= 0:
            val_idx = val_pool
        val_set = Subset(full_train_plain, val_idx.tolist())

        rng_te = np.random.default_rng(int(seed) + 23456)
        labels_te = np.array(test_set.targets)
        keep_prob_te = np.power(float(imbalance_base), labels_te)
        if min_keep_prob > 0.0:
            keep_prob_te = np.maximum(keep_prob_te, min_keep_prob)
        keep_mask_te = rng_te.random(len(labels_te)) < keep_prob_te
        te_idx = np.where(keep_mask_te)[0]
        if int(te_idx.size) <= 0:
            te_idx = np.arange(len(test_set))
        test_set_used = Subset(test_set, te_idx.tolist())
    else:
        if balanced_val:
            idx_val = _balanced_resample_from_pool(labels_plain, val_pool, seed=int(seed))
            val_set = Subset(full_train_plain, idx_val)
        else:
            # Backward-compat option (but requested default is 99/1 split).
            val_set = Subset(full_train_plain, val_pool.tolist())

    u_vec = compute_u_from_dataset(val_set)

    # Important: when `match_val_test_imbalance=True` and the validation split is small,
    # downsampling can easily make `len(val_set) < bs_val`. If `drop_last=True`, the val loader
    # becomes empty and algorithms that sample validation minibatches will crash with StopIteration.
    train_drop_last = bool(len(imbalanced_train_aug) >= int(bs_train))
    train_loader = DataLoader(imbalanced_train_aug, batch_size=int(bs_train), shuffle=True, drop_last=train_drop_last)
    bs_val_eff = int(len(val_set)) if bool(val_full_batch) else int(bs_val)
    bs_val_eff = max(1, bs_val_eff)
    val_loader = DataLoader(val_set, batch_size=bs_val_eff, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set_used, batch_size=256, shuffle=False)
    train_eval_loader = DataLoader(imbalanced_train_plain, batch_size=256, shuffle=False)
    val_eval_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    return (train_loader, val_loader, test_loader, train_eval_loader, val_eval_loader, u_vec)


def resolve_outer_iters(config: dict) -> int:
    if config.get("outer_iters") is not None:
        return int(config["outer_iters"])
    if config.get("outer_itr") is not None:
        return int(config["outer_itr"])
    if config.get("target_samples") is not None:
        return max(1, int(config["target_samples"]) // int(config["bs_train"]))
    raise KeyError("Provide one of: outer_iters, outer_itr, or target_samples (with bs_train).")


def maybe_restore_best(best: dict, *, model, delta_raw, gamma_raw) -> None:
    if best.get("state") is None:
        return
    model.load_state_dict(best["state"])
    delta_best, gamma_best = best["x"]
    with torch.no_grad():
        delta_raw.copy_(delta_best.to(device))
        gamma_raw.copy_(gamma_best.to(device))


def default_trainer_from_config(model, config: dict) -> FollowerTrainer:
    return FollowerTrainer(
        model,
        opt=config.get("y_opt", "adamw"),
        lr=config.get("y_lr", 1e-3),
        weight_decay=config.get("y_wd", 5e-4),
        momentum=config.get("y_momentum", 0.9),
        nesterov=True,
        eta_min_scale=0.2,
        T0=config.get("y_sched_T0", 200),
    )


def default_outer_from_config(delta_raw, gamma_raw, config: dict):
    outer_opt = str(config.get("outer_opt", "adamw")).lower()
    if outer_opt in {"sgd"}:
        outer = OuterSGD(
            delta_raw,
            gamma_raw,
            lr=config.get("x_lr", 1e-2),
            x_l2=config.get("x_l2", 0.0),
            grad_clip=config.get("x_grad_clip", None),
        )
        return outer, True
    if outer_opt == "bb":
        outer = OuterAdaptiveLR(
            alpha_init=config["x_lr_init"],
            alpha_min=config["x_alpha_min"],
            alpha_max=config["x_alpha_max"],
            ema=config["x_ema"],
            grad_clip=config.get("x_grad_clip", None),
            x_l2=config.get("x_l2", 0.0),
        )
        return outer, False

    outer = OuterAdam(
        delta_raw,
        gamma_raw,
        opt="adam" if outer_opt == "adam" else "adamw",
        lr=config.get("x_lr", 1e-3),
        betas=config.get("x_betas", (0.9, 0.999)),
        eps=config.get("x_eps", 1e-8),
        weight_decay=config.get("x_weight_decay", 0.0),
        x_l2=config.get("x_l2", 0.0),
        grad_clip=config.get("x_grad_clip", None),
        clip_mode=config.get("x_clip_mode", "value"),
        max_norm=config.get("x_clip_norm", 1.0),
    )
    return outer, True
