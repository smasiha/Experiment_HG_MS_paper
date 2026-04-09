# pbgd_param_loss_imbalanced.py
import os, ssl, math, time, random, copy, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# ----------------------------- SSL convenience for MNIST -----------------------------
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    ssl._create_default_https_context = ssl._create_unverified_context

# ----------------------------- Device -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.environ.get("HYPERGRAD_REQUIRE_CUDA", "0") == "1" and device.type != "cuda":
    raise RuntimeError("HYPERGRAD_REQUIRE_CUDA=1 but CUDA is not available.")

# ----------------------------- Dataset defaults -----------------------------
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
FASHION_MEAN, FASHION_STD = (0.2860,), (0.3530,)


def _dataset_spec(name: str):
    name = (name or "mnist").strip().lower()
    if name in {"fashion", "fashionmnist", "fashion-mnist"}:
        return datasets.FashionMNIST, FASHION_MEAN, FASHION_STD, "FashionMNIST"
    return datasets.MNIST, MNIST_MEAN, MNIST_STD, "MNIST"

# Keep the parametric loss close to identity.
# This prevents the leader variables from "blowing up" and destabilizing training.
DELTA_BOUND = 0.5
GAMMA_SPAN = 0.15

def print_device_banner(tag: str = "PBGD") -> None:
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

# ----------------------------- Repro utils -----------------------------
def set_seed(seed=2424):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ----------------------------- Data: imbalance + loaders -----------------------------
def create_imbalanced_dataset(dataset, imbalance_base=0.4):
    # Keep each class i with probability base**i (MNIST classes 0..9)
    labels = np.array(dataset.targets)
    keep_prob = np.power(imbalance_base, labels)
    keep_mask = np.random.rand(len(labels)) < keep_prob
    idx = np.where(keep_mask)[0]
    return Subset(dataset, idx)

@torch.no_grad()
def compute_u_from_dataset(ds, n_classes=10):
    counts = torch.zeros(n_classes, device=device)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    for _, labels in loader:
        labels = labels.to(device)
        counts += torch.bincount(labels, minlength=int(n_classes)).to(device)
    counts = torch.clamp(counts, min=1)
    u = 1.0 / counts
    return u

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
        if take < cls.size:
            chosen_set = set(int(i) for i in chosen.tolist())
            for idx in cls.tolist():
                if int(idx) not in chosen_set:
                    remaining.append(int(idx))

    if len(out) < n_total:
        rng.shuffle(remaining)
        out.extend(remaining[: (n_total - len(out))])
    rng.shuffle(out)
    return out[:n_total]


# ---------------- Spurious feature (train-only) ----------------
def _lcg_u32(x: int) -> int:
    return (int(x) * 1664525 + 1013904223) & 0xFFFFFFFF


def _u01_from_u32(x: int) -> float:
    return float(int(x) & 0xFFFFFFFF) / float(2**32)


def _spurious_patch_positions(size: int, n_positions: int) -> list[tuple[int, int]]:
    """
    Deterministic list of top-left patch coordinates on a 28x28 image.
    Keep patches on the border so they act as a clean shortcut feature.
    """
    size = int(size)
    n_positions = int(n_positions)
    if size <= 0:
        raise ValueError("spurious_patch_size must be > 0")
    if n_positions <= 0:
        raise ValueError("spurious_n_positions must be > 0")
    max_start = max(0, 28 - size)
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
    out = list(base)
    while len(out) < n_positions:
        out.extend(base)
    return out[:n_positions]


class SpuriousPatchDataset(torch.utils.data.Dataset):
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
        r0 = _lcg_u32(int(idx) ^ (_lcg_u32(self.seed) << 1))
        if self.prob < 1.0 and _u01_from_u32(r0) >= self.prob:
            return x, y
        label = int(y) if not torch.is_tensor(y) else int(y.item())
        patch_label = label
        if self.flip_prob > 0.0:
            r1 = _lcg_u32(r0 ^ 0xA5A5A5A5)
            if _u01_from_u32(r1) < self.flip_prob:
                r2 = _lcg_u32(r1 ^ 0x3C3C3C3C)
                patch_label = (label + 1 + (int(r2) % 9)) % 10
        r, c = self.positions[int(patch_label) % len(self.positions)]
        size = self.size
        x_out = x.clone()
        x_out[..., r : r + size, c : c + size] = float(self.value)
        return x_out, y


def prepare_data(
    seed=2424,
    imbalance_base=0.4,
    bs_train=64,
    bs_val=64,
    *,
    dataset: str = "mnist",
    min_keep_prob: float = 0.0,
    train_fraction: float = 0.99,
    augment_train: bool = True,
    balanced_val: bool = True,
    match_val_test_imbalance: bool = False,
    val_min_per_class: int = 0,
    val_full_batch: bool = False,
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
    labels_np = np.array(full_train_plain.targets)[train_pool]
    keep_prob = np.power(float(imbalance_base), labels_np)
    if min_keep_prob > 0.0:
        keep_prob = np.maximum(keep_prob, min_keep_prob)
    keep_mask = np.random.rand(len(labels_np)) < keep_prob
    train_idx = train_pool[keep_mask]
    imbalanced_train_aug = Subset(full_train_aug, train_idx.tolist())
    imbalanced_train_plain = Subset(full_train_plain, train_idx.tolist())

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

    # Validation is the remaining (1-train_fraction) split.
    # Default: resample to be roughly class-balanced for a higher-SNR f(val).
    # Optional: match the same train imbalance ratio on val/test too.
    test_set_used = test_set
    if bool(match_val_test_imbalance):
        rng_val = np.random.default_rng(int(seed) + 12345)
        labels_all = np.array(full_train_plain.targets)
        labels_val = labels_all[val_pool]
        keep_prob_val = np.power(float(imbalance_base), labels_val)
        if min_keep_prob > 0.0:
            keep_prob_val = np.maximum(keep_prob_val, min_keep_prob)
        keep_mask_val = rng_val.random(len(labels_val)) < keep_prob_val
        # Ensure the validation split contains at least a few examples of each class (if available),
        # so a "full-val batch" has support across classes and f(val) doesn't collapse to ~0 on a tiny subset.
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
            labels_all = np.array(full_train_plain.targets)
            idx_val = _balanced_resample_from_pool(labels_all, val_pool, seed=int(seed))
            val_set = Subset(full_train_plain, idx_val)
        else:
            val_set = Subset(full_train_plain, val_pool.tolist())

    u_vec = compute_u_from_dataset(val_set, n_classes=10)

    # Important: when `match_val_test_imbalance=True` and the validation split is small,
    # downsampling can easily make `len(val_set) < bs_val`. If `drop_last=True`, the val loader
    # becomes empty and algorithms that do `next(iter(val_loader))` will crash with StopIteration.
    train_drop_last = bool(len(imbalanced_train_aug) >= int(bs_train))
    train_loader = DataLoader(imbalanced_train_aug, batch_size=bs_train, shuffle=True, drop_last=train_drop_last)
    # Optionally use a "full validation batch" (one batch per epoch) to reduce val minibatch noise.
    bs_val_eff = int(len(val_set)) if bool(val_full_batch) else int(bs_val)
    bs_val_eff = max(1, bs_val_eff)
    val_loader   = DataLoader(val_set,           batch_size=bs_val_eff, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_set_used,     batch_size=256,      shuffle=False)

    # Eval on the *plain* (non-augmented) view for stable metrics.
    train_eval_loader = DataLoader(imbalanced_train_plain, batch_size=256,  shuffle=False)
    val_eval_loader   = DataLoader(val_set,      batch_size=256,      shuffle=False)

    return (train_loader, val_loader, test_loader, train_eval_loader, val_eval_loader, u_vec)

# ----------------------------- Model -----------------------------
class FollowerCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,5,padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 256), nn.ReLU(),
            nn.Linear(256, n_classes)
        )
    def forward(self, x): return self.fc(self.conv(x))

# ----------------------------- Param loss tuning: x unpack -----------------------------
def unpack_x(x_raw):
    delta_raw, gamma_raw = x_raw
    delta = DELTA_BOUND * torch.tanh(delta_raw)
    gamma = 1.0 + GAMMA_SPAN * torch.tanh(gamma_raw)
    return delta, gamma

# ----------------------------- Losses -----------------------------
def g_loss(model, x_raw, images, labels):
    logits = model(images)
    Δ, γ = unpack_x(x_raw)
    adjusted = logits * γ.view(1,-1) + Δ.view(1,-1)
    return F.cross_entropy(adjusted, labels)

def f_loss(model, x_raw, images, labels, u_vec, *, lam_delta: float = 0.0, lam_gamma: float = 0.0):
    logits = model(images)
    Δ, γ = unpack_x(x_raw)
    adjusted = logits * γ.view(1,-1) + Δ.view(1,-1)
    loss_vec = F.cross_entropy(adjusted, labels, reduction='none')
    w = u_vec[labels]
    base = (w * loss_vec).sum() / (w.sum() + 1e-12)
    # Optional stabilization: keep (Δ,γ) close to (0,1) to avoid degenerate logits scaling.
    if lam_delta > 0.0 or lam_gamma > 0.0:
        base = base + lam_delta * (Δ.pow(2).mean()) + lam_gamma * ((γ - 1.0).pow(2).mean())
    return base

@torch.no_grad()
def eval_balanced_acc(model, loader, x_raw, n_classes=10):
    """Balanced accuracy without sklearn warnings; averages recall over classes present in y_true."""
    model.eval()
    ys, ps = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        if x_raw is not None:
            Δ, γ = unpack_x(x_raw)
            logits = logits * γ.view(1, -1) + Δ.view(1, -1)
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
        mask = (y_true == c)
        denom = mask.sum()
        tp = (y_pred[mask] == c).sum() if denom > 0 else 0
        recalls.append(tp / max(denom, 1))
    return float(np.mean(recalls))

@torch.no_grad()
def eval_avg_losses(model, train_eval_loader, val_eval_loader, test_loader, x_raw, u_vec):
    """Returns (avg_g_train, avg_f_val, test_bal) for quick periodic eval."""
    model.eval()
    # g on train_eval
    tot_g, n = 0.0, 0
    for images, labels in train_eval_loader:
        images, labels = images.to(device), labels.to(device)
        tot_g += g_loss(model, x_raw, images, labels).item() * images.size(0)
        n += images.size(0)
    avg_g = tot_g / max(1,n)
    # f on val_eval
    tot_wloss, tot_w = 0.0, 0.0
    for images, labels in val_eval_loader:
        images, labels = images.to(device), labels.to(device)
        Δ, γ = unpack_x(x_raw)
        logits = model(images) * γ.view(1,-1) + Δ.view(1,-1)
        loss_vec = F.cross_entropy(logits, labels, reduction='none')
        w = u_vec[labels]
        tot_wloss += (w*loss_vec).sum().item(); tot_w += w.sum().item()
    avg_f = tot_wloss / max(tot_w,1e-12)
    # test balanced acc
    te_bal = eval_balanced_acc(model, test_loader, x_raw, n_classes=10)
    model.train()
    return avg_g, avg_f, te_bal

@torch.no_grad()
def eval_avg_g_on_loader(model, loader, x_raw):
    """Average g on an arbitrary loader (we use it for val_g)."""
    model.eval()
    tot, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        tot += g_loss(model, x_raw, images, labels).item() * images.size(0)
        n += images.size(0)
    model.train()
    return tot / max(1, n)

def l2_norm(params):
    return sum((p**2).sum() for p in params)

# ----------------------------- V-PBGD runner -----------------------------
def run_v_pbgd(config, common):
    (train_loader, val_loader, test_loader, train_eval_loader, val_eval_loader, u_vec) = common
    set_seed(config['seed'])
    print_device_banner("V-PBGD")
    model = FollowerCNN().to(device)

    # leader variables (raw)
    Δ_raw = torch.zeros(10, requires_grad=True, device=device)
    γ_raw = torch.zeros(10, requires_grad=True, device=device)
    x_raw = (Δ_raw, γ_raw)

    # optimizers
    opt_y = torch.optim.SGD(
        model.parameters(), lr=config["lry"], momentum=float(config.get("y_momentum", 0.9)), nesterov=False
    )
    opt_x = torch.optim.SGD(
        [Δ_raw, γ_raw], lr=config["lrx"], momentum=float(config.get("x_momentum", 0.9)), nesterov=False
    )

    # virtual follower
    omega = copy.deepcopy(model).to(device)
    opt_inner = torch.optim.SGD(
        omega.parameters(), lr=config["lr_inner"], momentum=float(config.get("inner_momentum", 0.0)), nesterov=False
    )

    # schedules
    gamma = config['gamma_init']
    gamma_step = (config['gamma_max'] - config['gamma_init']) / max(1, config['gamma_steps'])

    train_it, val_it = iter(train_loader), iter(val_loader)
    logs = {k: [] for k in ['iter','elapsed_s','train_g','val_f','val_g','train_bal_acc','test_bal_acc','gamma']}

    time_budget_s = config.get("time_budget_s", None)
    if time_budget_s is not None:
        try:
            time_budget_s = float(time_budget_s)
            if time_budget_s <= 0:
                time_budget_s = None
        except Exception:
            time_budget_s = None
    start_t = time.perf_counter()

    for itr in range(config['outer_itr']):
        if time_budget_s is not None and (time.perf_counter() - start_t) >= time_budget_s:
            break
        # fetch batches
        tb = next(train_it, None); vb = next(val_it, None)
        if tb is None: train_it = iter(train_loader); tb = next(train_it)
        if vb is None: val_it   = iter(val_loader);   vb = next(val_it)
        images_tr, labels_tr = tb[0].to(device), tb[1].to(device)
        images_va, labels_va = vb[0].to(device), vb[1].to(device)

        # --- inner loop on omega: T steps minimize g(omega; x) ---
        for _ in range(config['inner_itr']):
            opt_inner.zero_grad()
            loss_inner = g_loss(omega, x_raw, images_tr, labels_tr) + config['reg'] * l2_norm(omega.parameters())
            loss_inner.backward()
            if config['y_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(omega.parameters(), config['y_clip'])
            opt_inner.step()

        # --- outer: update theta (model) and x ---
        opt_y.zero_grad(); opt_x.zero_grad()

        # upper loss on validation
        f_y = f_loss(
            model,
            x_raw,
            images_va,
            labels_va,
            u_vec,
            lam_delta=float(config.get("lam_delta", 0.0)),
            lam_gamma=float(config.get("lam_gamma", 0.0)),
        )

        # lower loss on current model
        g_xy = g_loss(model, x_raw, images_tr, labels_tr) + config['reg'] * l2_norm(model.parameters())

        # virtual loss v_x: stop grads through omega, but keep grads through x.
        v_logits = omega(images_tr).detach()
        Δ, γ = unpack_x(x_raw)
        v_adj = v_logits * γ.view(1, -1) + Δ.view(1, -1)
        ce_v = F.cross_entropy(v_adj, labels_tr, reduction="none")
        v_x = ce_v.mean() + config["reg"] * sum((p.detach() ** 2).sum() for p in omega.parameters())

        penalty = f_y + gamma * (g_xy - v_x)
        if bool(config.get("use_decay_scaling", False)):
            decay = min(1.0 / (gamma + 1e-8), 1.0)
        else:
            decay = 1.0
        obj = float(decay) * penalty

        obj.backward()
        if config['x_clip'] is not None:
            torch.nn.utils.clip_grad_norm_([Δ_raw, γ_raw], config['x_clip'])
        if config['y_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['y_clip'])

        opt_x.step(); opt_y.step()

        # ramp gamma
        gamma = min(config['gamma_max'], gamma + gamma_step)

        # periodic eval
        if (itr % config['log_every'] == 0) or (itr == (int(config['outer_itr']) - 1)):
            tr_g, va_f, te_bal = eval_avg_losses(model, train_eval_loader, val_eval_loader, test_loader, x_raw, u_vec)
            va_g = eval_avg_g_on_loader(model, val_eval_loader, x_raw)            # NEW: g on validation
            tr_bal = eval_balanced_acc(model, train_eval_loader, x_raw, n_classes=10)
            logs['iter'].append(itr + 1)
            logs['elapsed_s'].append(float(time.perf_counter() - start_t))
            logs['train_g'].append(tr_g); logs['val_f'].append(va_f); logs['val_g'].append(va_g)
            logs['train_bal_acc'].append(tr_bal); logs['test_bal_acc'].append(te_bal)
            logs['gamma'].append(gamma)
            print(f"[V-PBGD] it {itr+1:5d} | g(train) {tr_g:.4f} | f(val) {va_f:.4f} | g(val) {va_g:.4f} | "
                  f"Train BalAcc {tr_bal:.4f} | Test BalAcc {te_bal:.4f} | γ {gamma:.3f}")

    logs['method'] = 'V-PBGD'
    logs['config'] = dict(config)
    return logs

# ----------------------------- G-PBGD runner -----------------------------
def run_g_pbgd(config, common):
    (train_loader, val_loader, test_loader, train_eval_loader, val_eval_loader, u_vec) = common
    set_seed(config['seed'])
    print_device_banner("G-PBGD")
    model = FollowerCNN().to(device)

    # leader variables (raw)
    Δ_raw = torch.zeros(10, requires_grad=True, device=device)
    γ_raw = torch.zeros(10, requires_grad=True, device=device)
    x_raw = (Δ_raw, γ_raw)

    # optimizers
    opt_y = torch.optim.SGD(
        model.parameters(), lr=config["lry"], momentum=float(config.get("y_momentum", 0.9)), nesterov=False
    )
    opt_x = torch.optim.SGD(
        [Δ_raw, γ_raw], lr=config["lrx"], momentum=float(config.get("x_momentum", 0.9)), nesterov=False
    )

    # schedules
    gamma = config['gamma_init']
    gamma_step = (config['gamma_max'] - config['gamma_init']) / max(1, config['gamma_steps'])

    train_it, val_it = iter(train_loader), iter(val_loader)
    logs = {k: [] for k in ['iter','elapsed_s','train_g','val_f','val_g','train_bal_acc','test_bal_acc','gamma']}
    def _fmt_loss(x: float) -> str:
        try:
            x = float(x)
        except Exception:
            return str(x)
        return f"{x:.4f}" if abs(x) >= 1e-3 else f"{x:.2e}"

    time_budget_s = config.get("time_budget_s", None)
    if time_budget_s is not None:
        try:
            time_budget_s = float(time_budget_s)
            if time_budget_s <= 0:
                time_budget_s = None
        except Exception:
            time_budget_s = None
    start_t = time.perf_counter()

    for itr in range(config['outer_itr']):
        if time_budget_s is not None and (time.perf_counter() - start_t) >= time_budget_s:
            break
        # fetch batches
        tb = next(train_it, None); vb = next(val_it, None)
        if tb is None: train_it = iter(train_loader); tb = next(train_it)
        if vb is None: val_it   = iter(val_loader);   vb = next(val_it)
        images_tr, labels_tr = tb[0].to(device), tb[1].to(device)
        images_va, labels_va = vb[0].to(device), vb[1].to(device)

        # upper loss
        f_y = f_loss(
            model,
            x_raw,
            images_va,
            labels_va,
            u_vec,
            lam_delta=float(config.get("lam_delta", 0.0)),
            lam_gamma=float(config.get("lam_gamma", 0.0)),
        )

        # lower loss + grad-norm penalty
        g_xy = g_loss(model, x_raw, images_tr, labels_tr) + config['reg'] * l2_norm(model.parameters())
        grads = torch.autograd.grad(g_xy, model.parameters(), create_graph=True, allow_unused=True)
        grads = [g for g in grads if g is not None]
        if grads:
            numel = sum(int(g.numel()) for g in grads)
            grad_norm2 = sum((g * g).sum() for g in grads) / max(1, numel)
        else:
            grad_norm2 = torch.tensor(0.0, device=device)

        penalty = f_y + 0.5 * gamma * grad_norm2
        if bool(config.get("use_decay_scaling", False)):
            decay = min(1.0 / (gamma + 1e-8), 1.0)
        else:
            decay = 1.0
        obj = float(decay) * penalty

        opt_y.zero_grad(); opt_x.zero_grad()
        obj.backward()
        if config['x_clip'] is not None:
            torch.nn.utils.clip_grad_norm_([Δ_raw, γ_raw], config['x_clip'])
        if config['y_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['y_clip'])
        opt_x.step(); opt_y.step()

        gamma = min(config['gamma_max'], gamma + gamma_step)

        # periodic eval
        if (itr % config['log_every'] == 0) or (itr == (int(config['outer_itr']) - 1)):
            tr_g, va_f, te_bal = eval_avg_losses(model, train_eval_loader, val_eval_loader, test_loader, x_raw, u_vec)
            va_g = eval_avg_g_on_loader(model, val_eval_loader, x_raw)            # NEW: g on validation
            tr_bal = eval_balanced_acc(model, train_eval_loader, x_raw, n_classes=10)
            logs['iter'].append(itr + 1)
            logs['elapsed_s'].append(float(time.perf_counter() - start_t))
            logs['train_g'].append(tr_g); logs['val_f'].append(va_f); logs['val_g'].append(va_g)
            logs['train_bal_acc'].append(tr_bal); logs['test_bal_acc'].append(te_bal)
            logs['gamma'].append(gamma)
            print(f"[G-PBGD] it {itr+1:5d} | g(train) {_fmt_loss(tr_g)} | f(val) {_fmt_loss(va_f)} | g(val) {_fmt_loss(va_g)} | "
                  f"Train BalAcc {tr_bal:.4f} | Test BalAcc {te_bal:.4f} | γ {gamma:.3f}")

    logs['method'] = 'G-PBGD'
    logs['config'] = dict(config)
    return logs

# ----------------------------- Plot compare -----------------------------
ALG_COLORS = {
    'V-PBGD': '#1f77b4',   # blue
    'G-PBGD': '#d62728',   # red
    'HG–NoSel': '#2ca02c', # green
    'HG–Sel': '#9467bd',   # purple
    'IAPTT-GM': '#ff7f0e', # orange
}
def _alg_color(tag):
    return ALG_COLORS.get(tag, '#333333')  # default gray if unknown


def compare_logs(logA, tagA, logB, tagB):
    itsA = logA['iter']; itsB = logB['iter']
    cA, cB = _alg_color(tagA), _alg_color(tagB)

    plt.figure(figsize=(14,4))

    # --- (1) Balanced Accuracies: Train + Test on same subplot ---
    plt.subplot(1,3,1)
    # Test (solid)
    plt.plot(itsA, logA['test_bal_acc'], '-',  label=f'{tagA} Test BalAcc',  color=cA)
    plt.plot(itsB, logB['test_bal_acc'], '-',  label=f'{tagB} Test BalAcc',  color=cB)
    # Train (dashed)
    if 'train_bal_acc' in logA:
        plt.plot(itsA, logA['train_bal_acc'], '--', label=f'{tagA} Train BalAcc', color=cA, alpha=0.8)
    if 'train_bal_acc' in logB:
        plt.plot(itsB, logB['train_bal_acc'], '--', label=f'{tagB} Train BalAcc', color=cB, alpha=0.8)

    plt.xlabel('Outer iter'); plt.ylabel('Balanced Acc')
    plt.title('Balanced Accuracy (Train + Test)')
    plt.grid(True); plt.legend()

    # --- (2) Losses: g(train), f(val), g(val) ---
    plt.subplot(1,3,2)
    plt.plot(itsA, logA['train_g'], '-',  label=f'{tagA} g(train)', color=cA)
    plt.plot(itsA, logA['val_f'],   '--', label=f'{tagA} f(val)',   color=cA, alpha=0.9)
    if 'val_g' in logA:
        plt.plot(itsA, logA['val_g'],   ':', label=f'{tagA} g(val)', color=cA, alpha=0.9)

    plt.plot(itsB, logB['train_g'], '-',  label=f'{tagB} g(train)', color=cB)
    plt.plot(itsB, logB['val_f'],   '--', label=f'{tagB} f(val)',   color=cB, alpha=0.9)
    if 'val_g' in logB:
        plt.plot(itsB, logB['val_g'],   ':', label=f'{tagB} g(val)', color=cB, alpha=0.9)

    plt.xlabel('Outer iter'); plt.ylabel('Loss')
    plt.title('Losses'); plt.grid(True); plt.legend()

    # --- (3) Gamma schedule ---
    plt.subplot(1,3,3)
    plt.plot(itsA, logA['gamma'], '-', label=f'{tagA} γ', color=cA)
    plt.plot(itsB, logB['gamma'], '-', label=f'{tagB} γ', color=cB)
    plt.xlabel('Outer iter'); plt.ylabel('gamma')
    plt.title('Penalty coefficient'); plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.savefig('compare_vpbgd_gpbgd_param_loss.png', dpi=300, bbox_inches='tight')
    plt.show()


# ----------------------------- Simple JSON saver -----------------------------
def save_logs(logs, path):
    safe = {k: v for k, v in logs.items() if isinstance(v, (list, dict, str, int, float))}
    with open(path, 'w') as f:
        json.dump(safe, f)

# ----------------------------- Main -----------------------------
def main():
    base = dict(
        seed=2424,
        imbalance_base=0.4,
        bs_train=64, bs_val=64,
        reg=0.0,
        y_clip=1.0,     # grad clip on model (None to disable)
        x_clip=None,    # grad clip on x (None to disable)
        outer_itr=100,  # keep moderate for demo; increase for stronger results
        gamma_init=0.0,
        log_every=10
    )
    common = prepare_data(seed=base['seed'], imbalance_base=base['imbalance_base'],
                          bs_train=base['bs_train'], bs_val=base['bs_val'])


    # ---- G-PBGD config ----
    cfg_g = dict(**base,
        lrx=1,
        lry=1,
        gamma_max=10,
        gamma_steps=5000
    )
    print("\nRunning G-PBGD on parametric loss tuning (imbalanced MNIST)...")
    logs_g = run_g_pbgd(cfg_g, common)

    # ---- V-PBGD config ----
    cfg_v = dict(**base,
        lrx=5e-1,       # lr for x (Δ_raw, γ_raw)
        lry=5e-1,       # lr for y (model)
        lr_inner=0.01,   # inner-loop lr for omega
        inner_itr=100,    # inner steps per outer iter
        gamma_max=50,
        gamma_steps=200
    )
    print("Running V-PBGD on parametric loss tuning (imbalanced MNIST)...")
    logs_v = run_v_pbgd(cfg_v, common)

    # Save for run_all_algorithms
    save_logs(logs_v, 'logs_vpbgd.json')
    save_logs(logs_g, 'logs_gpbgd.json')

    # ---- Compare ----
    compare_logs(logs_v, 'V-PBGD', logs_g, 'G-PBGD')

if __name__ == "__main__":
    main()
