# iaptt_gm_param_loss_imbalanced_fixed.py
import os, ssl, random, json, time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# SSL convenience for MNIST
# ---------------------------------------------------------------------
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------------------------------------------------------
# Device & seeds
# ---------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if os.environ.get("HYPERGRAD_REQUIRE_CUDA", "0") == "1" and device.type != "cuda":
    raise RuntimeError("HYPERGRAD_REQUIRE_CUDA=1 but CUDA is not available.")

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
FASHION_MEAN, FASHION_STD = (0.2860,), (0.3530,)


def _dataset_spec(name: str):
    name = (name or "mnist").strip().lower()
    if name in {"fashion", "fashionmnist", "fashion-mnist"}:
        return datasets.FashionMNIST, FASHION_MEAN, FASHION_STD, "FashionMNIST"
    return datasets.MNIST, MNIST_MEAN, MNIST_STD, "MNIST"

# Keep the parametric loss close to identity to avoid degenerate scaling/biasing of logits.
DELTA_BOUND = 0.5
GAMMA_SPAN = 0.15

def print_device_banner(tag: str = "IAPTT-GM") -> None:
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

def set_seed(seed=2424):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------
# Data prep: imbalanced MNIST and reweighting vector u for validation
# ---------------------------------------------------------------------
def create_imbalanced_indices(labels_numpy, base=0.4, *, min_keep_prob: float = 0.0):
    keep_prob = np.power(base, labels_numpy)   # p(y=i) ∝ base^i
    min_keep_prob = float(np.clip(float(min_keep_prob), 0.0, 1.0))
    if min_keep_prob > 0.0:
        keep_prob = np.maximum(keep_prob, min_keep_prob)
    keep_mask = np.random.rand(len(labels_numpy)) < keep_prob
    return np.where(keep_mask)[0]

@torch.no_grad()
def compute_u_from_loader(loader, n_classes=10):
    counts = torch.zeros(n_classes, device=device)
    for xb, yb in loader:
        yb = yb.to(device)
        counts += torch.bincount(yb, minlength=n_classes).to(device)
    counts = torch.clamp(counts, min=1)
    return 1.0 / counts

def tensorize_dataset(ds, batch_size=1024):
    """Materialize an entire (possibly Subset) torchvision dataset into tensors."""
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    xs, ys = [], []
    for xb, yb in dl:
        xs.append(xb); ys.append(yb)
    X = torch.cat(xs, 0)   # shape [N,784] or [N,1,28,28] depending on transform
    y = torch.cat(ys, 0)   # shape [N]
    return X, y


# ---------------- Spurious feature (train-only) ----------------
def _lcg_u32(x: int) -> int:
    return (int(x) * 1664525 + 1013904223) & 0xFFFFFFFF


def _u01_from_u32(x: int) -> float:
    return float(int(x) & 0xFFFFFFFF) / float(2**32)


def _spurious_patch_positions(size: int, n_positions: int) -> list[tuple[int, int]]:
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
        # Support both flattened ([784]) and image ([1,28,28]) tensors.
        if x_out.ndim == 1:
            x_img = x_out.view(1, 28, 28)
            x_img[..., r : r + size, c : c + size] = float(self.value)
            return x_img.view(-1), y
        x_out[..., r : r + size, c : c + size] = float(self.value)
        return x_out, y

def prepare_data(
    seed=2424,
    imbalance_base=0.4,
    val_size=100,
    *,
    dataset: str = "mnist",
    min_keep_prob: float = 0.0,
    train_fraction: float = 0.99,
    match_val_test_imbalance: bool = False,
    val_min_per_class: int = 0,
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

    # Train-only corruption: RandomErasing is applied before the spurious patch (which is injected after transforms),
    # so the shortcut remains a clean/easy feature even when the digit itself is partially erased.
    train_erase_p = float(np.clip(float(train_erase_p), 0.0, 1.0))
    tfm_plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(ds_mean, ds_std)])
    tfm_train = (
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(ds_mean, ds_std),
                transforms.RandomErasing(p=train_erase_p, scale=(0.15, 0.35), ratio=(0.3, 3.3), value=0.0),
            ]
        )
        if train_erase_p > 0.0
        else tfm_plain
    )

    full_train_train = ds_cls('.', train=True,  transform=tfm_train, download=True)
    full_train_plain = ds_cls('.', train=True,  transform=tfm_plain, download=True)
    full_test  = ds_cls('.', train=False, transform=tfm_plain, download=True)

    # 99/1 split of the original train split (requested).
    n = int(len(full_train_plain))
    perm = torch.randperm(n)
    n_train_pool = int(float(train_fraction) * n)
    n_train_pool = max(1, min(n - 1, n_train_pool))
    train_pool = perm[:n_train_pool].cpu().numpy()
    val_pool = perm[n_train_pool:].cpu().numpy()

    # Build imbalanced train subset from the train pool.
    labels_np = np.array(full_train_plain.targets)[train_pool]
    keep_idx_local = create_imbalanced_indices(labels_np, base=imbalance_base, min_keep_prob=min_keep_prob)
    train_idx = train_pool[keep_idx_local]
    train_set = torch.utils.data.Subset(full_train_train, train_idx.tolist())

    # Validation is the remaining split.
    # (val_size kept for backward-compat but ignored when using train_fraction split.)
    # Optional: match the same train imbalance ratio on val/test too.
    test_set = full_test
    if bool(match_val_test_imbalance):
        rng_val = np.random.default_rng(int(seed) + 12345)
        labels_val = np.array(full_train_plain.targets)[val_pool]
        keep_prob_val = np.power(float(imbalance_base), labels_val)
        if min_keep_prob > 0.0:
            keep_prob_val = np.maximum(keep_prob_val, min_keep_prob)
        keep_mask_val = rng_val.random(len(labels_val)) < keep_prob_val
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
        val_set = torch.utils.data.Subset(full_train_plain, val_idx.tolist())

        rng_te = np.random.default_rng(int(seed) + 23456)
        labels_te = np.array(full_test.targets)
        keep_prob_te = np.power(float(imbalance_base), labels_te)
        if min_keep_prob > 0.0:
            keep_prob_te = np.maximum(keep_prob_te, min_keep_prob)
        keep_mask_te = rng_te.random(len(labels_te)) < keep_prob_te
        te_idx = np.where(keep_mask_te)[0]
        if int(te_idx.size) <= 0:
            te_idx = np.arange(len(full_test))
        test_set = torch.utils.data.Subset(full_test, te_idx.tolist())
    else:
        val_set = torch.utils.data.Subset(full_train_plain, val_pool.tolist())

    # Optional: add a spurious, label-correlated patch to the *training* distribution only.
    if bool(spurious_train) and float(spurious_patch_prob) > 0.0:
        train_set = SpuriousPatchDataset(
            train_set,
            seed=int(seed),
            prob=float(spurious_patch_prob),
            flip_prob=float(spurious_patch_flip_prob),
            size=int(spurious_patch_size),
            value=float(spurious_patch_value),
            n_positions=int(spurious_n_positions),
        )

    # Tensorize
    Xtr, ytr = tensorize_dataset(train_set)   # [Ntr, 1, 28, 28], [Ntr]
    Xva, yva = tensorize_dataset(val_set)     # [Nva, 1, 28, 28], [Nva]
    Xte, yte = tensorize_dataset(test_set)    # [Nte, 1, 28, 28], [Nte]

    # Compute u on validation
    val_tensor_ds = torch.utils.data.TensorDataset(Xva, yva)
    val_loader = torch.utils.data.DataLoader(val_tensor_ds, batch_size=2048, shuffle=False)
    u_vec = compute_u_from_loader(val_loader, n_classes=10)

    # Move to device
    Xtr = Xtr.to(device); ytr = ytr.to(device)
    Xva = Xva.to(device); yva = yva.to(device)
    Xte = Xte.to(device); yte = yte.to(device)
    u_vec = u_vec.to(device)

    return (Xtr, ytr, Xva, yva, Xte, yte, u_vec)

# ---------------------------------------------------------------------
# Functional models (for clean inner dynamics)
# Default in this project: a small CNN (to match other methods).
# Legacy: two-layer MLP (set `IAPTT_MODEL=mlp`).
# ---------------------------------------------------------------------
def params_init(d_in, d_hidden=300, d_out=10):
    w1 = torch.empty(d_hidden, d_in,  device=device);  torch.nn.init.kaiming_uniform_(w1, nonlinearity='relu')
    b1 = torch.zeros(d_hidden,        device=device)
    w2 = torch.empty(d_out,   d_hidden, device=device); torch.nn.init.kaiming_uniform_(w2, nonlinearity='relu')
    b2 = torch.zeros(d_out,           device=device)
    return [torch.nn.Parameter(w1, requires_grad=True),
            torch.nn.Parameter(b1, requires_grad=True),
            torch.nn.Parameter(w2, requires_grad=True),
            torch.nn.Parameter(b2, requires_grad=True)]

def functional_forward(x_flat, params):
    W1, b1, W2, b2 = params
    h = F.relu(x_flat @ W1.t() + b1)
    return h @ W2.t() + b2  # logits (before parametric tuning)

def l2_norm(params):
    return sum((p**2).sum() for p in params)

def params_init_cnn(d_hidden=256, d_out=10):
    wc1 = torch.empty(32, 1, 5, 5, device=device);  torch.nn.init.kaiming_uniform_(wc1, nonlinearity='relu')
    bc1 = torch.zeros(32, device=device)
    wc2 = torch.empty(64, 32, 5, 5, device=device); torch.nn.init.kaiming_uniform_(wc2, nonlinearity='relu')
    bc2 = torch.zeros(64, device=device)
    w1 = torch.empty(d_hidden, 64 * 7 * 7, device=device); torch.nn.init.kaiming_uniform_(w1, nonlinearity='relu')
    b1 = torch.zeros(d_hidden, device=device)
    w2 = torch.empty(d_out, d_hidden, device=device); torch.nn.init.kaiming_uniform_(w2, nonlinearity='linear')
    b2 = torch.zeros(d_out, device=device)
    return [
        torch.nn.Parameter(wc1, requires_grad=True),
        torch.nn.Parameter(bc1, requires_grad=True),
        torch.nn.Parameter(wc2, requires_grad=True),
        torch.nn.Parameter(bc2, requires_grad=True),
        torch.nn.Parameter(w1, requires_grad=True),
        torch.nn.Parameter(b1, requires_grad=True),
        torch.nn.Parameter(w2, requires_grad=True),
        torch.nn.Parameter(b2, requires_grad=True),
    ]

def functional_forward_cnn(x_img, params):
    wc1, bc1, wc2, bc2, w1, b1, w2, b2 = params
    h = F.conv2d(x_img, wc1, bc1, stride=1, padding=2)
    h = F.relu(h)
    h = F.max_pool2d(h, kernel_size=2, stride=2)
    h = F.conv2d(h, wc2, bc2, stride=1, padding=2)
    h = F.relu(h)
    h = F.max_pool2d(h, kernel_size=2, stride=2)
    h = h.flatten(1)
    h = F.linear(h, w1, b1)
    h = F.relu(h)
    return F.linear(h, w2, b2)

def _forward_in_chunks(X, params, forward_fn, chunk_size: int):
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or int(X.shape[0]) <= chunk_size:
        return forward_fn(X, params)
    outs = []
    n = int(X.shape[0])
    for s in range(0, n, chunk_size):
        outs.append(forward_fn(X[s : s + chunk_size], params))
    return torch.cat(outs, dim=0)

# Global forward hook (set in `run_iaptt_gm`).
_IAPTT_FORWARD = functional_forward
_IAPTT_FORWARD_BATCH = 0

# ---------------------------------------------------------------------
# Parametric loss tuning: x = (Δ_raw, γ_raw)  -> (Δ, γ) with γ>0
# Regularize towards Δ=0 and γ≈1 in the UPPER objective.
# ---------------------------------------------------------------------
def unpack_x(delta_raw, gamma_raw):
    delta = DELTA_BOUND * torch.tanh(delta_raw)
    gamma = 1.0 + GAMMA_SPAN * torch.tanh(gamma_raw)
    return delta, gamma

# ---------------------------------------------------------------------
# Losses
#   g_LL: train CE (imbalanced), adjusted logits by (Δ,γ)
#   f_UL: reweighted val CE + x-regularization
# ---------------------------------------------------------------------
def g_LL(delta_raw, gamma_raw, y_params, Xtr, ytr, reg=0.0):
    Δ, γ = unpack_x(delta_raw, gamma_raw)
    logits = _IAPTT_FORWARD(Xtr, y_params)
    adj = logits * γ.view(1, -1) + Δ.view(1, -1)
    ce = F.cross_entropy(adj, ytr, reduction='mean')
    return ce + reg * l2_norm(y_params)

def f_UL(delta_raw, gamma_raw, y_params, Xva, yva, u_vec,
         lam_delta=1e-4, lam_gamma=1e-4):
    Δ, γ = unpack_x(delta_raw, gamma_raw)
    logits = _IAPTT_FORWARD(Xva, y_params)
    adj = logits * γ.view(1, -1) + Δ.view(1, -1)
    loss_vec = F.cross_entropy(adj, yva, reduction='none')
    w = u_vec[yva]
    base = (w * loss_vec).sum() / (w.sum() + 1e-12)
    # Upper-regularization for (Δ, γ)
    reg_x = lam_delta * (Δ**2).sum() + lam_gamma * ((γ - 1.0)**2).sum()
    return base + reg_x

@torch.no_grad()
def avg_g_on(loader_X, loader_y, delta_raw, gamma_raw, y_params):
    Δ, γ = unpack_x(delta_raw, gamma_raw)
    logits = _IAPTT_FORWARD(loader_X, y_params)
    adj = logits * γ.view(1, -1) + Δ.view(1, -1)
    return F.cross_entropy(adj, loader_y, reduction='mean').item()

@torch.no_grad()
def balanced_acc_on(X, y, delta_raw, gamma_raw, y_params, n_classes=10):
    Δ, γ = unpack_x(delta_raw, gamma_raw)
    logits = _IAPTT_FORWARD(X, y_params)
    adj = logits * γ.view(1, -1) + Δ.view(1, -1)
    preds = adj.argmax(1)
    yt = y
    yt_np = yt.detach().cpu().numpy()
    yp_np = preds.detach().cpu().numpy()
    present = np.unique(yt_np)
    if present.size == 0:
        return 0.0
    recalls = []
    for c in present:
        mask = (yt_np == c)
        denom = mask.sum()
        tp = (yp_np[mask] == c).sum() if denom > 0 else 0
        recalls.append(tp / max(denom, 1))
    return float(np.mean(recalls))

# ---------------------------------------------------------------------
# Utilities for soft truncation over trajectory
# ---------------------------------------------------------------------
def softmax_weights(vals, tau):
    # vals: [K] tensor
    z = (vals / max(tau, 1e-8))
    z = z - z.max()
    w = torch.exp(z)
    return w / (w.sum() + 1e-12)

def weighted_params_sum(params_list, weights):
    """
    params_list: list length K, each is a list of tensors [W1,b1,W2,b2]
    weights: [K] tensor
    returns: list of tensors (same shapes), weighted sum with graph
    """
    K = len(params_list)
    out = []
    for j in range(len(params_list[0])):
        acc = 0.0
        for k in range(K):
            acc = acc + weights[k] * params_list[k][j]
        out.append(acc)
    return out

# ---------------------------------------------------------------------
# IAPTT-GM main routine (with soft/hard truncation)
# ---------------------------------------------------------------------
def run_iaptt_gm(cfg):
    print_device_banner("IAPTT-GM")
    set_seed(cfg['seed'])
    Xtr, ytr, Xva, yva, Xte, yte, u_vec = prepare_data(
        seed=cfg['seed'],
        imbalance_base=cfg['imbalance_base'],
        val_size=cfg['val_size'],
        dataset=str(cfg.get("dataset", os.environ.get("IMB_DATASET", "mnist"))),
        min_keep_prob=float(cfg.get("min_keep_prob", os.environ.get("IMB_MIN_KEEP_PROB", 0.0)) or 0.0),
        train_fraction=float(cfg.get("train_fraction", 0.99)),
        match_val_test_imbalance=bool(cfg.get("match_val_test_imbalance", False)),
        val_min_per_class=int(cfg.get("val_min_per_class", 0) or 0),
        train_erase_p=float(cfg.get("train_erase_p", 0.0) or 0.0),
        spurious_train=bool(cfg.get("spurious_train", False)),
        spurious_patch_prob=float(cfg.get("spurious_patch_prob", 0.0)),
        spurious_patch_flip_prob=float(cfg.get("spurious_patch_flip_prob", 0.0)),
        spurious_patch_size=int(cfg.get("spurious_patch_size", 4)),
        spurious_patch_value=float(cfg.get("spurious_patch_value", 3.0)),
        spurious_n_positions=int(cfg.get("spurious_n_positions", 10)),
    )
    global _IAPTT_FORWARD, _IAPTT_FORWARD_BATCH
    model_type = str(cfg.get("model", os.environ.get("IAPTT_MODEL", "cnn"))).strip().lower()
    if model_type not in {"cnn", "mlp"}:
        model_type = "cnn"
    _IAPTT_FORWARD_BATCH = int(cfg.get("forward_batch", os.environ.get("IAPTT_FORWARD_BATCH", 4096 if model_type == "cnn" else 0)) or 0)

    # Upper variables: (Δ_raw, γ_raw) and z = initializer for y
    delta_raw = torch.nn.Parameter(torch.zeros(10, device=device), requires_grad=True)
    gamma_raw = torch.nn.Parameter(torch.zeros(10, device=device), requires_grad=True)
    if model_type == "cnn":
        z_params = params_init_cnn(d_hidden=int(cfg.get("d_hidden", 256)), d_out=10)  # learned initializer
        _IAPTT_FORWARD = lambda X, params: _forward_in_chunks(X, params, functional_forward_cnn, _IAPTT_FORWARD_BATCH)
    else:
        # Legacy MLP path: flatten images once.
        Xtr = Xtr.view(Xtr.size(0), -1)
        Xva = Xva.view(Xva.size(0), -1)
        Xte = Xte.view(Xte.size(0), -1)
        d_in = int(Xtr.size(1))
        z_params = params_init(d_in, d_hidden=cfg['d_hidden'], d_out=10)  # learned initializer
        _IAPTT_FORWARD = functional_forward

    # Optimizers for upper variables
    if cfg.get('use_adam', True):
        opt_x = torch.optim.Adam([delta_raw, gamma_raw], lr=cfg['lrx'])
        opt_z = torch.optim.Adam(z_params, lr=cfg['lrz'])
    else:
        opt_x = torch.optim.SGD([delta_raw, gamma_raw], lr=cfg['lrx'], momentum=0.9)
        opt_z = torch.optim.SGD(z_params, lr=cfg['lrz'], momentum=0.9)

    logs = {k: [] for k in ['iter','elapsed_s','k_bar','train_bal_acc','test_bal_acc','g_train','g_val','f_val']}

    time_budget_s = cfg.get("time_budget_s", None)
    if time_budget_s is not None:
        try:
            time_budget_s = float(time_budget_s)
            if time_budget_s <= 0:
                time_budget_s = None
        except Exception:
            time_budget_s = None
    start_t = time.perf_counter()

    def _fmt_loss(x: float) -> str:
        try:
            x = float(x)
        except Exception:
            return str(x)
        if not np.isfinite(x):
            return str(x)
        return f"{x:.4f}" if abs(x) >= 1e-3 else f"{x:.2e}"

    # Inner-loop stabilization knobs
    y_clip_inner = cfg.get("y_clip_inner", None)
    try:
        y_clip_inner = float(y_clip_inner) if y_clip_inner is not None else 0.0
    except Exception:
        y_clip_inner = 0.0
    y_clip_inner = float(y_clip_inner) if y_clip_inner and y_clip_inner > 0 else 0.0
    inner_retries = int(cfg.get("inner_retries", 2) or 0)
    inner_retries = max(0, inner_retries)
    inner_lr_decay = float(cfg.get("inner_lr_decay", 0.5) or 0.5)
    inner_lr_decay = float(inner_lr_decay) if 0.0 < float(inner_lr_decay) < 1.0 else 0.5

    def _all_finite(tensors) -> bool:
        for x in tensors:
            if not torch.isfinite(x).all().item():
                return False
        return True

    def _clip_grad_list(grads, max_norm: float):
        if not max_norm or max_norm <= 0:
            return grads
        with torch.no_grad():
            sq = 0.0
            for g in grads:
                sq = sq + float(g.detach().pow(2).sum().item())
            total = (sq ** 0.5) if sq > 0 else 0.0
            if not (total > 0.0):
                return grads
            coef = max_norm / (total + 1e-12)
            if coef >= 1.0:
                return grads
        return [g * float(coef) for g in grads]

    for itr in range(cfg['outer_itr']):
        if time_budget_s is not None and (time.perf_counter() - start_t) >= time_budget_s:
            break
        # ---- Build follower trajectory y_1..y_K from y_0=z ----
        # IMPORTANT: do NOT detach from z_params here.
        # We want gradients of the upper objective to flow back to z (learned initialization),
        # otherwise z never updates and the follower re-starts from the same random init each outer iter.
        traj_params, traj_Fvals = [], []
        lr_inner = float(cfg["lr_inner"])
        inner_ok = False
        y_star = list(z_params)

        for _attempt in range(inner_retries + 1):
            yk = list(z_params)
            traj_params, traj_Fvals = [], []
            inner_ok = True

            for _k in range(int(cfg["K_inner"])):
                loss_LL = g_LL(delta_raw, gamma_raw, yk, Xtr, ytr, reg=cfg["reg"])
                if not torch.isfinite(loss_LL).all().item():
                    inner_ok = False
                    break
                grads = torch.autograd.grad(loss_LL, yk, create_graph=True)
                if not _all_finite(grads):
                    inner_ok = False
                    break
                if y_clip_inner > 0:
                    grads = _clip_grad_list(grads, y_clip_inner)
                yk = [p - lr_inner * g for p, g in zip(yk, grads)]
                if not _all_finite(yk):
                    inner_ok = False
                    break

                F_val = f_UL(
                    delta_raw,
                    gamma_raw,
                    yk,
                    Xva,
                    yva,
                    u_vec,
                    lam_delta=cfg["lam_delta"],
                    lam_gamma=cfg["lam_gamma"],
                )
                if not torch.isfinite(F_val).all().item():
                    inner_ok = False
                    break
                traj_params.append(yk)
                traj_Fvals.append(F_val)

            if inner_ok and len(traj_Fvals) > 0:
                break
            lr_inner = lr_inner * inner_lr_decay

        if not (inner_ok and len(traj_Fvals) > 0):
            # Skip this outer update rather than poisoning the run with NaNs.
            if (itr % cfg["eval_every"] == 0) or (itr == (int(cfg["outer_itr"]) - 1)):
                g_tr = avg_g_on(Xtr, ytr, delta_raw, gamma_raw, y_star)
                g_va = avg_g_on(Xva, yva, delta_raw, gamma_raw, y_star)
                f_va = f_UL(
                    delta_raw,
                    gamma_raw,
                    y_star,
                    Xva,
                    yva,
                    u_vec,
                    lam_delta=cfg["lam_delta"],
                    lam_gamma=cfg["lam_gamma"],
                ).item()
                tr_bal = balanced_acc_on(Xtr, ytr, delta_raw, gamma_raw, y_star)
                te_bal = balanced_acc_on(Xte, yte, delta_raw, gamma_raw, y_star)
                logs["iter"].append(itr + 1)
                logs["elapsed_s"].append(float(time.perf_counter() - start_t))
                logs["k_bar"].append(0)
                logs["g_train"].append(g_tr)
                logs["g_val"].append(g_va)
                logs["f_val"].append(f_va)
                logs["train_bal_acc"].append(tr_bal)
                logs["test_bal_acc"].append(te_bal)
                print(
                    f"[IAPTT-GM/ParamLoss] it {itr+1:5d} | inner_failed | "
                    f"g(train) {_fmt_loss(g_tr)} | g(val) {_fmt_loss(g_va)} | f(val) {_fmt_loss(f_va)} | "
                    f"Train BalAcc {tr_bal:.4f} | Test BalAcc {te_bal:.4f}"
                )
            continue

        # ---- Trajectory Truncation (min/max over the LL trajectory) ----
        # For this task (minimize validation loss), the safe default is select_mode="min".
        select_mode = str(cfg.get("select_mode", "min")).strip().lower()
        if select_mode not in {"min", "max"}:
            raise ValueError("select_mode must be 'min' or 'max'.")
        want_min = (select_mode == "min")

        vals = torch.stack(traj_Fvals)  # [K]
        if cfg['tau'] is None:
            # hard argmin/argmax
            with torch.no_grad():
                k_bar = int(torch.argmin(vals).item()) if want_min else int(torch.argmax(vals).item())
            y_star = traj_params[k_bar]
            F_star = traj_Fvals[k_bar]
        else:
            # soft selection
            score = (-vals).detach() if want_min else vals.detach()
            w = softmax_weights(score, cfg['tau'])  # stop grads through weights for stability
            k_bar = int(torch.argmin(vals.detach()).item()) if want_min else int(torch.argmax(vals.detach()).item())
            y_star = weighted_params_sum(traj_params, w)    # weighted params
            # Evaluate F at the weighted parameters (keeps graph to x & z via y_star)
            F_star = f_UL(delta_raw, gamma_raw, y_star, Xva, yva, u_vec,
                          lam_delta=cfg['lam_delta'], lam_gamma=cfg['lam_gamma'])

        # ---- Upper updates: ∇ wrt (Δ_raw, γ_raw) and z
        opt_x.zero_grad(); opt_z.zero_grad()
        F_star.backward()
        # If grads go non-finite, skip the update (keeps run usable under tight budgets).
        grads_ok = True
        if delta_raw.grad is None or gamma_raw.grad is None:
            grads_ok = False
        else:
            grads_ok = grads_ok and torch.isfinite(delta_raw.grad).all().item()
            grads_ok = grads_ok and torch.isfinite(gamma_raw.grad).all().item()
        if grads_ok:
            for p in z_params:
                if p.grad is None or (not torch.isfinite(p.grad).all().item()):
                    grads_ok = False
                    break
        if not grads_ok:
            opt_x.zero_grad()
            opt_z.zero_grad()
            continue
        if cfg['x_clip']: torch.nn.utils.clip_grad_norm_([delta_raw, gamma_raw], cfg['x_clip'])
        if cfg['z_clip']: torch.nn.utils.clip_grad_norm_(z_params, cfg['z_clip'])
        opt_x.step(); opt_z.step()

        # ---- Logging
        if (itr % cfg['eval_every'] == 0) or (itr == (int(cfg['outer_itr']) - 1)):
            g_tr = avg_g_on(Xtr, ytr, delta_raw, gamma_raw, y_star)
            g_va = avg_g_on(Xva, yva, delta_raw, gamma_raw, y_star)
            f_va = f_UL(delta_raw, gamma_raw, y_star, Xva, yva, u_vec,
                        lam_delta=cfg['lam_delta'], lam_gamma=cfg['lam_gamma']).item()
            tr_bal = balanced_acc_on(Xtr, ytr, delta_raw, gamma_raw, y_star)
            te_bal = balanced_acc_on(Xte, yte, delta_raw, gamma_raw, y_star)

            logs['iter'].append(itr + 1)
            logs['elapsed_s'].append(float(time.perf_counter() - start_t))
            logs['k_bar'].append(k_bar+1)
            logs['g_train'].append(g_tr); logs['g_val'].append(g_va); logs['f_val'].append(f_va)
            logs['train_bal_acc'].append(tr_bal); logs['test_bal_acc'].append(te_bal)
            print(f"[IAPTT-GM/ParamLoss] it {itr+1:5d} | k_bar={k_bar+1}/{cfg['K_inner']} | "
                  f"g(train) {_fmt_loss(g_tr)} | g(val) {_fmt_loss(g_va)} | f(val) {_fmt_loss(f_va)} | "
                  f"Train BalAcc {tr_bal:.4f} | Test BalAcc {te_bal:.4f}")

    logs['method'] = 'IAPTT-GM-ParamLoss'
    logs['config'] = dict(cfg)
    return logs

def save_logs(logs, path):
    safe = {k: v for k, v in logs.items() if isinstance(v, (list, dict, str, int, float))}
    with open(path, 'w') as f:
        json.dump(safe, f)

# ---- Consistent colors per algorithm ----
ALG_COLORS = {
    'IAPTT-GM': '#ff7f0e',   # orange
    'V-PBGD':   '#1f77b4',   # blue
    'G-PBGD':   '#d62728',   # red
    'HG–NoSel': '#2ca02c',   # green
    'HG–Sel':   '#9467bd',   # purple
}
def _alg_color(tag): return ALG_COLORS.get(tag, '#333333')

def compare_logs(logA, tagA, logB, tagB, save_as='compare_iaptt_vs_other.png'):
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
        plt.plot(itsA, logA['train_bal_acc'], '--', label=f'{tagA} Train BalAcc', color=cA, alpha=0.85)
    if 'train_bal_acc' in logB:
        plt.plot(itsB, logB['train_bal_acc'], '--', label=f'{tagB} Train BalAcc', color=cB, alpha=0.85)
    plt.xlabel('Outer iter'); plt.ylabel('Balanced Acc')
    plt.title('Balanced Accuracy (Train + Test)')
    plt.grid(True); plt.legend()

    # --- (2) Losses: g(train), f(val), g(val) ---
    plt.subplot(1,3,2)
    plt.plot(itsA, logA['g_train'], '-',  label=f'{tagA} g(train)', color=cA)
    plt.plot(itsA, logA['f_val'],   '--', label=f'{tagA} f(val)',   color=cA)
    if 'g_val' in logA:
        plt.plot(itsA, logA['g_val'],   ':', label=f'{tagA} g(val)', color=cA)
    plt.plot(itsB, logB['g_train'], '-',  label=f'{tagB} g(train)', color=cB)
    plt.plot(itsB, logB['f_val'],   '--', label=f'{tagB} f(val)',   color=cB)
    if 'g_val' in logB:
        plt.plot(itsB, logB['g_val'],   ':', label=f'{tagB} g(val)', color=cB)
    plt.xlabel('Outer iter'); plt.ylabel('Loss')
    plt.title('Losses'); plt.grid(True); plt.legend()

    # --- (3) IAPTT indicator (k̄) or other schedule ---
    plt.subplot(1,3,3)
    if 'k_bar' in logA:
        plt.plot(itsA, logA['k_bar'], '-', label=f'{tagA} k̄ (pess. step)', color=cA)
    if 'k_bar' in logB:
        plt.plot(itsB, logB['k_bar'], '-', label=f'{tagB} k̄ (pess. step)', color=cB)
    plt.xlabel('Outer iter'); plt.ylabel('k̄ (1..K)')
    plt.title('Trajectory Truncation Index'); plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.savefig(save_as, dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg = dict(
        seed=2424,
        imbalance_base=0.4,
        val_size=100,
        d_hidden=300,

        # IAPTT-GM follower
        K_inner=10,        # ↑ longer trajectory
        lr_inner=0.1,      # ↑ stronger steps
        reg=5e-4,           # small inner weight decay

        # Upper regularization for (Δ, γ)
        lam_delta=1e-4,
        lam_gamma=1e-4,

        # Outer optim
        use_adam=True,
        lrx=0.1,
        lrz=0.1,

        # Gradient clipping
        x_clip=5.0,
        z_clip=5.0,

        # Truncation selector: set tau=None for hard argmax
        select_mode="min",  # "min" is appropriate for minimizing validation loss (default).
        tau=0.5,            # soft-min: more stable than hard selection; try 0.3–1.0

        outer_itr=100,
        eval_every=10
    )
    logs = run_iaptt_gm(cfg)
    save_logs(logs, 'logs_iaptt_gm_paramloss_fixed.json')

if __name__ == "__main__":
    main()
