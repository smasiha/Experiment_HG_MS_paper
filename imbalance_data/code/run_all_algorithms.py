import os
import json
import argparse
import importlib.util
import importlib.machinery
import sys

import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Defaults (overridable via CLI)
OUTER_ITERS = 100
SEED = 2424
SAVE_DIR = "."
TIME_BUDGET_S = None
TIME_GRID_S = 1.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, None)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _call_with_supported_kwargs(func, kwargs: dict):
    """
    Call `func(**kwargs)` but only with keyword args supported by its signature.
    This lets us share global experiment knobs across different algorithm files.
    """
    try:
        import inspect

        sig = inspect.signature(func)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(**supported)
    except Exception:
        return func(**kwargs)


# ---------------- Experiment-wide knobs (env/preset) ----------------
# Why this exists:
# - Minima selection depends on a *reliable* upper-level signal f(val).
# - With a tiny validation set (99/1 split) and many candidates, selection becomes noisy and conservative gates
#   tend to keep the base candidate, making HG-minsel look similar to HG-baseline.
#
# Use `IMB_PRESET=minsel_effective` to increase validation signal and branch diversity by default.
IMB_PRESET = (os.environ.get("IMB_PRESET", "") or "").strip().lower()
IMB_DATASET = (os.environ.get("IMB_DATASET", "mnist") or "mnist").strip().lower()
# NOTE: `imbalance_base` too small can drop entire classes from the LL training data
# (since keep_prob = imbalance_base ** label). In that regime HG cannot recover the missing
# classes (Δ,γ are too limited), so test balanced accuracy collapses.
# Presets:
# - default: original-ish behavior
# - minsel_effective: slightly larger validation split + safer imbalance for selection experiments
# - minsel_mismatch: stronger train–val mismatch via more severe LL imbalance + train-only corruption
# - minsel_showcase: adds a *train-only* spurious shortcut feature (label-correlated patch) to make
#   the LL problem more multi-solution, so minima selection can matter more.
_default_imb_base = (
    0.2
    if IMB_PRESET in {"minsel_mismatch"}
    else (0.5 if IMB_PRESET in {"minsel_effective", "minsel_showcase"} else 0.4)
)
# For minima selection to be meaningful, f(val) needs enough signal. However, making the
# validation split too large can also shrink the LL train pool and (with strong imbalance)
# drop rare classes. 5% validation is a safer default than 10% in this setup.
_default_train_frac = (
    0.95 if IMB_PRESET in {"minsel_effective", "minsel_mismatch", "minsel_showcase", "minsel_spurious"} else 0.99
)  # 5% vs 1%
_default_balanced_val = True
_default_augment_train = False if IMB_PRESET in {"minsel_mismatch", "minsel_showcase"} else True

IMB_IMBALANCE_BASE = float(_env_float("IMB_IMBALANCE_BASE", _default_imb_base))
# Prevent the class-keep probability base**label from becoming so tiny that some classes disappear entirely.
# Setting this >0 makes the imbalance task harder but still well-posed for balanced-accuracy evaluation.
_default_min_keep_prob = 0.003 if IMB_PRESET == "minsel_mismatch" else 0.0
IMB_MIN_KEEP_PROB = float(np.clip(_env_float("IMB_MIN_KEEP_PROB", _default_min_keep_prob), 0.0, 1.0))
IMB_TRAIN_FRACTION = float(np.clip(_env_float("IMB_TRAIN_FRACTION", _default_train_frac), 0.50, 0.999))
# Convenience override: set the validation-set size directly (MNIST train has 60,000 examples).
# Example: IMB_VAL_SIZE=600 -> train_fraction = 1 - 600/60000 = 0.99.
_mnist_train_n = 60000
_val_size_env = os.environ.get("IMB_VAL_SIZE", None)
IMB_VAL_SIZE = None
if _val_size_env is not None:
    try:
        _val_n = int(_val_size_env)
        _val_n = max(1, min(_mnist_train_n - 1, _val_n))
        IMB_TRAIN_FRACTION = float(np.clip(1.0 - (float(_val_n) / float(_mnist_train_n)), 0.50, 0.999))
        IMB_VAL_SIZE = _val_n
    except Exception:
        IMB_VAL_SIZE = None
if IMB_VAL_SIZE is None:
    _val_n = int(round(float(_mnist_train_n) * (1.0 - float(IMB_TRAIN_FRACTION))))
    IMB_VAL_SIZE = max(1, min(_mnist_train_n - 1, _val_n))
IMB_BALANCED_VAL = _env_bool("IMB_BALANCED_VAL", _default_balanced_val)
IMB_AUGMENT_TRAIN = _env_bool("IMB_AUGMENT_TRAIN", _default_augment_train)
# If enabled, apply the same class-imbalance mechanism (keep_prob = base**label) to val/test too.
# This makes train/val/test label marginals more similar (at the cost of noisier evaluation on rare classes).
IMB_MATCH_SPLIT_IMBALANCE = _env_bool("IMB_MATCH_SPLIT_IMBALANCE", False)
# When val/test are also downsampled, validation can get very small and drop classes entirely.
# These knobs make the validation batches higher quality for algorithms that consume val minibatches.
IMB_VAL_MIN_PER_CLASS = max(0, _env_int("IMB_VAL_MIN_PER_CLASS", 5 if IMB_MATCH_SPLIT_IMBALANCE else 0))
_default_val_full_batch = bool(IMB_MATCH_SPLIT_IMBALANCE) or (
    IMB_PRESET in {"minsel_effective", "minsel_mismatch", "minsel_showcase", "minsel_spurious"}
)
IMB_VAL_FULL_BATCH = _env_bool("IMB_VAL_FULL_BATCH", _default_val_full_batch)

# Train-only spurious shortcut feature: a small label-correlated patch on *train* images only.
# This intentionally creates an objective mismatch (g(train) sees the shortcut, f(val)/test doesn't),
# which can create multiple LL basins (shortcut vs real features) and make minima selection more impactful.
_default_spurious_train = IMB_PRESET in {"minsel_showcase", "minsel_spurious"}
IMB_SPURIOUS_TRAIN = _env_bool("IMB_SPURIOUS_TRAIN", _default_spurious_train)
#
# NOTE: In the showcase preset we intentionally make the shortcut strong and consistent to
# bias LL training toward the spurious basin; HG-MS can then use candidate diversity and
# explicit selection to recover a val/test-friendly solution under a fixed time budget.
if IMB_PRESET == "minsel_showcase":
    _default_patch_prob = 1.0
    _default_patch_flip = 0.0
    _default_patch_size = 12
    _default_patch_value = 3.0
elif IMB_PRESET == "minsel_spurious":
    # Softer spurious feature: creates a train–val mismatch, but leaves room for some LL solutions
    # that ignore the shortcut (so minima selection can meaningfully improve test accuracy).
    _default_patch_prob = 0.5
    _default_patch_flip = 0.2
    _default_patch_size = 6
    _default_patch_value = 2.0
else:
    _default_patch_prob = 0.0
    _default_patch_flip = 0.0
    _default_patch_size = 4
    _default_patch_value = 3.0
IMB_SPURIOUS_PATCH_PROB = float(np.clip(_env_float("IMB_SPURIOUS_PATCH_PROB", _default_patch_prob), 0.0, 1.0))
IMB_SPURIOUS_PATCH_FLIP_PROB = float(np.clip(_env_float("IMB_SPURIOUS_PATCH_FLIP_PROB", _default_patch_flip), 0.0, 1.0))
IMB_SPURIOUS_PATCH_SIZE = max(1, _env_int("IMB_SPURIOUS_PATCH_SIZE", _default_patch_size))
IMB_SPURIOUS_PATCH_VALUE = float(_env_float("IMB_SPURIOUS_PATCH_VALUE", _default_patch_value))
IMB_SPURIOUS_N_POSITIONS = max(1, _env_int("IMB_SPURIOUS_N_POSITIONS", 10))

# Extra train-only corruption to amplify train–val mismatch (spurious shortcut becomes more attractive).
# This is useful when MNIST is otherwise too easy and all baselines saturate under long budgets.
_default_train_erase_p = 0.40 if IMB_PRESET == "minsel_showcase" else (0.20 if IMB_PRESET == "minsel_mismatch" else 0.0)
IMB_TRAIN_ERASE_P = float(np.clip(_env_float("IMB_TRAIN_ERASE_P", _default_train_erase_p), 0.0, 1.0))

# Candidate filenames (resolved relative to this file)
CANDIDATE_FILES = {
    "PBGD": ["V-PBGD_G-PBGD.py", "pbgd_param_loss_imbalanced.py"],
    "IAPTT": ["IAPTT-GM.py", "iaptt_gm_param_loss_imbalanced.py"],
    "RHG": ["RHG.py", "rhg_param_loss_imbalanced.py"],
    "HG": ["Hyp_grad_method_SGLD.py", "hypergradient_param_loss_selection.py"],
    "HG_Acc_NoSel": ["HG_acc_param_loss_baseline.py", "HG_acc_param_loss_improved.py", "HG_acc_param_loss.py"],
    "HG_Acc_Sel": ["HG_acc_param_loss_minsel.py", "HG_acc_param_loss_improved.py", "HG_acc_param_loss.py"],
}

# Standardized output filenames (in the per-run SAVE_DIR)
LOG_FILENAMES = {
    "V-PBGD": "logs_vpbgd.json",
    "G-PBGD": "logs_gpbgd.json",
    "IAPTT-GM": "logs_iaptt_gm_paramloss_fixed.json",
    "RHG": "logs_rhg_paramloss_fixed.json",
    "HG–NoSel": "logs_hg_nosel.json",
    "HG–Sel": "logs_hg_sel.json",
    "HG_acc–NoSel": "logs_hg_acc_nosel.json",
    "HG_acc–Sel": "logs_hg_acc_sel.json",
}

# Consistent colors across plots
ALG_COLORS = {
    "V-PBGD": "#1f77b4",  # blue
    "G-PBGD": "#d62728",  # red
    "IAPTT-GM": "#ff7f0e",  # orange
    "RHG": "#17becf",  # teal
    "HG–NoSel": "#2ca02c",  # green
    "HG–Sel": "#9467bd",  # purple
    "HG_acc–NoSel": "#2ca02c",  # green
    "HG_acc–Sel": "#9467bd",  # purple
}

DISPLAY_NAMES = {
    "HG_acc–NoSel": "HG-baseline",
    "HG_acc–Sel": "HG-MS",
}


def _display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _log_path(method_name: str) -> str:
    return os.path.join(SAVE_DIR, LOG_FILENAMES[method_name])


def _load_module_from_candidates(key: str):
    for rel in CANDIDATE_FILES.get(key, []):
        candidate = rel if os.path.isabs(rel) else os.path.join(SCRIPT_DIR, rel)
        if not os.path.exists(candidate):
            continue
        mod_name = os.path.splitext(os.path.basename(candidate))[0]
        loader = importlib.machinery.SourceFileLoader(mod_name, candidate)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module, candidate
    return None, None


def _save_logs(logs, path: str):
    safe = {k: v for k, v in logs.items() if isinstance(v, (list, dict, str, int, float))}
    with open(path, "w") as f:
        json.dump(safe, f)


def _load_logs(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _logs_match_config(raw_logs, outer_iters: int, seed: int, *, time_budget_s=None) -> bool:
    if raw_logs is None:
        return False
    cfg = raw_logs.get("config", {})
    if not isinstance(cfg, dict):
        return False

    # Seed match if present
    if "seed" in cfg and int(cfg["seed"]) != int(seed):
        return False

    # Outer-iteration match
    it_ok = False
    if cfg.get("outer_iters") == outer_iters or cfg.get("outer_itr") == outer_iters:
        it_ok = True
    try:
        target_samples = cfg.get("target_samples", None)
        bs_train = cfg.get("bs_train", None)
        if target_samples is not None and bs_train is not None and int(bs_train) > 0:
            if int(target_samples) // int(bs_train) == outer_iters:
                it_ok = True
    except Exception:
        pass
    if not it_ok:
        return False

    # Time-budget match (if requested)
    if time_budget_s is None:
        # If the old log was time-budgeted but we're not, force rerun.
        if cfg.get("time_budget_s", None) is not None:
            return False
        return True

    try:
        tb = cfg.get("time_budget_s", None)
        if tb is None:
            return False
        return abs(float(tb) - float(time_budget_s)) <= 1e-6
    except Exception:
        return False


def _standardize_logs(raw_logs, method: str):
    if raw_logs is None:
        return None
    out = {
        "method": method,
        "iter": [],
        "elapsed_s": [],
        "train_bal_acc": [],
        "test_bal_acc": [],
        "g_train": [],
        "f_val": [],
    }

    # PBGD / IAPTT / RHG-style
    if "iter" in raw_logs and ("train_bal_acc" in raw_logs or "test_bal_acc" in raw_logs):
        out["iter"] = raw_logs.get("iter", [])
        out["elapsed_s"] = raw_logs.get("elapsed_s", raw_logs.get("elapsed", [])) or []
        out["train_bal_acc"] = raw_logs.get("train_bal_acc", [])
        out["test_bal_acc"] = raw_logs.get("test_bal_acc", [])
        out["g_train"] = raw_logs.get("g_train", raw_logs.get("train_g", []))
        out["f_val"] = raw_logs.get("f_val", raw_logs.get("val_f", []))
        return out

    # HG / HG_acc-style
    if all(k in raw_logs for k in ["iterations", "bal_train_acc", "bal_test_acc", "train_loss", "val_loss"]):
        out["iter"] = raw_logs["iterations"]
        out["elapsed_s"] = raw_logs.get("elapsed_s", raw_logs.get("elapsed", [])) or []
        out["train_bal_acc"] = raw_logs["bal_train_acc"]
        out["test_bal_acc"] = raw_logs["bal_test_acc"]
        out["g_train"] = raw_logs["train_loss"]
        out["f_val"] = raw_logs["val_loss"]
        return out

    return out


def _z_or_t_multiplier(confidence: float, n: int) -> float:
    if n <= 1:
        return 0.0
    alpha = 1.0 - float(confidence)
    q = 1.0 - alpha / 2.0
    try:
        from scipy.stats import t as student_t  # type: ignore

        return float(student_t.ppf(q, df=n - 1))
    except Exception:
        # Normal approximation
        return 1.959963984540054


def _mean_and_ci(values_2d: np.ndarray, confidence: float = 0.95):
    """
    values_2d: shape (n_runs, n_points)
    returns: mean, halfwidth arrays (n_points,)
    """
    if values_2d.size == 0:
        return np.array([]), np.array([])
    n = int(values_2d.shape[0])
    mean = np.nanmean(values_2d, axis=0)
    if n <= 1:
        return mean, np.zeros_like(mean)
    std = np.nanstd(values_2d, axis=0, ddof=1)
    se = std / np.sqrt(n)
    mult = _z_or_t_multiplier(confidence, n)
    return mean, mult * se


def _fmt_loss(x: float) -> str:
    """
    Print losses in fixed-point unless they're tiny, to avoid confusing `0.0000` displays.
    """
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return str(x)
    return f"{x:.4f}" if abs(x) >= 1e-3 else f"{x:.2e}"


def _align_runs_by_iter(logs_list, metric_key: str):
    """
    Align multiple standardized logs by common iteration indices.
    Returns: common_iters (list[int]), values_2d (n_runs, n_points)
    """
    if not logs_list:
        return [], np.zeros((0, 0))
    iter_sets = []
    per_run = []
    for log in logs_list:
        iters = log.get("iter", []) or []
        vals = log.get(metric_key, []) or []
        if len(iters) != len(vals) or len(iters) == 0:
            continue
        mapping = {int(i): float(v) for i, v in zip(iters, vals)}
        iter_sets.append(set(mapping.keys()))
        per_run.append(mapping)
    if not per_run:
        return [], np.zeros((0, 0))
    common = sorted(set.intersection(*iter_sets)) if iter_sets else []
    if not common:
        return [], np.zeros((0, 0))
    values = np.array([[m[i] for i in common] for m in per_run], dtype=float)
    return common, values


def _align_runs_by_time(logs_list, metric_key: str, *, time_grid: np.ndarray):
    """
    Resample multiple standardized logs to a shared time grid via linear interpolation.
    Returns: time_grid (list[float]), values_2d (n_runs, n_points)
    """
    if not logs_list:
        return [], np.zeros((0, 0))
    per_run = []
    for log in logs_list:
        ts = log.get("elapsed_s", []) or []
        vs = log.get(metric_key, []) or []
        if len(ts) != len(vs) or len(ts) == 0:
            continue
        try:
            t = np.asarray([float(x) for x in ts], dtype=float)
            v = np.asarray([float(x) for x in vs], dtype=float)
        except Exception:
            continue
        # Drop non-finite points; they otherwise poison interpolation and can hide valid signal.
        mask = np.isfinite(t) & np.isfinite(v)
        if not bool(mask.any()):
            continue
        t = t[mask]
        v = v[mask]
        # Ensure sorted by time and unique times
        order = np.argsort(t)
        t = t[order]
        v = v[order]
        # De-dup identical time stamps (keep last)
        uniq_t, uniq_v = [], []
        last_t = None
        for ti, vi in zip(t.tolist(), v.tolist()):
            if last_t is not None and float(ti) == float(last_t):
                uniq_v[-1] = float(vi)
            else:
                uniq_t.append(float(ti))
                uniq_v.append(float(vi))
                last_t = float(ti)
        t = np.asarray(uniq_t, dtype=float)
        v = np.asarray(uniq_v, dtype=float)
        if t.size == 0:
            continue
        # Interpolate onto grid; clamp outside range.
        # If there is only one point, treat it as constant over time (common when a method is slow).
        if int(t.size) == 1:
            v_grid = np.full_like(time_grid, float(v[0]), dtype=float)
        else:
            v_grid = np.interp(time_grid, t, v, left=v[0], right=v[-1])
        per_run.append(v_grid.astype(float))
    if not per_run:
        return [], np.zeros((0, 0))
    values = np.stack(per_run, axis=0)
    return time_grid.tolist(), values


def aggregate_results(all_results_per_run, confidence: float = 0.95):
    """
    all_results_per_run: list of {method_name: standardized_log}
    Returns: aggregated dict keyed by method_name with mean+ci arrays for each metric.
    """
    method_names = sorted({name for r in all_results_per_run for name in r.keys()})
    metrics = ["train_bal_acc", "test_bal_acc", "g_train", "f_val"]
    aggregated = {}

    for name in method_names:
        logs_list = [r[name] for r in all_results_per_run if name in r and r[name] is not None]
        if not logs_list:
            continue
        agg = {"method": name, "n_runs": len(logs_list)}
        agg_iters = None

        for mk in metrics:
            iters, values = _align_runs_by_iter(logs_list, mk)
            if len(iters) == 0:
                continue
            mean, ci_hw = _mean_and_ci(values, confidence=confidence)
            if agg_iters is None:
                agg_iters = iters
            else:
                common = sorted(set(agg_iters).intersection(iters))
                if common != agg_iters:
                    idx_old = [agg_iters.index(i) for i in common]
                    for stored in list(agg.keys()):
                        if stored.endswith("_mean") or stored.endswith("_ci"):
                            agg[stored] = [agg[stored][j] for j in idx_old]
                    agg_iters = common
                idx_new = [iters.index(i) for i in agg_iters]
                mean = mean[idx_new]
                ci_hw = ci_hw[idx_new]

            agg[f"{mk}_mean"] = mean.tolist()
            agg[f"{mk}_ci"] = ci_hw.tolist()

        agg["iter"] = agg_iters or []
        aggregated[name] = agg

    return aggregated


def aggregate_results_time(
    all_results_per_run,
    *,
    time_budget_s: float,
    time_grid_s: float = 1.0,
    confidence: float = 0.95,
):
    """
    Aggregate standardized logs by wall-clock time (seconds).
    """
    time_budget_s = float(time_budget_s)
    time_grid_s = float(time_grid_s) if float(time_grid_s) > 0 else 1.0
    grid = np.arange(0.0, time_budget_s + 1e-12, time_grid_s, dtype=float)

    method_names = sorted({name for r in all_results_per_run for name in r.keys()})
    metrics = ["train_bal_acc", "test_bal_acc", "g_train", "f_val"]
    aggregated = {}

    for name in method_names:
        logs_list = [r[name] for r in all_results_per_run if name in r and r[name] is not None]
        if not logs_list:
            continue
        agg = {"method": name, "n_runs": len(logs_list)}
        for mk in metrics:
            times, values = _align_runs_by_time(logs_list, mk, time_grid=grid)
            if len(times) == 0:
                continue
            mean, ci_hw = _mean_and_ci(values, confidence=confidence)
            agg[f"{mk}_mean"] = mean.tolist()
            agg[f"{mk}_ci"] = ci_hw.tolist()
        agg["elapsed_s"] = grid.tolist()
        aggregated[name] = agg
    return aggregated


def run_or_load_vg_pbgd(outer_iters: int, seed: int):
    vp_path = _log_path("V-PBGD")
    gp_path = _log_path("G-PBGD")
    logs_v = _load_logs(vp_path)
    logs_g = _load_logs(gp_path)
    if _logs_match_config(logs_v, outer_iters, seed, time_budget_s=TIME_BUDGET_S) and _logs_match_config(
        logs_g, outer_iters, seed, time_budget_s=TIME_BUDGET_S
    ):
        return logs_v, logs_g

    mod, _ = _load_module_from_candidates("PBGD")
    if mod is None:
        return logs_v, logs_g

    common = (
        _call_with_supported_kwargs(
            mod.prepare_data,
            dict(
                dataset=IMB_DATASET,
                seed=seed,
                imbalance_base=IMB_IMBALANCE_BASE,
                min_keep_prob=IMB_MIN_KEEP_PROB,
                bs_train=64,
                bs_val=64,
                train_fraction=IMB_TRAIN_FRACTION,
                augment_train=IMB_AUGMENT_TRAIN,
                balanced_val=(IMB_BALANCED_VAL and (not IMB_MATCH_SPLIT_IMBALANCE)),
                match_val_test_imbalance=IMB_MATCH_SPLIT_IMBALANCE,
                val_min_per_class=IMB_VAL_MIN_PER_CLASS,
                val_full_batch=IMB_VAL_FULL_BATCH,
                spurious_train=IMB_SPURIOUS_TRAIN,
                spurious_patch_prob=IMB_SPURIOUS_PATCH_PROB,
                spurious_patch_flip_prob=IMB_SPURIOUS_PATCH_FLIP_PROB,
                spurious_patch_size=IMB_SPURIOUS_PATCH_SIZE,
                spurious_patch_value=IMB_SPURIOUS_PATCH_VALUE,
                spurious_n_positions=IMB_SPURIOUS_N_POSITIONS,
                train_erase_p=IMB_TRAIN_ERASE_P,
            ),
        )
        if hasattr(mod, "prepare_data")
        else None
    )
    log_every = max(1, int(os.environ.get("IMB_LOG_EVERY", 20) or 20)) if TIME_BUDGET_S is not None else max(1, outer_iters // 10)
    base = dict(
        seed=seed,
        imbalance_base=IMB_IMBALANCE_BASE,
        dataset=IMB_DATASET,
        min_keep_prob=IMB_MIN_KEEP_PROB,
        spurious_train=IMB_SPURIOUS_TRAIN,
        spurious_patch_prob=IMB_SPURIOUS_PATCH_PROB,
        spurious_patch_flip_prob=IMB_SPURIOUS_PATCH_FLIP_PROB,
        spurious_patch_size=IMB_SPURIOUS_PATCH_SIZE,
        spurious_patch_value=IMB_SPURIOUS_PATCH_VALUE,
        spurious_n_positions=IMB_SPURIOUS_N_POSITIONS,
        train_erase_p=IMB_TRAIN_ERASE_P,
        bs_train=64,
        bs_val=64,
        reg=5e-4,
        y_clip=1.0,
        x_clip=1.0,
        outer_itr=outer_iters,
        gamma_init=0.0,
        log_every=log_every,
        outer_iters=outer_iters,
        # Keep x close to identity for stability (works well with bounded unpack_x()).
        lam_delta=1e-2,
        lam_gamma=1e-2,
        # Old code used a 1/gamma scaling that effectively kills the upper loss for large gamma.
        # Keep it off by default and tune gamma_max instead.
        use_decay_scaling=False,
        # Momentum helps these methods converge in a reasonable number of outer iterations.
        y_momentum=0.9,
        x_momentum=0.9,
        inner_momentum=0.0,
    )
    if TIME_BUDGET_S is not None:
        base["time_budget_s"] = float(TIME_BUDGET_S)

    if hasattr(mod, "run_v_pbgd"):
        # V-PBGD is sensitive to the quality of the "virtual follower" omega.
        # A few more inner steps + a moderate gamma schedule makes it much more reliable.
        gamma_steps = outer_iters
        if TIME_BUDGET_S is not None:
            # In time-budget mode `outer_iters` is often set very large; tying `gamma_steps` to that
            # keeps gamma ~0 throughout the budget. Use a fixed horizon instead (overridable via env).
            gamma_steps = max(1, _env_int("PBGD_GAMMA_STEPS", 200))

        cfg_v = dict(
            **base,
            lrx=2e-2,
            lry=5e-2,
            lr_inner=5e-2,
            inner_itr=20,
            gamma_max=10,
            gamma_steps=gamma_steps,
        )
        logs_v = mod.run_v_pbgd(cfg_v, common) if common else mod.run_v_pbgd(cfg_v, mod.prepare_data())
        _save_logs(logs_v, vp_path)

    if hasattr(mod, "run_g_pbgd"):
        # G-PBGD: avoid huge gamma/lr combos (they often collapse to a single-class predictor).
        cfg_g = dict(
            **base,
            lrx=2e-2,
            lry=5e-2,
            gamma_max=10,
            gamma_steps=gamma_steps,
        )
        logs_g = mod.run_g_pbgd(cfg_g, common) if common else mod.run_g_pbgd(cfg_g, mod.prepare_data())
        _save_logs(logs_g, gp_path)

    return _load_logs(vp_path), _load_logs(gp_path)


def run_or_load_iaptt(outer_iters: int, seed: int):
    out_path = _log_path("IAPTT-GM")
    logs = _load_logs(out_path)
    if _logs_match_config(logs, outer_iters, seed, time_budget_s=TIME_BUDGET_S):
        return logs

    mod, _ = _load_module_from_candidates("IAPTT")
    if mod is None or not hasattr(mod, "run_iaptt_gm"):
        return None

    # IAPTT-GM is sensitive to its inner/outer step sizes. Expose key knobs via env vars so you can tune
    # without editing the Python. Defaults are chosen to be reasonably stable on this MNIST setup.
    preset_showcase = (IMB_PRESET == "minsel_showcase")
    time_budget_mode = (TIME_BUDGET_S is not None)
    # Conservative defaults in time-budget mode to avoid NaNs from unrolled inner dynamics.
    # (Override via env vars in sbatch.)
    iaptt_k_inner = max(1, _env_int("IAPTT_K_INNER", 5 if time_budget_mode else (20 if preset_showcase else 10)))
    iaptt_lr_inner = float(np.clip(_env_float("IAPTT_LR_INNER", 0.03 if time_budget_mode else 0.1), 1e-4, 1.0))
    iaptt_lrx = float(np.clip(_env_float("IAPTT_LRX", 5e-4 if time_budget_mode else 1e-2), 1e-5, 1.0))
    iaptt_lrz = float(np.clip(_env_float("IAPTT_LRZ", 5e-4 if time_budget_mode else 1e-2), 1e-5, 1.0))
    iaptt_tau = _env_float("IAPTT_TAU", 0.5)
    # For CNN this is the hidden width of the final MLP head; for MLP it's the hidden layer width.
    iaptt_d_hidden = max(50, _env_int("IAPTT_D_HIDDEN", 256))
    iaptt_reg = float(np.clip(_env_float("IAPTT_REG", 5e-4), 0.0, 1.0))
    iaptt_lam_delta = float(np.clip(_env_float("IAPTT_LAM_DELTA", 1e-4), 0.0, 1.0))
    iaptt_lam_gamma = float(np.clip(_env_float("IAPTT_LAM_GAMMA", 1e-4), 0.0, 1.0))
    iaptt_model = (os.environ.get("IAPTT_MODEL", "cnn") or "cnn").strip().lower()
    if iaptt_model not in {"cnn", "mlp"}:
        iaptt_model = "cnn"
    iaptt_forward_batch = max(0, _env_int("IAPTT_FORWARD_BATCH", 4096))
    iaptt_y_clip_inner = float(
        np.clip(_env_float("IAPTT_Y_CLIP_INNER", 5.0 if time_budget_mode else 0.0), 0.0, 1e6)
    )
    iaptt_inner_retries = max(0, _env_int("IAPTT_INNER_RETRIES", 2 if time_budget_mode else 0))

    eval_every = (
        max(1, int(os.environ.get("IMB_EVAL_EVERY", 10) or 10))
        if TIME_BUDGET_S is not None
        else max(1, outer_iters // 10)
    )
    cfg = dict(
        seed=seed,
        imbalance_base=IMB_IMBALANCE_BASE,
        train_fraction=IMB_TRAIN_FRACTION,
        dataset=IMB_DATASET,
        min_keep_prob=IMB_MIN_KEEP_PROB,
        match_val_test_imbalance=IMB_MATCH_SPLIT_IMBALANCE,
        val_min_per_class=IMB_VAL_MIN_PER_CLASS,
        spurious_train=IMB_SPURIOUS_TRAIN,
        spurious_patch_prob=IMB_SPURIOUS_PATCH_PROB,
        spurious_patch_flip_prob=IMB_SPURIOUS_PATCH_FLIP_PROB,
        spurious_patch_size=IMB_SPURIOUS_PATCH_SIZE,
        spurious_patch_value=IMB_SPURIOUS_PATCH_VALUE,
        spurious_n_positions=IMB_SPURIOUS_N_POSITIONS,
        train_erase_p=IMB_TRAIN_ERASE_P,
        val_size=IMB_VAL_SIZE,
        model=iaptt_model,
        forward_batch=iaptt_forward_batch,
        d_hidden=iaptt_d_hidden,
        # Inner trajectory: this is the main knob for accuracy vs runtime.
        K_inner=iaptt_k_inner,
        lr_inner=iaptt_lr_inner,
        reg=iaptt_reg,
        # Upper learning rates: keep conservative; the bilevel signal is noisy early on.
        lrx=iaptt_lrx,
        lrz=iaptt_lrz,
        x_clip=0.5 if time_budget_mode else 1.0,
        z_clip=2.0 if time_budget_mode else 5.0,
        y_clip_inner=iaptt_y_clip_inner,
        inner_retries=iaptt_inner_retries,
        outer_itr=outer_iters,
        eval_every=eval_every,
        lam_delta=iaptt_lam_delta,
        lam_gamma=iaptt_lam_gamma,
        # Min-selection matches our objective (minimize validation loss).
        select_mode="min",
        # Soft-min (tau>0) is much more stable than hard argmin/argmax.
        tau=iaptt_tau,
        use_adam=True,
        outer_iters=outer_iters,
    )
    if TIME_BUDGET_S is not None:
        cfg["time_budget_s"] = float(TIME_BUDGET_S)
    logs = mod.run_iaptt_gm(cfg)
    _save_logs(logs, out_path)
    return _load_logs(out_path)


def run_or_load_rhg(outer_iters: int, seed: int):
    out_path = _log_path("RHG")
    logs = _load_logs(out_path)
    if _logs_match_config(logs, outer_iters, seed, time_budget_s=TIME_BUDGET_S):
        return logs

    mod, _ = _load_module_from_candidates("RHG")
    if mod is None or not hasattr(mod, "run_rhg"):
        return None

    cfg = dict(
        seed=seed,
        imbalance_base=IMB_IMBALANCE_BASE,
        train_fraction=IMB_TRAIN_FRACTION,
        match_val_test_imbalance=IMB_MATCH_SPLIT_IMBALANCE,
        val_min_per_class=IMB_VAL_MIN_PER_CLASS,
        spurious_train=IMB_SPURIOUS_TRAIN,
        spurious_patch_prob=IMB_SPURIOUS_PATCH_PROB,
        spurious_patch_flip_prob=IMB_SPURIOUS_PATCH_FLIP_PROB,
        spurious_patch_size=IMB_SPURIOUS_PATCH_SIZE,
        spurious_patch_value=IMB_SPURIOUS_PATCH_VALUE,
        spurious_n_positions=IMB_SPURIOUS_N_POSITIONS,
        val_size=IMB_VAL_SIZE,
        d_hidden=300,
        outer_itr=outer_iters,
        T=outer_iters,
        K=outer_iters,
        lr_y=5e-2,
        lr_x=5e-2,
        reg=0.0,
        x_clip=1.0,
        eval_every=max(1, outer_iters // 10),
        lam_delta=1e-4,
        lam_gamma=1e-4,
        warm_start=True,
        use_adam_x=True,
        outer_iters=outer_iters,
    )
    if TIME_BUDGET_S is not None:
        cfg["time_budget_s"] = float(TIME_BUDGET_S)
    logs = mod.run_rhg(cfg)
    _save_logs(logs, out_path)
    return _load_logs(out_path)


def run_or_load_hg(outer_iters: int, seed: int, *, run_nosel: bool = True, run_sel: bool = True):
    nosel_path = _log_path("HG–NoSel")
    sel_path = _log_path("HG–Sel")
    nosel = _load_logs(nosel_path) if run_nosel else None
    sel = _load_logs(sel_path) if run_sel else None
    need_nosel = bool(run_nosel) and (not _logs_match_config(nosel, outer_iters, seed, time_budget_s=TIME_BUDGET_S))
    need_sel = bool(run_sel) and (not _logs_match_config(sel, outer_iters, seed, time_budget_s=TIME_BUDGET_S))
    if not (need_nosel or need_sel):
        return nosel, sel

    mod, _ = _load_module_from_candidates("HG")
    if mod is None or not hasattr(mod, "prepare_data"):
        return nosel, sel

    bs_train = 64
    base_cfg = dict(
        # Experiment context (recorded into logs for reproducibility)
        dataset=IMB_DATASET,
        preset=IMB_PRESET,
        train_fraction=IMB_TRAIN_FRACTION,
        val_size=IMB_VAL_SIZE,
        balanced_val=bool(IMB_BALANCED_VAL),
        match_split_imbalance=bool(IMB_MATCH_SPLIT_IMBALANCE),
        val_min_per_class=int(IMB_VAL_MIN_PER_CLASS),
        val_full_batch=bool(IMB_VAL_FULL_BATCH),
        augment_train=bool(IMB_AUGMENT_TRAIN),
        train_erase_p=float(IMB_TRAIN_ERASE_P),
        imbalance_base=IMB_IMBALANCE_BASE,
        spurious_train=IMB_SPURIOUS_TRAIN,
        spurious_patch_prob=IMB_SPURIOUS_PATCH_PROB,
        spurious_patch_flip_prob=IMB_SPURIOUS_PATCH_FLIP_PROB,
        spurious_patch_size=IMB_SPURIOUS_PATCH_SIZE,
        spurious_patch_value=IMB_SPURIOUS_PATCH_VALUE,
        spurious_n_positions=IMB_SPURIOUS_N_POSITIONS,
        bs_train=bs_train,
        bs_val=64,
        seed=seed,
        outer_iters=outer_iters,
        outer_itr=outer_iters,
        target_samples=outer_iters * bs_train,
        cg_steps=10,
        cg_damping=1e-3,
        x_lr_init=5e-3,
        x_alpha_min=1e-5,
        x_alpha_max=1e-2,
        x_ema=0.8,
        x_grad_clip=None,
        x_l2=1e-4,
    )
    if TIME_BUDGET_S is not None:
        base_cfg["time_budget_s"] = float(TIME_BUDGET_S)
    common = _call_with_supported_kwargs(
        mod.prepare_data,
        dict(
            imbalance_base=base_cfg["imbalance_base"],
            bs_train=base_cfg["bs_train"],
            bs_val=base_cfg["bs_val"],
            seed=base_cfg["seed"],
            train_fraction=IMB_TRAIN_FRACTION,
            augment_train=IMB_AUGMENT_TRAIN,
            balanced_val=(IMB_BALANCED_VAL and (not IMB_MATCH_SPLIT_IMBALANCE)),
            match_val_test_imbalance=IMB_MATCH_SPLIT_IMBALANCE,
            val_min_per_class=IMB_VAL_MIN_PER_CLASS,
            val_full_batch=IMB_VAL_FULL_BATCH,
            spurious_train=IMB_SPURIOUS_TRAIN,
            spurious_patch_prob=IMB_SPURIOUS_PATCH_PROB,
            spurious_patch_flip_prob=IMB_SPURIOUS_PATCH_FLIP_PROB,
            spurious_patch_size=IMB_SPURIOUS_PATCH_SIZE,
            spurious_patch_value=IMB_SPURIOUS_PATCH_VALUE,
            spurious_n_positions=IMB_SPURIOUS_N_POSITIONS,
        ),
    )

    if need_nosel and hasattr(mod, "run_hg_no_selection"):
        cfg_n = dict(**base_cfg, warm_y_steps=100, y_steps_per_outer=10, y_lr=0.05, y_wd=0.0, y_clip=1.0, eval_every=5)
        print("Running HG–NoSel ...")
        nosel = mod.run_hg_no_selection(cfg_n, common)
        _save_logs(nosel, nosel_path)

    if need_sel and hasattr(mod, "run_hg_with_selection"):
        cfg_s = dict(
            **base_cfg,
            select_mode="min",
            sample_every=1,
            sgld_chains=4,
            sgld_burn_in=20,
            sgld_steps=20,
            sgld_thin=0,
            sgld_eps0=1e-4,
            sgld_alpha=0.55,
            sgld_temperature=0,
            sgld_wd=1e-5,
            sgld_rho=0.99,
            sgld_eps=1e-8,
            sgld_clip=1.0,
            sgld_target_ratio=0.02,
            sgld_inc=1.05,
            sgld_dec=0.95,
            sgld_scale_bounds=(0.1, 10.0),
            sgld_eval_batches=0,
            eval_every=2,
        )
        print("Running HG–Sel ...")
        sel = mod.run_hg_with_selection(cfg_s, common)
        _save_logs(sel, sel_path)

    return (_load_logs(nosel_path) if run_nosel else None), (_load_logs(sel_path) if run_sel else None)


def run_or_load_hg_acc(outer_iters: int, seed: int, *, run_nosel: bool = True, run_sel: bool = True):
    nosel_path = _log_path("HG_acc–NoSel")
    sel_path = _log_path("HG_acc–Sel")
    nosel = _load_logs(nosel_path) if run_nosel else None
    sel = _load_logs(sel_path) if run_sel else None
    need_nosel = bool(run_nosel) and (not _logs_match_config(nosel, outer_iters, seed, time_budget_s=TIME_BUDGET_S))
    need_sel = bool(run_sel) and (not _logs_match_config(sel, outer_iters, seed, time_budget_s=TIME_BUDGET_S))
    if not (need_nosel or need_sel):
        return nosel, sel

    mod_nosel, _ = _load_module_from_candidates("HG_Acc_NoSel")
    mod_sel, _ = _load_module_from_candidates("HG_Acc_Sel")
    mod_data = mod_nosel or mod_sel
    if mod_data is None or not hasattr(mod_data, "prepare_data"):
        return nosel, sel

    # Treat the spurious/mismatch presets as "minsel-style" too: they are explicitly designed
    # to showcase candidate diversity + selection, so default HG-MS knobs should not be overly
    # conservative (EMA/significance/min-improve gates).
    preset_minsel = IMB_PRESET in {"minsel_effective", "minsel_mismatch", "minsel_showcase", "minsel_spurious"}
    time_budget_mode = (TIME_BUDGET_S is not None)

    # HG_acc–Sel minima-selection backend knobs (overridable via environment).
    # - HG_ACC_MINSEL_BACKEND: "adamw" (deterministic branches) or "psgld" (Langevin chains)
    # - HG_ACC_EXTRA_BRANCHES: number of extra candidates besides the base model
    # - HG_ACC_BRANCH_STEPS: lower-level steps per outer iteration for each extra branch (adamw backend)
    hg_acc_minsel_backend = (os.environ.get("HG_ACC_MINSEL_BACKEND", "adamw") or "adamw").strip()
    # Default candidate count: in time-budget mode, keep this small so we don't lose too many outer steps.
    default_extra = 4 if time_budget_mode else 9
    default_branch_steps = 5 if time_budget_mode else (10 if preset_minsel else 5)
    hg_acc_extra_branches = max(0, _env_int("HG_ACC_EXTRA_BRANCHES", default_extra))
    hg_acc_branch_steps = max(0, _env_int("HG_ACC_BRANCH_STEPS", default_branch_steps))
    # NOTE: With persistent branches, the branches must be advanced roughly as often as the base model;
    # otherwise they fall behind and selection never picks them. Default to selecting every outer step.
    hg_acc_sample_every = max(1, _env_int("HG_ACC_SAMPLE_EVERY", 1))
    hg_acc_branch_init_noise_std = max(0.0, _env_float("HG_ACC_BRANCH_INIT_NOISE_STD", 1e-2 if preset_minsel else 1e-3))

    # Deterministic-branch diversity (minsel_backend="adamw"):
    # Spread y_lr / y_wd across branches to encourage different LL basins (often makes selection more impactful).
    # Keep branch hyperparameter ranges moderate by default; extreme ranges can produce unstable LL dynamics.
    # Stronger diversity in the "showcase" regimes.
    preset_showcase = IMB_PRESET in {"minsel_showcase", "minsel_spurious"}
    _default_lr_mult_min = 0.25 if preset_showcase else (0.5 if preset_minsel else 1.0)
    _default_lr_mult_max = 4.0 if preset_showcase else (2.0 if preset_minsel else 1.0)
    _default_wd_mult_min = 0.25 if preset_showcase else (0.5 if preset_minsel else 1.0)
    _default_wd_mult_max = 4.0 if preset_showcase else (2.0 if preset_minsel else 1.0)
    hg_acc_branch_lr_mult_min = max(0.0, _env_float("HG_ACC_BRANCH_LR_MULT_MIN", _default_lr_mult_min))
    hg_acc_branch_lr_mult_max = max(0.0, _env_float("HG_ACC_BRANCH_LR_MULT_MAX", _default_lr_mult_max))
    hg_acc_branch_wd_mult_min = max(0.0, _env_float("HG_ACC_BRANCH_WD_MULT_MIN", _default_wd_mult_min))
    hg_acc_branch_wd_mult_max = max(0.0, _env_float("HG_ACC_BRANCH_WD_MULT_MAX", _default_wd_mult_max))
    hg_acc_branch_restart = _env_bool("HG_ACC_BRANCH_RESTART", False)
    hg_acc_branch_restart_noise_std = float(
        _env_float("HG_ACC_BRANCH_RESTART_NOISE_STD", hg_acc_branch_init_noise_std)
    )
    hg_acc_branch_accept_mode = (os.environ.get("HG_ACC_BRANCH_ACCEPT_MODE", "swap") or "swap").strip().lower()
    if hg_acc_branch_accept_mode not in {"swap", "swap_state", "copy", "copy_weights", "swap_weights"}:
        hg_acc_branch_accept_mode = "swap"
    hg_acc_outer_opt = (os.environ.get("HG_ACC_OUTER_OPT", "adamw") or "adamw").strip().lower()
    if hg_acc_outer_opt not in {"sgd", "adam", "adamw"}:
        hg_acc_outer_opt = "adamw"

    # Selection aggressiveness knobs (overridable via environment).
    # These let you tune minima-selection behavior from sbatch without editing Python.
    # Selection metric: "f" (weighted val loss) or "bal_acc" (val balanced accuracy).
    hg_acc_selection_metric = (os.environ.get("HG_ACC_SELECTION_METRIC", "f") or "f").strip().lower()
    if hg_acc_selection_metric not in {"f", "loss", "val_loss", "acc", "bal_acc", "balanced_acc"}:
        hg_acc_selection_metric = "f"
    hg_acc_selection_min_improve = max(0.0, _env_float("HG_ACC_SELECTION_MIN_IMPROVE", 0.0 if preset_minsel else 0.01))
    hg_acc_selection_use_ema = _env_bool("HG_ACC_SELECTION_USE_EMA", False if preset_minsel else True)
    hg_acc_selection_ema = float(np.clip(_env_float("HG_ACC_SELECTION_EMA", 0.9), 0.0, 0.999))
    hg_acc_selection_rank_by_ema = _env_bool("HG_ACC_SELECTION_RANK_BY_EMA", False if preset_minsel else True)
    hg_acc_selection_ema_override = max(0.0, _env_float("HG_ACC_SELECTION_EMA_OVERRIDE", 0.05))
    hg_acc_selection_use_significance = _env_bool("HG_ACC_SELECTION_USE_SIGNIFICANCE", False if preset_minsel else True)
    hg_acc_selection_z = max(0.0, _env_float("HG_ACC_SELECTION_Z", 1.96))
    hg_acc_selection_eval_batches = max(0, _env_int("HG_ACC_SELECTION_EVAL_BATCHES", 0 if time_budget_mode else (0 if preset_minsel else 20)))

    hg_acc_sel_polish_y_steps = max(0, _env_int("HG_ACC_SEL_POLISH_Y_STEPS", 5 if time_budget_mode else (20 if preset_minsel else 10)))

    # Reporting / stopping behavior (HG_acc–Sel only)
    hg_acc_report_best_at_eval = _env_bool("HG_ACC_REPORT_BEST_AT_EVAL", False if time_budget_mode else True)
    hg_acc_early_stop_patience = max(0, _env_int("HG_ACC_EARLY_STOP_PATIENCE", 0))
    hg_acc_early_stop_min_delta = max(0.0, _env_float("HG_ACC_EARLY_STOP_MIN_DELTA", 0.0))

    # LL budget per outer iteration.
    # Too few LL steps can make the implicit HG direction noisy (y not close to a g-minimizer),
    # which often hurts f(val)/test accuracy more than it helps selection "visibility".
    hg_acc_y_steps_per_outer_default = max(0, _env_int("HG_ACC_Y_STEPS_PER_OUTER", 30 if preset_minsel else 50))
    # Optional per-variant override (useful in time-budget mode to reallocate LL steps between base vs branches).
    hg_acc_y_steps_per_outer_nosel = max(0, _env_int("HG_ACC_Y_STEPS_PER_OUTER_NOSEL", hg_acc_y_steps_per_outer_default))
    hg_acc_y_steps_per_outer_sel = max(0, _env_int("HG_ACC_Y_STEPS_PER_OUTER_SEL", hg_acc_y_steps_per_outer_default))

    # Core HG_acc optimization knobs (shared by HG-MS and HG-baseline).
    # Exposed as env vars so we can tune stability under a wall-clock budget without editing Python.
    hg_acc_cg_steps = max(1, _env_int("HG_ACC_CG_STEPS", 10))
    hg_acc_cg_damping = max(0.0, _env_float("HG_ACC_CG_DAMPING", 1e-3))
    hg_acc_x_lr = max(0.0, _env_float("HG_ACC_X_LR", 1e-2))
    hg_acc_x_l2 = max(0.0, _env_float("HG_ACC_X_L2", 1e-4))
    hg_acc_x_grad_clip = float(_env_float("HG_ACC_X_GRAD_CLIP", 0.5))
    hg_acc_y_lr = max(0.0, _env_float("HG_ACC_Y_LR", 1e-3))
    hg_acc_y_wd = max(0.0, _env_float("HG_ACC_Y_WD", 5e-4))
    hg_acc_warm_y_steps = max(0, _env_int("HG_ACC_WARM_Y_STEPS", 60))
    hg_acc_use_amp = _env_bool("HG_ACC_USE_AMP", False)
    hg_acc_hg_train_cap = max(0, _env_int("HG_ACC_HG_TRAIN_CAP", 512))
    hg_acc_hg_val_cap = max(0, _env_int("HG_ACC_HG_VAL_CAP", 512))

    # pSGLD (LMC) backend knobs (only used when HG_ACC_MINSEL_BACKEND=psgld).
    hg_acc_sgld_temperature = max(0.0, _env_float("HG_ACC_SGLD_TEMPERATURE", 0.0))
    hg_acc_sgld_steps = max(0, _env_int("HG_ACC_SGLD_STEPS", 20))
    hg_acc_sgld_burn_in = max(0, _env_int("HG_ACC_SGLD_BURN_IN", 20))

    bs_train = 64
    eval_every = (
        max(1, int(os.environ.get("IMB_EVAL_EVERY", 10) or 10))
        if TIME_BUDGET_S is not None
        else max(1, outer_iters // 10)
    )
    base_cfg = dict(
        # Experiment context (recorded into logs for reproducibility)
        dataset=IMB_DATASET,
        preset=IMB_PRESET,
        train_fraction=IMB_TRAIN_FRACTION,
        val_size=IMB_VAL_SIZE,
        balanced_val=bool(IMB_BALANCED_VAL),
        match_split_imbalance=bool(IMB_MATCH_SPLIT_IMBALANCE),
        val_min_per_class=int(IMB_VAL_MIN_PER_CLASS),
        val_full_batch=bool(IMB_VAL_FULL_BATCH),
        augment_train=bool(IMB_AUGMENT_TRAIN),
        train_erase_p=float(IMB_TRAIN_ERASE_P),
        spurious_train=bool(IMB_SPURIOUS_TRAIN),
        spurious_patch_prob=float(IMB_SPURIOUS_PATCH_PROB),
        spurious_patch_flip_prob=float(IMB_SPURIOUS_PATCH_FLIP_PROB),
        spurious_patch_size=int(IMB_SPURIOUS_PATCH_SIZE),
        spurious_patch_value=float(IMB_SPURIOUS_PATCH_VALUE),
        spurious_n_positions=int(IMB_SPURIOUS_N_POSITIONS),
        imbalance_base=IMB_IMBALANCE_BASE,
        bs_train=bs_train,
        bs_val=64,
        seed=seed,
        outer_iters=outer_iters,
        outer_itr=outer_iters,
        target_samples=outer_iters * bs_train,
        cg_steps=hg_acc_cg_steps,
        cg_damping=hg_acc_cg_damping,
        eval_every=eval_every,
        # Upper-level optimizer for HG_acc–Sel (baseline overrides this to plain SGD below).
        outer_opt=hg_acc_outer_opt,
        x_lr=hg_acc_x_lr,
        x_l2=hg_acc_x_l2,
        x_grad_clip=hg_acc_x_grad_clip,
        x_clip_mode="value",
        x_clip_norm=1.0,
        y_opt="adamw",
        y_lr=hg_acc_y_lr,
        y_wd=hg_acc_y_wd,
        y_momentum=0.9,
        y_sched_T0=20,
        y_clip=1.0,
        y_grad_accum=1,
        warm_y_steps=hg_acc_warm_y_steps,
        y_steps_warm=30,
        y_steps_per_outer=hg_acc_y_steps_per_outer_default,
        freeze_x_outer=0,
        use_amp=bool(hg_acc_use_amp),
        hg_train_cap=(hg_acc_hg_train_cap if hg_acc_hg_train_cap > 0 else None),
        hg_val_cap=(hg_acc_hg_val_cap if hg_acc_hg_val_cap > 0 else None),
        select_mode="min",
        selection_metric=hg_acc_selection_metric,
        sample_every=hg_acc_sample_every,
        # Minima-selection backend:
        # - "adamw": persistent deterministic branches trained on g(train) with y_opt (default: AdamW)
        # - "psgld": persistent Langevin chains trained on g(train) (exploration)
        minsel_backend=hg_acc_minsel_backend,
        # Total candidates = 1 (base) + `minsel_extra_branches` (extra branches/chains).
        minsel_extra_branches=hg_acc_extra_branches,
        # Backward compatibility: the selection code also falls back to `sgld_chains` as the branch count.
        sgld_chains=hg_acc_extra_branches,
        # For the deterministic-branch backend, keep this small to control runtime (per branch per outer iter).
        branch_steps=hg_acc_branch_steps,
        branch_init_noise_std=hg_acc_branch_init_noise_std,
        branch_restart_each_sample=hg_acc_branch_restart,
        branch_restart_noise_std=hg_acc_branch_restart_noise_std,
        branch_lr_mult_min=hg_acc_branch_lr_mult_min,
        branch_lr_mult_max=hg_acc_branch_lr_mult_max,
        branch_wd_mult_min=hg_acc_branch_wd_mult_min,
        branch_wd_mult_max=hg_acc_branch_wd_mult_max,
        branch_accept_mode=hg_acc_branch_accept_mode,
        sgld_burn_in=hg_acc_sgld_burn_in,
        sgld_steps=hg_acc_sgld_steps,
        sgld_thin=0,
        sgld_eps0=1e-4,
        sgld_alpha=0.55,
        sgld_temperature=hg_acc_sgld_temperature,
        sgld_wd=1e-5,
        sgld_rho=0.99,
        sgld_eps=1e-8,
        sgld_clip=1.0,
        sgld_target_ratio=0.02,
        sgld_inc=1.05,
        sgld_dec=0.95,
        sgld_scale_bounds=(0.1, 10.0),
        # When sgld_temperature=0, chains must see different minibatches (or different inits) to diversify.
        sgld_per_chain_minibatch=True,
        sgld_init_noise_std=1e-3,
        # Candidate-scoring budget: 0 means full val set. Larger val sets reduce selection noise but cost more.
        selection_eval_batches=hg_acc_selection_eval_batches,
        sgld_eval_batches=hg_acc_selection_eval_batches,
        # Conservative switching to reduce "lucky" candidate picks.
        selection_min_improve=hg_acc_selection_min_improve,
        selection_use_significance=hg_acc_selection_use_significance,
        selection_z=hg_acc_selection_z,
        # Extra stabilization for tiny val sets: EMA ranking reduces selection noise.
        selection_use_ema=hg_acc_selection_use_ema,
        selection_ema=hg_acc_selection_ema,
        selection_rank_by_ema=hg_acc_selection_rank_by_ema,
        # If EMA ranking keeps "sticking" to the base while some branch is clearly better on the current f(val),
        # override the EMA choice once the raw improvement exceeds this threshold.
        selection_ema_override_improve=hg_acc_selection_ema_override,
        elitism_chain=True,
        # After accepting a non-base candidate, run a few standard training steps on g before
        # computing the implicit hypergradient. This makes the chosen y closer to a local minimizer
        # (more stable CG system, better outer update).
        sel_polish_y_steps=hg_acc_sel_polish_y_steps,
        # For time-budget plots we always report the *current* iterate (no best-so-far).
        # For fixed-iteration mode we keep the old behavior unless overridden.
        report_best_at_eval=(False if TIME_BUDGET_S is not None else bool(hg_acc_report_best_at_eval)),
        # Optional early stopping once f(val) stops improving; default 0 (disabled).
        early_stop_patience=hg_acc_early_stop_patience,
        early_stop_min_delta=hg_acc_early_stop_min_delta,
        # Parallelize SGLD chains on GPU via CUDA streams (HG_acc_sel only).
        sgld_parallel_chains=True,
    )
    if TIME_BUDGET_S is not None:
        base_cfg["time_budget_s"] = float(TIME_BUDGET_S)

    common = _call_with_supported_kwargs(
        mod_data.prepare_data,
        dict(
            dataset=IMB_DATASET,
            imbalance_base=base_cfg["imbalance_base"],
            min_keep_prob=IMB_MIN_KEEP_PROB,
            bs_train=base_cfg["bs_train"],
            bs_val=base_cfg["bs_val"],
            seed=base_cfg["seed"],
            train_fraction=IMB_TRAIN_FRACTION,
            balanced_val=(IMB_BALANCED_VAL and (not IMB_MATCH_SPLIT_IMBALANCE)),
            match_val_test_imbalance=IMB_MATCH_SPLIT_IMBALANCE,
            val_min_per_class=IMB_VAL_MIN_PER_CLASS,
            val_full_batch=IMB_VAL_FULL_BATCH,
            augment_train=IMB_AUGMENT_TRAIN,
            spurious_train=IMB_SPURIOUS_TRAIN,
            spurious_patch_prob=IMB_SPURIOUS_PATCH_PROB,
            spurious_patch_flip_prob=IMB_SPURIOUS_PATCH_FLIP_PROB,
            spurious_patch_size=IMB_SPURIOUS_PATCH_SIZE,
            spurious_patch_value=IMB_SPURIOUS_PATCH_VALUE,
            spurious_n_positions=IMB_SPURIOUS_N_POSITIONS,
            train_erase_p=IMB_TRAIN_ERASE_P,
            per_class_val=500,
        ),
    )

    if need_nosel and mod_nosel is not None and hasattr(mod_nosel, "run_hg_acc_no_selection"):
        print("Running HG_acc–NoSel ...")
        # Plain baseline: no best-checkpoint restore.
        cfg_nosel = dict(base_cfg)
        # Allow per-variant overrides for stability sweeps without changing the other variant.
        # (If unset, falls back to the shared `HG_ACC_*` knobs used in `base_cfg`.)
        _x_lr_nosel = os.environ.get("HG_ACC_X_LR_NOSEL", None)
        if _x_lr_nosel is not None:
            cfg_nosel["x_lr"] = float(_x_lr_nosel)
        _cg_damp_nosel = os.environ.get("HG_ACC_CG_DAMPING_NOSEL", None)
        if _cg_damp_nosel is not None:
            cfg_nosel["cg_damping"] = float(_cg_damp_nosel)
        # By default match the HG-MS outer optimizer so the only difference is minima selection.
        # Override via env var if you want a strictly "fixed-step SGD" baseline.
        baseline_outer_opt = (os.environ.get("HG_ACC_BASELINE_OUTER_OPT", "") or "").strip().lower()
        if baseline_outer_opt:
            if baseline_outer_opt not in {"sgd", "adam", "adamw"}:
                raise ValueError("HG_ACC_BASELINE_OUTER_OPT must be one of: sgd, adam, adamw")
            cfg_nosel["outer_opt"] = baseline_outer_opt
        cfg_nosel["restore_best"] = False
        cfg_nosel["y_steps_per_outer"] = hg_acc_y_steps_per_outer_nosel
        nosel = mod_nosel.run_hg_acc_no_selection(cfg_nosel, common)
        _save_logs(nosel, nosel_path)
    if need_sel and mod_sel is not None and hasattr(mod_sel, "run_hg_acc_sel"):
        print("Running HG_acc–Sel ...")
        cfg_sel = dict(base_cfg)
        _x_lr_sel = os.environ.get("HG_ACC_X_LR_SEL", None)
        if _x_lr_sel is not None:
            cfg_sel["x_lr"] = float(_x_lr_sel)
        _cg_damp_sel = os.environ.get("HG_ACC_CG_DAMPING_SEL", None)
        if _cg_damp_sel is not None:
            cfg_sel["cg_damping"] = float(_cg_damp_sel)
        cfg_sel["y_steps_per_outer"] = hg_acc_y_steps_per_outer_sel
        sel = mod_sel.run_hg_acc_sel(cfg_sel, common)
        _save_logs(sel, sel_path)

    return (_load_logs(nosel_path) if run_nosel else None), (_load_logs(sel_path) if run_sel else None)


def _parse_methods_arg(methods: str):
    methods = (methods or "").strip().lower()
    if methods in ("", "all", "*"):
        return {"pbgd", "iaptt", "rhg", "hg", "hg_acc"}
    parts = [p.strip() for p in methods.split(",") if p.strip()]
    out = set()
    for p in parts:
        if p in {"pbgd", "vpbgd", "gpbgd"}:
            out.add("pbgd")
        elif p in {"iaptt", "iaptt-gm", "iaptt_gm"}:
            out.add("iaptt")
        elif p in {"rhg"}:
            out.add("rhg")
        elif p in {"hg"}:
            out.add("hg")
        elif p in {"hg_sel", "hg-sel", "hgsel"}:
            out.add("hg_sel")
        elif p in {"hg_nosel", "hg-nosel", "hgnosel"}:
            out.add("hg_nosel")
        elif p in {"hg_acc", "hg-acc", "hgacc"}:
            out.add("hg_acc")
        elif p in {"hg_acc_sel", "hg-acc-sel", "hgaccsel", "hg_min_sel", "hg-min-sel", "hg_minsel", "hg-minsel"}:
            out.add("hg_acc_sel")
        elif p in {"hg_acc_nosel", "hg-acc-nosel", "hgaccnosel"}:
            out.add("hg_acc_nosel")
        else:
            raise ValueError(
                f"Unknown method '{p}'. Use --methods all or a comma-list from: "
                "pbgd,iaptt,rhg,hg,hg_sel,hg_nosel,hg_acc,hg_acc_sel,hg_acc_nosel"
            )
    return out


def collect_all(outer_iters: int, seed: int, enabled_methods=None):
    enabled_methods = enabled_methods or {"pbgd", "iaptt", "rhg", "hg", "hg_acc"}
    results = {}

    if "pbgd" in enabled_methods:
        try:
            logs_v, logs_g = run_or_load_vg_pbgd(outer_iters, seed)
            if logs_v:
                results["V-PBGD"] = _standardize_logs(logs_v, "V-PBGD")
            if logs_g:
                results["G-PBGD"] = _standardize_logs(logs_g, "G-PBGD")
        except Exception as e:
            import traceback

            print(f"[WARN] PBGD failed: {e!r}")
            traceback.print_exc()

    if "iaptt" in enabled_methods:
        try:
            logs_i = run_or_load_iaptt(outer_iters, seed)
            if logs_i:
                results["IAPTT-GM"] = _standardize_logs(logs_i, "IAPTT-GM")
        except Exception as e:
            import traceback

            print(f"[WARN] IAPTT failed: {e!r}")
            traceback.print_exc()

    if "rhg" in enabled_methods:
        try:
            logs_r = run_or_load_rhg(outer_iters, seed)
            if logs_r:
                results["RHG"] = _standardize_logs(logs_r, "RHG")
        except Exception as e:
            import traceback

            print(f"[WARN] RHG failed: {e!r}")
            traceback.print_exc()

    run_hg_nosel = ("hg" in enabled_methods) or ("hg_nosel" in enabled_methods)
    run_hg_sel = ("hg" in enabled_methods) or ("hg_sel" in enabled_methods)
    if run_hg_nosel or run_hg_sel:
        try:
            logs_hn, logs_hs = run_or_load_hg(outer_iters, seed, run_nosel=run_hg_nosel, run_sel=run_hg_sel)
            if logs_hn:
                results["HG–NoSel"] = _standardize_logs(logs_hn, "HG–NoSel")
            if logs_hs:
                results["HG–Sel"] = _standardize_logs(logs_hs, "HG–Sel")
        except Exception as e:
            import traceback

            print(f"[WARN] HG failed: {e!r}")
            traceback.print_exc()

    run_hg_acc_nosel = ("hg_acc" in enabled_methods) or ("hg_acc_nosel" in enabled_methods)
    run_hg_acc_sel = ("hg_acc" in enabled_methods) or ("hg_acc_sel" in enabled_methods)
    if run_hg_acc_nosel or run_hg_acc_sel:
        try:
            logs_han, logs_has = run_or_load_hg_acc(
                outer_iters, seed, run_nosel=run_hg_acc_nosel, run_sel=run_hg_acc_sel
            )
            if logs_han:
                results["HG_acc–NoSel"] = _standardize_logs(logs_han, "HG_acc–NoSel")
            if logs_has:
                results["HG_acc–Sel"] = _standardize_logs(logs_has, "HG_acc–Sel")
        except Exception as e:
            import traceback

            print(f"[WARN] HG_acc failed: {e!r}")
            traceback.print_exc()

    return results


def plot_accuracies_mean_ci(
    agg_results,
    *,
    save_as: str,
    confidence: float,
    show: bool,
    x_key: str = "iter",
    x_label: str = "Outer iter",
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
    ax_tr, ax_te = axes

    for name, log in agg_results.items():
        color = ALG_COLORS.get(name, "#333333")
        disp = _display_name(name)
        xs = log.get(x_key, [])
        if not xs:
            continue
        if "train_bal_acc_mean" in log:
            m = np.array(log["train_bal_acc_mean"], dtype=float)
            c = np.array(log.get("train_bal_acc_ci", np.zeros_like(m)), dtype=float)
            ax_tr.plot(xs, m, "-", label=f"{disp}", color=color, linewidth=1.8)
            ax_tr.fill_between(xs, m - c, m + c, color=color, alpha=0.18, linewidth=0)
        if "test_bal_acc_mean" in log:
            m = np.array(log["test_bal_acc_mean"], dtype=float)
            c = np.array(log.get("test_bal_acc_ci", np.zeros_like(m)), dtype=float)
            ax_te.plot(xs, m, "-", label=f"{disp}", color=color, linewidth=1.8)
            ax_te.fill_between(xs, m - c, m + c, color=color, alpha=0.18, linewidth=0)

    ax_tr.set_title(f"Train Balanced Accuracy (mean ± {int(confidence*100)}% CI)")
    ax_te.set_title(f"Test Balanced Accuracy (mean ± {int(confidence*100)}% CI)")
    for ax in axes:
        ax.set_xlabel(x_label)
        ax.set_ylabel("Balanced accuracy")
        ax.grid(True)
        ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, save_as), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_losses_mean_ci(
    agg_results,
    *,
    save_as: str,
    confidence: float,
    show: bool,
    x_key: str = "iter",
    x_label: str = "Outer iter",
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
    ax_g, ax_f = axes

    for name, log in agg_results.items():
        color = ALG_COLORS.get(name, "#333333")
        disp = _display_name(name)
        xs = log.get(x_key, [])
        if not xs:
            continue
        if "g_train_mean" in log:
            m = np.array(log["g_train_mean"], dtype=float)
            c = np.array(log.get("g_train_ci", np.zeros_like(m)), dtype=float)
            ax_g.plot(xs, m, "-", label=f"{disp}", color=color, linewidth=1.8)
            ax_g.fill_between(xs, m - c, m + c, color=color, alpha=0.18, linewidth=0)
        if "f_val_mean" in log:
            m = np.array(log["f_val_mean"], dtype=float)
            c = np.array(log.get("f_val_ci", np.zeros_like(m)), dtype=float)
            ax_f.plot(xs, m, "-", label=f"{disp}", color=color, linewidth=1.8)
            ax_f.fill_between(xs, m - c, m + c, color=color, alpha=0.18, linewidth=0)

    ax_g.set_title(f"g(train) (mean ± {int(confidence*100)}% CI)")
    ax_f.set_title(f"f(val) (mean ± {int(confidence*100)}% CI)")
    for ax in axes:
        ax.set_xlabel(x_label)
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, save_as), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    global OUTER_ITERS, SEED, SAVE_DIR, TIME_BUDGET_S, TIME_GRID_S
    parser = argparse.ArgumentParser(description="Run and compare loss-tuning algorithms on imbalanced MNIST.")
    parser.add_argument("--outer-iters", type=int, default=OUTER_ITERS, help="Max outer updates per method.")
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="If set, stop each method after this wall-clock budget (seconds) and aggregate/plot vs time.",
    )
    parser.add_argument(
        "--time-grid-seconds",
        type=float,
        default=TIME_GRID_S,
        help="Time-grid resolution (seconds) for mean/CI plots in time-budget mode.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Base random seed (repeat i uses seed+ i).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="Directory to write logs and figures.")
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help=(
            "Comma-list: pbgd,iaptt,rhg,hg,hg_sel,hg_nosel,hg_acc,hg_acc_sel,hg_acc_nosel "
            "(default: all)."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of runs (different seeds) to average over.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for CI bands (default: 0.95).")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate plots (still writes JSON manifests).")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show() (useful on headless).")
    parser.add_argument(
        "--merge-time-manifests",
        nargs="+",
        default=None,
        help=(
            "Merge existing per-run `aggregate_time_manifest.json` files (no training). "
            "Provide file paths or directories containing `aggregate_time_manifest.json`."
        ),
    )
    args = parser.parse_args()

    def _resolve_time_manifest_path(p: str) -> str:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            cand = os.path.join(p, "aggregate_time_manifest.json")
            return cand
        return p

    def _merge_aggregate_time_manifests(paths, confidence: float = 0.95):
        manifests = []
        for raw in paths:
            p = _resolve_time_manifest_path(raw)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing time-manifest: {p}")
            with open(p, "r") as f:
                manifests.append(json.load(f))

        if not manifests:
            return {}

        # common methods across manifests
        common = set(manifests[0].keys())
        for m in manifests[1:]:
            common &= set(m.keys())
        common = sorted(common)
        if not common:
            raise ValueError("No common methods found across time-manifests.")

        merged = {}
        n_runs = len(manifests)
        for name in common:
            first = manifests[0][name]
            xs = first.get("elapsed_s", [])
            if not xs:
                continue

            out = {
                "method": first.get("method", name),
                "n_runs": n_runs,
                "elapsed_s": xs,
            }

            # Metrics in time manifests are stored as *_mean arrays (and *_ci arrays).
            for metric in ["train_bal_acc", "test_bal_acc", "g_train", "f_val"]:
                key = f"{metric}_mean"
                series = []
                for mf in manifests:
                    log = mf.get(name, {})
                    vals = log.get(key, [])
                    if not vals:
                        vals = [float("nan")] * len(xs)
                    if len(vals) != len(xs):
                        # Align by interpolation onto xs if needed
                        x0 = np.array(log.get("elapsed_s", []), dtype=float)
                        y0 = np.array(vals, dtype=float)
                        if x0.size == 0 or y0.size == 0:
                            y = np.full(len(xs), np.nan, dtype=float)
                        else:
                            xq = np.array(xs, dtype=float)
                            y = np.interp(xq, x0, y0, left=y0[0], right=y0[-1])
                        series.append(y)
                    else:
                        series.append(np.array(vals, dtype=float))

                arr = np.stack(series, axis=0)
                mean, ci = _mean_and_ci(arr, confidence=confidence)
                out[key] = mean.tolist()
                out[f"{metric}_ci"] = ci.tolist()

            merged[name] = out
        return merged

    OUTER_ITERS = int(args.outer_iters)
    TIME_BUDGET_S = float(args.time_budget_seconds) if args.time_budget_seconds is not None else None
    if TIME_BUDGET_S is not None and TIME_BUDGET_S <= 0:
        TIME_BUDGET_S = None
    TIME_GRID_S = float(args.time_grid_seconds) if float(args.time_grid_seconds) > 0 else 1.0
    base_seed = int(args.seed)
    base_save_dir = args.save_dir
    os.makedirs(base_save_dir, exist_ok=True)

    if args.merge_time_manifests:
        # Merge-only path: no method execution, just aggregate existing time manifests.
        SAVE_DIR = base_save_dir
        confidence = float(args.confidence)
        show = (not args.no_show)
        agg = _merge_aggregate_time_manifests(args.merge_time_manifests, confidence=confidence)
        with open(os.path.join(SAVE_DIR, "aggregate_time_manifest.json"), "w") as f:
            json.dump(agg, f)

        # Save summary CSV at the last time point.
        import csv

        summary_path = os.path.join(SAVE_DIR, "summary_last_time_mean_ci.csv")
        with open(summary_path, "w", newline="") as f:
            fieldnames = [
                "algorithm",
                "last_time_s",
                "elapsed_s",
                "train_bal_acc_mean",
                "train_bal_acc_ci_low",
                "train_bal_acc_ci_high",
                "test_bal_acc_mean",
                "test_bal_acc_ci_low",
                "test_bal_acc_ci_high",
                "g_train_mean",
                "g_train_ci_low",
                "g_train_ci_high",
                "f_val_mean",
                "f_val_ci_low",
                "f_val_ci_high",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for name, log in agg.items():
                disp = _display_name(name)
                xs = log.get("elapsed_s", [])
                if not xs:
                    continue
                last = -1

                def _m_ci(metric: str):
                    m = log.get(f"{metric}_mean", [])
                    c = log.get(f"{metric}_ci", [])
                    if not m:
                        return None
                    mean = float(m[last])
                    ci = float(c[last]) if c else 0.0
                    return mean, mean - ci, mean + ci

                tr = _m_ci("train_bal_acc")
                te = _m_ci("test_bal_acc")
                gg = _m_ci("g_train")
                ff = _m_ci("f_val")
                if tr is None or te is None:
                    continue
                row = {
                    "algorithm": disp,
                    "last_time_s": float(xs[last]),
                    "elapsed_s": float(xs[last]),
                    "train_bal_acc_mean": tr[0],
                    "train_bal_acc_ci_low": tr[1],
                    "train_bal_acc_ci_high": tr[2],
                    "test_bal_acc_mean": te[0],
                    "test_bal_acc_ci_low": te[1],
                    "test_bal_acc_ci_high": te[2],
                    "g_train_mean": gg[0] if gg else "",
                    "g_train_ci_low": gg[1] if gg else "",
                    "g_train_ci_high": gg[2] if gg else "",
                    "f_val_mean": ff[0] if ff else "",
                    "f_val_ci_low": ff[1] if ff else "",
                    "f_val_ci_high": ff[2] if ff else "",
                }
                w.writerow(row)

        if not args.no_plot:
            plot_accuracies_mean_ci(
                agg,
                save_as="compare_acc_vs_time_mean_ci.png",
                confidence=confidence,
                show=show,
                x_key="elapsed_s",
                x_label="Time (s)",
            )
            # Do not emit loss-vs-time by default when merge-only (paper uses accuracies).

        print(f"[merge] wrote {summary_path}")
        return

    preset_label = IMB_PRESET if IMB_PRESET else "default"
    print(
        f"[Imbalance] preset={preset_label} dataset={IMB_DATASET} "
        f"imbalance_base={IMB_IMBALANCE_BASE} min_keep_prob={IMB_MIN_KEEP_PROB} "
        f"train_fraction={IMB_TRAIN_FRACTION} val_size≈{IMB_VAL_SIZE} "
        f"balanced_val={IMB_BALANCED_VAL} match_split_imbalance={IMB_MATCH_SPLIT_IMBALANCE} "
        f"val_min_per_class={IMB_VAL_MIN_PER_CLASS} val_full_batch={IMB_VAL_FULL_BATCH} "
        f"augment_train={IMB_AUGMENT_TRAIN} train_erase_p={IMB_TRAIN_ERASE_P} "
        f"spurious_train={IMB_SPURIOUS_TRAIN} spurious_prob={IMB_SPURIOUS_PATCH_PROB} "
        f"spurious_flip_prob={IMB_SPURIOUS_PATCH_FLIP_PROB}"
    )

    enabled_methods = _parse_methods_arg(args.methods)
    print(f"[Imbalance] enabled_methods={sorted(enabled_methods)} (raw={args.methods!r})")
    repeats = max(1, int(args.repeats))
    confidence = float(args.confidence)
    show = (not args.no_show)
    if TIME_BUDGET_S is not None:
        print(f"[Imbalance] mode=time_budget time_budget_s={TIME_BUDGET_S} time_grid_s={TIME_GRID_S} max_outer_iters={OUTER_ITERS}")
    else:
        print(f"[Imbalance] mode=iterations outer_iters={OUTER_ITERS}")

    all_runs = []
    for r in range(repeats):
        SEED = base_seed + r
        SAVE_DIR = os.path.join(base_save_dir, f"run_{r:02d}_seed{SEED}")
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"\n=== Repeat {r+1}/{repeats} | seed={SEED} | out={SAVE_DIR} ===")
        one = collect_all(OUTER_ITERS, SEED, enabled_methods=enabled_methods)
        all_runs.append(one)

        # Per-run manifest (standardized)
        with open(os.path.join(SAVE_DIR, "logs_all_manifest.json"), "w") as f:
            json.dump(one, f)

    # Aggregate (write to base save-dir)
    SAVE_DIR = base_save_dir
    if TIME_BUDGET_S is not None:
        agg = aggregate_results_time(
            all_runs,
            time_budget_s=float(TIME_BUDGET_S),
            time_grid_s=float(TIME_GRID_S),
            confidence=confidence,
        )
        with open(os.path.join(SAVE_DIR, "aggregate_time_manifest.json"), "w") as f:
            json.dump(agg, f)

        # Save summary CSV at the last time point.
        import csv

        summary_path = os.path.join(SAVE_DIR, "summary_last_time_mean_ci.csv")
        with open(summary_path, "w", newline="") as f:
            fieldnames = [
                "algorithm",
                "last_time_s",
                "elapsed_s",
                "train_bal_acc_mean",
                "train_bal_acc_ci_low",
                "train_bal_acc_ci_high",
                "test_bal_acc_mean",
                "test_bal_acc_ci_low",
                "test_bal_acc_ci_high",
                "g_train_mean",
                "g_train_ci_low",
                "g_train_ci_high",
                "f_val_mean",
                "f_val_ci_low",
                "f_val_ci_high",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for name, log in agg.items():
                disp = _display_name(name)
                xs = log.get("elapsed_s", [])
                if not xs:
                    continue
                last = -1

                def _m_ci(metric: str):
                    m = log.get(f"{metric}_mean", [])
                    c = log.get(f"{metric}_ci", [])
                    if not m:
                        return None
                    mean = float(m[last])
                    ci = float(c[last]) if c else 0.0
                    return mean, mean - ci, mean + ci

                tr = _m_ci("train_bal_acc")
                te = _m_ci("test_bal_acc")
                gg = _m_ci("g_train")
                ff = _m_ci("f_val")
                if tr is None or te is None:
                    continue
                row = {
                    "algorithm": disp,
                    "last_time_s": float(xs[last]),
                    "elapsed_s": float(xs[last]),
                    "train_bal_acc_mean": tr[0],
                    "train_bal_acc_ci_low": tr[1],
                    "train_bal_acc_ci_high": tr[2],
                    "test_bal_acc_mean": te[0],
                    "test_bal_acc_ci_low": te[1],
                    "test_bal_acc_ci_high": te[2],
                    "g_train_mean": gg[0] if gg else "",
                    "g_train_ci_low": gg[1] if gg else "",
                    "g_train_ci_high": gg[2] if gg else "",
                    "f_val_mean": ff[0] if ff else "",
                    "f_val_ci_low": ff[1] if ff else "",
                    "f_val_ci_high": ff[2] if ff else "",
                }
                w.writerow(row)

        print("\n=== Summary (last time point): mean ± CI ===")
        for name, log in agg.items():
            xs = log.get("elapsed_s", [])
            if not xs:
                continue
            last = -1
            n_runs = log.get("n_runs", 0)
            parts = [f"{name} (n={n_runs}) @ t={xs[last]:.0f}s"]
            if "test_bal_acc_mean" in log:
                parts.append(
                    f"Test BalAcc {log['test_bal_acc_mean'][last]:.4f} ± {log.get('test_bal_acc_ci',[0])[last]:.4f}"
                )
            if "train_bal_acc_mean" in log:
                parts.append(
                    f"Train BalAcc {log['train_bal_acc_mean'][last]:.4f} ± {log.get('train_bal_acc_ci',[0])[last]:.4f}"
                )
            if "f_val_mean" in log:
                parts.append(f"f(val) {_fmt_loss(log['f_val_mean'][last])} ± {_fmt_loss(log.get('f_val_ci',[0])[last])}")
            if "g_train_mean" in log:
                parts.append(
                    f"g(train) {_fmt_loss(log['g_train_mean'][last])} ± {_fmt_loss(log.get('g_train_ci',[0])[last])}"
                )
            print(" | ".join(parts))

        if not args.no_plot:
            plot_accuracies_mean_ci(
                agg,
                save_as="compare_acc_vs_time_mean_ci.png",
                confidence=confidence,
                show=show,
                x_key="elapsed_s",
                x_label="Time (s)",
            )
            plot_losses_mean_ci(
                agg,
                save_as="compare_losses_vs_time_mean_ci.png",
                confidence=confidence,
                show=show,
                x_key="elapsed_s",
                x_label="Time (s)",
            )
    else:
        agg = aggregate_results(all_runs, confidence=confidence)
        with open(os.path.join(SAVE_DIR, "aggregate_manifest.json"), "w") as f:
            json.dump(agg, f)

        print("\n=== Summary (final logged point): mean ± CI ===")
        for name, log in agg.items():
            iters = log.get("iter", [])
            if not iters:
                continue
            last = -1
            n_runs = log.get("n_runs", 0)
            parts = [f"{name} (n={n_runs}) @ iter {iters[last]}"]
            if "test_bal_acc_mean" in log:
                parts.append(
                    f"Test BalAcc {log['test_bal_acc_mean'][last]:.4f} ± {log.get('test_bal_acc_ci',[0])[last]:.4f}"
                )
            if "train_bal_acc_mean" in log:
                parts.append(
                    f"Train BalAcc {log['train_bal_acc_mean'][last]:.4f} ± {log.get('train_bal_acc_ci',[0])[last]:.4f}"
                )
            if "f_val_mean" in log:
                parts.append(f"f(val) {_fmt_loss(log['f_val_mean'][last])} ± {_fmt_loss(log.get('f_val_ci',[0])[last])}")
            if "g_train_mean" in log:
                parts.append(
                    f"g(train) {_fmt_loss(log['g_train_mean'][last])} ± {_fmt_loss(log.get('g_train_ci',[0])[last])}"
                )
            print(" | ".join(parts))

        if not args.no_plot:
            plot_accuracies_mean_ci(agg, save_as="compare_acc_mean_ci.png", confidence=confidence, show=show)
            plot_losses_mean_ci(agg, save_as="compare_losses_mean_ci.png", confidence=confidence, show=show)


if __name__ == "__main__":
    main()
