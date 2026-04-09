"""
run_all_algorithms.py

Imports the algorithms from your .py files and runs them with a shared
number of outer iterations. It aggregates histories, compares last-iteration
train/test accuracies, and saves plots of training accuracy, test accuracy,
and upper-level (validation) loss over the outer-loop trajectory.

Files expected in the same directory:
- V-PBGD.py  (runner name resolved from: run_VPBG, run_vpbgd, run_VPGD, run_vpgd, run_VPBGD)
- G-PBGD.py  (runner name resolved from: run_GPBG, run_gpbgd, run_GPGD, run_gpgd, run_GPBGD)
- IAPTT-GM.py              (runner name resolved from: run_IAPTT_GM, run_iaptt_gm, run_IAPTT, run_iaptt)
- ITD_baseline.py            (exports: run_ITD_baseline)
- ITD_minima_selection.py    (exports: run_ITD_Sel   or run_itd_minima_selection)
- AID_baseline.py            (exports: run_AID_baseline)
- AID_with_Sel.py            (exports: run_AID_Sel   or run_aid_minima_selection)
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure local sibling imports work when this file is imported as a module.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ---- Direct imports for 2 algorithms ----
from ITD_baseline import run_ITD_baseline
from ITD_minima_selection import run_ITD_Sel as run_ITD_Sel
# from AID_baseline import run_AID_baseline
# from AID_with_Sel import run_AID_Sel as run_AID_Sel

# ---- Dynamic import helpers for hyphenated files ----
import importlib.util
from importlib.machinery import SourceFileLoader
import inspect  # for robust arg filtering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = HERE

# Matplotlib guidance per environment:
# - use matplotlib (no seaborn)
# - single chart per figure (no subplots)
# - do not set custom colors or styles

def _load_module_from_path(mod_name: str, filepath: str):
    loader = SourceFileLoader(mod_name, filepath)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module

def _resolve_vpgd_runner(vp_module):
    # Prefer "core" runners (avoid wrappers that print config unconditionally).
    for name in ("run_vpbgd", "run_vpgd", "run_VPGD", "run_VPBGD", "run_VPBG"):
        fn = getattr(vp_module, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(
        "V-PBGD runner not found in V-PBGD.py. Define one of: run_VPBG, run_vpbgd, run_VPGD, run_vpgd, run_VPBGD."
    )

def _resolve_gpbgd_runner(gp_module):
    # Prefer "core" runners (avoid wrappers that print config unconditionally).
    for name in ("run_gpbgd", "run_gpgd", "run_GPGD", "run_GPBGD", "run_GPBG"):
        fn = getattr(gp_module, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(
        "G-PBGD runner not found in G-PBGD.py. Define one of: run_GPBG, run_gpbgd, run_GPGD, run_gpgd, run_GPBGD."
    )

def _resolve_iaptt_gm_runner(iaptt_module):
    for name in ("run_IAPTT_GM", "run_iaptt_gm", "run_IAPTT", "run_iaptt"):
        fn = getattr(iaptt_module, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(
        "IAPTT-GM runner not found in IAPTT-GM.py. Define one of: run_IAPTT_GM, run_iaptt_gm, run_IAPTT, run_iaptt."
    )

def _norm_return(fn, **kwargs):
    """
    Call `fn` but drop any kwargs it doesn't accept (handles both config-style
    and explicit-args runners). Normalize the result to (final, history|None).
    """
    sig = inspect.signature(fn)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not accepts_var_kw:
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    out = fn(**kwargs)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, None

def _history_to_df(history: List[Dict[str, Any]], algo_name: str) -> pd.DataFrame:
    if history is None or len(history) == 0:
        return pd.DataFrame(columns=["algorithm","iteration","elapsed_s","train_acc","test_acc","val_loss"])
    df = pd.DataFrame(history).copy()
    for col in ("iteration", "elapsed_s", "train_acc", "test_acc", "val_loss"):
        if col not in df.columns:
            df[col] = None
    df["algorithm"] = algo_name
    base = ["algorithm", "iteration", "elapsed_s", "train_acc", "test_acc", "val_loss"]
    extras = [c for c in df.columns if c not in set(base)]
    return df[base + extras]

def _t_critical_95(df: int) -> float:
    """
    2-sided 95% t critical values for df=1..30, else normal approx (1.96).
    """
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(int(df), 1.96)

def _aggregate_mean_std_ci(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    """
    Returns df grouped by group_cols with columns:
      <metric>_mean, <metric>_std, <metric>_n, <metric>_ci_low, <metric>_ci_high
    using a two-sided 95% t-interval.
    """
    grouped = df.groupby(group_cols, dropna=False)
    means = grouped[metric_cols].mean(numeric_only=True).add_suffix("_mean")
    stds = grouped[metric_cols].std(numeric_only=True).add_suffix("_std")
    counts = grouped[metric_cols].count().add_suffix("_n")
    out = pd.concat([means, stds, counts], axis=1).reset_index()

    for m in metric_cols:
        n_col = f"{m}_n"
        std_col = f"{m}_std"
        mean_col = f"{m}_mean"
        low_col = f"{m}_ci_low"
        high_col = f"{m}_ci_high"

        n = out[n_col].astype(float)
        se = out[std_col] / np.sqrt(n)
        tcrit = n.apply(lambda k: _t_critical_95(int(k) - 1) if k >= 2 else np.nan)
        half = tcrit * se
        out[low_col] = out[mean_col] - half
        out[high_col] = out[mean_col] + half

    return out

def _save_plot_line(df: pd.DataFrame, y_col: str, outpath: str, title: str):
    plt.figure()
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        if sub[y_col].notna().any():
            plt.plot(sub["iteration"], sub[y_col], label=algo)
    plt.xlabel("Outer iteration")
    plt.ylabel(y_col.replace("_", " "))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _save_plot_mean_ci(
    df: pd.DataFrame,
    y_col: str,
    outpath: str,
    title: str,
    *,
    x_col: str = "iteration",
    x_label: str = "Outer iteration",
):
    """
    Expects columns: {x_col}, algorithm, <y>_mean, <y>_ci_low, <y>_ci_high.
    """
    y_mean = f"{y_col}_mean"
    y_low = f"{y_col}_ci_low"
    y_high = f"{y_col}_ci_high"

    plt.figure()
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo].sort_values(x_col)
        if sub.empty or not sub[y_mean].notna().any():
            continue
        (line,) = plt.plot(sub[x_col], sub[y_mean], label=algo)
        if y_low in sub.columns and y_high in sub.columns and sub[y_low].notna().any() and sub[y_high].notna().any():
            plt.fill_between(
                sub[x_col].to_numpy(),
                sub[y_low].to_numpy(),
                sub[y_high].to_numpy(),
                color=line.get_color(),
                alpha=0.2,
            )
    plt.xlabel(x_label)
    plt.ylabel(y_col.replace("_", " "))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _resample_history_time_grid(
    raw: pd.DataFrame,
    *,
    time_grid: np.ndarray,
    metric_cols: List[str],
) -> pd.DataFrame:
    """
    For each (algorithm, run), linearly interpolate metrics onto a shared `time_grid`.
    Requires columns: algorithm, run, elapsed_s.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    if "elapsed_s" not in raw.columns or "run" not in raw.columns:
        return pd.DataFrame()

    out: List[pd.DataFrame] = []
    for (algo, run_idx), sub in raw.groupby(["algorithm", "run"], dropna=False):
        sub = sub.dropna(subset=["elapsed_s"]).sort_values("elapsed_s")
        if sub.empty:
            continue
        sub = sub.groupby("elapsed_s", as_index=False).last()
        t = sub["elapsed_s"].to_numpy(dtype=float)
        if len(t) == 0:
            continue

        rows: Dict[str, Any] = {"algorithm": algo, "run": run_idx, "elapsed_s": time_grid.astype(float)}
        for m in metric_cols:
            if m not in sub.columns:
                continue
            v = sub[m].to_numpy(dtype=float)
            if len(v) != len(t):
                continue
            rows[m] = np.interp(time_grid, t, v, left=v[0], right=v[-1])
        out.append(pd.DataFrame(rows))

    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)

def run_all_algorithms(
    # Shared
    outer_iterations: int = 1000,
    return_histories: bool = True,
    n_runs: int = 10,
    seed_offset: int = 0,
    seed_stride: int = 1,
    out_dir: Optional[str] = None,
    time_budget_s: Optional[float] = None,
    time_grid_s: float = 1.0,
    only_hg: bool = False,

    # ---- V-PBGD ----
    vp_seed: int = 2424,
    vp_pollute_rate: float = 0.8,
    vp_lrx: float = 0.1,
    vp_lry: float = 0.1,
    vp_lr_inner: float = 0.001,
    vp_gamma_init: float = 0.0,
    vp_gamma_max: float = 0.2,
    vp_gamma_steps: int = 30000,
    vp_inner_itr: int = 1,
    vp_reg: float = 0.0,
    vp_hidden: int = 300,
    vp_verbose: bool = True,

    # ---- G-PBGD (GPGD) ----
    gp_seed: int = 2424,
    gp_pollute_rate: float = 0.8,
    gp_lrx: float = 0.1,
    gp_lry: float = 0.1,
    gp_lr_inner: float = 0.001,   # kept for parity (unused)
    gp_gamma_init: float = 0.0,
    gp_gamma_max: float = 0.2,
    gp_gamma_steps: int = 30000,
    gp_inner_itr: int = 1,        # kept for parity (unused)
    gp_reg: float = 0.0,
    gp_hidden: int = 300,
    gp_verbose: bool = True,

    # ---- IAPTT-GM ----
    iaptt_seed: int = 2424,
    iaptt_pollute_rate: float = 0.8,
    iaptt_print_every: int = 1,
    iaptt_lrx: float = 0.10,
    iaptt_lrz: float = 0.10,
    iaptt_K_inner: int = 5,
    iaptt_lr_inner: float = 0.01,
    iaptt_reg: float = 0.0,
    iaptt_hidden: int = 300,
    iaptt_num_classes: int = 10,
    iaptt_verbose: bool = True,

    # ---- ITD baseline ----
    itd_seed: int = 2424,
    itd_pollute_rate: float = 0.8,
    itd_print_every: int = 1,
    itd_D_inner_gd: int = 20,
    itd_lr_inner_gd: float = 0.005,
    itd_inner_reg: float = 0.0,
    # kept for parity (unused baseline)
    itd_M_candidates: int = 5,
    itd_D_inner_sgld: int = 20,
    itd_lr_inner_sgld: float = 0.02,
    itd_sgld_temp: float = 0.0,
    itd_lr_x: float = 0.1,
    itd_x_clip: float = 5.0,
    itd_hyper_clip: float = 0.0,
    itd_reg: float = 1e-4,
    itd_hidden: int = 300,
    itd_num_classes: int = 10,
    itd_init_scale: float = 0.1,
    itd_verbose: bool = True,

    # ---- ITD + Minima Selection ----
    itdms_seed: int = 2424,
    itdms_pollute_rate: float = 0.8,
    itdms_print_every: int = 1,
    # HG-MinSel-LMC defaults: no unrolled inner GD for hypergrad (optional polish only).
    itdms_D_inner_gd: int = 0,
    itdms_lr_inner_gd: float = 0.005,
    itdms_inner_reg: float = 0.0,
    # Persistent parallel LMC chains for minima selection.
    itdms_M_candidates: int = 8,
    itdms_D_inner_sgld: int = 20,
    itdms_lr_inner_sgld: float = 0.02,
    itdms_sgld_temp: float = 0.0,
    itdms_sgld_batch_size: int = 256,
    itdms_persistent_chains: bool = True,
    itdms_elitism_chain: bool = True,
    itdms_include_base_candidate: bool = True,
    itdms_minsel_backend: str = "adamw",
    itdms_branch_init_noise_std: float = 0.0,
    itdms_branch_lr_mult_min: float = 1.0,
    itdms_branch_lr_mult_max: float = 1.0,
    itdms_branch_weight_decay: float = 0.0,
    itdms_selection_min_improve: float = 0.01,
    itdms_selection_use_significance: bool = True,
    itdms_selection_z: float = 1.96,
    itdms_ms_every: int = 1,
    itdms_lr_x: float = 0.1,
    # Adaptive outer (hyper) step-size for HG-MS
    itdms_lr_x_schedule: str = "cosine",
    itdms_lr_x_final: float = 0.01,
    itdms_lr_x_decay_steps: int = 200,
    # Implicit hypergradient via CG/HVP (HG-MinSel-LMC)
    itdms_cg_tol: float = 1e-3,
    itdms_cg_max_it: int = 30,
    itdms_cg_ridge: float = 1e-2,
    itdms_cg_batch_size: int = 0,
    itdms_x_clip: float = 5.0,
    itdms_hyper_clip: float = 0.0,
    itdms_reg: float = 1e-4,
    itdms_hidden: int = 300,
    itdms_num_classes: int = 10,
    itdms_init_scale: float = 0.1,
    itdms_verbose: bool = True,

    # # ---- AID baseline ----
    # aid_seed: int = 2424,
    # aid_pollute_rate: float = 0.5,
    # aid_print_every: int = 1,
    # aid_D_inner_gd: int = 10,
    # aid_lr_inner_gd: float = 0.01,
    # aid_inner_momentum: float = 0.0,
    # aid_inner_clip: float = 5.0,
    # # parity only
    # aid_M_candidates: int = 5,
    # aid_D_inner_sgld: int = 10,
    # aid_lr_inner_sgld: float = 0.01,
    # aid_sgld_temp: float = 0.0,
    # aid_init_jitter: float = 3e-2,
    # aid_lr_x: float = 0.1,
    # aid_hyper_clip: float = 100.0,
    # aid_reg: float = 1e-4,
    # aid_cg_max_it: int = 30,
    # aid_cg_tol: float = 1e-6,
    # aid_cg_damping: float = 1e-2,
    # aid_hidden: int = 300,
    # aid_verbose: bool = True,

    # # ---- AID + Minima Selection ----
    # aidms_seed: int = 2424,
    # aidms_pollute_rate: float = 0.5,
    # aidms_print_every: int = 1,
    # aidms_D_inner_gd: int = 10,          # parity
    # aidms_lr_inner_gd: float = 0.01,    # parity
    # aidms_inner_momentum: float = 0.0,
    # aidms_inner_clip: float = 5.0,
    # aidms_M_candidates: int = 5,
    # aidms_D_inner_sgld: int = 10,
    # aidms_lr_inner_sgld: float = 0.01,
    # aidms_sgld_temp: float = 0.0,
    # aidms_init_jitter: float = 3e-2,
    # aidms_lr_x: float = 0.1,
    # aidms_hyper_clip: float = 100.0,
    # aidms_reg: float = 1e-4,
    # aidms_cg_max_it: int = 30,
    # aidms_cg_tol: float = 1e-6,
    # aidms_cg_damping: float = 1e-2,
    # aidms_hidden: int = 300,
    # aidms_verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs the algorithms with a *fixed* outer_iterations for comparability.
    Repeats each algorithm `n_runs` times (seeded) and reports mean/std and 95% CI.
    """
    if out_dir is None:
        out_dir = DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    if time_budget_s is not None:
        time_budget_s = float(time_budget_s)
        if time_budget_s <= 0:
            time_budget_s = None
    time_grid_s = float(time_grid_s)
    if time_grid_s <= 0:
        time_grid_s = 1.0

    vpgd_fn = None
    gpbgd_fn = None
    iaptt_fn = None
    if not bool(only_hg):
        # --- Load V-PBGD dynamically (hyphen in filename) ---
        vp_mod = _load_module_from_path("vpgd_mod", os.path.join(DATA_DIR, "V-PBGD.py"))
        vpgd_fn = _resolve_vpgd_runner(vp_mod)

        # --- Load G-PBGD dynamically (hyphen in filename) ---
        gp_mod = _load_module_from_path("gpbgd_mod", os.path.join(DATA_DIR, "G-PBGD.py"))
        gpbgd_fn = _resolve_gpbgd_runner(gp_mod)

        # --- Load IAPTT-GM dynamically (hyphen in filename) ---
        iaptt_mod = _load_module_from_path("iaptt_gm_mod", os.path.join(DATA_DIR, "IAPTT-GM.py"))
        iaptt_fn = _resolve_iaptt_gm_runner(iaptt_mod)

    histories: List[pd.DataFrame] = []
    finals_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    algo_labels = {
        "ITD": "HG-baseline",
        "ITD+MS": "HG-MS",
    }

    # --------- Repeat runs ---------
    for run_idx in range(int(n_runs)):
        seed_bump = seed_offset + run_idx * seed_stride

        def _capture_final(run_name: str, final: Any):
            row = {"algorithm": run_name, "run": run_idx}
            if isinstance(final, dict):
                row.update(final)
                iters = row.get("iterations_ran", None)
                elapsed = row.get("elapsed_s", None)
                if iters is not None and elapsed is not None:
                    try:
                        iters_f = float(iters)
                        elapsed_f = float(elapsed)
                        if iters_f > 0 and elapsed_f >= 0:
                            row["sec_per_outer"] = elapsed_f / iters_f
                    except Exception:
                        pass
            else:
                row["final"] = final
            finals_rows.append(row)

        if not bool(only_hg):
            # --------- V-PBGD ---------
            try:
                vp_final, vp_hist = _norm_return(
                    vpgd_fn,
                    seed=vp_seed + seed_bump, pollute_rate=vp_pollute_rate,
                    lrx=vp_lrx, lry=vp_lry, lr_inner=vp_lr_inner,
                    gamma_init=vp_gamma_init, gamma_max=vp_gamma_max, gamma_steps=vp_gamma_steps,
                    outer_itr=outer_iterations, inner_itr=vp_inner_itr,
                    reg=vp_reg, hidden=vp_hidden,
                    verbose=vp_verbose, log_every=1, return_history=return_histories,
                    time_budget_s=time_budget_s,
                )
                _capture_final("V-PBGD", vp_final)
                hist_df = _history_to_df(vp_hist, "V-PBGD")
                if not hist_df.empty:
                    hist_df["run"] = run_idx
                    histories.append(hist_df)
            except Exception as e:
                errors.append({"algorithm": "V-PBGD", "run": run_idx, "error": repr(e)})

            # --------- G-PBGD ---------
            try:
                gp_final, gp_hist = _norm_return(
                    gpbgd_fn,
                    seed=gp_seed + seed_bump, pollute_rate=gp_pollute_rate,
                    lrx=gp_lrx, lry=gp_lry, lr_inner=gp_lr_inner,
                    gamma_init=gp_gamma_init, gamma_max=gp_gamma_max, gamma_steps=gp_gamma_steps,
                    outer_itr=outer_iterations, inner_itr=gp_inner_itr,
                    reg=gp_reg, hidden=gp_hidden,
                    verbose=gp_verbose, log_every=1, return_history=return_histories,
                    time_budget_s=time_budget_s,
                )
                _capture_final("G-PBGD", gp_final)
                hist_df = _history_to_df(gp_hist, "G-PBGD")
                if not hist_df.empty:
                    hist_df["run"] = run_idx
                    histories.append(hist_df)
            except Exception as e:
                errors.append({"algorithm": "G-PBGD", "run": run_idx, "error": repr(e)})

        # --------- ITD baseline ---------
        try:
            itd_b_final, itd_b_hist = _norm_return(
                run_ITD_baseline,
                seed=itd_seed + seed_bump, pollute_rate=itd_pollute_rate,
                outer_itr=outer_iterations, print_every=itd_print_every,
                D_inner_gd=itd_D_inner_gd, lr_inner_gd=itd_lr_inner_gd, inner_reg=itd_inner_reg,
                M_candidates=itd_M_candidates, D_inner_sgld=itd_D_inner_sgld,
                lr_inner_sgld=itd_lr_inner_sgld, sgld_temp=itd_sgld_temp,
                lr_x=itd_lr_x, x_clip=itd_x_clip, hyper_clip=itd_hyper_clip,
                reg=itd_reg, hidden=itd_hidden, num_classes=itd_num_classes, init_scale=itd_init_scale,
                verbose=itd_verbose, log_every=1, return_history=return_histories,
                time_budget_s=time_budget_s,
            )
            _capture_final(algo_labels["ITD"], itd_b_final)
            hist_df = _history_to_df(itd_b_hist, algo_labels["ITD"])
            if not hist_df.empty:
                hist_df["run"] = run_idx
                histories.append(hist_df)
        except Exception as e:
            errors.append({"algorithm": algo_labels["ITD"], "run": run_idx, "error": repr(e)})

        if not bool(only_hg):
            # --------- IAPTT-GM ---------
            try:
                iaptt_final, iaptt_hist = _norm_return(
                    iaptt_fn,
                    seed=iaptt_seed + seed_bump, pollute_rate=iaptt_pollute_rate,
                    outer_itr=outer_iterations, print_every=iaptt_print_every,
                    lrx=iaptt_lrx, lrz=iaptt_lrz,
                    K_inner=iaptt_K_inner, lr_inner=iaptt_lr_inner,
                    reg=iaptt_reg, hidden=iaptt_hidden, num_classes=iaptt_num_classes,
                    verbose=iaptt_verbose, log_every=1, return_history=return_histories,
                    time_budget_s=time_budget_s,
                )
                _capture_final("IAPTT-GM", iaptt_final)
                hist_df = _history_to_df(iaptt_hist, "IAPTT-GM")
                if not hist_df.empty:
                    hist_df["run"] = run_idx
                    histories.append(hist_df)
            except Exception as e:
                errors.append({"algorithm": "IAPTT-GM", "run": run_idx, "error": repr(e)})

        # --------- ITD + minima selection ---------
        try:
            itd_s_final, itd_s_hist = _norm_return(
                run_ITD_Sel,
                seed=itdms_seed + seed_bump, pollute_rate=itdms_pollute_rate,
                outer_itr=outer_iterations, print_every=itdms_print_every,
                D_inner_gd=itdms_D_inner_gd, lr_inner_gd=itdms_lr_inner_gd, inner_reg=itdms_inner_reg,
                M_candidates=itdms_M_candidates, D_inner_sgld=itdms_D_inner_sgld,
                lr_inner_sgld=itdms_lr_inner_sgld, sgld_temp=itdms_sgld_temp,
                sgld_batch_size=itdms_sgld_batch_size,
                persistent_chains=itdms_persistent_chains,
                elitism_chain=itdms_elitism_chain,
                include_base_candidate=itdms_include_base_candidate,
                minsel_backend=itdms_minsel_backend,
                branch_init_noise_std=itdms_branch_init_noise_std,
                branch_lr_mult_min=itdms_branch_lr_mult_min,
                branch_lr_mult_max=itdms_branch_lr_mult_max,
                branch_weight_decay=itdms_branch_weight_decay,
                selection_min_improve=itdms_selection_min_improve,
                selection_use_significance=itdms_selection_use_significance,
                selection_z=itdms_selection_z,
                ms_every=itdms_ms_every,
                lr_x=itdms_lr_x,
                lr_x_schedule=itdms_lr_x_schedule,
                lr_x_final=itdms_lr_x_final,
                lr_x_decay_steps=itdms_lr_x_decay_steps,
                cg_tol=itdms_cg_tol,
                cg_max_it=itdms_cg_max_it,
                cg_ridge=itdms_cg_ridge,
                cg_batch_size=itdms_cg_batch_size,
                x_clip=itdms_x_clip, hyper_clip=itdms_hyper_clip,
                reg=itdms_reg, hidden=itdms_hidden, num_classes=itdms_num_classes, init_scale=itdms_init_scale,
                verbose=itdms_verbose, log_every=1, return_history=return_histories,
                time_budget_s=time_budget_s,
            )
            _capture_final(algo_labels["ITD+MS"], itd_s_final)
            hist_df = _history_to_df(itd_s_hist, algo_labels["ITD+MS"])
            if not hist_df.empty:
                hist_df["run"] = run_idx
                histories.append(hist_df)
        except Exception as e:
            errors.append({"algorithm": algo_labels["ITD+MS"], "run": run_idx, "error": repr(e)})

    # --------- (optional) AID variants ---------
    # aid_b_final, aid_b_hist = ...
    # aid_s_final, aid_s_hist = ...

    # ---------------- Aggregation ----------------
    finals_raw = pd.DataFrame(finals_rows)

    # ---------------- Aggregation ----------------
    all_hist_raw = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    metric_cols_hist = [c for c in ("train_acc", "test_acc", "val_loss", "g_train") if c in all_hist_raw.columns]
    metric_cols_final = [
        c for c in ("train_acc", "test_acc", "val_loss", "g_train", "precision", "recall", "f1", "iterations_ran", "elapsed_s", "sec_per_outer")
        if c in finals_raw.columns
    ]

    mode = "time" if time_budget_s is not None else "iterations"

    history_resampled = pd.DataFrame()
    history_mean_ci = pd.DataFrame()
    history_best_raw = pd.DataFrame()
    history_best_resampled = pd.DataFrame()
    history_best_mean_ci = pd.DataFrame()
    if return_histories and not all_hist_raw.empty and metric_cols_hist:
        # Build a "best-so-far" version of the history (monotone by construction):
        # - accuracies: cumulative max
        # - losses: cumulative min
        best_rows: List[pd.DataFrame] = []
        for (algo, run_idx_), sub in all_hist_raw.groupby(["algorithm", "run"], dropna=False):
            sub = sub.sort_values("elapsed_s" if "elapsed_s" in sub.columns else "iteration").copy()
            if "train_acc" in metric_cols_hist and "train_acc" in sub.columns:
                sub["train_acc"] = sub["train_acc"].cummax()
            if "test_acc" in metric_cols_hist and "test_acc" in sub.columns:
                sub["test_acc"] = sub["test_acc"].cummax()
            if "val_loss" in metric_cols_hist and "val_loss" in sub.columns:
                sub["val_loss"] = sub["val_loss"].cummin()
            if "g_train" in metric_cols_hist and "g_train" in sub.columns:
                sub["g_train"] = sub["g_train"].cummin()
            best_rows.append(sub)
        history_best_raw = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()

        if mode == "iterations":
            history_mean_ci = _aggregate_mean_std_ci(
                all_hist_raw,
                group_cols=["algorithm", "iteration"],
                metric_cols=metric_cols_hist,
            ).sort_values(["algorithm", "iteration"])

            if not history_best_raw.empty:
                history_best_mean_ci = _aggregate_mean_std_ci(
                    history_best_raw,
                    group_cols=["algorithm", "iteration"],
                    metric_cols=metric_cols_hist,
                ).sort_values(["algorithm", "iteration"])
        else:
            grid = np.arange(0.0, time_budget_s + 1e-12, time_grid_s, dtype=float)
            history_resampled = _resample_history_time_grid(
                all_hist_raw,
                time_grid=grid,
                metric_cols=metric_cols_hist,
            )
            history_mean_ci = _aggregate_mean_std_ci(
                history_resampled,
                group_cols=["algorithm", "elapsed_s"],
                metric_cols=metric_cols_hist,
            ).sort_values(["algorithm", "elapsed_s"])

            if not history_best_raw.empty:
                history_best_resampled = _resample_history_time_grid(
                    history_best_raw,
                    time_grid=grid,
                    metric_cols=metric_cols_hist,
                )
                history_best_mean_ci = _aggregate_mean_std_ci(
                    history_best_resampled,
                    group_cols=["algorithm", "elapsed_s"],
                    metric_cols=metric_cols_hist,
                ).sort_values(["algorithm", "elapsed_s"])

    finals_mean_std = (
        _aggregate_mean_std_ci(
            finals_raw,
            group_cols=["algorithm"],
            metric_cols=metric_cols_final,
        ).sort_values(["algorithm"])
        if (not finals_raw.empty and metric_cols_final)
        else pd.DataFrame()
    )

    last_point_summary = pd.DataFrame()
    if not history_mean_ci.empty:
        if mode == "iterations" and "iteration" in history_mean_ci.columns:
            last_x = history_mean_ci["iteration"].max()
            last_point_summary = history_mean_ci[history_mean_ci["iteration"] == last_x].copy()
            last_point_summary.insert(1, "last_iteration", int(last_x))
        elif mode == "time" and "elapsed_s" in history_mean_ci.columns:
            last_x = history_mean_ci["elapsed_s"].max()
            last_point_summary = history_mean_ci[history_mean_ci["elapsed_s"] == last_x].copy()
            last_point_summary.insert(1, "last_time_s", float(last_x))

    best_point_summary = pd.DataFrame()
    if not history_best_mean_ci.empty:
        if mode == "iterations" and "iteration" in history_best_mean_ci.columns:
            last_x = history_best_mean_ci["iteration"].max()
            best_point_summary = history_best_mean_ci[history_best_mean_ci["iteration"] == last_x].copy()
            best_point_summary.insert(1, "best_up_to_iteration", int(last_x))
        elif mode == "time" and "elapsed_s" in history_best_mean_ci.columns:
            last_x = history_best_mean_ci["elapsed_s"].max()
            best_point_summary = history_best_mean_ci[history_best_mean_ci["elapsed_s"] == last_x].copy()
            best_point_summary.insert(1, "best_up_to_time_s", float(last_x))

    # ---------------- Plots ----------------
    plots: Dict[str, str] = {}
    if not history_mean_ci.empty:
        if mode == "iterations":
            train_plot = os.path.join(out_dir, "plot_train_accuracy_vs_outer_mean_ci.png")
            _save_plot_mean_ci(history_mean_ci, "train_acc", train_plot, "Training Accuracy vs Outer Iterations (mean ± 95% CI)")
        else:
            train_plot = os.path.join(out_dir, "plot_train_accuracy_vs_time_mean_ci.png")
            _save_plot_mean_ci(history_mean_ci, "train_acc", train_plot, "Training Accuracy vs Time (mean ± 95% CI)", x_col="elapsed_s", x_label="Time (s)")
        plots["train_accuracy_mean_ci"] = train_plot

        if mode == "iterations":
            test_plot = os.path.join(out_dir, "plot_test_accuracy_vs_outer_mean_ci.png")
            _save_plot_mean_ci(history_mean_ci, "test_acc", test_plot, "Test Accuracy vs Outer Iterations (mean ± 95% CI)")
        else:
            test_plot = os.path.join(out_dir, "plot_test_accuracy_vs_time_mean_ci.png")
            _save_plot_mean_ci(history_mean_ci, "test_acc", test_plot, "Test Accuracy vs Time (mean ± 95% CI)", x_col="elapsed_s", x_label="Time (s)")
        plots["test_accuracy_mean_ci"] = test_plot

        if mode == "iterations":
            f_plot = os.path.join(out_dir, "plot_upper_val_loss_vs_outer_mean_ci.png")
            _save_plot_mean_ci(history_mean_ci, "val_loss", f_plot, "Upper-level (Validation) Loss vs Outer Iterations (mean ± 95% CI)")
        else:
            f_plot = os.path.join(out_dir, "plot_upper_val_loss_vs_time_mean_ci.png")
            _save_plot_mean_ci(history_mean_ci, "val_loss", f_plot, "Upper-level (Validation) Loss vs Time (mean ± 95% CI)", x_col="elapsed_s", x_label="Time (s)")
        plots["upper_val_loss_mean_ci"] = f_plot

        if "g_train_mean" in history_mean_ci.columns:
            if mode == "iterations":
                g_plot = os.path.join(out_dir, "plot_lower_train_loss_vs_outer_mean_ci.png")
                _save_plot_mean_ci(history_mean_ci, "g_train", g_plot, "Lower-level (Training) Loss g(x,y) vs Outer Iterations (mean ± 95% CI)")
            else:
                g_plot = os.path.join(out_dir, "plot_lower_train_loss_vs_time_mean_ci.png")
                _save_plot_mean_ci(history_mean_ci, "g_train", g_plot, "Lower-level (Training) Loss g(x,y) vs Time (mean ± 95% CI)", x_col="elapsed_s", x_label="Time (s)")
            plots["lower_train_loss_mean_ci"] = g_plot

    if not history_best_mean_ci.empty:
        if mode == "iterations":
            train_plot = os.path.join(out_dir, "plot_train_accuracy_best_vs_outer_mean_ci.png")
            _save_plot_mean_ci(history_best_mean_ci, "train_acc", train_plot, "Best-So-Far Training Accuracy vs Outer Iterations (mean ± 95% CI)")
        else:
            train_plot = os.path.join(out_dir, "plot_train_accuracy_best_vs_time_mean_ci.png")
            _save_plot_mean_ci(history_best_mean_ci, "train_acc", train_plot, "Best-So-Far Training Accuracy vs Time (mean ± 95% CI)", x_col="elapsed_s", x_label="Time (s)")
        plots["train_accuracy_best_mean_ci"] = train_plot

        if mode == "iterations":
            test_plot = os.path.join(out_dir, "plot_test_accuracy_best_vs_outer_mean_ci.png")
            _save_plot_mean_ci(history_best_mean_ci, "test_acc", test_plot, "Best-So-Far Test Accuracy vs Outer Iterations (mean ± 95% CI)")
        else:
            test_plot = os.path.join(out_dir, "plot_test_accuracy_best_vs_time_mean_ci.png")
            _save_plot_mean_ci(history_best_mean_ci, "test_acc", test_plot, "Best-So-Far Test Accuracy vs Time (mean ± 95% CI)", x_col="elapsed_s", x_label="Time (s)")
        plots["test_accuracy_best_mean_ci"] = test_plot

        if mode == "iterations":
            f_plot = os.path.join(out_dir, "plot_upper_val_loss_best_vs_outer_mean_ci.png")
            _save_plot_mean_ci(history_best_mean_ci, "val_loss", f_plot, "Best-So-Far Upper-level (Validation) Loss vs Outer Iterations (mean ± 95% CI)")
        else:
            f_plot = os.path.join(out_dir, "plot_upper_val_loss_best_vs_time_mean_ci.png")
            _save_plot_mean_ci(history_best_mean_ci, "val_loss", f_plot, "Best-So-Far Upper-level (Validation) Loss vs Time (mean ± 95% CI)", x_col="elapsed_s", x_label="Time (s)")
        plots["upper_val_loss_best_mean_ci"] = f_plot

        if "g_train_mean" in history_best_mean_ci.columns:
            if mode == "iterations":
                g_plot = os.path.join(out_dir, "plot_lower_train_loss_best_vs_outer_mean_ci.png")
                _save_plot_mean_ci(history_best_mean_ci, "g_train", g_plot, "Best-So-Far Lower-level (Training) Loss g(x,y) vs Outer Iterations (mean ± 95% CI)")
            else:
                g_plot = os.path.join(out_dir, "plot_lower_train_loss_best_vs_time_mean_ci.png")
                _save_plot_mean_ci(history_best_mean_ci, "g_train", g_plot, "Best-So-Far Lower-level (Training) Loss g(x,y) vs Time (mean ± 95% CI)", x_col="elapsed_s", x_label="Time (s)")
            plots["lower_train_loss_best_mean_ci"] = g_plot

    return {
        "mode": mode,
        "n_runs": int(n_runs),
        "seed_offset": int(seed_offset),
        "seed_stride": int(seed_stride),
        "time_budget_s": time_budget_s,
        "time_grid_s": time_grid_s,
        "finals_raw": finals_raw,
        "finals_mean_std": finals_mean_std,
        "history_raw": all_hist_raw,
        "history_resampled": history_resampled,
        "history_mean_ci": history_mean_ci,
        "summary_last_point": last_point_summary,
        "history_best_raw": history_best_raw,
        "history_best_resampled": history_best_resampled,
        "history_best_mean_ci": history_best_mean_ci,
        "summary_best_point": best_point_summary,
        "plots": plots,
        "errors": errors,
    }


if __name__ == "__main__":
    import argparse
    import json

    from gpu_utils import print_device_info, resolve_device

    parser = argparse.ArgumentParser(description="Run and compare V-PBGD, G-PBGD, ITD (baseline & minima selection), and AID (baseline & minima selection).")
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "minsel-gap", "minsel-gap-all"],
        help=(
            "Convenience preset. "
            "'minsel-gap' runs only HG-baseline vs HG-MS with a short budget. "
            "'minsel-gap-all' runs the same short budget but includes V-PBGD, G-PBGD and IAPTT-GM too."
        ),
    )
    parser.add_argument("--outer-iterations", type=int, default=1000, help="Max outer iterations (also used when not time-budgeted).")
    parser.add_argument("--time-budget-seconds", type=float, default=None, help="If set, stop each algorithm when its own wall-clock training time reaches this budget.")
    parser.add_argument("--time-grid-seconds", type=float, default=1.0, help="Time grid (seconds) for mean/CI curves when using --time-budget-seconds.")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated runs (different seeds) per algorithm.")
    parser.add_argument("--pollute-rate", type=float, default=0.8, help="Label-noise rate (applied to all algorithms).")
    parser.add_argument(
        "--only-hg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, run only HG-baseline and HG-MS (skips V-PBGD/G-PBGD/IAPTT-GM).",
    )
    parser.add_argument("--seed-offset", type=int, default=0, help="Seed offset applied before run index bump.")
    parser.add_argument("--seed-stride", type=int, default=1, help="Seed stride multiplied by run index bump.")
    parser.add_argument("--plots-dir", type=str, default=".", help="Directory to write plots and CSVs.")
    parser.add_argument("--no-history", action="store_true", help="If set, do not return/save per-iteration history.")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-iteration logs from algorithms (use --no-verbose to silence).",
    )
    parser.add_argument("--hg-inner-steps", type=int, default=20, help="HG-baseline inner GD steps (unrolled).")
    parser.add_argument("--hg-inner-lr", type=float, default=0.005, help="HG-baseline inner GD learning rate.")
    parser.add_argument("--hg-lr-x", type=float, default=0.1, help="HG-baseline outer lr_x (hyper step size).")
    parser.add_argument("--hgms-lr-x", type=float, default=0.1, help="HG-MS initial lr_x (hyper step size).")
    parser.add_argument(
        "--hgms-lr-x-schedule",
        type=str,
        default="cosine",
        choices=["constant", "cosine", "linear", "inv"],
        help="HG-MS lr_x schedule.",
    )
    parser.add_argument("--hgms-lr-x-final", type=float, default=0.01, help="HG-MS lr_x final value (cosine/linear).")
    parser.add_argument("--hgms-lr-x-decay-steps", type=int, default=200, help="HG-MS lr_x decay steps (inv).")
    parser.add_argument("--hgms-m-candidates", type=int, default=8, help="HG-MS number of parallel LMC chains (m).")
    parser.add_argument("--hgms-lmc-steps", type=int, default=20, help="HG-MS LMC steps per outer iteration (K).")
    parser.add_argument("--hgms-lmc-step-size", type=float, default=0.02, help="HG-MS LMC step size (h).")
    parser.add_argument("--hgms-sgld-temp", type=float, default=0.0, help="HG-MS LMC temperature (lambda).")
    parser.add_argument("--hgms-sgld-batch-size", type=int, default=256, help="HG-MS minibatch size for LMC gradients.")
    parser.add_argument(
        "--hgms-persistent-chains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HG-MS persistent chains across outer iterations.",
    )
    parser.add_argument(
        "--hgms-elitism-chain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HG-MS keep an elite chain equal to the selected best state.",
    )
    parser.add_argument(
        "--hgms-include-base-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HG-MS include base candidate in minima selection.",
    )
    parser.add_argument(
        "--hgms-backend",
        type=str,
        default="adamw",
        choices=["adamw", "lmc"],
        help="HG-MS minima selection backend (default: adamw parallel branches).",
    )
    parser.add_argument("--hgms-branch-init-noise-std", type=float, default=0.0, help="HG-MS branch init noise std (diversity).")
    parser.add_argument("--hgms-branch-lr-mult-min", type=float, default=1.0, help="HG-MS min branch lr multiplier (log-spaced).")
    parser.add_argument("--hgms-branch-lr-mult-max", type=float, default=1.0, help="HG-MS max branch lr multiplier (log-spaced).")
    parser.add_argument("--hgms-branch-weight-decay", type=float, default=0.0, help="HG-MS branch AdamW weight decay.")
    parser.add_argument("--hgms-selection-min-improve", type=float, default=0.01, help="HG-MS min val-loss improvement to switch from base.")
    parser.add_argument(
        "--hgms-selection-use-significance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HG-MS require paired-difference significance gate when switching candidates.",
    )
    parser.add_argument("--hgms-selection-z", type=float, default=1.96, help="HG-MS z-threshold for significance gate (e.g., 1.96 ~ 95%).")
    parser.add_argument("--hgms-ms-every", type=int, default=1, help="HG-MS run minima selection every N outer iterations.")
    parser.add_argument("--hgms-polish-gd-steps", type=int, default=0, help="HG-MS optional GD polish steps on g after selection.")
    parser.add_argument("--hgms-cg-tol", type=float, default=1e-3, help="HG-MS CG relative tolerance.")
    parser.add_argument("--hgms-cg-max-it", type=int, default=30, help="HG-MS CG max iterations.")
    parser.add_argument("--hgms-cg-ridge", type=float, default=1e-2, help="HG-MS CG ridge (gamma) for conditioning.")
    parser.add_argument("--hgms-cg-batch-size", type=int, default=0, help="HG-MS batch size for CG/HVP computations (0=full batch).")

    # Two-pass parsing: first read --preset, then apply preset defaults, then parse fully.
    pre_args, _ = parser.parse_known_args()
    if pre_args.preset == "minsel-gap":
        parser.set_defaults(
            outer_iterations=50,
            runs=3,
            only_hg=True,
        )
    elif pre_args.preset == "minsel-gap-all":
        parser.set_defaults(
            outer_iterations=50,
            runs=3,
            only_hg=False,
        )
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    print_device_info(resolve_device(), tag="run_all_algorithms")

    results = run_all_algorithms(
        outer_iterations=args.outer_iterations,
        return_histories=not args.no_history,
        n_runs=args.runs,
        seed_offset=args.seed_offset,
        seed_stride=args.seed_stride,
        out_dir=args.plots_dir,
        time_budget_s=args.time_budget_seconds,
        time_grid_s=args.time_grid_seconds,
        only_hg=args.only_hg,
        vp_pollute_rate=args.pollute_rate,
        gp_pollute_rate=args.pollute_rate,
        iaptt_pollute_rate=args.pollute_rate,
        itd_pollute_rate=args.pollute_rate,
        itdms_pollute_rate=args.pollute_rate,
        vp_verbose=args.verbose,
        gp_verbose=args.verbose,
        iaptt_verbose=args.verbose,
        itd_verbose=args.verbose,
        itdms_verbose=args.verbose,
        itd_D_inner_gd=args.hg_inner_steps,
        itd_lr_inner_gd=args.hg_inner_lr,
        itd_lr_x=args.hg_lr_x,
        itdms_D_inner_gd=args.hgms_polish_gd_steps,
        itdms_lr_x=args.hgms_lr_x,
        itdms_lr_x_schedule=args.hgms_lr_x_schedule,
        itdms_lr_x_final=args.hgms_lr_x_final,
        itdms_lr_x_decay_steps=args.hgms_lr_x_decay_steps,
        itdms_M_candidates=args.hgms_m_candidates,
        itdms_D_inner_sgld=args.hgms_lmc_steps,
        itdms_lr_inner_sgld=args.hgms_lmc_step_size,
        itdms_sgld_temp=args.hgms_sgld_temp,
        itdms_sgld_batch_size=args.hgms_sgld_batch_size,
        itdms_persistent_chains=args.hgms_persistent_chains,
        itdms_elitism_chain=args.hgms_elitism_chain,
        itdms_include_base_candidate=args.hgms_include_base_candidate,
        itdms_minsel_backend=args.hgms_backend,
        itdms_branch_init_noise_std=args.hgms_branch_init_noise_std,
        itdms_branch_lr_mult_min=args.hgms_branch_lr_mult_min,
        itdms_branch_lr_mult_max=args.hgms_branch_lr_mult_max,
        itdms_branch_weight_decay=args.hgms_branch_weight_decay,
        itdms_selection_min_improve=args.hgms_selection_min_improve,
        itdms_selection_use_significance=args.hgms_selection_use_significance,
        itdms_selection_z=args.hgms_selection_z,
        itdms_ms_every=args.hgms_ms_every,
        itdms_cg_tol=args.hgms_cg_tol,
        itdms_cg_max_it=args.hgms_cg_max_it,
        itdms_cg_ridge=args.hgms_cg_ridge,
        itdms_cg_batch_size=args.hgms_cg_batch_size,
    )

    # Save CSVs
    summary_name = "summary_last_time_mean_ci.csv" if results["mode"] == "time" else "summary_last_iteration_mean_ci.csv"
    summary_path = os.path.join(args.plots_dir, summary_name)
    results["summary_last_point"].to_csv(summary_path, index=False)

    summary_best_name = "summary_best_time_mean_ci.csv" if results["mode"] == "time" else "summary_best_iteration_mean_ci.csv"
    summary_best_path = os.path.join(args.plots_dir, summary_best_name)
    if results.get("summary_best_point") is not None and not results["summary_best_point"].empty:
        results["summary_best_point"].to_csv(summary_best_path, index=False)
    else:
        summary_best_path = None

    finals_path = os.path.join(args.plots_dir, "finals_mean_std.csv")
    results["finals_mean_std"].to_csv(finals_path, index=False)

    if not args.no_history:
        history_name = "history_time_mean_ci.csv" if results["mode"] == "time" else "history_mean_ci.csv"
        history_path = os.path.join(args.plots_dir, history_name)
        results["history_mean_ci"].to_csv(history_path, index=False)

        history_best_name = "history_best_time_mean_ci.csv" if results["mode"] == "time" else "history_best_mean_ci.csv"
        history_best_path = os.path.join(args.plots_dir, history_best_name)
        if results.get("history_best_mean_ci") is not None and not results["history_best_mean_ci"].empty:
            results["history_best_mean_ci"].to_csv(history_best_path, index=False)
        else:
            history_best_path = None

        history_raw_path = os.path.join(args.plots_dir, "history_all_runs.csv")
        results["history_raw"].to_csv(history_raw_path, index=False)

        history_best_raw_path = os.path.join(args.plots_dir, "history_best_all_runs.csv")
        if results.get("history_best_raw") is not None and not results["history_best_raw"].empty:
            results["history_best_raw"].to_csv(history_best_raw_path, index=False)
        else:
            history_best_raw_path = None

        if results["mode"] == "time" and not results["history_resampled"].empty:
            history_resampled_path = os.path.join(args.plots_dir, "history_time_resampled_all_runs.csv")
            results["history_resampled"].to_csv(history_resampled_path, index=False)
        else:
            history_resampled_path = None

        if results["mode"] == "time" and results.get("history_best_resampled") is not None and not results["history_best_resampled"].empty:
            history_best_resampled_path = os.path.join(args.plots_dir, "history_best_time_resampled_all_runs.csv")
            results["history_best_resampled"].to_csv(history_best_resampled_path, index=False)
        else:
            history_best_resampled_path = None
    else:
        history_path = None
        history_best_path = None
        history_raw_path = None
        history_best_raw_path = None
        history_resampled_path = None
        history_best_resampled_path = None

    plot_out = results["plots"]

    print(json.dumps({
        "outer_iterations": args.outer_iterations,
        "mode": results["mode"],
        "time_budget_s": results["time_budget_s"],
        "time_grid_s": results["time_grid_s"],
        "runs": args.runs,
        "summary_csv": summary_path,
        "summary_best_csv": summary_best_path,
        "finals_csv": finals_path,
        "history_csv": history_path,
        "history_best_csv": history_best_path,
        "history_raw_csv": history_raw_path,
        "history_best_raw_csv": history_best_raw_path,
        "history_resampled_csv": history_resampled_path,
        "history_best_resampled_csv": history_best_resampled_path,
        "plots": plot_out,
        "errors": results["errors"],
    }, indent=2))
