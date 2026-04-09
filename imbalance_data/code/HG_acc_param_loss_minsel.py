import time
from copy import deepcopy
import math

import torch

from hg_acc_param_loss_core import (
    _init_kaiming,
    balanced_acc_no_warn,
    default_outer_from_config,
    default_trainer_from_config,
    device,
    eval_avg_f_loss,
    eval_avg_g_loss,
    implicit_hypergrad_x,
    maybe_restore_best,
    psgld_advance_state,
    psgld_init_state,
    print_device_banner,
    prepare_data,
    resolve_outer_iters,
    select_by_bal_acc_conservative,
    select_by_f_conservative,
    set_seed,
    FollowerCNN,
)


def run_hg_acc_sel(config, common):
    (train_loader, val_loader, test_loader, train_eval_loader, val_eval_loader, u_vec) = common

    def _is_full_batch_loader(loader) -> bool:
        try:
            bs = getattr(loader, "batch_size", None)
            ds = getattr(loader, "dataset", None)
            if bs is None or ds is None:
                return False
            return int(bs) >= int(len(ds))
        except Exception:
            return False

    config = dict(config)
    set_seed(int(config["seed"]))
    print_device_banner("HG_acc_sel")

    model = FollowerCNN().to(device)
    model.apply(_init_kaiming)
    trainer = default_trainer_from_config(model, config)

    delta_raw = torch.zeros(10, requires_grad=True, device=device)
    gamma_raw = torch.zeros(10, requires_grad=True, device=device)
    x_raw = (delta_raw, gamma_raw)

    outer, use_adam = default_outer_from_config(delta_raw, gamma_raw, config)

    iters_total = resolve_outer_iters(config)
    config["outer_iters"] = iters_total
    config["target_samples"] = iters_total * int(config["bs_train"])

    train_it, val_it = iter(train_loader), iter(val_loader)
    # NOTE: We keep both "current" metrics and a "best-so-far (by f(val))" view.
    # The best-so-far view makes the reported Test BalAcc much more stable and typically non-decreasing,
    # without using the test metric for selection (selection is still by f(val)).
    logs = {
        k: []
        for k in [
            "iterations",
            "elapsed_s",
            # Reported metrics (default: best-so-far by f(val))
            "bal_train_acc",
            "bal_test_acc",
            "train_loss",
            "val_loss",
            # Raw/current metrics (for debugging fluctuations)
            "bal_train_acc_curr",
            "bal_test_acc_curr",
            "train_loss_curr",
            "val_loss_curr",
            # Misc
            "best_iter",
            "x_alpha",
            "iteration_times",
        ]
    }

    best = {"f": float("inf"), "x": None, "state": None, "t": None, "metrics": None}
    # Safe default: report the best-so-far checkpoint (by f(val)) at eval points.
    report_best_at_eval = bool(config.get("report_best_at_eval", True))
    # Optional: stop once we stop improving f(val) for a while (disabled by default).
    early_stop_patience = max(0, int(config.get("early_stop_patience", 0) or 0))
    early_stop_min_delta = max(0.0, float(config.get("early_stop_min_delta", 0.0) or 0.0))
    no_improve_evals = 0

    trainer.step_k(
        model,
        x_raw,
        train_loader,
        steps=int(config["warm_y_steps"]),
        grad_clip=float(config["y_clip"]),
        accum=int(config.get("y_grad_accum", 1)),
    )

    time_budget_s = config.get("time_budget_s", None)
    if time_budget_s is not None:
        try:
            time_budget_s = float(time_budget_s)
            if time_budget_s <= 0:
                time_budget_s = None
        except Exception:
            time_budget_s = None
    start_t = time.perf_counter()

    # ------------------------------------------------------------------
    # Minima selection backend
    # ------------------------------------------------------------------
    # Two supported backends:
    #   1) "psgld": persistent pSGLD chains (stochastic exploration of g-minima)
    #   2) "adamw": persistent deterministic branches trained by standard optimizer (AdamW/SGD) on g
    #
    # Using "adamw" matches the idea: run multiple LL optimizers in parallel and pick the branch
    # with the best UL score f at each selection step.
    minsel_backend = str(config.get("minsel_backend", "psgld")).strip().lower()

    # Number of extra branches (besides the base model).
    # For backward compatibility we reuse `sgld_chains` as the extra-branch count.
    extra_branches = int(config.get("minsel_extra_branches", config.get("sgld_chains", 4)))
    extra_branches = max(0, extra_branches)

    # Shared EMA buffer for selection (candidate 0 is base, candidates[1:] are extra branches).
    minsel_f_ema = [None for _ in range(1 + extra_branches)]

    # --- Backend state ---
    chains_state = None
    branch_models = None
    branch_trainers = None
    branch_cfgs = None

    if minsel_backend in {"psgld", "sgld", "lmc"}:
        # HG-MinSel-LMC style persistent chains:
        # - keep multiple persistent Langevin chains over the follower params (y)
        # - at each outer step, advance each chain a bit using (p)SGLD on g, then pick the best by f
        chains_state = psgld_init_state(
            deepcopy(model).to(device),
            chains=int(extra_branches),
            seed=int(config["seed"]),
            init_noise_std=float(config.get("sgld_init_noise_std", 0.0) or 0.0),
        )
        chains_state["f_ema"] = minsel_f_ema

        # One-time burn-in (optional): helps chains move away from init before selection starts.
        if int(config.get("sgld_burn_in", 0)) > 0:
            chains_state = psgld_advance_state(
                chains_state,
                x_raw=x_raw,
                train_loader=train_loader,
                steps=int(config["sgld_burn_in"]),
                eps0=float(config["sgld_eps0"]),
                alpha=float(config["sgld_alpha"]),
                temperature=float(config.get("sgld_temperature", 0.0)),
                weight_decay=float(config["sgld_wd"]),
                rms_rho=float(config["sgld_rho"]),
                rms_eps=float(config["sgld_eps"]),
                grad_clip=float(config["sgld_clip"]),
                auto_tune_eps=True,
                target_update_ratio=float(config["sgld_target_ratio"]),
                inc=float(config["sgld_inc"]),
                dec=float(config["sgld_dec"]),
                eps_scale_bounds=tuple(config["sgld_scale_bounds"]),
                parallel_streams=bool(config.get("sgld_parallel_chains", True)),
                per_chain_minibatch=bool(config.get("sgld_per_chain_minibatch", True)),
            )

    elif minsel_backend in {"adamw", "sgd", "gd", "branches"}:
        # Persistent deterministic branches:
        # Each branch maintains its own optimizer/scheduler state and runs a few LL steps on g.
        # To prevent all branches collapsing to the same solution, add small init noise.
        branch_models = [deepcopy(model).to(device) for _ in range(int(extra_branches))]
        init_noise_std = float(config.get("branch_init_noise_std", config.get("sgld_init_noise_std", 0.0)) or 0.0)
        if init_noise_std > 0:
            with torch.no_grad():
                for m in branch_models:
                    for p in m.parameters():
                        p.add_(torch.randn_like(p) * init_noise_std)

        # Optional branch diversity via optimizer hyperparameters (same objective g(train), different dynamics).
        # This often makes minima selection much more "visible" in accuracy, because candidates explore
        # meaningfully different basins / implicit regularization regimes.
        #
        # Safe defaults keep all multipliers at 1.0 (no change). To enable:
        #   branch_lr_mult_min < 1 < branch_lr_mult_max  (log-spaced across branches)
        #   branch_wd_mult_min < 1 < branch_wd_mult_max
        def _logspace(min_v: float, max_v: float, n: int) -> list[float]:
            n = int(n)
            if n <= 0:
                return []
            if n == 1:
                return [float(min_v)]
            if min_v <= 0 or max_v <= 0:
                return [1.0 for _ in range(n)]
            a, b = math.log10(float(min_v)), math.log10(float(max_v))
            return [10 ** (a + (b - a) * (i / (n - 1))) for i in range(n)]

        lr_mult_min = float(config.get("branch_lr_mult_min", 1.0) or 1.0)
        lr_mult_max = float(config.get("branch_lr_mult_max", 1.0) or 1.0)
        wd_mult_min = float(config.get("branch_wd_mult_min", 1.0) or 1.0)
        wd_mult_max = float(config.get("branch_wd_mult_max", 1.0) or 1.0)

        lr_mults = _logspace(lr_mult_min, lr_mult_max, int(extra_branches))
        wd_mults = _logspace(wd_mult_min, wd_mult_max, int(extra_branches))

        base_y_lr = float(config.get("y_lr", 1e-3))
        base_y_wd = float(config.get("y_wd", 0.0))
        branch_cfgs = []
        branch_trainers = []
        for i, m in enumerate(branch_models):
            cfg_i = dict(config)
            if i < len(lr_mults):
                cfg_i["y_lr"] = float(base_y_lr) * float(lr_mults[i])
            if i < len(wd_mults):
                cfg_i["y_wd"] = float(base_y_wd) * float(wd_mults[i])
            branch_cfgs.append(cfg_i)
            branch_trainers.append(default_trainer_from_config(m, cfg_i))

        # Optional one-time burn-in for branches (reuses `sgld_burn_in` to avoid adding new knobs).
        burn = int(config.get("branch_burn_in", config.get("sgld_burn_in", 0)) or 0)
        if burn > 0:
            for m, tr in zip(branch_models, branch_trainers):
                tr.step_k(
                    m,
                    x_raw,
                    train_loader,
                    steps=burn,
                    grad_clip=float(config["y_clip"]),
                    accum=int(config.get("y_grad_accum", 1)),
                )
        # Store EMA buffer in a tiny state dict for consistency with the pSGLD path.
        chains_state = {"f_ema": minsel_f_ema}
    else:
        raise ValueError("minsel_backend must be one of: psgld, adamw (or aliases).")

    # Selection diagnostics (stored at eval points)
    last_picked_idx = 0
    last_base_score = float("nan")
    last_best_score = float("nan")
    last_raw_best_idx = 0
    last_picked_is_raw_best = True
    last_picked_score = float("nan")
    last_rank_best_idx = 0
    last_rank_best_score = float("nan")
    last_eps_scale = float("nan")

    selection_print = bool(config.get("selection_print", True))
    selection_print_every = max(1, int(config.get("selection_print_every", int(config.get("eval_every", 5)))))

    for t in range(iters_total):
        if time_budget_s is not None and (time.perf_counter() - start_t) >= time_budget_s:
            break
        tb = next(train_it, None)
        vb = next(val_it, None)
        if tb is None:
            train_it = iter(train_loader)
            tb = next(train_it)
        if vb is None:
            val_it = iter(val_loader)
            vb = next(val_it)

        step_start = time.perf_counter()

        if t < int(config.get("freeze_x_outer", 0)):
            trainer.step_k(
                model,
                x_raw,
                train_loader,
                steps=int(config.get("y_steps_warm", 200)),
                grad_clip=float(config["y_clip"]),
                accum=int(config.get("y_grad_accum", 1)),
            )
            alpha_used = float("nan")
        else:
            # (Minima selection) advance candidates on g(train) every outer step, but only *evaluate/switch*
            # every `sample_every` outer iterations (reduces time-budget overhead while keeping branches "fresh").
            sample_every = max(1, int(config.get("sample_every", 1) or 1))
            do_select = ((t + 1) % int(sample_every)) == 0

            # Candidate 0 is always the current `model`, followed by extra chains/branches.
            # Avoid deepcopy here: selection only needs forward passes.
            if minsel_backend in {"psgld", "sgld", "lmc"}:
                assert chains_state is not None
                chains_state = psgld_advance_state(
                    chains_state,
                    x_raw=x_raw,
                    train_loader=train_loader,
                    steps=int(config["sgld_steps"]),
                    eps0=float(config["sgld_eps0"]),
                    alpha=float(config["sgld_alpha"]),
                    temperature=float(config.get("sgld_temperature", 0.0)),
                    weight_decay=float(config["sgld_wd"]),
                    rms_rho=float(config["sgld_rho"]),
                    rms_eps=float(config["sgld_eps"]),
                    grad_clip=float(config["sgld_clip"]),
                    auto_tune_eps=True,
                    target_update_ratio=float(config["sgld_target_ratio"]),
                    inc=float(config["sgld_inc"]),
                    dec=float(config["sgld_dec"]),
                    eps_scale_bounds=tuple(config["sgld_scale_bounds"]),
                    parallel_streams=bool(config.get("sgld_parallel_chains", True)),
                    per_chain_minibatch=bool(config.get("sgld_per_chain_minibatch", True)),
                )
                candidates = [model] + list(chains_state["chains_models"])
            else:
                assert branch_models is not None and branch_trainers is not None and chains_state is not None
                branch_steps = int(config.get("branch_steps", config.get("sgld_steps", 20)) or 0)
                branch_steps = max(0, branch_steps)
                if do_select and bool(config.get("branch_restart_each_sample", False)):
                    assert branch_cfgs is not None
                    restart_noise = float(config.get("branch_restart_noise_std", config.get("branch_init_noise_std", 0.0)) or 0.0)
                    base_state = model.state_dict()
                    for i, m in enumerate(branch_models):
                        m.load_state_dict(base_state)
                        if restart_noise and restart_noise > 0:
                            with torch.no_grad():
                                for p in m.parameters():
                                    p.add_(torch.randn_like(p) * float(restart_noise))
                        branch_trainers[i] = default_trainer_from_config(m, branch_cfgs[i])
                if branch_steps > 0:
                    # If branches are restarted from the base at selection steps, advancing them on
                    # non-selection iterations is wasted work under a wall-clock budget.
                    # In that mode, only step branches when we are about to score/select candidates.
                    restart_each = bool(config.get("branch_restart_each_sample", False))
                    if (not restart_each) or bool(do_select):
                        for m, tr in zip(branch_models, branch_trainers):
                            tr.step_k(
                                m,
                                x_raw,
                                train_loader,
                                steps=branch_steps,
                                grad_clip=float(config["y_clip"]),
                                accum=int(config.get("y_grad_accum", 1)),
                            )
                candidates = [model] + list(branch_models)

            if do_select:
                ema_scores = None
                if bool(config.get("selection_use_ema", True)):
                    if not isinstance(chains_state.get("f_ema", None), list) or len(chains_state.get("f_ema", [])) != len(candidates):
                        chains_state["f_ema"] = [None for _ in range(len(candidates))]
                    ema_scores = chains_state["f_ema"]
                selection_metric = str(config.get("selection_metric", "f") or "f").strip().lower()
                metric_select_mode = "max" if selection_metric in {"acc", "bal_acc", "balanced_acc"} else str(config["select_mode"])
                selector = (
                    select_by_bal_acc_conservative
                    if selection_metric in {"acc", "bal_acc", "balanced_acc"}
                    else select_by_f_conservative
                )
                # Selection can dominate runtime under wall-clock budgets. If the validation loader is
                # configured as a full-batch loader (val_full_batch=True in the experiment runner),
                # use it directly for candidate scoring (deterministic and cheap).
                selection_loader = val_loader if _is_full_batch_loader(val_loader) else val_eval_loader
                picked, scores, picked_idx = selector(
                    candidates,
                    x_raw,
                    selection_loader,
                    u_vec,
                    select_mode=str(metric_select_mode),
                    eval_batches=int(config.get("selection_eval_batches", config.get("sgld_eval_batches", 0))),
                    min_improve_over_base=float(config.get("selection_min_improve", 0.0) or 0.0),
                    use_significance=bool(config.get("selection_use_significance", True)),
                    z_value=float(config.get("selection_z", 1.96) or 1.96),
                    ema_scores=ema_scores,
                    ema_beta=float(config.get("selection_ema", 0.9) or 0.9),
                    rank_by_ema=bool(config.get("selection_rank_by_ema", True)),
                )

                # If we are ranking by EMA, it can take a while for a genuinely better branch to "win" the EMA.
                # To avoid missing clear improvements, optionally override the EMA choice when the raw best
                # candidate beats the base by a large margin on the current f(val).
                select_mode = str(metric_select_mode).strip().lower()
                ema_override = float(config.get("selection_ema_override_improve", 0.0) or 0.0)
                if scores and ema_override > 0 and bool(config.get("selection_rank_by_ema", True)):
                    base_score = float(scores[0])
                    raw_best_idx = int(
                        min(range(len(scores)), key=lambda i: float(scores[i]))
                        if select_mode == "min"
                        else max(range(len(scores)), key=lambda i: float(scores[i]))
                    )
                    raw_best_score = float(scores[raw_best_idx])
                    raw_improve = (base_score - raw_best_score) if select_mode == "min" else (raw_best_score - base_score)
                    if raw_best_idx != int(picked_idx) and raw_best_idx != 0 and raw_improve >= ema_override:
                        picked_idx = int(raw_best_idx)
                        picked = candidates[int(raw_best_idx)]

                last_picked_idx = int(picked_idx)
                if scores:
                    last_base_score = float(scores[0])
                    if select_mode == "min":
                        last_raw_best_idx = int(min(range(len(scores)), key=lambda i: float(scores[i])))
                        last_best_score = float(min(scores))
                    else:
                        last_raw_best_idx = int(max(range(len(scores)), key=lambda i: float(scores[i])))
                        last_best_score = float(max(scores))
                    last_picked_score = float(scores[int(picked_idx)])
                    last_picked_is_raw_best = bool(int(picked_idx) == int(last_raw_best_idx))

                    # If we are ranking by EMA, report the EMA-best index too.
                    if bool(config.get("selection_rank_by_ema", True)) and ema_scores is not None:
                        rank_scores = [
                            float(e) if e is not None else float(s) for e, s in zip(ema_scores, scores)
                        ]
                        if select_mode == "min":
                            last_rank_best_idx = int(min(range(len(rank_scores)), key=lambda i: float(rank_scores[i])))
                        else:
                            last_rank_best_idx = int(max(range(len(rank_scores)), key=lambda i: float(rank_scores[i])))
                        last_rank_best_score = float(rank_scores[int(last_rank_best_idx)])
                    else:
                        last_rank_best_idx = int(last_raw_best_idx)
                        last_rank_best_score = float(last_best_score)
                last_eps_scale = float(chains_state.get("eps_scale", float("nan")))

                if selection_print and (
                    (t % selection_print_every == 0)
                    or (int(last_picked_idx) != 0)
                    or (not bool(last_picked_is_raw_best))
                ):
                    tag = "chain" if minsel_backend in {"psgld", "sgld", "lmc"} else "branch"
                    branch = "base" if int(last_picked_idx) == 0 else f"{tag}{int(last_picked_idx) - 1}"
                    print(
                        f"[HG_acc_sel] select | iter {t:4d} | pick={branch} idx={int(last_picked_idx)} "
                        f"f_pick={float(last_picked_score):.4f} | best_idx={int(last_raw_best_idx)} "
                        f"best_f={float(last_best_score):.4f} | base_f={float(last_base_score):.4f} | "
                        f"pick_is_best={bool(last_picked_is_raw_best)} | "
                        f"rank_best_idx={int(last_rank_best_idx)} rank_best={float(last_rank_best_score):.4f} | "
                                f"eps_scale={float(last_eps_scale):.3f}"
                    )

                # Only switch the running model if a non-base candidate was accepted.
                if int(picked_idx) != 0:
                    if minsel_backend in {"psgld", "sgld", "lmc"}:
                        # pSGLD chains have no optimizer state we can swap; copy weights into the running model.
                        with torch.no_grad():
                            for p_dst, p_src in zip(model.parameters(), picked.parameters()):
                                p_dst.copy_(p_src)
                    else:
                        # Deterministic branches can optionally "swap state" (model+optimizer) or only accept weights.
                        assert branch_models is not None and branch_trainers is not None
                        assert branch_cfgs is not None
                        bi = int(picked_idx) - 1
                        if 0 <= bi < len(branch_models):
                            # Keep the EMA buffer aligned with the "accepted" candidate: after acceptance,
                            # candidate 0 should carry the accepted candidate's EMA state.
                            if isinstance(ema_scores, list) and 0 <= int(picked_idx) < len(ema_scores):
                                ema_scores[0], ema_scores[int(picked_idx)] = (ema_scores[int(picked_idx)], ema_scores[0])

                            accept_mode = str(config.get("branch_accept_mode", "swap") or "swap").strip().lower()
                            if accept_mode in {"swap", "swap_state"}:
                                # Full state swap: keep optimizer/scheduler state from the chosen branch.
                                branch_models[bi], model = model, branch_models[bi]
                                branch_trainers[bi], trainer = trainer, branch_trainers[bi]
                            elif accept_mode in {"copy", "copy_weights"}:
                                # Accept only weights into the running model; keep the running optimizer/hparams.
                                model.load_state_dict(branch_models[bi].state_dict())
                            elif accept_mode in {"swap_weights"}:
                                # Swap weights between the running model and the chosen branch, but keep the
                                # running optimizer/hparams (more stable under time budgets).
                                base_state = deepcopy(model.state_dict())
                                model.load_state_dict(branch_models[bi].state_dict())
                                branch_models[bi].load_state_dict(base_state)
                                # Optimizer state no longer matches after a weight swap; reinit the branch trainer.
                                branch_trainers[bi] = default_trainer_from_config(branch_models[bi], branch_cfgs[bi])
                            else:
                                raise ValueError("branch_accept_mode must be one of: swap, copy_weights, swap_weights")
                        else:
                            raise IndexError("picked_idx out of range for branch_models")

                    # Optional "polish" to move the selected candidate closer to a g-minimizer before hypergrad.
                    polish_steps = int(config.get("sel_polish_y_steps", 0) or 0)
                    if polish_steps > 0:
                        trainer.step_k(
                            model,
                            x_raw,
                            train_loader,
                            steps=polish_steps,
                            grad_clip=float(config["y_clip"]),
                            accum=int(config.get("y_grad_accum", 1)),
                        )

                    if minsel_backend in {"psgld", "sgld", "lmc"}:
                        # Elitism: if we accepted a chain candidate, make it chain 0 and preserve its RMS state.
                        assert chains_state is not None
                        if bool(config.get("elitism_chain", True)) and chains_state["chains_models"]:
                            chain_i = int(picked_idx) - 1  # candidates[1:] corresponds to chains_models[0:]
                            if chain_i >= 0 and chain_i < len(chains_state["chains_models"]):
                                if chain_i != 0:
                                    # Keep EMA buffer aligned with chain ordering after elitism swap.
                                    f_ema = chains_state.get("f_ema", None)
                                    if isinstance(f_ema, list) and len(f_ema) == (1 + len(chains_state["chains_models"])):
                                        a, b = 1, int(chain_i) + 1
                                        f_ema[a], f_ema[b] = f_ema[b], f_ema[a]
                                    chains_state["chains_models"][0], chains_state["chains_models"][chain_i] = (
                                        chains_state["chains_models"][chain_i],
                                        chains_state["chains_models"][0],
                                    )
                                    chains_state["sq_avg"][0], chains_state["sq_avg"][chain_i] = (
                                        chains_state["sq_avg"][chain_i],
                                        chains_state["sq_avg"][0],
                                    )
                                # Sync elite chain weights to the (possibly polished) running model.
                                chains_state["chains_models"][0].load_state_dict(model.state_dict())

            d_delta, d_gamma = implicit_hypergrad_x(
                model,
                x_raw,
                tb,
                vb,
                u_vec,
                cg_steps=int(config["cg_steps"]),
                damping=float(config["cg_damping"]),
                use_amp=bool(config.get("use_amp", False)),
                hg_train_cap=config.get("hg_train_cap", None),
                hg_val_cap=config.get("hg_val_cap", None),
            )

            if use_adam:
                alpha_used = float(outer.step(d_delta, d_gamma))
            else:
                alpha_used = float(outer.step(delta_raw, gamma_raw, d_delta, d_gamma))
            x_raw = (delta_raw, gamma_raw)

            trainer.step_k(
                model,
                x_raw,
                train_loader,
                steps=int(config["y_steps_per_outer"]),
                grad_clip=float(config["y_clip"]),
                accum=int(config.get("y_grad_accum", 1)),
            )

        logs["iteration_times"].append(time.perf_counter() - step_start)

        if (t % int(config["eval_every"]) == 0) or (t == (iters_total - 1)):
            tr_acc_curr = balanced_acc_no_warn(model, train_eval_loader, x_raw)
            te_acc_curr = balanced_acc_no_warn(model, test_loader, x_raw)
            tr_g_curr = eval_avg_g_loss(model, train_eval_loader, x_raw)
            va_f_curr = eval_avg_f_loss(model, val_eval_loader, u_vec, x_raw)

            improved = bool(va_f_curr < (float(best["f"]) - float(early_stop_min_delta)))
            if improved:
                best["f"] = float(va_f_curr)
                best["t"] = int(t + 1)
                best["x"] = (delta_raw.detach().cpu().clone(), gamma_raw.detach().cpu().clone())
                best["state"] = deepcopy(model.state_dict())
                best["metrics"] = dict(
                    bal_train_acc=float(tr_acc_curr),
                    bal_test_acc=float(te_acc_curr),
                    train_loss=float(tr_g_curr),
                )
                no_improve_evals = 0
            else:
                no_improve_evals += 1

            # Report either the current point or the best-so-far-by-f(val) checkpoint.
            if report_best_at_eval and best.get("metrics") is not None:
                tr_acc = float(best["metrics"]["bal_train_acc"])
                te_acc = float(best["metrics"]["bal_test_acc"])
                tr_g = float(best["metrics"]["train_loss"])
                va_f = float(best["f"])
                best_iter = int(best.get("t") or (t + 1))
            else:
                tr_acc = float(tr_acc_curr)
                te_acc = float(te_acc_curr)
                tr_g = float(tr_g_curr)
                va_f = float(va_f_curr)
                best_iter = int(best.get("t") or (t + 1))

            logs["iterations"].append(t + 1)
            logs["elapsed_s"].append(float(time.perf_counter() - start_t))
            logs["bal_train_acc"].append(tr_acc)
            logs["bal_test_acc"].append(te_acc)
            logs["train_loss"].append(tr_g)
            logs["val_loss"].append(va_f)
            logs["bal_train_acc_curr"].append(float(tr_acc_curr))
            logs["bal_test_acc_curr"].append(float(te_acc_curr))
            logs["train_loss_curr"].append(float(tr_g_curr))
            logs["val_loss_curr"].append(float(va_f_curr))
            logs["best_iter"].append(int(best_iter))
            logs["x_alpha"].append(alpha_used)
            # Selection diagnostics at eval points
            logs.setdefault("picked_idx", []).append(int(last_picked_idx))
            logs.setdefault("picked_base_f", []).append(float(last_base_score))
            logs.setdefault("picked_best_f", []).append(float(last_best_score))
            logs.setdefault("picked_score", []).append(float(last_picked_score))
            logs.setdefault("raw_best_idx", []).append(int(last_raw_best_idx))
            logs.setdefault("picked_is_best", []).append(bool(last_picked_is_raw_best))
            logs.setdefault("rank_best_idx", []).append(int(last_rank_best_idx))
            logs.setdefault("rank_best_score", []).append(float(last_rank_best_score))
            logs.setdefault("sgld_eps_scale", []).append(float(last_eps_scale))

            best_tag = f"best@{best_iter}" if report_best_at_eval else "curr"
            print(
                f"HG_acc_sel({config['select_mode']}) | Iter {t:4d} | {best_tag} | Train BalAcc {tr_acc:.4f} | "
                f"Test BalAcc {te_acc:.4f} | g(train) {tr_g:.4f} | f(val) {va_f:.4f} | "
                f"pick_idx {int(last_picked_idx)} best_idx {int(last_raw_best_idx)} pick_is_best {bool(last_picked_is_raw_best)} | "
                f"α/η {alpha_used}"
            )

            if early_stop_patience > 0 and no_improve_evals >= early_stop_patience:
                print(
                    f"[HG_acc_sel] early_stop: no f(val) improvement for {no_improve_evals} evals "
                    f"(patience={early_stop_patience})."
                )
                break

    maybe_restore_best(best, model=model, delta_raw=delta_raw, gamma_raw=gamma_raw)

    logs["method"] = f"HG_acc_sel({config['select_mode']})"
    logs["config"] = dict(config)
    return logs


__all__ = ["prepare_data", "run_hg_acc_sel"]
