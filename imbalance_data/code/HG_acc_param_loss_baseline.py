import time
from copy import deepcopy

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
    print_device_banner,
    prepare_data,
    resolve_outer_iters,
    set_seed,
    FollowerCNN,
)


def run_hg_acc_no_selection(config, common):
    (train_loader, val_loader, test_loader, train_eval_loader, val_eval_loader, u_vec) = common
    config = dict(config)
    set_seed(int(config["seed"]))
    print_device_banner("HG_acc_no_sel")
    # Plain baseline: by default do NOT restore the best checkpoint (this can make the baseline look
    # artificially strong by reporting a best-of-trajectory model rather than the final iterate).
    restore_best = bool(config.get("restore_best", False))

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
    logs = {
        k: []
        for k in [
            "iterations",
            "elapsed_s",
            "bal_train_acc",
            "bal_test_acc",
            "train_loss",
            "val_loss",
            "x_alpha",
            "iteration_times",
        ]
    }

    best = {"f": float("inf"), "x": None, "state": None} if restore_best else None

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
            tr_acc = balanced_acc_no_warn(model, train_eval_loader, x_raw)
            te_acc = balanced_acc_no_warn(model, test_loader, x_raw)
            tr_g = eval_avg_g_loss(model, train_eval_loader, x_raw)
            va_f = eval_avg_f_loss(model, val_eval_loader, u_vec, x_raw)

            if restore_best and best is not None and va_f < best["f"]:
                best["f"] = va_f
                best["x"] = (delta_raw.detach().cpu().clone(), gamma_raw.detach().cpu().clone())
                best["state"] = deepcopy(model.state_dict())

            logs["iterations"].append(t + 1)
            logs["elapsed_s"].append(float(time.perf_counter() - start_t))
            logs["bal_train_acc"].append(tr_acc)
            logs["bal_test_acc"].append(te_acc)
            logs["train_loss"].append(tr_g)
            logs["val_loss"].append(va_f)
            logs["x_alpha"].append(alpha_used)

            print(
                f"HG_acc_no_sel | Iter {t:4d} | Train BalAcc {tr_acc:.4f} | Test BalAcc {te_acc:.4f} | "
                f"g(train) {tr_g:.4f} | f(val) {va_f:.4f} | α/η {alpha_used}"
            )

    if restore_best and best is not None:
        maybe_restore_best(best, model=model, delta_raw=delta_raw, gamma_raw=gamma_raw)

    logs["method"] = "HG_acc_no_sel"
    logs["config"] = dict(config)
    return logs


__all__ = ["prepare_data", "run_hg_acc_no_selection"]
