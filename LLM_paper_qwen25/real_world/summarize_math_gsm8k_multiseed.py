#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict


def mean_std(values):
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean, std


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    if pct:
        return f"{100.0 * x:.2f}"
    return f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "*", "metrics.json")))
    if not paths:
        raise SystemExit(f"No metrics found under {args.root}")

    grouped = defaultdict(list)
    for path in paths:
        with open(path) as f:
            metrics = json.load(f)
        method = metrics.get("method", "unknown")
        grouped[method].append((os.path.basename(os.path.dirname(path)), metrics))

    order = ["uniform", "less", "rho_loss", "scalebio", "hg_single", "hg_ms"]
    for method in sorted(grouped.keys(), key=lambda m: (order.index(m) if m in order else 999, m)):
        runs = grouped[method]
        model_names = sorted({m.get("model_name", "unknown") for _, m in runs})
        tune_losses = [m.get("final_ft_tune_loss") for _, m in runs if m.get("final_ft_tune_loss") is not None]
        gsm8k_accs = [m.get("gsm8k_accuracy") for _, m in runs if m.get("gsm8k_accuracy") is not None]
        reweight_secs = [m.get("reweight_elapsed_sec") for _, m in runs if m.get("reweight_elapsed_sec") is not None]
        reweight_steps = [m.get("reweight_steps_completed") for _, m in runs if m.get("reweight_steps_completed") is not None]

        tune_mean, tune_std = mean_std(tune_losses)
        acc_mean, acc_std = mean_std(gsm8k_accs)
        sec_mean, sec_std = mean_std(reweight_secs)
        step_mean, step_std = mean_std(reweight_steps)

        print(f"{method}")
        print(f"  model_names         = {', '.join(model_names)}")
        print(f"  n                   = {len(runs)}")
        print(f"  final_ft_tune_loss  = {fmt(tune_mean)} ± {fmt(tune_std)}")
        print(f"  gsm8k_accuracy      = {fmt(acc_mean, pct=True)} ± {fmt(acc_std, pct=True)}")
        print(f"  reweight_elapsed_s  = {fmt(sec_mean)} ± {fmt(sec_std)}")
        print(f"  reweight_steps      = {fmt(step_mean)} ± {fmt(step_std)}")
        for run_name, metrics in sorted(runs):
            print(
                f"    seed={metrics.get('seed')} run={run_name} "
                f"tune={fmt(metrics.get('final_ft_tune_loss'))} "
                f"acc={fmt(metrics.get('gsm8k_accuracy'), pct=True)}"
            )
        print()


if __name__ == "__main__":
    main()
