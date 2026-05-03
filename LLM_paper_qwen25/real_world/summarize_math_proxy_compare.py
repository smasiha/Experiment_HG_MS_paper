#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def max_dev_from_uniform(lambdas: list[float]) -> float:
    if not lambdas:
        return float("nan")
    uniform = 1.0 / len(lambdas)
    return max(abs(value - uniform) for value in lambdas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(Path(args.root).glob("*/metrics.json")):
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append(
            {
                "run": metrics_path.parent.name,
                "method": metrics.get("method"),
                "gsm8k_accuracy": metrics.get("gsm8k_accuracy"),
                "final_ft_tune_loss": metrics.get("final_ft_tune_loss"),
                "best_tune_loss_reweight": metrics.get("best_tune_loss_reweight"),
                "max_dev": max_dev_from_uniform(metrics.get("learned_lambda", [])),
            }
        )

    rows.sort(
        key=lambda row: (
            -(row["gsm8k_accuracy"] if row["gsm8k_accuracy"] is not None else float("-inf")),
            row["final_ft_tune_loss"] if row["final_ft_tune_loss"] is not None else float("inf"),
        )
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
