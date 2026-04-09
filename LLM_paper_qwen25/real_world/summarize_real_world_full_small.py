from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--math-root", required=True)
    parser.add_argument("--instruction-root", required=True)
    return parser.parse_args()


def max_deviation_from_uniform(weights: list[float] | None) -> float | None:
    if not weights:
        return None
    uniform = 1.0 / len(weights)
    return max(abs(weight - uniform) for weight in weights)


def collect(root: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        with open(metrics_path) as handle:
            metrics = json.load(handle)
        metrics["run"] = metrics_path.parent.name
        metrics["max_lambda_dev_from_uniform"] = max_deviation_from_uniform(metrics.get("learned_lambda"))
        rows.append(metrics)
    return rows


def print_rows(task: str, rows: list[dict]) -> None:
    print(task)
    if task == "math":
        print("method\tbest_reweight_tune\tfinal_tune\tgsm8k\tmax_lambda_dev\trun")
        for row in rows:
            print(
                f"{row['method']}\t{row.get('best_tune_loss_reweight', float('nan')):.4f}\t"
                f"{row.get('final_ft_tune_loss', float('nan')):.4f}\t"
                f"{row.get('gsm8k_accuracy', float('nan')):.4f}\t"
                f"{row.get('max_lambda_dev_from_uniform', float('nan')):.6f}\t{row['run']}"
            )
    else:
        print("method\tbest_reweight_tune\tfinal_tune\ttest_loss\tmax_lambda_dev\trun")
        for row in rows:
            print(
                f"{row['method']}\t{row.get('best_tune_loss_reweight', float('nan')):.4f}\t"
                f"{row.get('final_ft_tune_loss', float('nan')):.4f}\t"
                f"{row.get('test_loss', float('nan')):.4f}\t"
                f"{row.get('max_lambda_dev_from_uniform', float('nan')):.6f}\t{row['run']}"
            )
    print()


def main() -> None:
    args = parse_args()
    math_rows = collect(Path(args.math_root))
    instruction_rows = collect(Path(args.instruction_root))
    print_rows("math", math_rows)
    print_rows("instruction", instruction_rows)
    print(
        json.dumps(
            {
                "math_root": args.math_root,
                "instruction_root": args.instruction_root,
                "math_rows": math_rows,
                "instruction_rows": instruction_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
