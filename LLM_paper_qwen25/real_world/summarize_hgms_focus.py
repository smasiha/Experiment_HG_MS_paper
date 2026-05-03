from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_last_history_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_row: dict[str, Any] | None = None
    with path.open() as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            last_row = json.loads(line)
    return last_row


def lambda_entropy(lambda_values: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in lambda_values)


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())
    history_last = load_last_history_row(run_dir / "reweight_history.jsonl")
    learned_lambda = metrics.get("learned_lambda") or []
    uniform = 1.0 / len(learned_lambda) if learned_lambda else 0.0
    return {
        "run_dir": str(run_dir),
        "profile": run_dir.parents[1].name,
        "model": run_dir.parents[0].name,
        "run_name": run_dir.name,
        "test_loss": metrics.get("test_loss"),
        "final_ft_tune_loss": metrics.get("final_ft_tune_loss"),
        "best_tune_loss_reweight": metrics.get("best_tune_loss_reweight"),
        "reweight_steps_completed": metrics.get("reweight_steps_completed"),
        "reweight_elapsed_sec": metrics.get("reweight_elapsed_sec"),
        "mtbench": (((metrics.get("mtbench") or {}).get("judging") or {}).get("mean_question_score")),
        "lambda_max_dev_from_uniform": (
            max(abs(value - uniform) for value in learned_lambda)
            if learned_lambda
            else None
        ),
        "lambda_entropy": lambda_entropy(learned_lambda) if learned_lambda else None,
        "top_sources": sorted(
            zip(
                learned_lambda,
                metrics.get("source_names", []),
                metrics.get("source_counts", []),
            ),
            reverse=True,
        )[:5],
        "last_theta_grad_norm": history_last.get("theta_grad_norm") if history_last else None,
        "last_selected_branch": ((history_last or {}).get("selected") or {}).get("branch_index"),
    }


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("**/metrics.json")):
        row = summarize_run(metrics_path.parent)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"No completed runs found under {runs_root}")
        return

    rows.sort(key=lambda row: (row["model"], row["profile"], row["test_loss"] if row["test_loss"] is not None else 1e9))
    print(
        "profile\tmodel\ttest_loss\tft_tune\treweight_best\tsteps\telapsed_sec\tlambda_dev\tentropy\ttheta_grad"
    )
    for row in rows:
        print(
            "\t".join(
                [
                    row["profile"],
                    row["model"],
                    format_value(row["test_loss"]),
                    format_value(row["final_ft_tune_loss"]),
                    format_value(row["best_tune_loss_reweight"]),
                    format_value(row["reweight_steps_completed"]),
                    format_value(row["reweight_elapsed_sec"]),
                    format_value(row["lambda_max_dev_from_uniform"]),
                    format_value(row["lambda_entropy"]),
                    format_value(row["last_theta_grad_norm"]),
                ]
            )
        )
        for weight, name, count in row["top_sources"]:
            print(f"  top_source\t{weight:.4f}\t{count}\t{name}")


if __name__ == "__main__":
    main()
