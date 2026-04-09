from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows
    for metrics_path in sorted(root.glob("*/metrics.json")):
        with metrics_path.open() as f:
            payload = json.load(f)
        payload["path"] = str(metrics_path.parent)
        rows.append(payload)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.root))
    if not rows:
        print("No metrics found.")
        return

    metric_keys = [
        "best_tune_loss_reweight",
        "final_ft_tune_loss",
        "gsm8k_accuracy",
        "test_loss",
        "reweight_elapsed_sec",
        "total_elapsed_sec",
        "reweight_steps_completed",
        "reweight_budget_seconds",
        "reweight_budget_hit",
    ]
    header = ["method", "path"] + metric_keys
    print("\t".join(header))
    for row in rows:
        values = [str(row.get("method", "")), row["path"]]
        for key in metric_keys:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        print("\t".join(values))


if __name__ == "__main__":
    main()
