from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in sorted(Path(args.root).rglob("metrics.json")):
        with open(metrics_path) as handle:
            metrics = json.load(handle)
        run_dir = metrics_path.parent
        tag = run_dir.parent.parent.name if run_dir.parent.name in {"denoising", "multilingual"} else run_dir.parent.name
        rows.append(
            {
                "tag": tag,
                "task": metrics.get("task"),
                "method": metrics.get("method"),
                "best_val_loss": metrics.get("best_val_loss"),
                "test_loss": metrics.get("test_loss"),
                "learned_lambda": metrics.get("learned_lambda"),
                "target_error_l1": metrics.get("target_error_l1"),
                "path": str(run_dir),
            }
        )

    rows.sort(key=lambda row: (row["task"] or "", row["target_error_l1"] if row["target_error_l1"] is not None else 1e9, row["best_val_loss"] if row["best_val_loss"] is not None else 1e9))
    print("tag\ttask\tmethod\tbest_val_loss\ttest_loss\ttarget_error_l1\tlearned_lambda\tpath")
    for row in rows:
        print(
            f"{row['tag']}\t{row['task']}\t{row['method']}\t"
            f"{row['best_val_loss']:.4f}\t{row['test_loss']:.4f}\t{row['target_error_l1']:.4f}\t"
            f"{row['learned_lambda']}\t{row['path']}"
        )


if __name__ == "__main__":
    main()
