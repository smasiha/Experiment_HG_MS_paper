from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for path in sorted(Path(args.root).rglob("metrics.json")):
        with open(path) as f:
            payload = json.load(f)
        payload["path"] = str(path.parent)
        rows.append(payload)

    if not rows:
        print("No metrics.json files found.")
        return

    rows.sort(key=lambda row: (row["method"], row["seed"]))
    print("method\tseed\tbest_val_loss\tbest_val_test_accuracy\tbest_test_accuracy\telapsed_min\tpath")
    for row in rows:
        print(
            f"{row['method']}\t{row['seed']}\t{row['best_val_loss']:.4f}\t"
            f"{row.get('best_val_test_accuracy', float('nan')):.4f}\t"
            f"{row['best_test_accuracy']:.4f}\t{row['elapsed_min']:.2f}\t{row['path']}"
        )

    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    print("\nAggregated")
    print("method\tn\tmean_best_val\tstd_best_val\tmean_best_val_test\tstd_best_val_test\tmean_best_test\tstd_best_test")
    for method in sorted(by_method):
        group = by_method[method]
        best_vals = [row["best_val_loss"] for row in group]
        best_val_tests = [row.get("best_val_test_accuracy", float("nan")) for row in group]
        best_tests = [row["best_test_accuracy"] for row in group]
        val_std = statistics.pstdev(best_vals) if len(best_vals) > 1 else 0.0
        best_val_test_std = statistics.pstdev(best_val_tests) if len(best_val_tests) > 1 else 0.0
        test_std = statistics.pstdev(best_tests) if len(best_tests) > 1 else 0.0
        print(
            f"{method}\t{len(group)}\t{statistics.mean(best_vals):.4f}\t{val_std:.4f}\t"
            f"{statistics.mean(best_val_tests):.4f}\t{best_val_test_std:.4f}\t"
            f"{statistics.mean(best_tests):.4f}\t{test_std:.4f}"
        )


if __name__ == "__main__":
    main()
