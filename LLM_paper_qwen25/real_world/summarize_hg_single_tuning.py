#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_metrics(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Tuning output root containing run subdirs")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for metrics_path in root.rglob("metrics.json"):
        data = load_metrics(metrics_path)
        if not data:
            continue
        if data.get("method") != "hg_single":
            continue
        best_tune = data.get("best_tune_loss_reweight")
        if best_tune is None:
            continue
        rows.append(
            {
                "best_tune_loss_reweight": best_tune,
                "run_dir": metrics_path.parent,
                "run_name": metrics_path.parent.name,
                "reweight_steps_completed": data.get("reweight_steps_completed"),
                "reweight_elapsed_sec": data.get("reweight_elapsed_sec"),
            }
        )

    rows = [r for r in rows if isinstance(r["best_tune_loss_reweight"], (int, float))]
    rows.sort(key=lambda r: r["best_tune_loss_reweight"])

    if not rows:
        print("No completed hg_single metrics found.")
        return

    print(f"Found {len(rows)} hg_single runs. Top {min(args.top, len(rows))} by best_tune_loss_reweight:\n")
    for idx, row in enumerate(rows[: args.top], start=1):
        print(
            f"{idx:02d}. loss={row['best_tune_loss_reweight']:.6f} steps={row['reweight_steps_completed']} "
            f"elapsed={row['reweight_elapsed_sec']:.1f}s run={row['run_name']}"
        )


if __name__ == "__main__":
    main()
