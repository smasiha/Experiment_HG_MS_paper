from __future__ import annotations

import argparse
import json
from pathlib import Path


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
                "math_accuracy": metrics.get("math_accuracy"),
                "best_tune_loss_reweight": metrics.get("best_tune_loss_reweight"),
                "final_ft_tune_loss": metrics.get("final_ft_tune_loss"),
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
