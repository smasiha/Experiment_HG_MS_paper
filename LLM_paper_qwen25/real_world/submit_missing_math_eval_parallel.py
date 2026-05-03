#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


DEFAULT_MODELS = [
    {
        "tag": "llama",
        "preset": "llama3_8b",
        "time_limit": "08:00:00",
        "runs_root": "/scratch/masiha/math_gsm8k_llama3_8b_final_multiseed_20260410_172321/math",
    },
    {
        "tag": "qwen",
        "preset": "qwen2_7b",
        "time_limit": "12:00:00",
        "runs_root": "/scratch/masiha/math_gsm8k_qwen2_7b_final_multiseed_20260410_172419/math",
    },
    {
        "tag": "gemma",
        "preset": "gemma2_9b",
        "time_limit": "20:00:00",
        "runs_root": "/scratch/masiha/math_gsm8k_gemma2_9b_final_multiseed_20260410_172435/math",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit one Slurm MATH-eval job per unfinished run. "
            "This keeps the evaluation protocol fixed to the full Hendrycks MATH test set "
            "(math_eval_size=0) while avoiding the serial 24h bottleneck."
        )
    )
    parser.add_argument(
        "--only-tag",
        action="append",
        choices=[item["tag"] for item in DEFAULT_MODELS],
        help="Restrict submission to one or more model tags.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def has_finite_math_accuracy(metrics_path: Path) -> bool:
    if not metrics_path.exists():
        return False
    try:
        payload = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return False
    value = payload.get("math_accuracy")
    return isinstance(value, (int, float)) and math.isfinite(value)


def iter_pending_runs(runs_root: Path):
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        state_path = run_dir / "final_ft_best_adapter_state.pt"
        if not config_path.exists() or not metrics_path.exists() or not state_path.exists():
            continue
        if has_finite_math_accuracy(metrics_path):
            continue
        config = json.loads(config_path.read_text())
        yield run_dir, config


def job_name(tag: str, method: str, seed: int) -> str:
    compact_method = (
        method.replace("hg_single", "hgs")
        .replace("hg_ms", "hgm")
        .replace("rho_loss", "rho")
        .replace("scalebio", "sbo")
        .replace("uniform", "uni")
        .replace("less", "les")
    )
    return f"math_{tag}_{compact_method}_{seed}"


def main() -> int:
    args = parse_args()
    selected = {item["tag"] for item in DEFAULT_MODELS} if not args.only_tag else set(args.only_tag)
    sbatch_script = Path(__file__).with_name("run_saved_math_eval.sbatch")

    submitted = []
    for model_cfg in DEFAULT_MODELS:
        if model_cfg["tag"] not in selected:
            continue
        runs_root = Path(model_cfg["runs_root"])
        for run_dir, config in iter_pending_runs(runs_root):
            method = str(config.get("method"))
            seed = int(config.get("seed"))
            name = job_name(model_cfg["tag"], method, seed)
            export = (
                "ALL,"
                f"MODEL_PRESET={model_cfg['preset']},"
                f"RUN_DIR={run_dir},"
                "MATH_EVAL_SIZE=0,"
                "FORCE_EVAL=0"
            )
            cmd = [
                "sbatch",
                "--parsable",
                "--job-name",
                name,
                "--time",
                model_cfg["time_limit"],
                "--export",
                export,
                str(sbatch_script),
            ]
            if args.dry_run:
                print(json.dumps({"dry_run": cmd}))
                continue
            job_id = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
            submitted.append(
                {
                    "job_id": job_id,
                    "job_name": name,
                    "time_limit": model_cfg["time_limit"],
                    "run_dir": str(run_dir),
                    "method": method,
                    "seed": seed,
                    "tag": model_cfg["tag"],
                }
            )
            print(json.dumps(submitted[-1]))

    print(json.dumps({"submitted_jobs": len(submitted)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
