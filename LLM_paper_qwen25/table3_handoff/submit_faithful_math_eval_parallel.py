#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit one MATH-eval Slurm job per finished faithful run that has a saved full-model checkpoint."
    )
    parser.add_argument("--runs-root", action="append", default=[])
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


def time_limit_for_model(model_name: str) -> str:
    name = model_name.lower()
    if "gemma-2-9b" in name:
        return "20:00:00"
    if "qwen2-7b" in name:
        return "12:00:00"
    if "llama-3-8b" in name:
        return "10:00:00"
    return "12:00:00"


def cache_tag_for_model(model_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name).strip("_")


def iter_pending_runs(root: Path):
    if not root.exists():
        return
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        state_path = run_dir / "final_full_state.pt"
        if not config_path.exists() or not state_path.exists():
            continue
        if has_finite_math_accuracy(metrics_path):
            continue
        config = json.loads(config_path.read_text())
        yield run_dir, config


def compact_method_name(method: str) -> str:
    return (
        method.replace("uniform", "uni")
        .replace("scalebio", "sbo")
        .replace("hg_single", "hgs")
        .replace("hg_ms", "hgm")
        .replace("rho_loss", "rho")
        .replace("less", "les")
    )


def infer_method(run_dir: Path, config: dict[str, object]) -> str:
    method = config.get("method")
    if method:
        return str(method)
    name = run_dir.name.lower()
    if "uniform" in name:
        return "uniform"
    if "scalebio" in name:
        return "scalebio"
    return run_dir.name


def main() -> int:
    args = parse_args()
    if args.runs_root:
        roots = [Path(item) for item in args.runs_root]
    else:
        scratch_root = Path("/scratch") / os.environ.get("USER", "unknown")
        roots = sorted(path for path in scratch_root.glob("faithful_table4_*") if path.is_dir())
    sbatch_script = Path(__file__).with_name("run_saved_faithful_math_eval.sbatch")

    submitted = []
    for root in roots:
        for run_dir, config in iter_pending_runs(root):
            model_name = str(config["model_name"])
            method = infer_method(run_dir, config)
            seed = int(config.get("seed", 0))
            time_limit = time_limit_for_model(model_name)
            cache_tag = cache_tag_for_model(model_name)
            job_name = f"fm_{compact_method_name(method)}_{seed}"
            cache_root = f"/scratch/{os.environ.get('USER', 'unknown')}/hf_cache_faithful_math_eval/{cache_tag}"
            export = (
                "ALL,"
                f"RUN_DIR={run_dir},"
                "MATH_EVAL_SIZE=0,"
                "FORCE_EVAL=0,"
                "LOCAL_MODEL_ONLY=0,"
                f"HF_CACHE_ROOT={cache_root}"
            )
            cmd = [
                "sbatch",
                "--parsable",
                "--job-name",
                job_name,
                "--time",
                time_limit,
                "--export",
                export,
                str(sbatch_script),
            ]
            if args.dry_run:
                print(json.dumps({"dry_run": cmd}))
                continue
            job_id = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
            row = {
                "job_id": job_id,
                "job_name": job_name,
                "time_limit": time_limit,
                "run_dir": str(run_dir),
                "model_name": model_name,
                "method": method,
                "seed": seed,
            }
            submitted.append(row)
            print(json.dumps(row))

    print(json.dumps({"submitted_jobs": len(submitted)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
