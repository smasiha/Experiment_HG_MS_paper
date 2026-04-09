# LLM Paper Experiments

This folder is a git-ready subset of the LLM experiments currently used in the paper draft.

Scope:
- keep the existing **controlled** small-scale LLM source-reweighting experiments,
- keep the existing **real-world** Qwen2.5-based reproductions,
- exclude the newer multi-model `Llama` / `Gemma` / `Qwen-2-7B` expansion work,
- exclude raw Slurm logs and large output directories.

Structure:
- `controlled/`: the existing two-source ScaleBiO-style controlled tasks used in the paper (`denoising`, `multilingual`)
- `real_world/`: the existing single-H100 real-world code path centered on `Qwen2.5-0.5B-Instruct`
- `requirements.txt`: copied from `LLM_minima_selection/requirements.txt`

Notes:
- The controlled experiments use the existing GPT-2 setup from the draft. They are kept here because they are part of the current LLM section.
- The real-world experiments in this package are the existing `Qwen2.5-0.5B-Instruct + LoRA` reproductions.
- This package is intentionally narrow. It does not include the recent larger-model GSM8K calibration work.

Paper-facing snapshots:
- controlled-task table snapshot: `controlled/results_summary.md`
- real-world instruction snapshot: `real_world/results_summary.md`

Recommended git additions:
- track this folder
- do **not** track `outputs/`, `cache/`, `__pycache__/`, or Slurm logs

