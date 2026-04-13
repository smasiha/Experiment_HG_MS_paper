# Table 3 Handoff Bundle

This directory is a self-contained bundle for finishing the remaining `Table 3` math-reasoning experiments on another H100 server.

## Scope

The bundle supports two experiment tracks:

- faithful baselines:
  - `Uniform`
  - `LESS`
  - `RHO-LOSS`
  - `ScaleBiO`
- adapted hypergradient checks:
  - `HG-single`
  - `HG-MS`

The intent is:

1. keep `ScaleBiO` and the main baselines on the faithful path;
2. keep `HG-single` and `HG-MS` on the adapted path;
3. minimize GPU cost by staging the expensive jobs.

## What Is Already Known

Faithful `Llama-3-8B` progress at handoff time:

- `Uniform`: GSM8K `60.42`, MATH `22.01`
- `RHO-LOSS`: GSM8K `52.99`, MATH `22.23`
- `LESS`: GSM8K `56.10`, MATH `22.19`
- `ScaleBiO`: GSM8K `68.92`, MATH `24.81`

These values are progress-tracking results for the faithful path, not replacements for the current main `Table 3` yet.

## Directory Contents

- launchers:
  - `submit_faithful_uniform_canary.sh`
  - `submit_faithful_baseline_stage.sh`
  - `submit_faithful_scalebio_canary.sh`
  - `submit_math_hg_finalization_checks.sh`
- Slurm wrappers:
  - `run_exact_math_reasoning.sbatch`
  - `run_faithful_uniform_math_fsdp.sbatch`
  - `run_faithful_scalebio_math_fsdp.sbatch`
  - `run_faithful_weighted_math_fsdp.sbatch`
  - `run_saved_faithful_math_eval.sbatch`
- Python code:
  - `exact_math_reasoning.py`
  - `exact_math_reasoning_data.py`
  - `faithful_uniform_math_fsdp.py`
  - `faithful_scalebio_math_fsdp.py`
  - `faithful_weighted_math_fsdp.py`
  - `evaluate_saved_faithful_math_runs.py`
  - `submit_faithful_math_eval_parallel.py`
  - `check_model_access.py`
- configuration/docs:
  - `model_presets.sh`
  - `TABLE3_HANDOFF_PLAN.md`

All shell launchers in this directory default `ROOT_DIR` to this directory, so they can be run directly after cloning.

## Recommended Order

### 1. Finish `Qwen`

Run:

```bash
cd /path/to/paper_experiments/LLM_paper_qwen25/table3_handoff

MODEL_PRESET=qwen2_7b SEEDS="2424" bash submit_faithful_uniform_canary.sh
MODEL_PRESET=qwen2_7b STAGE=reweights METHODS="less rho_loss" SEEDS="2424" bash submit_faithful_baseline_stage.sh
```

After `Uniform` finishes cleanly:

```bash
MODEL_PRESET=qwen2_7b STAGE=finals METHODS="less rho_loss" SEEDS="2424" \
  REWEIGHT_OUTPUT_ROOT=/scratch/$USER/faithful_math_baselines_qwen2_7b_<tag>/reweight \
  FINAL_OUTPUT_ROOT=/scratch/$USER/faithful_math_baselines_qwen2_7b_<tag>/final \
  bash submit_faithful_baseline_stage.sh

MODEL_PRESET=qwen2_7b SEEDS="2424" REWEIGHT_MODEL_LR=1e-5 bash submit_faithful_scalebio_canary.sh
```

Only if the first `ScaleBiO` canary is weak or unstable, retry with `REWEIGHT_MODEL_LR=8e-6`.

### 2. Then `Gemma`

Use the same sequence:

```bash
MODEL_PRESET=gemma2_9b SEEDS="2424" bash submit_faithful_uniform_canary.sh
MODEL_PRESET=gemma2_9b STAGE=reweights METHODS="less rho_loss" SEEDS="2424" bash submit_faithful_baseline_stage.sh
```

Then release its faithful finals and one `ScaleBiO` canary after the `Uniform` canary is validated.

### 3. Only Then Expand Seeds

After seed `2424` is clean for a model/method:

- rerun for `2525` and `2626`
- run MATH evals with:

```bash
python submit_faithful_math_eval_parallel.py --runs-root /scratch/$USER/<faithful-final-root>
```

### 4. Cheap HG Checks Before More HG Runs

```bash
MODEL_PRESET=llama3_8b SEED=2424 bash submit_math_hg_finalization_checks.sh
```

Interpretation:

- if `HG-single` still shows zero `theta_grad_norm` and no lambda motion, do not launch more `HG-single`
- if `HG-MS` is stable and moves lambda, then use that config for the next adapted full run

## Cost Policy

Use this bundle under a strict low-cost policy:

- at most one model at a time on the 8-GPU faithful path
- 1-GPU reweight jobs may run in parallel if they shorten the next decision
- do not launch 8-GPU `LESS` / `RHO-LOSS` finals before the matching `Uniform` canary passes
- do not launch `ScaleBiO` for a model before its `Uniform` canary passes

## Environment Assumptions

- Slurm with an `h100` partition
- 4 H100 GPUs per node; the faithful path uses `2` nodes (`8` GPUs total)
- Python environment compatible with the original scripts, typically:
  - `ENV_DIR=/home/masiha/venvs/hypergrad_llm_env`
- scratch storage at `/scratch/$USER`

If the collaborator uses a different virtualenv path, pass `ENV_DIR=/path/to/env` when invoking the scripts.
