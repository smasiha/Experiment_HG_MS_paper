# Table 3 Handoff Plan

This note packages the remaining `Table 3` work into a low-cost staged plan that a collaborator can continue on another H100 server.

If that server does not use Slurm, keep the same staged order and call the Python entrypoints directly with `python` / `torchrun`. The exact non-Slurm launch patterns are documented in `README.md`.

## Goal

Fill `Hyper_grad_paper/Minima_Selection/example_paper.tex` `Table 3` with:

- faithful `ScaleBiO` baselines;
- faithful `Uniform`, `LESS`, and `RHO-LOSS` baselines for the updated comparison;
- adapted `HG-single` and `HG-MS` runs, after one short configuration check.

The cost-control rule is strict:

1. only one model at a time on the expensive faithful 8-GPU path;
2. run cheap 1-GPU reweight jobs in parallel only when they shorten the next decision;
3. do not launch 8-GPU `LESS` / `RHO-LOSS` finals until the matching `Uniform` canary finishes cleanly;
4. do not launch `ScaleBiO` for a model until that model's `Uniform` canary is clean.

## Current Status

Llama faithful progress:

- `Uniform`: GSM8K `60.42`, MATH `22.01`
- `RHO-LOSS`: GSM8K `52.99`, MATH `22.23`
- `LESS`: GSM8K `56.10`, MATH `22.19`
- `ScaleBiO`: GSM8K `68.92`, MATH `24.81`

Current live jobs at the time this handoff file was written:

- `Qwen` faithful `Uniform` canary
- `Qwen` faithful `LESS` reweight

## Scripts Added For Handoff

- `submit_faithful_uniform_canary.sh`
  - one faithful 8-GPU `Uniform` canary per model/seed
- `submit_faithful_baseline_stage.sh`
  - `STAGE=reweights`: submit only `LESS` / `RHO-LOSS` 1-GPU reweight jobs
  - `STAGE=finals`: submit only the dependent 8-GPU faithful final-training jobs from completed reweight metrics
- `submit_faithful_scalebio_canary.sh`
  - one faithful 8-GPU `ScaleBiO` canary per model/seed
- `submit_math_hg_finalization_checks.sh`
  - cheap adapted HG checks before any heavy `HG-single` / `HG-MS` reruns

## Recommended Execution Order

### 1. Finish `Qwen`

Run in this order:

1. `Uniform` canary
2. `LESS` / `RHO-LOSS` reweights
3. if `Uniform` is clean, `LESS` / `RHO-LOSS` faithful finals
4. one faithful `ScaleBiO` canary at `REWEIGHT_MODEL_LR=1e-5`
5. only if needed, retry `ScaleBiO` with `REWEIGHT_MODEL_LR=8e-6`

Commands:

```bash
cd /path/to/paper_experiments/LLM_paper_qwen25/table3_handoff

MODEL_PRESET=qwen2_7b SEEDS="2424" bash submit_faithful_uniform_canary.sh
MODEL_PRESET=qwen2_7b STAGE=reweights METHODS="less rho_loss" SEEDS="2424" bash submit_faithful_baseline_stage.sh

# after the Qwen uniform canary finishes cleanly:
MODEL_PRESET=qwen2_7b STAGE=finals METHODS="less rho_loss" SEEDS="2424" \
  REWEIGHT_OUTPUT_ROOT=/scratch/$USER/faithful_math_baselines_qwen2_7b_<tag>/reweight \
  FINAL_OUTPUT_ROOT=/scratch/$USER/faithful_math_baselines_qwen2_7b_<tag>/final \
  bash submit_faithful_baseline_stage.sh

MODEL_PRESET=qwen2_7b SEEDS="2424" REWEIGHT_MODEL_LR=1e-5 bash submit_faithful_scalebio_canary.sh
```

### 2. Then `Gemma`

Use the same sequence, but only after `Qwen` is resolved:

```bash
MODEL_PRESET=gemma2_9b SEEDS="2424" bash submit_faithful_uniform_canary.sh
MODEL_PRESET=gemma2_9b STAGE=reweights METHODS="less rho_loss" SEEDS="2424" bash submit_faithful_baseline_stage.sh

# after the Gemma uniform canary finishes cleanly:
MODEL_PRESET=gemma2_9b STAGE=finals METHODS="less rho_loss" SEEDS="2424" \
  REWEIGHT_OUTPUT_ROOT=/scratch/$USER/faithful_math_baselines_gemma2_9b_<tag>/reweight \
  FINAL_OUTPUT_ROOT=/scratch/$USER/faithful_math_baselines_gemma2_9b_<tag>/final \
  bash submit_faithful_baseline_stage.sh

MODEL_PRESET=gemma2_9b SEEDS="2424" REWEIGHT_MODEL_LR=1e-5 bash submit_faithful_scalebio_canary.sh
```

### 3. Expand To Three Seeds

Only after each model has a clean seed-`2424` canary:

- rerun the same method/model combination for `2525` and `2626`;
- launch MATH evaluation only for completed faithful checkpoints:

```bash
python submit_faithful_math_eval_parallel.py --runs-root /scratch/$USER/<faithful-final-root>
```

- update the appendix progress table first, then the main table.

### 4. Finalize HG Configs Cheaply

Before any more heavy adapted `HG-single` / `HG-MS` runs:

```bash
MODEL_PRESET=llama3_8b SEED=2424 bash submit_math_hg_finalization_checks.sh
```

Interpretation:

- if `hg_single_mlr1e4_check` or `hg_single_mlr1e5_check` still shows zero `theta_grad_norm` and no lambda motion, do not spend more GPU on `HG-single` yet;
- if `hg_ms_curr_check` shows stable motion and finite tune loss, use that configuration for the next adapted full run.

## What To Upload To Git

Bundle contents:

- `run_faithful_uniform_math_fsdp.sbatch`
- `run_faithful_scalebio_math_fsdp.sbatch`
- `run_faithful_weighted_math_fsdp.sbatch`
- `run_saved_faithful_math_eval.sbatch`
- `faithful_uniform_math_fsdp.py`
- `faithful_scalebio_math_fsdp.py`
- `faithful_weighted_math_fsdp.py`
- `evaluate_saved_faithful_math_runs.py`
- `submit_faithful_uniform_canary.sh`
- `submit_faithful_baseline_stage.sh`
- `submit_faithful_scalebio_canary.sh`
- `submit_math_hg_finalization_checks.sh`
- `TABLE3_HANDOFF_PLAN.md`
