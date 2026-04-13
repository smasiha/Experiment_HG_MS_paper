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

## If There Is No Slurm

The bundle is still usable without `sbatch`. The Slurm files are thin wrappers around the actual Python entrypoints.

Use this rule:

- ignore `submit_*.sh` and `run_*.sbatch` as launchers;
- read the matching `run_*.sbatch` file only to copy the environment variables and command-line flags;
- run the Python entrypoints directly with `python` or `torchrun`.

### Direct Entry Points

- faithful `Uniform`:
  - `faithful_uniform_math_fsdp.py`
- faithful `ScaleBiO`:
  - `faithful_scalebio_math_fsdp.py`
- faithful `LESS` / `RHO-LOSS` final training:
  - `faithful_weighted_math_fsdp.py`
- adapted HG checks:
  - `exact_math_reasoning.py`
- faithful MATH evaluation:
  - `evaluate_saved_faithful_math_runs.py`

### GPU Requirement Without Slurm

For the faithful baselines, the target setup is still `8` H100 GPUs total.

- if the collaborator has one `8`-GPU machine, use `torchrun --standalone --nproc-per-node=8 ...`
- if the collaborator has only `4` GPUs on one machine, the run is no longer faithful to the target setup
- if the collaborator has `2` machines with `4` GPUs each, use multi-node `torchrun` manually with `--nnodes=2`, `--node-rank`, `--master-addr`, and `--master-port`

### Direct Launch Examples

Activate the same environment first:

```bash
cd /path/to/Experiment_HG_MS_paper/LLM_paper_qwen25/table3_handoff
source /path/to/env/bin/activate
export HF_HUB_CACHE=/scratch/$USER/hf_cache_jobs/example/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache_jobs/example/transformers
export HF_ASSETS_CACHE=/scratch/$USER/hf_cache_jobs/example/assets
```

Faithful `Uniform` on one `8`-GPU machine:

```bash
torchrun --standalone --nproc-per-node=8 faithful_uniform_math_fsdp.py \
  --model-name Qwen/Qwen2-7B \
  --output-dir /scratch/$USER/faithful_qwen_uniform_seed2424 \
  --seed 2424 \
  --max-length 1024 \
  --reweight-train-per-source 3000 \
  --train-raw-oversample 4.0 \
  --val-per-source 280 \
  --val-raw-oversample 1.2 \
  --final-train-size 20000 \
  --final-ft-lr 8e-6 \
  --final-ft-epochs 1 \
  --per-device-batch-size 1 \
  --grad-accum 8 \
  --eval-batch-size 4 \
  --eval-max-new-tokens 256 \
  --gsm8k-eval-size 0 \
  --bf16 \
  --gradient-checkpointing
```

Faithful `ScaleBiO` on one `8`-GPU machine:

```bash
torchrun --standalone --nproc-per-node=8 faithful_scalebio_math_fsdp.py \
  --model-name Qwen/Qwen2-7B \
  --output-dir /scratch/$USER/faithful_qwen_scalebio_seed2424 \
  --seed 2424 \
  --max-length 1024 \
  --reweight-train-per-source 3000 \
  --train-raw-oversample 4.0 \
  --val-per-source 280 \
  --val-raw-oversample 1.2 \
  --reweight-epochs 3 \
  --per-source-batch 1 \
  --val-batch-size 8 \
  --reweight-model-lr 1e-5 \
  --reweight-weight-decay 0.0 \
  --reweight-outer-lr 1e-2 \
  --scalebio-alpha 100.0 \
  --tune-every 50 \
  --theta-optimizer sgd \
  --theta-grad-clip 0.0 \
  --final-train-size 20000 \
  --final-ft-lr 8e-6 \
  --final-ft-epochs 1 \
  --per-device-batch-size 1 \
  --grad-accum 8 \
  --eval-batch-size 4 \
  --eval-max-new-tokens 256 \
  --gsm8k-eval-size 0 \
  --bf16 \
  --gradient-checkpointing
```

Faithful `LESS` / `RHO-LOSS` final training from an existing reweight metrics file:

```bash
torchrun --standalone --nproc-per-node=8 faithful_weighted_math_fsdp.py \
  --method less \
  --model-name Qwen/Qwen2-7B \
  --reweight-metrics-path /scratch/$USER/<reweight-root>/less_seed2424_qwen2_7b_faithful_reweight/metrics.json \
  --output-dir /scratch/$USER/faithful_qwen_less_seed2424 \
  --seed 2424 \
  --max-length 1024 \
  --reweight-train-per-source 3000 \
  --train-raw-oversample 4.0 \
  --val-per-source 280 \
  --val-raw-oversample 1.2 \
  --final-train-size 20000 \
  --final-ft-lr 8e-6 \
  --final-ft-epochs 1 \
  --per-device-batch-size 1 \
  --grad-accum 8 \
  --eval-batch-size 4 \
  --eval-max-new-tokens 256 \
  --gsm8k-eval-size 0 \
  --bf16 \
  --use-best-reweight-checkpoint \
  --gradient-checkpointing
```

Adapted HG checks on one GPU:

```bash
python -u exact_math_reasoning.py \
  --method hg_single \
  --model-name meta-llama/Meta-Llama-3-8B \
  --output-dir /scratch/$USER/hg_single_check \
  --seed 2424 \
  --cache-path /scratch/$USER/hf_cache_jobs/hg_single_check \
  --reweight-train-per-source 3000 \
  --val-per-source 280 \
  --train-raw-oversample 4.0 \
  --val-raw-oversample 1.2 \
  --reweight-epochs 1 \
  --reweight-wall-clock-seconds 3600 \
  --per-source-batch 1 \
  --val-batch-size 8 \
  --selection-val-batch-size 1 \
  --selection-val-batches 4 \
  --hypergrad-val-batches 4 \
  --selection-train-tolerance 0.05 \
  --selection-train-penalty 0.0 \
  --inner-steps 1 \
  --selection-inner-steps 1 \
  --hypergrad-inner-steps 1 \
  --m-branches 1 \
  --reweight-model-lr 1e-5 \
  --reweight-outer-lr 5.0 \
  --theta-optimizer sgd \
  --final-train-size 20000 \
  --final-ft-lr 8e-6 \
  --final-ft-epochs 1 \
  --final-ft-micro-batch 1 \
  --final-ft-grad-accum 8 \
  --max-length 256 \
  --eval-max-new-tokens 256 \
  --gsm8k-eval-size 0 \
  --lora-r 4 \
  --lora-alpha 8 \
  --lora-dropout 0.0 \
  --bf16 \
  --gradient-checkpointing \
  --stop-after-reweight
```

Faithful MATH evaluation of a finished checkpoint:

```bash
python -u evaluate_saved_faithful_math_runs.py \
  --run-dir /scratch/$USER/<faithful-run-dir> \
  --math-eval-size 0
```

### Practical Note

If the collaborator does not use Slurm, the fastest way to stay aligned is:

1. run faithful baselines on a single `8`-GPU machine with `torchrun --standalone`
2. run reweight-only jobs and MATH evals as plain `python` commands
3. keep the staged order from `TABLE3_HANDOFF_PLAN.md`

Do not change the method order or the cost policy just because the scheduler changed.
