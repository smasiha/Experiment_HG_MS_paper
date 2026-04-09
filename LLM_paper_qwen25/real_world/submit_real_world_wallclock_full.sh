#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SEED="${SEED:-2424}"
TAG="${TAG:-realworld_wallclock_full_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

# Calibrated from the tuned HG-MS reweight runs.
MATH_BUDGET_SECONDS="${MATH_BUDGET_SECONDS:-576}"
INSTRUCTION_BUDGET_SECONDS="${INSTRUCTION_BUDGET_SECONDS:-1004}"

cd "${ROOT_DIR}"

submit_math_job() {
  local method="$1"
  local run_name="$2"
  local outer_lr="$3"
  local inner_steps="$4"
  local m_branches="$5"
  local selection_batches="$6"
  local hypergrad_batches="$7"
  local selection_batch_size="$8"
  local branch_noise="$9"
  local train_tolerance="${10}"
  local train_penalty="${11}"
  local reweight_steps="${12}"

  sbatch \
    --job-name="wcfull_math_${method}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
REWEIGHT_WALL_CLOCK_SECONDS="${MATH_BUDGET_SECONDS}",\
OUTPUT_ROOT="${OUTPUT_ROOT}/math",\
RUN_NAME="${run_name}",\
MAX_LENGTH=512,\
VAL_BATCH_SIZE=1,\
SELECTION_VAL_BATCH_SIZE="${selection_batch_size}",\
EVAL_BATCH_SIZE=1,\
REWEIGHT_MODEL_LR=1e-4,\
REWEIGHT_OUTER_LR="${outer_lr}",\
BRANCH_NOISE_STD="${branch_noise}",\
REWEIGHT_TRAIN_PER_SOURCE=500,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH=1,\
REWEIGHT_STEPS="${reweight_steps}",\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${hypergrad_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${train_tolerance}",\
SELECTION_TRAIN_PENALTY="${train_penalty}",\
INNER_STEPS="${inner_steps}",\
M_BRANCHES="${m_branches}",\
FINAL_TRAIN_SIZE=20000,\
FINAL_FT_MICRO_BATCH=1,\
FINAL_FT_GRAD_ACCUM=32,\
GSM8K_EVAL_SIZE=0,\
REFERENCE_BATCH_SIZE=2,\
REFERENCE_FT_STEPS=100 \
    run_exact_math_reasoning.sbatch
}

submit_instruction_job() {
  local method="$1"
  local run_name="$2"
  local outer_lr="$3"
  local theta_optimizer="$4"
  local reweight_model_lr="$5"
  local reweight_steps="$6"
  local m_branches="$7"
  local per_source_batch="$8"
  local selection_batches="$9"
  local hypergrad_batches="${10}"
  local train_tolerance="${11}"
  local train_penalty="${12}"

  sbatch \
    --job-name="wcfull_instr_${method}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
REWEIGHT_WALL_CLOCK_SECONDS="${INSTRUCTION_BUDGET_SECONDS}",\
OUTPUT_ROOT="${OUTPUT_ROOT}/instruction",\
RUN_NAME="${run_name}",\
MAX_LENGTH=256,\
VAL_BATCH_SIZE=16,\
SELECTION_VAL_BATCH_SIZE=16,\
EVAL_BATCH_SIZE=16,\
THETA_OPTIMIZER="${theta_optimizer}",\
REWEIGHT_MODEL_LR="${reweight_model_lr}",\
REWEIGHT_OUTER_LR="${outer_lr}",\
BRANCH_NOISE_STD="5e-4",\
REWEIGHT_TRAIN_PER_SOURCE=3000,\
REWEIGHT_TRAIN_RAW_OVERSAMPLE=1.5,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH="${per_source_batch}",\
REWEIGHT_STEPS="${reweight_steps}",\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${hypergrad_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${train_tolerance}",\
SELECTION_TRAIN_PENALTY="${train_penalty}",\
INNER_STEPS="1",\
M_BRANCHES="${m_branches}",\
FINAL_TRAIN_SIZE=10000,\
FINAL_FT_MICRO_BATCH=1,\
FINAL_FT_GRAD_ACCUM=64,\
FINAL_TRAIN_RAW_OVERSAMPLE=1.1,\
REFERENCE_BATCH_SIZE=8,\
REFERENCE_FT_STEPS=200 \
    run_real_world_instruction.sbatch
}

echo "Wall-clock full-run output root: ${OUTPUT_ROOT}"
echo "Math reweight budget (sec): ${MATH_BUDGET_SECONDS}"
echo "Instruction reweight budget (sec): ${INSTRUCTION_BUDGET_SECONDS}"

# Math reasoning
submit_math_job "uniform"   "uniform_wcfull"   "0.1"  "1" "1" "1" "1" "1" "0" "0.05" "0.0"  "1"
submit_math_job "less"      "less_wcfull"      "0.1"  "1" "1" "1" "1" "1" "0" "0.05" "0.0"  "1"
submit_math_job "rho_loss"  "rho_wcfull"       "0.1"  "1" "1" "1" "1" "1" "0" "0.05" "0.0"  "1"
submit_math_job "scalebio"  "scalebio_wcfull"  "1.0"  "1" "1" "1" "1" "1" "0" "0.05" "0.0"  "700"
submit_math_job "hg_single" "single_wcfull"    "5.0"  "1" "1" "4" "4" "2" "0" "0.05" "0.0"  "300"
submit_math_job "hg_ms"     "ms_wcfull"        "10.0" "2" "4" "6" "6" "2" "0" "0.30" "0.25" "200"

# Instruction following
submit_instruction_job "uniform"   "uniform_wcfull"   "0.01" "sgd"  "1e-5" "1"    "1"  "1" "1" "1" "0.05" "0.0"
submit_instruction_job "less"      "less_wcfull"      "0.01" "sgd"  "1e-5" "1"    "1"  "1" "1" "1" "0.05" "0.0"
submit_instruction_job "rho_loss"  "rho_wcfull"       "0.01" "sgd"  "1e-5" "1"    "1"  "1" "1" "1" "0.05" "0.0"
submit_instruction_job "scalebio"  "scalebio_wcfull"  "0.01" "sgd"  "1e-5" "2000" "1"  "1" "1" "1" "0.05" "0.0"
submit_instruction_job "hg_single" "single_wcfull"    "10.0" "adam" "2e-4" "400"  "1"  "2" "6" "6" "0.50" "0.0"
submit_instruction_job "hg_ms"     "ms_wcfull"        "0.05" "adam" "2e-4" "250"  "8"  "2" "6" "6" "0.50" "0.0"
