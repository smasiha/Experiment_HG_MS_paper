#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SEED="${SEED:-2424}"
TAG="${TAG:-realworld_wallclock_valalign_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}/math}"
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS:-576}"

cd "${ROOT_DIR}"

submit_job() {
  local run_name="$1"
  local val_kinds_spec="$2"
  local selection_inner_steps="$3"
  local selection_batches="$4"

  sbatch \
    --job-name="wcfull_val_${run_name}" \
    --export=ALL,\
METHOD="hg_ms",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
VAL_KINDS_SPEC="${val_kinds_spec}",\
MAX_LENGTH=512,\
VAL_BATCH_SIZE=1,\
SELECTION_VAL_BATCH_SIZE=2,\
EVAL_BATCH_SIZE=1,\
REWEIGHT_MODEL_LR=1e-4,\
REWEIGHT_OUTER_LR=10,\
THETA_OPTIMIZER="sgd",\
BRANCH_NOISE_STD=0,\
REWEIGHT_TRAIN_PER_SOURCE=500,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH=1,\
REWEIGHT_STEPS=400,\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES=6,\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=0,\
SELECTION_TRAIN_TOLERANCE=0.30,\
SELECTION_TRAIN_PENALTY=0.25,\
INNER_STEPS=2,\
SELECTION_INNER_STEPS="${selection_inner_steps}",\
HYPERGRAD_INNER_STEPS=2,\
M_BRANCHES=4 \
    run_exact_math_reasoning.sbatch
}

echo "Math validation-aligned full output root: ${OUTPUT_ROOT}"
echo "Reweight budget (sec): ${REWEIGHT_WALL_CLOCK_SECONDS}"

submit_job "ms_curr_gsm8k_wcfull" "gsm8k" 2 6
submit_job "ms_curr_gsmath_wcfull" "gsm8k+math" 2 6
submit_job "ms_s1h2b3_gsm8k_wcfull" "gsm8k" 1 3
submit_job "ms_s1h2b3_gsmath_wcfull" "gsm8k+math" 1 3
