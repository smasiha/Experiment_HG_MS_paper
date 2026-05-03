#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SEED="${SEED:-2424}"
TAG="${TAG:-math_hg_sweep_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

cd "${ROOT_DIR}"

submit_job() {
  local method="$1"
  local name="$2"
  local outer_lr="$3"
  local inner_steps="$4"
  local m_branches="$5"
  local selection_batches="$6"
  local hypergrad_batches="$7"
  local selection_batch_size="$8"
  local branch_noise="$9"
  local train_tolerance="${10}"
  local train_penalty="${11}"

  sbatch \
    --job-name="math_${method}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
STOP_AFTER_REWEIGHT=1,\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
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
REWEIGHT_STEPS=50,\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${hypergrad_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${train_tolerance}",\
SELECTION_TRAIN_PENALTY="${train_penalty}",\
INNER_STEPS="${inner_steps}",\
M_BRANCHES="${m_branches}",\
RUN_NAME="${name}" \
    run_exact_math_reasoning.sbatch
}

echo "Math HG sweep output root: ${OUTPUT_ROOT}"

submit_job "hg_single" "single_ref" "5.0" "1" "1" "4" "4" "2" "0" "0.05" "0.0"
submit_job "hg_ms" "ms_m2_b4" "5.0" "1" "2" "4" "4" "2" "0" "0.05" "0.0"
submit_job "hg_ms" "ms_m4_b4" "5.0" "1" "4" "4" "4" "2" "0" "0.05" "0.0"
submit_job "hg_ms" "ms_m4_b8" "5.0" "1" "4" "8" "8" "2" "0" "0.05" "0.0"
submit_job "hg_ms" "ms_m4_inner2" "5.0" "2" "4" "4" "4" "2" "0" "0.05" "0.0"
submit_job "hg_ms" "ms_m4_noise" "5.0" "1" "4" "4" "4" "2" "1e-4" "0.05" "0.0"
submit_job "hg_ms" "ms_m4_outer10" "10.0" "1" "4" "4" "4" "2" "0" "0.05" "0.0"

