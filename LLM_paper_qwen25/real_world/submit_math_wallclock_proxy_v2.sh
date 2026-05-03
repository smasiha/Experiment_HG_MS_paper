#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SEED="${SEED:-2424}"
TAG="${TAG:-math_wallclock_proxy_v2_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS:-576}"

cd "${ROOT_DIR}"

submit_job() {
  local method="$1"
  local run_name="$2"
  local outer_lr="$3"
  local theta_optimizer="$4"
  local inner_steps="$5"
  local selection_inner_steps="$6"
  local hypergrad_inner_steps="$7"
  local m_branches="$8"
  local selection_batches="$9"
  local hypergrad_batches="${10}"
  local selection_batch_size="${11}"
  local branch_noise="${12}"
  local train_tolerance="${13}"
  local train_penalty="${14}"

  sbatch \
    --job-name="wcproxy2_${method}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
MAX_LENGTH=512,\
VAL_BATCH_SIZE=1,\
SELECTION_VAL_BATCH_SIZE="${selection_batch_size}",\
EVAL_BATCH_SIZE=1,\
REWEIGHT_MODEL_LR=1e-4,\
REWEIGHT_OUTER_LR="${outer_lr}",\
THETA_OPTIMIZER="${theta_optimizer}",\
BRANCH_NOISE_STD="${branch_noise}",\
REWEIGHT_TRAIN_PER_SOURCE=500,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH=1,\
REWEIGHT_STEPS=400,\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${hypergrad_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=0,\
SELECTION_TRAIN_TOLERANCE="${train_tolerance}",\
SELECTION_TRAIN_PENALTY="${train_penalty}",\
INNER_STEPS="${inner_steps}",\
SELECTION_INNER_STEPS="${selection_inner_steps}",\
HYPERGRAD_INNER_STEPS="${hypergrad_inner_steps}",\
M_BRANCHES="${m_branches}",\
FINAL_TRAIN_SIZE=6000,\
FINAL_FT_MICRO_BATCH=1,\
FINAL_FT_GRAD_ACCUM=32,\
GSM8K_EVAL_SIZE=256,\
REFERENCE_BATCH_SIZE=2,\
REFERENCE_FT_STEPS=100 \
    run_exact_math_reasoning.sbatch
}

echo "Math wall-clock proxy v2 output root: ${OUTPUT_ROOT}"
echo "Reweight budget (sec): ${REWEIGHT_WALL_CLOCK_SECONDS}"

submit_job "less"      "less_wcproxy2"                    "0.1"  "sgd"  "1" "1" "1" "1" "1" "1" "1" "0" "0.05" "0.0"
submit_job "hg_single" "single_wcproxy2"                  "5.0"  "sgd"  "1" "1" "1" "1" "4" "4" "2" "0" "0.05" "0.0"

submit_job "hg_ms"     "ms_curr_wcproxy2"                 "10.0" "sgd"  "2" "2" "2" "4" "6" "6" "2" "0" "0.30" "0.25"
submit_job "hg_ms"     "ms_s1h2_b3_sgd_wcproxy2"          "10.0" "sgd"  "2" "1" "2" "4" "3" "6" "2" "0" "0.30" "0.25"
submit_job "hg_ms"     "ms_s1h2_b3_adam05_wcproxy2"       "0.5"  "adam" "2" "1" "2" "4" "3" "6" "2" "0" "0.30" "0.25"
submit_job "hg_ms"     "ms_s1h2_b3_adam1_wcproxy2"        "1.0"  "adam" "2" "1" "2" "4" "3" "6" "2" "0" "0.30" "0.25"
submit_job "hg_ms"     "ms_s1h2_b4_adam05_wcproxy2"       "0.5"  "adam" "2" "1" "2" "4" "4" "6" "2" "0" "0.30" "0.25"
submit_job "hg_ms"     "ms_m3_s1h2_b3_adam05_wcproxy2"    "0.5"  "adam" "2" "1" "2" "3" "3" "6" "2" "0" "0.30" "0.25"
