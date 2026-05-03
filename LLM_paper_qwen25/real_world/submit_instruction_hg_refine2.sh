#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SEED="${SEED:-2424}"
TAG="${TAG:-instruction_hg_refine2_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

cd "${ROOT_DIR}"

submit_job() {
  local run_name="$1"
  local outer_lr="$2"
  local reweight_steps="$3"
  local selection_batches="$4"
  local m_branches="$5"
  local train_penalty="$6"
  local train_tolerance="$7"

  sbatch \
    --job-name="instr_hg_ms" \
    --export=ALL,\
METHOD="hg_ms",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
STOP_AFTER_REWEIGHT=0,\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
MAX_LENGTH=256,\
VAL_BATCH_SIZE=16,\
SELECTION_VAL_BATCH_SIZE=16,\
EVAL_BATCH_SIZE=16,\
THETA_OPTIMIZER="adam",\
REWEIGHT_MODEL_LR="2e-4",\
REWEIGHT_OUTER_LR="${outer_lr}",\
BRANCH_NOISE_STD="5e-4",\
REWEIGHT_TRAIN_PER_SOURCE=3000,\
REWEIGHT_TRAIN_RAW_OVERSAMPLE=1.5,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH="2",\
REWEIGHT_STEPS="${reweight_steps}",\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${selection_batches}",\
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

echo "Instruction HG refine2 output root: ${OUTPUT_ROOT}"

submit_job "ms_adam_lr0p03_ml2_s100"       "0.03" "100" "6" "4" "0.05" "0.30"
submit_job "ms_adam_lr0p05_ml2_s100"       "0.05" "100" "6" "4" "0.05" "0.30"
submit_job "ms_adam_lr0p07_ml2_s100"       "0.07" "100" "6" "4" "0.05" "0.30"
submit_job "ms_adam_lr0p05_ml2_s150"       "0.05" "150" "6" "4" "0.05" "0.30"
submit_job "ms_adam_lr0p03_ml2_s150"       "0.03" "150" "6" "4" "0.05" "0.30"
submit_job "ms_adam_lr0p05_ml2_b8"         "0.05" "100" "8" "4" "0.05" "0.30"
submit_job "ms_adam_lr0p05_ml2_m8"         "0.05" "100" "6" "8" "0.05" "0.30"
submit_job "ms_adam_lr0p05_ml2_pen0p02"    "0.05" "100" "6" "4" "0.02" "0.30"
submit_job "ms_adam_lr0p05_ml2_pen0"       "0.05" "100" "6" "4" "0.00" "0.30"
