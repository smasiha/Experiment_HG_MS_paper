#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SEED="${SEED:-2424}"
TAG="${TAG:-instruction_hg_refine3_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

cd "${ROOT_DIR}"

submit_job() {
  local run_name="$1"
  local outer_lr="$2"
  local reweight_steps="$3"
  local selection_batches="$4"
  local train_penalty="$5"
  local branch_noise="$6"

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
BRANCH_NOISE_STD="${branch_noise}",\
REWEIGHT_TRAIN_PER_SOURCE=3000,\
REWEIGHT_TRAIN_RAW_OVERSAMPLE=1.5,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH="2",\
REWEIGHT_STEPS="${reweight_steps}",\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${selection_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="0.30",\
SELECTION_TRAIN_PENALTY="${train_penalty}",\
INNER_STEPS="1",\
M_BRANCHES="8",\
FINAL_TRAIN_SIZE=10000,\
FINAL_FT_MICRO_BATCH=1,\
FINAL_FT_GRAD_ACCUM=64,\
FINAL_TRAIN_RAW_OVERSAMPLE=1.1,\
REFERENCE_BATCH_SIZE=8,\
REFERENCE_FT_STEPS=200 \
    run_real_world_instruction.sbatch
}

echo "Instruction HG refine3 output root: ${OUTPUT_ROOT}"

submit_job "ms_adam_lr0p05_ml2_m8_s150"         "0.05" "150" "6" "0.05" "5e-4"
submit_job "ms_adam_lr0p03_ml2_m8_s150"         "0.03" "150" "6" "0.05" "5e-4"
submit_job "ms_adam_lr0p05_ml2_m8_s200"         "0.05" "200" "6" "0.05" "5e-4"
submit_job "ms_adam_lr0p05_ml2_m8_s150_pen0"    "0.05" "150" "6" "0.00" "5e-4"
submit_job "ms_adam_lr0p05_ml2_m8_s150_pen0p02" "0.05" "150" "6" "0.02" "5e-4"
submit_job "ms_adam_lr0p05_ml2_m8_s150_b8"      "0.05" "150" "8" "0.05" "5e-4"
submit_job "ms_adam_lr0p05_ml2_m8_s150_noise0"  "0.05" "150" "6" "0.05" "0"
submit_job "ms_adam_lr0p05_ml2_m8_s150_noise1e3" "0.05" "150" "6" "0.05" "1e-3"
