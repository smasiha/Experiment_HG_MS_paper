#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SEED="${SEED:-2424}"
TAG="${TAG:-instruction_hg_refine_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

cd "${ROOT_DIR}"

submit_job() {
  local method="$1"
  local run_name="$2"
  local theta_optimizer="$3"
  local outer_lr="$4"
  local model_lr="$5"
  local inner_steps="$6"
  local m_branches="$7"
  local selection_batches="$8"
  local selection_batch_size="$9"
  local branch_noise="${10}"
  local train_tolerance="${11}"
  local train_penalty="${12}"
  local reweight_steps="${13}"
  local per_source_batch="${14}"

  sbatch \
    --job-name="instr_${method}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
STOP_AFTER_REWEIGHT=0,\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
MAX_LENGTH=256,\
VAL_BATCH_SIZE=16,\
SELECTION_VAL_BATCH_SIZE="${selection_batch_size}",\
EVAL_BATCH_SIZE=16,\
REWEIGHT_MODEL_LR="${model_lr}",\
REWEIGHT_OUTER_LR="${outer_lr}",\
THETA_OPTIMIZER="${theta_optimizer}",\
BRANCH_NOISE_STD="${branch_noise}",\
REWEIGHT_TRAIN_PER_SOURCE=3000,\
REWEIGHT_TRAIN_RAW_OVERSAMPLE=1.5,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH="${per_source_batch}",\
REWEIGHT_STEPS="${reweight_steps}",\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${selection_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${train_tolerance}",\
SELECTION_TRAIN_PENALTY="${train_penalty}",\
INNER_STEPS="${inner_steps}",\
M_BRANCHES="${m_branches}",\
FINAL_TRAIN_SIZE=10000,\
FINAL_FT_MICRO_BATCH=1,\
FINAL_FT_GRAD_ACCUM=64,\
FINAL_TRAIN_RAW_OVERSAMPLE=1.1,\
REFERENCE_BATCH_SIZE=8,\
REFERENCE_FT_STEPS=200 \
    run_real_world_instruction.sbatch
}

echo "Instruction HG refine output root: ${OUTPUT_ROOT}"

submit_job "hg_single" "single_sgd_s100"        "sgd"  "10.0" "1e-4" "1" "1" "6" "16" "0"    "0.05" "0.0"  "100" "1"
submit_job "hg_ms"     "ms_sgd_m4_s100"         "sgd"  "10.0" "1e-4" "1" "4" "6" "16" "5e-4" "0.30" "0.05" "100" "1"
submit_job "hg_ms"     "ms_adam_lr0p02_s100"    "adam" "0.02" "1e-4" "1" "4" "6" "16" "5e-4" "0.30" "0.05" "100" "1"
submit_job "hg_ms"     "ms_adam_lr0p05_s100"    "adam" "0.05" "1e-4" "1" "4" "6" "16" "5e-4" "0.30" "0.05" "100" "1"
submit_job "hg_ms"     "ms_adam_lr0p10_s100"    "adam" "0.10" "1e-4" "1" "4" "6" "16" "5e-4" "0.30" "0.05" "100" "1"
submit_job "hg_ms"     "ms_adam_lr0p05_s200"    "adam" "0.05" "1e-4" "1" "4" "6" "16" "5e-4" "0.30" "0.05" "200" "1"
submit_job "hg_ms"     "ms_adam_lr0p10_s200"    "adam" "0.10" "1e-4" "1" "4" "6" "16" "5e-4" "0.30" "0.05" "200" "1"
submit_job "hg_ms"     "ms_adam_lr0p05_m8_s100" "adam" "0.05" "1e-4" "1" "8" "6" "16" "5e-4" "0.30" "0.05" "100" "1"
submit_job "hg_ms"     "ms_adam_lr0p05_ml2"     "adam" "0.05" "2e-4" "1" "4" "6" "16" "5e-4" "0.30" "0.05" "100" "2"
