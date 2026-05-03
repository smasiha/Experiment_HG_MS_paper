#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_PRESET="${MODEL_PRESET:-llama3_8b}"
SEED="${SEED:-2424}"
TAG="${TAG:-math_hg_finalization_${MODEL_PRESET}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

source "${ROOT_DIR}/model_presets.sh"
apply_math_model_preset "${MODEL_PRESET}"

cd "${ROOT_DIR}"

submit_job() {
  local method="$1"
  local run_name="$2"
  local outer_lr="$3"
  local model_lr="$4"
  local inner_steps="$5"
  local m_branches="$6"
  local selection_batches="$7"
  local hypergrad_batches="$8"
  local selection_batch_size="$9"
  local branch_noise="${10}"
  local train_tolerance="${11}"
  local train_penalty="${12}"

  sbatch \
    --job-name="chk_${method}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
HF_CACHE_ROOT="/scratch/${USER}/hf_cache_jobs/${MODEL_PRESET}/${run_name}",\
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY}",\
STOP_AFTER_REWEIGHT=1,\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
VAL_KINDS_SPEC=gsm8k,\
MAX_LENGTH="${MAX_LENGTH}",\
VAL_BATCH_SIZE="${VAL_BATCH_SIZE}",\
SELECTION_VAL_BATCH_SIZE="${selection_batch_size}",\
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}",\
REWEIGHT_MODEL_LR="${model_lr}",\
REWEIGHT_OUTER_LR="${outer_lr}",\
BRANCH_NOISE_STD="${branch_noise}",\
REWEIGHT_TRAIN_PER_SOURCE=500,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH="${PER_SOURCE_BATCH}",\
REWEIGHT_STEPS=100,\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${hypergrad_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${train_tolerance}",\
SELECTION_TRAIN_PENALTY="${train_penalty}",\
INNER_STEPS="${inner_steps}",\
SELECTION_INNER_STEPS="${inner_steps}",\
HYPERGRAD_INNER_STEPS="${inner_steps}",\
M_BRANCHES="${m_branches}",\
FINAL_TRAIN_SIZE=5000,\
FINAL_FT_MICRO_BATCH="${FINAL_FT_MICRO_BATCH}",\
FINAL_FT_GRAD_ACCUM="${FINAL_FT_GRAD_ACCUM}",\
GSM8K_EVAL_SIZE=256,\
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE}",\
REFERENCE_FT_STEPS=100,\
LORA_R="${LORA_R}",\
LORA_ALPHA="${LORA_ALPHA}",\
LORA_DROPOUT="${LORA_DROPOUT}" \
    run_exact_math_reasoning.sbatch
}

echo "Model preset: ${MODEL_PRESET}"
echo "Model: ${MODEL_NAME}"
echo "Reference model: ${REFERENCE_MODEL_NAME}"
echo "Seed: ${SEED}"
echo "Output root: ${OUTPUT_ROOT}"

# HG-single: compare the old stable model LR with the recent low-LR setting that degenerated.
submit_job "hg_single" "hg_single_mlr1e4_check" "5.0" "1e-4" "1" "1" "4" "4" "2" "0" "0.05" "0.0"
submit_job "hg_single" "hg_single_mlr1e5_check" "5.0" "1e-5" "1" "1" "4" "4" "2" "0" "0.05" "0.0"

# HG-MS: current best adapted configuration from the paper draft.
submit_job "hg_ms" "hg_ms_curr_check" "10.0" "1e-4" "2" "4" "6" "6" "2" "0" "0.30" "0.25"
