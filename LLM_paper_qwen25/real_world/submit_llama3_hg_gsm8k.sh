#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_PRESET="llama3_8b"
SEED="${SEED:-2424}"
METHODS="${METHODS:-hg_single hg_ms}"
MODE="${MODE:-full}"
TAG="${TAG:-llama3_8b_hg_gsm8k_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS:-3600}"

source "${ROOT_DIR}/model_presets.sh"
apply_math_model_preset "${MODEL_PRESET}"

cd "${ROOT_DIR}"

submit_job() {
  local method="$1"
  local stop_after_reweight="0"
  local final_train_size="20000"
  local gsm8k_eval_size="0"
  local run_name="${method}_seed${SEED}_${MODEL_PRESET}_gsm8k"

  local reweight_model_lr="1e-4"
  local reweight_outer_lr="500.0"
  local inner_steps="1"
  local selection_inner_steps="1"
  local hypergrad_inner_steps="1"
  local m_branches="1"
  local selection_batches="1"
  local hypergrad_batches="1"
  local selection_batch_size="1"
  local train_tolerance="0.05"
  local train_penalty="0.0"

  case "${method}" in
    hg_single)
      reweight_outer_lr="500.0"
      ;;
    hg_ms)
      reweight_outer_lr="1000.0"
      m_branches="4"
      train_tolerance="0.30"
      train_penalty="0.05"
      ;;
    *)
      echo "Unsupported method: ${method}" >&2
      exit 1
      ;;
  esac

  case "${MODE}" in
    proof)
      stop_after_reweight="1"
      final_train_size="5000"
      gsm8k_eval_size="256"
      ;;
    full)
      ;;
    *)
      echo "Unsupported MODE: ${MODE}" >&2
      exit 1
      ;;
  esac

  sbatch \
    --job-name="llama_${method}_${MODE}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
HF_CACHE_ROOT="/scratch/${USER}/hf_cache_jobs/${MODEL_PRESET}/${run_name}",\
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY}",\
STOP_AFTER_REWEIGHT="${stop_after_reweight}",\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
VAL_KINDS_SPEC=gsm8k,\
MAX_LENGTH="${MAX_LENGTH}",\
VAL_BATCH_SIZE="${VAL_BATCH_SIZE}",\
SELECTION_VAL_BATCH_SIZE="${selection_batch_size}",\
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}",\
REWEIGHT_MODEL_LR="${reweight_model_lr}",\
REWEIGHT_OUTER_LR="${reweight_outer_lr}",\
BRANCH_NOISE_STD=0,\
REWEIGHT_TRAIN_PER_SOURCE=500,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH=1,\
REWEIGHT_STEPS=100000,\
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS}",\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${hypergrad_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${train_tolerance}",\
SELECTION_TRAIN_PENALTY="${train_penalty}",\
INNER_STEPS="${inner_steps}",\
SELECTION_INNER_STEPS="${selection_inner_steps}",\
HYPERGRAD_INNER_STEPS="${hypergrad_inner_steps}",\
M_BRANCHES="${m_branches}",\
FINAL_TRAIN_SIZE="${final_train_size}",\
FINAL_FT_MICRO_BATCH="${FINAL_FT_MICRO_BATCH}",\
FINAL_FT_GRAD_ACCUM="${FINAL_FT_GRAD_ACCUM}",\
GSM8K_EVAL_SIZE="${gsm8k_eval_size}",\
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
echo "Methods: ${METHODS}"
echo "Mode: ${MODE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Reweight/search budget (sec): ${REWEIGHT_WALL_CLOCK_SECONDS}"

for method in ${METHODS}; do
  submit_job "${method}"
done
