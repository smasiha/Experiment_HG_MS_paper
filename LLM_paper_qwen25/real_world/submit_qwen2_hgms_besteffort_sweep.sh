#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_PRESET="qwen2_7b"
SEED="${SEED:-2424}"
TAG="${TAG:-qwen2_7b_hgms_besteffort_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

source "${ROOT_DIR}/model_presets.sh"
apply_math_model_preset "${MODEL_PRESET}"

cd "${ROOT_DIR}"

CACHE_PATH="${CACHE_PATH:-/scratch/${USER}/hypergrad_exact_cache/exact_splits_seed${SEED}_train500_val280_tov4.0_vov1.2.json}"

submit_cfg() {
  local suffix="$1"
  local tolerance="$2"
  local penalty="$3"

  local run_name="hg_ms_${suffix}_seed${SEED}_${MODEL_PRESET}_gsm8k"
  sbatch \
    --job-name="qwen_hgms_${suffix}" \
    --export=ALL,\
METHOD=hg_ms,\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
HF_CACHE_ROOT="/scratch/${USER}/hf_cache_jobs/${MODEL_PRESET}/${run_name}",\
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY}",\
STOP_AFTER_REWEIGHT=0,\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
VAL_KINDS_SPEC=gsm8k,\
MAX_LENGTH="${MAX_LENGTH}",\
VAL_BATCH_SIZE="${VAL_BATCH_SIZE}",\
SELECTION_VAL_BATCH_SIZE=1,\
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}",\
REWEIGHT_MODEL_LR=1e-4,\
REWEIGHT_OUTER_LR=10.0,\
BRANCH_NOISE_STD=0,\
REWEIGHT_TRAIN_PER_SOURCE=500,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH=1,\
REWEIGHT_STEPS=100,\
REWEIGHT_WALL_CLOCK_SECONDS=0,\
SELECTION_VAL_BATCHES=4,\
HYPERGRAD_VAL_BATCHES=4,\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${tolerance}",\
SELECTION_TRAIN_PENALTY="${penalty}",\
INNER_STEPS=1,\
SELECTION_INNER_STEPS=1,\
HYPERGRAD_INNER_STEPS=1,\
M_BRANCHES=2,\
FINAL_TRAIN_SIZE=20000,\
FINAL_FT_MICRO_BATCH="${FINAL_FT_MICRO_BATCH}",\
FINAL_FT_GRAD_ACCUM="${FINAL_FT_GRAD_ACCUM}",\
GSM8K_EVAL_SIZE=0,\
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE}",\
REFERENCE_FT_STEPS=100,\
LORA_R="${LORA_R}",\
LORA_ALPHA="${LORA_ALPHA}",\
LORA_DROPOUT="${LORA_DROPOUT}",\
CACHE_PATH="${CACHE_PATH}" \
    run_exact_math_reasoning.sbatch
}

echo "Model preset: ${MODEL_PRESET}"
echo "Model: ${MODEL_NAME}"
echo "Seed: ${SEED}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Cache path: ${CACHE_PATH}"

# Best current run is tolerance=0.30, penalty=0.20; we search nearby settings only.
submit_cfg "tol20_pen20" "0.20" "0.20"
submit_cfg "tol40_pen20" "0.40" "0.20"
submit_cfg "tol30_pen15" "0.30" "0.15"
submit_cfg "tol30_pen10" "0.30" "0.10"
