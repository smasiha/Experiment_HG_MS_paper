#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
SEED="${SEED:-2424}"
MODELS="${MODELS:-llama3_8b qwen2_7b gemma2_9b}"
METHODS="${METHODS:-scalebio hg_single hg_ms}"
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS:-10800}"
TAG="${TAG:-iterative_longbudget_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

source "${ROOT_DIR}/model_presets.sh"
cd "${ROOT_DIR}"

submit_job() {
  local model_preset="$1"
  local method="$2"

  unset MODEL_NAME REFERENCE_MODEL_NAME MAX_LENGTH LORA_R LORA_ALPHA LORA_DROPOUT \
    FINAL_FT_MICRO_BATCH FINAL_FT_GRAD_ACCUM VAL_BATCH_SIZE SELECTION_VAL_BATCH_SIZE \
    EVAL_BATCH_SIZE PER_SOURCE_BATCH REFERENCE_BATCH_SIZE LOCAL_MODEL_ONLY
  apply_math_model_preset "${model_preset}"

  local model_lr outer_lr branches sel_batches hyp_batches tolerance penalty
  local run_name="${method}_seed${SEED}_${model_preset}_gsm8k"

  case "${model_preset}:${method}" in
    llama3_8b:scalebio|qwen2_7b:scalebio|gemma2_9b:scalebio)
      model_lr="1e-4"; outer_lr="1.0"; branches="1"; sel_batches="1"; hyp_batches="1"; tolerance="0.05"; penalty="0.0"
      ;;
    llama3_8b:hg_single)
      model_lr="1e-5"; outer_lr="5.0"; branches="1"; sel_batches="4"; hyp_batches="4"; tolerance="0.05"; penalty="0.0"
      ;;
    llama3_8b:hg_ms)
      model_lr="1e-4"; outer_lr="10.0"; branches="2"; sel_batches="4"; hyp_batches="4"; tolerance="0.30"; penalty="0.25"
      ;;
    qwen2_7b:hg_single)
      model_lr="1e-4"; outer_lr="5.0"; branches="1"; sel_batches="4"; hyp_batches="4"; tolerance="0.05"; penalty="0.0"
      ;;
    qwen2_7b:hg_ms)
      model_lr="1e-4"; outer_lr="10.0"; branches="2"; sel_batches="4"; hyp_batches="4"; tolerance="0.30"; penalty="0.20"
      ;;
    gemma2_9b:hg_single)
      model_lr="1e-4"; outer_lr="500.0"; branches="1"; sel_batches="1"; hyp_batches="1"; tolerance="0.05"; penalty="0.0"
      ;;
    gemma2_9b:hg_ms)
      model_lr="1e-4"; outer_lr="1000.0"; branches="4"; sel_batches="1"; hyp_batches="1"; tolerance="0.30"; penalty="0.05"
      ;;
    *)
      echo "Unsupported config: ${model_preset}:${method}" >&2
      exit 1
      ;;
  esac

  sbatch \
    --job-name="${model_preset}_${method}_long" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
HF_CACHE_ROOT="/scratch/${USER}/hf_cache_jobs/${model_preset}/${run_name}",\
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
REWEIGHT_MODEL_LR="${model_lr}",\
REWEIGHT_OUTER_LR="${outer_lr}",\
BRANCH_NOISE_STD=0,\
REWEIGHT_TRAIN_PER_SOURCE=500,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH=1,\
REWEIGHT_STEPS=100000,\
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS}",\
SELECTION_VAL_BATCHES="${sel_batches}",\
HYPERGRAD_VAL_BATCHES="${hyp_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${tolerance}",\
SELECTION_TRAIN_PENALTY="${penalty}",\
INNER_STEPS=1,\
SELECTION_INNER_STEPS=1,\
HYPERGRAD_INNER_STEPS=1,\
M_BRANCHES="${branches}",\
FINAL_TRAIN_SIZE=20000,\
FINAL_FT_MICRO_BATCH="${FINAL_FT_MICRO_BATCH}",\
FINAL_FT_GRAD_ACCUM="${FINAL_FT_GRAD_ACCUM}",\
GSM8K_EVAL_SIZE=0,\
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE}",\
REFERENCE_FT_STEPS=100,\
SCALEBIO_ALPHA=10.0,\
LORA_R="${LORA_R}",\
LORA_ALPHA="${LORA_ALPHA}",\
LORA_DROPOUT="${LORA_DROPOUT}",\
CACHE_PATH="/scratch/${USER}/hypergrad_exact_cache/exact_splits_seed${SEED}_train500_val280_tov4.0_vov1.2.json" \
    run_exact_math_reasoning.sbatch
}

echo "Output root: ${OUTPUT_ROOT}"
echo "Seed: ${SEED}"
echo "Models: ${MODELS}"
echo "Methods: ${METHODS}"
echo "Reweight budget (sec): ${REWEIGHT_WALL_CLOCK_SECONDS}"

for model in ${MODELS}; do
  for method in ${METHODS}; do
    submit_job "${model}" "${method}"
  done
done
