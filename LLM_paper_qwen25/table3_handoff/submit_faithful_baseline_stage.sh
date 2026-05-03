#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_PRESET="${MODEL_PRESET:-qwen2_7b}"
SEEDS="${SEEDS:-2424}"
METHODS="${METHODS:-less rho_loss}"
STAGE="${STAGE:-reweights}"
TAG="${TAG:-faithful_math_baselines_${MODEL_PRESET}_$(date +%Y%m%d_%H%M%S)}"
REWEIGHT_OUTPUT_ROOT="${REWEIGHT_OUTPUT_ROOT:-/scratch/${USER}/${TAG}/reweight}"
FINAL_OUTPUT_ROOT="${FINAL_OUTPUT_ROOT:-/scratch/${USER}/${TAG}/final}"
REFERENCE_MODEL_SHARED="${REFERENCE_MODEL_SHARED:-Qwen/Qwen2-1.5B}"
BASELINE_SCORE_SAMPLES_PER_SOURCE="${BASELINE_SCORE_SAMPLES_PER_SOURCE:-0}"

case "${MODEL_PRESET}" in
  llama3_8b)
    MODEL_NAME="meta-llama/Meta-Llama-3-8B"
    ;;
  qwen2_7b)
    MODEL_NAME="Qwen/Qwen2-7B"
    ;;
  gemma2_9b)
    MODEL_NAME="google/gemma-2-9b"
    ;;
  *)
    echo "Unknown MODEL_PRESET: ${MODEL_PRESET}" >&2
    exit 1
    ;;
esac

mkdir -p "${REWEIGHT_OUTPUT_ROOT}" "${FINAL_OUTPUT_ROOT}"
cd "${ROOT_DIR}"

submit_reweight() {
  local method="$1"
  local seed="$2"
  local reference_model_name="${REFERENCE_MODEL_SHARED}"
  local reference_ft_epochs="1"
  local reweight_run_name="${method}_seed${seed}_${MODEL_PRESET}_faithful_reweight"
  local reweight_hf_cache_root="/scratch/${USER}/hf_cache_jobs/${MODEL_PRESET}/${reweight_run_name}"

  if [[ "${method}" == "less" ]]; then
    reference_model_name="${MODEL_NAME}"
    reference_ft_epochs="4"
  fi

  sbatch \
    --job-name="fb_${MODEL_PRESET}_${method}_${seed}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${reference_model_name}",\
HF_CACHE_ROOT="${reweight_hf_cache_root}",\
SEED="${seed}",\
OUTPUT_ROOT="${REWEIGHT_OUTPUT_ROOT}",\
RUN_NAME="${reweight_run_name}",\
LOCAL_MODEL_ONLY=0,\
STOP_AFTER_REWEIGHT=1,\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
MAX_LENGTH=1024,\
REWEIGHT_TRAIN_PER_SOURCE=3000,\
VAL_PER_SOURCE=280,\
TRAIN_RAW_OVERSAMPLE=4.0,\
VAL_RAW_OVERSAMPLE=1.2,\
REFERENCE_FT_EPOCHS="${reference_ft_epochs}",\
REFERENCE_BATCH_SIZE=1,\
REFERENCE_GRAD_ACCUM=32,\
REFERENCE_FT_LR=1e-5,\
BASELINE_SCORE_SAMPLES_PER_SOURCE="${BASELINE_SCORE_SAMPLES_PER_SOURCE}",\
LORA_R=128,\
LORA_ALPHA=512,\
LORA_DROPOUT=0.1,\
GRADIENT_CHECKPOINTING=1 \
    run_exact_math_reasoning.sbatch
}

submit_final() {
  local method="$1"
  local seed="$2"
  local final_run_name="${method}_seed${seed}_${MODEL_PRESET}_faithful"
  local final_hf_cache_root="/scratch/${USER}/hf_cache_jobs/faithful_${method}/${final_run_name}"
  local reweight_metrics_path="${REWEIGHT_OUTPUT_ROOT}/${method}_seed${seed}_${MODEL_PRESET}_faithful_reweight/metrics.json"

  if [[ ! -f "${reweight_metrics_path}" ]]; then
    echo "Missing reweight metrics: ${reweight_metrics_path}" >&2
    return 1
  fi

  sbatch \
    --job-name="ff_${MODEL_PRESET}_${method}_${seed}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
HF_CACHE_ROOT="${final_hf_cache_root}",\
SEED="${seed}",\
OUTPUT_ROOT="${FINAL_OUTPUT_ROOT}",\
RUN_NAME="${final_run_name}",\
REWEIGHT_METRICS_PATH="${reweight_metrics_path}",\
MAX_LENGTH=1024,\
REWEIGHT_TRAIN_PER_SOURCE=3000,\
VAL_PER_SOURCE=280,\
TRAIN_RAW_OVERSAMPLE=4.0,\
VAL_RAW_OVERSAMPLE=1.2,\
FINAL_TRAIN_SIZE=20000,\
FINAL_FT_LR=8e-6,\
FINAL_FT_EPOCHS=1,\
PER_DEVICE_BATCH_SIZE=1,\
GRAD_ACCUM=8,\
EVAL_BATCH_SIZE=4,\
GSM8K_EVAL_SIZE=0,\
GRADIENT_CHECKPOINTING=1 \
    run_faithful_weighted_math_fsdp.sbatch
}

echo "Model preset: ${MODEL_PRESET}"
echo "Model name: ${MODEL_NAME}"
echo "Methods: ${METHODS}"
echo "Seeds: ${SEEDS}"
echo "Stage: ${STAGE}"
echo "Reweight output root: ${REWEIGHT_OUTPUT_ROOT}"
echo "Final output root: ${FINAL_OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    case "${STAGE}" in
      reweights)
        submit_reweight "${method}" "${seed}"
        ;;
      finals)
        submit_final "${method}" "${seed}"
        ;;
      both)
        echo "STAGE=both is intentionally unsupported in the low-cost handoff flow." >&2
        exit 1
        ;;
      *)
        echo "Unknown STAGE: ${STAGE}" >&2
        exit 1
        ;;
    esac
  done
done
