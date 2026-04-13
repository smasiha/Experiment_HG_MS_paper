#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_PRESET="${MODEL_PRESET:-qwen2_7b}"
SEEDS="${SEEDS:-2424}"
REWEIGHT_MODEL_LR="${REWEIGHT_MODEL_LR:-1e-5}"
TAG="${TAG:-faithful_table4_${MODEL_PRESET}_scalebio_canary_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

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

cd "${ROOT_DIR}"

echo "Model preset: ${MODEL_PRESET}"
echo "Model name: ${MODEL_NAME}"
echo "Seeds: ${SEEDS}"
echo "Reweight model lr: ${REWEIGHT_MODEL_LR}"
echo "Output root: ${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  run_name="scalebio_seed${seed}_${MODEL_PRESET}_faithful_lr${REWEIGHT_MODEL_LR//./p}"
  hf_cache_root="/scratch/${USER}/hf_cache_jobs/faithful_${MODEL_PRESET}_scalebio/${run_name}"
  sbatch \
    --job-name="fs_${MODEL_PRESET}_sbo_${seed}" \
    --export=ALL,\
MODEL_NAME="${MODEL_NAME}",\
SEED="${seed}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
HF_CACHE_ROOT="${hf_cache_root}",\
MAX_LENGTH=1024,\
REWEIGHT_TRAIN_PER_SOURCE=3000,\
TRAIN_RAW_OVERSAMPLE=4.0,\
VAL_PER_SOURCE=280,\
VAL_RAW_OVERSAMPLE=1.2,\
REWEIGHT_EPOCHS=3,\
PER_SOURCE_BATCH=1,\
VAL_BATCH_SIZE=8,\
REWEIGHT_MODEL_LR="${REWEIGHT_MODEL_LR}",\
REWEIGHT_WEIGHT_DECAY=0.0,\
REWEIGHT_OUTER_LR=1e-2,\
SCALEBIO_ALPHA=100.0,\
TUNE_EVERY=50,\
FINAL_TRAIN_SIZE=20000,\
FINAL_FT_LR=8e-6,\
FINAL_FT_EPOCHS=1,\
PER_DEVICE_BATCH_SIZE=1,\
GRAD_ACCUM=8,\
EVAL_BATCH_SIZE=4,\
GSM8K_EVAL_SIZE=0,\
GRADIENT_CHECKPOINTING=1 \
    run_faithful_scalebio_math_fsdp.sbatch
done
