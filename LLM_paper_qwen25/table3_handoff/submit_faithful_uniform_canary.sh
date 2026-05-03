#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_PRESET="${MODEL_PRESET:-qwen2_7b}"
SEEDS="${SEEDS:-2424}"
TAG="${TAG:-faithful_table4_${MODEL_PRESET}_uniform_canary_$(date +%Y%m%d_%H%M%S)}"
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
echo "Output root: ${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  run_name="uniform_seed${seed}_${MODEL_PRESET}_faithful"
  hf_cache_root="/scratch/${USER}/hf_cache_jobs/faithful_${MODEL_PRESET}_uniform/${run_name}"
  sbatch \
    --job-name="fu_${MODEL_PRESET}_uni_${seed}" \
    --export=ALL,\
MODEL_NAME="${MODEL_NAME}",\
SEED="${seed}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
HF_CACHE_ROOT="${hf_cache_root}",\
MAX_LENGTH=1024,\
FINAL_TRAIN_SIZE=20000,\
FINAL_FT_LR=8e-6,\
FINAL_FT_EPOCHS=1,\
PER_DEVICE_BATCH_SIZE=1,\
GRAD_ACCUM=8,\
EVAL_BATCH_SIZE=4,\
GSM8K_EVAL_SIZE=0,\
REWEIGHT_TRAIN_PER_SOURCE=3000,\
TRAIN_RAW_OVERSAMPLE=4.0,\
VAL_PER_SOURCE=280,\
VAL_RAW_OVERSAMPLE=1.2,\
GRADIENT_CHECKPOINTING=1 \
    run_faithful_uniform_math_fsdp.sbatch
done
