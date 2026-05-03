#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-2424}"
MODE="${MODE:-smoke}"
METHODS="${METHODS:-uniform scalebio_transfer hg_ms_transfer}"
TAG="${TAG:-large_qwen25_32b_fsdp_final_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct}"
HGMS_TRANSFER_METRICS_PATH="${HGMS_TRANSFER_METRICS_PATH:-/scratch/${USER}/hg_math_nonuniform_qwen_budget_20260425_111706/hg_ms_seed2424_qwen2_7b_gsm8k_math_nonuniform_9h/metrics.json}"
SCALEBIO_TRANSFER_METRICS_PATH="${SCALEBIO_TRANSFER_METRICS_PATH:-/scratch/${USER}/hg_math_nonuniform_qwen_budget_20260425_111706/scalebio_seed2424_qwen2_7b_gsm8k_math_9h/metrics.json}"

case "${MODE}" in
  smoke)
    SBATCH_TIME="${SBATCH_TIME:-02:00:00}"
    FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-256}"
    REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-80}"
    VAL_PER_SOURCE="${VAL_PER_SOURCE:-30}"
    GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE:-32}"
    MATH_EVAL_SIZE="${MATH_EVAL_SIZE:-32}"
    EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
    ;;
  full)
    SBATCH_TIME="${SBATCH_TIME:-2-00:00:00}"
    FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-20000}"
    REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-500}"
    VAL_PER_SOURCE="${VAL_PER_SOURCE:-280}"
    GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE:-0}"
    MATH_EVAL_SIZE="${MATH_EVAL_SIZE:-0}"
    EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use MODE=smoke or MODE=full." >&2
    exit 1
    ;;
esac

MAX_LENGTH="${MAX_LENGTH:-768}"
TRAIN_RAW_OVERSAMPLE="${TRAIN_RAW_OVERSAMPLE:-4.0}"
VAL_RAW_OVERSAMPLE="${VAL_RAW_OVERSAMPLE:-1.2}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"

SBATCH_NODES="${SBATCH_NODES:-1}"
SBATCH_GPUS_PER_NODE="${SBATCH_GPUS_PER_NODE:-4}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-48}"
SBATCH_MEM="${SBATCH_MEM:-256G}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

submit_uniform() {
  local run_name="uniform_qwen25_32b_fsdp_${MODE}"
  echo "Submitting uniform -> ${OUTPUT_ROOT}/${run_name}"
  sbatch \
    --job-name="lg32_uniform_fsdp" \
    --nodes="${SBATCH_NODES}" \
    --gres="gpu:${SBATCH_GPUS_PER_NODE}" \
    --cpus-per-task="${SBATCH_CPUS_PER_TASK}" \
    --mem="${SBATCH_MEM}" \
    --time="${SBATCH_TIME}" \
    --export=ALL,\
ROOT_DIR="${SCRIPT_DIR}",\
MODEL_NAME="${MODEL_NAME}",\
SEED="${SEED}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
HF_CACHE_ROOT="/scratch/${USER}/hf_cache_jobs/qwen25_32b_fsdp/${run_name}",\
MAX_LENGTH="${MAX_LENGTH}",\
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE}",\
REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE}",\
TRAIN_RAW_OVERSAMPLE="${TRAIN_RAW_OVERSAMPLE}",\
VAL_PER_SOURCE="${VAL_PER_SOURCE}",\
VAL_RAW_OVERSAMPLE="${VAL_RAW_OVERSAMPLE}",\
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}",\
GRAD_ACCUM="${GRAD_ACCUM}",\
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}",\
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}",\
GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE}",\
MATH_EVAL_SIZE="${MATH_EVAL_SIZE}",\
EVAL_MATH=1,\
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL}",\
NPROC_PER_NODE="${NPROC_PER_NODE}",\
GRADIENT_CHECKPOINTING=1 \
    "${SCRIPT_DIR}/run_faithful_uniform_math_fsdp.sbatch"
}

submit_hgms_transfer() {
  local run_name="hg_ms_transfer_qwen25_32b_fsdp_${MODE}"
  echo "Submitting HG-MS transfer -> ${OUTPUT_ROOT}/${run_name}"
  sbatch \
    --job-name="lg32_hgms_xfer_fsdp" \
    --nodes="${SBATCH_NODES}" \
    --gres="gpu:${SBATCH_GPUS_PER_NODE}" \
    --cpus-per-task="${SBATCH_CPUS_PER_TASK}" \
    --mem="${SBATCH_MEM}" \
    --time="${SBATCH_TIME}" \
    --export=ALL,\
ROOT_DIR="${SCRIPT_DIR}",\
METHOD="hg_ms_transfer",\
MODEL_NAME="${MODEL_NAME}",\
SEED="${SEED}",\
REWEIGHT_METRICS_PATH="${HGMS_TRANSFER_METRICS_PATH}",\
LAMBDA_KEY="final_train_lambda",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
HF_CACHE_ROOT="/scratch/${USER}/hf_cache_jobs/qwen25_32b_fsdp/${run_name}",\
MAX_LENGTH="${MAX_LENGTH}",\
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE}",\
REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE}",\
TRAIN_RAW_OVERSAMPLE="${TRAIN_RAW_OVERSAMPLE}",\
VAL_PER_SOURCE="${VAL_PER_SOURCE}",\
VAL_RAW_OVERSAMPLE="${VAL_RAW_OVERSAMPLE}",\
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}",\
GRAD_ACCUM="${GRAD_ACCUM}",\
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}",\
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}",\
GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE}",\
MATH_EVAL_SIZE="${MATH_EVAL_SIZE}",\
EVAL_MATH=1,\
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL}",\
NPROC_PER_NODE="${NPROC_PER_NODE}",\
GRADIENT_CHECKPOINTING=1 \
    "${SCRIPT_DIR}/run_faithful_weighted_math_fsdp.sbatch"
}

submit_scalebio_transfer() {
  local run_name="scalebio_transfer_qwen25_32b_fsdp_${MODE}"
  echo "Submitting ScaleBiO transfer -> ${OUTPUT_ROOT}/${run_name}"
  sbatch \
    --job-name="lg32_sbio_xfer_fsdp" \
    --nodes="${SBATCH_NODES}" \
    --gres="gpu:${SBATCH_GPUS_PER_NODE}" \
    --cpus-per-task="${SBATCH_CPUS_PER_TASK}" \
    --mem="${SBATCH_MEM}" \
    --time="${SBATCH_TIME}" \
    --export=ALL,\
ROOT_DIR="${SCRIPT_DIR}",\
METHOD="scalebio_transfer",\
MODEL_NAME="${MODEL_NAME}",\
SEED="${SEED}",\
REWEIGHT_METRICS_PATH="${SCALEBIO_TRANSFER_METRICS_PATH}",\
LAMBDA_KEY="final_train_lambda",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
HF_CACHE_ROOT="/scratch/${USER}/hf_cache_jobs/qwen25_32b_fsdp/${run_name}",\
MAX_LENGTH="${MAX_LENGTH}",\
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE}",\
REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE}",\
TRAIN_RAW_OVERSAMPLE="${TRAIN_RAW_OVERSAMPLE}",\
VAL_PER_SOURCE="${VAL_PER_SOURCE}",\
VAL_RAW_OVERSAMPLE="${VAL_RAW_OVERSAMPLE}",\
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}",\
GRAD_ACCUM="${GRAD_ACCUM}",\
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}",\
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}",\
GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE}",\
MATH_EVAL_SIZE="${MATH_EVAL_SIZE}",\
EVAL_MATH=1,\
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL}",\
NPROC_PER_NODE="${NPROC_PER_NODE}",\
GRADIENT_CHECKPOINTING=1 \
    "${SCRIPT_DIR}/run_faithful_weighted_math_fsdp.sbatch"
}

echo "Mode: ${MODE}"
echo "Methods: ${METHODS}"
echo "Model: ${MODEL_NAME}"
echo "Output root: ${OUTPUT_ROOT}"
echo "FSDP shape: nodes=${SBATCH_NODES}, gpus/node=${SBATCH_GPUS_PER_NODE}, nproc/node=${NPROC_PER_NODE}"
echo "Skip final eval: ${SKIP_FINAL_EVAL}"

for method in ${METHODS}; do
  case "${method}" in
    uniform) submit_uniform ;;
    scalebio_transfer) submit_scalebio_transfer ;;
    hg_ms_transfer) submit_hgms_transfer ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 1
      ;;
  esac
done
