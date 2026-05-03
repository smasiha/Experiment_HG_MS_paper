#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/..}"
SEED="${SEED:-2424}"
MODE="${MODE:-smoke}"
METHODS="${METHODS:-hg_ms}"
TAG="${TAG:-large_qwen25_32b_math_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
HGMS_TRANSFER_METRICS_PATH="${HGMS_TRANSFER_METRICS_PATH:-/scratch/${USER}/hg_math_nonuniform_qwen_budget_20260425_111706/hg_ms_seed2424_qwen2_7b_gsm8k_math_nonuniform_9h/metrics.json}"

case "${MODE}" in
  smoke)
    SBATCH_TIME="${SBATCH_TIME:-08:00:00}"
    SBATCH_MEM="${SBATCH_MEM:-90G}"
    SBATCH_CPUS="${SBATCH_CPUS:-16}"
    MAX_LENGTH="${MAX_LENGTH:-768}"
    REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-80}"
    VAL_PER_SOURCE="${VAL_PER_SOURCE:-30}"
    REWEIGHT_STEPS="${REWEIGHT_STEPS:-20}"
    REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS:-1800}"
    FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-256}"
    GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE:-32}"
    MATH_EVAL_SIZE="${MATH_EVAL_SIZE:-32}"
    EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
    LORA_R="${LORA_R:-4}"
    LORA_ALPHA="${LORA_ALPHA:-8}"
    ;;
  full)
    SBATCH_TIME="${SBATCH_TIME:-2-00:00:00}"
    SBATCH_MEM="${SBATCH_MEM:-90G}"
    SBATCH_CPUS="${SBATCH_CPUS:-16}"
    MAX_LENGTH="${MAX_LENGTH:-768}"
    REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-500}"
    VAL_PER_SOURCE="${VAL_PER_SOURCE:-280}"
    REWEIGHT_STEPS="${REWEIGHT_STEPS:-100000}"
    REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS:-32400}"
    FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-20000}"
    GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE:-0}"
    MATH_EVAL_SIZE="${MATH_EVAL_SIZE:-0}"
    EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
    LORA_R="${LORA_R:-8}"
    LORA_ALPHA="${LORA_ALPHA:-16}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use MODE=smoke or MODE=full." >&2
    exit 1
    ;;
esac

MAX_LENGTH="${MAX_LENGTH:-768}"
TRAIN_RAW_OVERSAMPLE="${TRAIN_RAW_OVERSAMPLE:-4.0}"
VAL_RAW_OVERSAMPLE="${VAL_RAW_OVERSAMPLE:-1.2}"
CACHE_PATH="${CACHE_PATH:-/scratch/${USER}/hypergrad_exact_cache/qwen25_32b_seed${SEED}_train${REWEIGHT_TRAIN_PER_SOURCE}_val${VAL_PER_SOURCE}_tov${TRAIN_RAW_OVERSAMPLE}_vov${VAL_RAW_OVERSAMPLE}.json}"

submit_job() {
  local method_key="$1"
  local method=""
  local run_name=""
  local extra_export=""
  local job_name=""

  case "${method_key}" in
    uniform)
      method="uniform"
      run_name="uniform_qwen25_32b_${MODE}"
      job_name="lg32_uniform"
      ;;
    hg_ms_transfer)
      method="fixed_lambda"
      run_name="hg_ms_transfer_qwen25_32b_${MODE}"
      job_name="lg32_hgms_xfer"
      extra_export=",INITIAL_LAMBDA_METRICS_PATH=${HGMS_TRANSFER_METRICS_PATH},INITIAL_LAMBDA_KEY=final_train_lambda,INITIAL_LAMBDA_BLEND=1.0"
      ;;
    scalebio)
      method="scalebio"
      run_name="scalebio_qwen25_32b_${MODE}"
      job_name="lg32_scalebio"
      extra_export=",SCALEBIO_ALPHA=10.0"
      ;;
    hg_ms)
      method="hg_ms"
      run_name="hg_ms_qwen25_32b_${MODE}"
      job_name="lg32_hgms"
      extra_export=",THETA_OPTIMIZER=adam,THETA_INIT_STD=1.0,M_BRANCHES=4,SELECTION_VAL_BATCHES=4,HYPERGRAD_VAL_BATCHES=4,SELECTION_TRAIN_TOLERANCE=0.05,SELECTION_TRAIN_PENALTY=0.0"
      ;;
    *)
      echo "Unknown method key: ${method_key}" >&2
      exit 1
      ;;
  esac

  echo "Submitting ${method_key} (${method}) -> ${OUTPUT_ROOT}/${run_name}"
  sbatch \
    --job-name="${job_name}" \
    --time="${SBATCH_TIME}" \
    --mem="${SBATCH_MEM}" \
    --cpus-per-task="${SBATCH_CPUS}" \
    --export=ALL,\
ROOT_DIR="${ROOT_DIR}",\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${SEED}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
HF_CACHE_ROOT="/scratch/${USER}/hf_cache_jobs/qwen25_32b/${run_name}",\
LOCAL_MODEL_ONLY=0,\
CACHE_PATH="${CACHE_PATH}",\
VAL_KINDS_SPEC=gsm8k+math,\
MAX_LENGTH="${MAX_LENGTH}",\
ATTN_IMPLEMENTATION=sdpa,\
GRADIENT_CHECKPOINTING=1,\
LORA_R="${LORA_R}",\
LORA_ALPHA="${LORA_ALPHA}",\
LORA_DROPOUT=0.05,\
REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE}",\
VAL_PER_SOURCE="${VAL_PER_SOURCE}",\
TRAIN_RAW_OVERSAMPLE="${TRAIN_RAW_OVERSAMPLE}",\
VAL_RAW_OVERSAMPLE="${VAL_RAW_OVERSAMPLE}",\
ALLOW_TRAIN_SHORTFALL=1,\
REWEIGHT_STEPS="${REWEIGHT_STEPS}",\
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS}",\
REWEIGHT_MODEL_LR=1e-5,\
REWEIGHT_OUTER_LR=0.05,\
PER_SOURCE_BATCH=1,\
VAL_BATCH_SIZE=1,\
SELECTION_VAL_BATCH_SIZE=1,\
EVAL_BATCH_SIZE=1,\
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE}",\
FINAL_FT_MICRO_BATCH=1,\
FINAL_FT_GRAD_ACCUM=64,\
GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE}",\
MATH_EVAL_SIZE="${MATH_EVAL_SIZE}",\
EVAL_MATH=1,\
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}",\
USE_BEST_REWEIGHT_CHECKPOINT=0\
"${extra_export}" \
    "${SCRIPT_DIR}/run_exact_math_reasoning.sbatch"
}

echo "Mode: ${MODE}"
echo "Methods: ${METHODS}"
echo "Model: ${MODEL_NAME}"
echo "Output root: ${OUTPUT_ROOT}"

for method_key in ${METHODS}; do
  submit_job "${method_key}"
done
