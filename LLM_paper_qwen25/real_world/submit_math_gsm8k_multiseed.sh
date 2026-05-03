#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_PRESET="${MODEL_PRESET:-llama3_8b}"
SEEDS="${SEEDS:-2424 2525 2626}"
METHODS="${METHODS:-uniform less rho_loss scalebio hg_single hg_ms}"
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS:-3600}"
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-20000}"
GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE:-0}"
TAG="${TAG:-math_gsm8k_${MODEL_PRESET}_multiseed_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}/math}"
MODEL_PREFLIGHT="${MODEL_PREFLIGHT:-1}"
PREFLIGHT_HF_CACHE_ROOT="${PREFLIGHT_HF_CACHE_ROOT:-/scratch/${USER}/hf_cache_prewarm/${MODEL_PRESET}}"

source "${ROOT_DIR}/model_presets.sh"
apply_math_model_preset "${MODEL_PRESET}"

cd "${ROOT_DIR}"

mkdir -p \
  "${PREFLIGHT_HF_CACHE_ROOT}/hub" \
  "${PREFLIGHT_HF_CACHE_ROOT}/transformers" \
  "${PREFLIGHT_HF_CACHE_ROOT}/assets"

if [[ "${MODEL_PREFLIGHT}" == "1" ]]; then
  source /home/masiha/venvs/hypergrad_llm_env/bin/activate
  preflight_args=()
  if [[ "${LOCAL_MODEL_ONLY}" == "1" ]]; then
    preflight_args+=(--local-files-only)
  fi
  HF_HUB_CACHE="${PREFLIGHT_HF_CACHE_ROOT}/hub" \
  HUGGINGFACE_HUB_CACHE="${PREFLIGHT_HF_CACHE_ROOT}/hub" \
  TRANSFORMERS_CACHE="${PREFLIGHT_HF_CACHE_ROOT}/transformers" \
  HF_ASSETS_CACHE="${PREFLIGHT_HF_CACHE_ROOT}/assets" \
    ./check_model_access.py --model "${MODEL_NAME}" "${preflight_args[@]}"
  HF_HUB_CACHE="${PREFLIGHT_HF_CACHE_ROOT}/hub" \
  HUGGINGFACE_HUB_CACHE="${PREFLIGHT_HF_CACHE_ROOT}/hub" \
  TRANSFORMERS_CACHE="${PREFLIGHT_HF_CACHE_ROOT}/transformers" \
  HF_ASSETS_CACHE="${PREFLIGHT_HF_CACHE_ROOT}/assets" \
    ./check_model_access.py --model "${REFERENCE_MODEL_NAME}" "${preflight_args[@]}"
fi

submit_job() {
  local method="$1"
  local seed="$2"
  local run_name="${method}_seed${seed}_${MODEL_PRESET}_gsm8k"
  local job_hf_cache_root="/scratch/${USER}/hf_cache_jobs/${MODEL_PRESET}/${run_name}"

  local outer_lr inner_steps selection_inner_steps hypergrad_inner_steps
  local m_branches selection_batches hypergrad_batches
  local selection_batch_size branch_noise train_tolerance train_penalty reweight_steps
  local theta_optimizer="sgd"

  case "${method}" in
    uniform)
      outer_lr="0.1"; inner_steps="1"; selection_inner_steps="1"; hypergrad_inner_steps="1"; m_branches="1"; selection_batches="1"; hypergrad_batches="1"; selection_batch_size="${SELECTION_VAL_BATCH_SIZE}"; branch_noise="0"; train_tolerance="0.05"; train_penalty="0.0"; reweight_steps="1"
      ;;
    less)
      outer_lr="0.1"; inner_steps="1"; selection_inner_steps="1"; hypergrad_inner_steps="1"; m_branches="1"; selection_batches="1"; hypergrad_batches="1"; selection_batch_size="${SELECTION_VAL_BATCH_SIZE}"; branch_noise="0"; train_tolerance="0.05"; train_penalty="0.0"; reweight_steps="1"
      ;;
    rho_loss)
      outer_lr="0.1"; inner_steps="1"; selection_inner_steps="1"; hypergrad_inner_steps="1"; m_branches="1"; selection_batches="1"; hypergrad_batches="1"; selection_batch_size="${SELECTION_VAL_BATCH_SIZE}"; branch_noise="0"; train_tolerance="0.05"; train_penalty="0.0"; reweight_steps="1"
      ;;
    scalebio)
      outer_lr="1.0"; inner_steps="1"; selection_inner_steps="1"; hypergrad_inner_steps="1"; m_branches="1"; selection_batches="1"; hypergrad_batches="1"; selection_batch_size="${SELECTION_VAL_BATCH_SIZE}"; branch_noise="0"; train_tolerance="0.05"; train_penalty="0.0"; reweight_steps="100000"
      ;;
    hg_single)
      outer_lr="500.0"; inner_steps="1"; selection_inner_steps="1"; hypergrad_inner_steps="1"; m_branches="1"; selection_batches="4"; hypergrad_batches="4"; selection_batch_size="${SELECTION_VAL_BATCH_SIZE}"; branch_noise="0"; train_tolerance="0.05"; train_penalty="0.0"; reweight_steps="100000"
      ;;
    hg_ms)
      outer_lr="1000.0"; inner_steps="2"; selection_inner_steps="2"; hypergrad_inner_steps="2"; m_branches="4"; selection_batches="6"; hypergrad_batches="6"; selection_batch_size="${SELECTION_VAL_BATCH_SIZE}"; branch_noise="0"; train_tolerance="0.30"; train_penalty="0.05"; reweight_steps="100000"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 1
      ;;
  esac

  case "${MODEL_PRESET}" in
    llama3_8b|qwen2_7b|gemma2_9b)
      case "${method}" in
        scalebio)
          selection_batch_size="1"
          ;;
        hg_single)
          selection_batches="1"
          hypergrad_batches="1"
          selection_batch_size="1"
          ;;
        hg_ms)
          inner_steps="1"
          selection_inner_steps="1"
          hypergrad_inner_steps="1"
          m_branches="4"
          selection_batches="1"
          hypergrad_batches="1"
          selection_batch_size="1"
          train_tolerance="0.30"
          train_penalty="0.05"
          ;;
      esac
      ;;
    qwen25_32b)
      case "${method}" in
        scalebio)
          selection_batch_size="1"
          ;;
        hg_single)
          selection_batches="2"
          hypergrad_batches="2"
          selection_batch_size="1"
          ;;
        hg_ms)
          inner_steps="1"
          selection_inner_steps="1"
          hypergrad_inner_steps="1"
          m_branches="2"
          selection_batches="2"
          hypergrad_batches="2"
          selection_batch_size="1"
          train_tolerance="0.10"
          train_penalty="0.0"
          ;;
      esac
      ;;
  esac

  sbatch \
    --job-name="gsm8k_${MODEL_PRESET}_${method}_${seed}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
HF_CACHE_ROOT="${job_hf_cache_root}",\
SEED="${seed}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY}",\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
VAL_KINDS_SPEC=gsm8k,\
EVAL_MATH=0,\
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS}",\
MAX_LENGTH="${MAX_LENGTH}",\
VAL_BATCH_SIZE="${VAL_BATCH_SIZE}",\
SELECTION_VAL_BATCH_SIZE="${selection_batch_size}",\
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}",\
REWEIGHT_MODEL_LR=1e-4,\
REWEIGHT_OUTER_LR="${outer_lr}",\
THETA_OPTIMIZER="${theta_optimizer}",\
BRANCH_NOISE_STD="${branch_noise}",\
REWEIGHT_TRAIN_PER_SOURCE=500,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH="${PER_SOURCE_BATCH}",\
REWEIGHT_STEPS="${reweight_steps}",\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${hypergrad_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${train_tolerance}",\
SELECTION_TRAIN_PENALTY="${train_penalty}",\
INNER_STEPS="${inner_steps}",\
SELECTION_INNER_STEPS="${selection_inner_steps}",\
HYPERGRAD_INNER_STEPS="${hypergrad_inner_steps}",\
M_BRANCHES="${m_branches}",\
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE}",\
FINAL_FT_MICRO_BATCH="${FINAL_FT_MICRO_BATCH}",\
FINAL_FT_GRAD_ACCUM="${FINAL_FT_GRAD_ACCUM}",\
GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE}",\
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE}",\
REFERENCE_FT_STEPS=100,\
LORA_R="${LORA_R}",\
LORA_ALPHA="${LORA_ALPHA}",\
LORA_DROPOUT="${LORA_DROPOUT}" \
    run_exact_math_reasoning.sbatch
}

echo "Math GSM8K output root: ${OUTPUT_ROOT}"
echo "Model preset: ${MODEL_PRESET}"
echo "Model: ${MODEL_NAME}"
echo "Reference model: ${REFERENCE_MODEL_NAME}"
echo "Seeds: ${SEEDS}"
echo "Methods: ${METHODS}"
echo "Reweight/search wall-clock budget (sec): ${REWEIGHT_WALL_CLOCK_SECONDS}"

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    submit_job "${method}" "${seed}"
  done
done
