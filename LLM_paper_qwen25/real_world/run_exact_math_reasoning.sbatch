#!/bin/bash
#SBATCH --job-name=llm_exact
#SBATCH --output=/scratch/%u/slurm/llm_exact_%j.out
#SBATCH --error=/scratch/%u/slurm/llm_exact_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
HF_CACHE_ROOT_DEFAULT="/scratch/${USER}/hf_cache_jobs/${SLURM_JOB_ID}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${HF_CACHE_ROOT_DEFAULT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_CACHE_ROOT}/assets}"
mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_ASSETS_CACHE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${PACKAGE_ROOT_DEFAULT}}"
ENV_DIR="${ENV_DIR:-/home/masiha/venvs/hypergrad_llm_env}"
METHOD="${METHOD:-hg_ms}"
SEED="${SEED:-2424}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-7B}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2-1.5B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/real_world_math/${SLURM_JOB_ID}}"
RUN_NAME="${RUN_NAME:-${METHOD}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-3000}"
VAL_PER_SOURCE="${VAL_PER_SOURCE:-280}"
TRAIN_RAW_OVERSAMPLE="${TRAIN_RAW_OVERSAMPLE:-4.0}"
VAL_RAW_OVERSAMPLE="${VAL_RAW_OVERSAMPLE:-1.2}"
VAL_KINDS_CSV="${VAL_KINDS_CSV:-}"
VAL_KINDS_SPEC="${VAL_KINDS_SPEC:-}"
CACHE_PATH="${CACHE_PATH:-/scratch/${USER}/hypergrad_exact_cache/exact_splits_seed${SEED}_train${REWEIGHT_TRAIN_PER_SOURCE:-3000}_val${VAL_PER_SOURCE:-280}_tov${TRAIN_RAW_OVERSAMPLE}_vov${VAL_RAW_OVERSAMPLE}.json}"
ALLOW_TRAIN_SHORTFALL="${ALLOW_TRAIN_SHORTFALL:-0}"
REWEIGHT_EPOCHS="${REWEIGHT_EPOCHS:-3}"
REWEIGHT_STEPS="${REWEIGHT_STEPS:-0}"
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS:-0}"
PER_SOURCE_BATCH="${PER_SOURCE_BATCH:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
SELECTION_VAL_BATCH_SIZE="${SELECTION_VAL_BATCH_SIZE:-4}"
SELECTION_VAL_BATCHES="${SELECTION_VAL_BATCHES:-1}"
HYPERGRAD_VAL_BATCHES="${HYPERGRAD_VAL_BATCHES:-1}"
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD="${REUSE_SELECTION_BATCHES_FOR_HYPERGRAD:-1}"
SELECTION_TRAIN_TOLERANCE="${SELECTION_TRAIN_TOLERANCE:-0.05}"
SELECTION_TRAIN_PENALTY="${SELECTION_TRAIN_PENALTY:-0.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
INNER_STEPS="${INNER_STEPS:-1}"
SELECTION_INNER_STEPS="${SELECTION_INNER_STEPS:-0}"
HYPERGRAD_INNER_STEPS="${HYPERGRAD_INNER_STEPS:-0}"
M_BRANCHES="${M_BRANCHES:-2}"
BRANCH_NOISE_STD="${BRANCH_NOISE_STD:-5e-4}"
REWEIGHT_MODEL_LR="${REWEIGHT_MODEL_LR:-1e-5}"
REWEIGHT_OUTER_LR="${REWEIGHT_OUTER_LR:-1e-2}"
THETA_OPTIMIZER="${THETA_OPTIMIZER:-sgd}"
THETA_ADAM_BETA1="${THETA_ADAM_BETA1:-0.9}"
THETA_ADAM_BETA2="${THETA_ADAM_BETA2:-0.999}"
THETA_ADAM_EPS="${THETA_ADAM_EPS:-1e-8}"
THETA_GRAD_CLIP="${THETA_GRAD_CLIP:-0}"
SCALEBIO_ALPHA="${SCALEBIO_ALPHA:-10.0}"
TUNE_EVERY="${TUNE_EVERY:-50}"
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-20000}"
FINAL_FT_LR="${FINAL_FT_LR:-8e-6}"
FINAL_FT_EPOCHS="${FINAL_FT_EPOCHS:-1}"
FINAL_FT_MICRO_BATCH="${FINAL_FT_MICRO_BATCH:-1}"
FINAL_FT_GRAD_ACCUM="${FINAL_FT_GRAD_ACCUM:-64}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE:-0}"
MATH_EVAL_SIZE="${MATH_EVAL_SIZE:-0}"
REFERENCE_FT_STEPS="${REFERENCE_FT_STEPS:-200}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-8}"
REFERENCE_FT_LR="${REFERENCE_FT_LR:-2e-4}"
BASELINE_SCORE_SAMPLES_PER_SOURCE="${BASELINE_SCORE_SAMPLES_PER_SOURCE:-256}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
EVAL_MATH="${EVAL_MATH:-0}"
STOP_AFTER_REWEIGHT="${STOP_AFTER_REWEIGHT:-0}"
USE_BEST_REWEIGHT_CHECKPOINT="${USE_BEST_REWEIGHT_CHECKPOINT:-0}"
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY:-1}"

mkdir -p "/scratch/${USER}/slurm"
mkdir -p "$(dirname "${CACHE_PATH}")"
mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}/real_world"

source "${ENV_DIR}/bin/activate"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export LOCAL_MODEL_ONLY
python -V
nvidia-smi -L

EXTRA_ARGS=()
if [[ "${EVAL_MATH}" == "1" ]]; then
  EXTRA_ARGS+=(--eval-math)
fi
if [[ "${STOP_AFTER_REWEIGHT}" == "1" ]]; then
  EXTRA_ARGS+=(--stop-after-reweight)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  EXTRA_ARGS+=(--gradient-checkpointing)
fi
if [[ "${ALLOW_TRAIN_SHORTFALL}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-train-shortfall)
fi
if [[ "${USE_BEST_REWEIGHT_CHECKPOINT}" == "1" ]]; then
  EXTRA_ARGS+=(--use-best-reweight-checkpoint)
fi
VAL_KIND_SOURCE="${VAL_KINDS_SPEC:-${VAL_KINDS_CSV}}"
if [[ -n "${VAL_KIND_SOURCE}" ]]; then
  VAL_KIND_SOURCE="${VAL_KIND_SOURCE//+/,}"
  VAL_KIND_SOURCE="${VAL_KIND_SOURCE//|/,}"
  VAL_KIND_SOURCE="${VAL_KIND_SOURCE//:/,}"
  IFS=',' read -r -a VAL_KIND_ARRAY <<< "${VAL_KIND_SOURCE}"
  EXTRA_ARGS+=(--val-kinds "${VAL_KIND_ARRAY[@]}")
fi

python -u exact_math_reasoning.py \
  --method "${METHOD}" \
  --model-name "${MODEL_NAME}" \
  --reference-model-name "${REFERENCE_MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --cache-path "${CACHE_PATH}" \
  --reweight-train-per-source "${REWEIGHT_TRAIN_PER_SOURCE}" \
  --val-per-source "${VAL_PER_SOURCE}" \
  --train-raw-oversample "${TRAIN_RAW_OVERSAMPLE}" \
  --val-raw-oversample "${VAL_RAW_OVERSAMPLE}" \
  --reweight-epochs "${REWEIGHT_EPOCHS}" \
  --reweight-steps "${REWEIGHT_STEPS}" \
  --reweight-wall-clock-seconds "${REWEIGHT_WALL_CLOCK_SECONDS}" \
  --per-source-batch "${PER_SOURCE_BATCH}" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --selection-val-batch-size "${SELECTION_VAL_BATCH_SIZE}" \
  --selection-val-batches "${SELECTION_VAL_BATCHES}" \
  --hypergrad-val-batches "${HYPERGRAD_VAL_BATCHES}" \
  --selection-train-tolerance "${SELECTION_TRAIN_TOLERANCE}" \
  --selection-train-penalty "${SELECTION_TRAIN_PENALTY}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --inner-steps "${INNER_STEPS}" \
  --selection-inner-steps "${SELECTION_INNER_STEPS}" \
  --hypergrad-inner-steps "${HYPERGRAD_INNER_STEPS}" \
  --m-branches "${M_BRANCHES}" \
  --branch-noise-std "${BRANCH_NOISE_STD}" \
  --reweight-model-lr "${REWEIGHT_MODEL_LR}" \
  --reweight-outer-lr "${REWEIGHT_OUTER_LR}" \
  --theta-optimizer "${THETA_OPTIMIZER}" \
  --theta-adam-beta1 "${THETA_ADAM_BETA1}" \
  --theta-adam-beta2 "${THETA_ADAM_BETA2}" \
  --theta-adam-eps "${THETA_ADAM_EPS}" \
  --theta-grad-clip "${THETA_GRAD_CLIP}" \
  --scalebio-alpha "${SCALEBIO_ALPHA}" \
  --tune-every "${TUNE_EVERY}" \
  --final-train-size "${FINAL_TRAIN_SIZE}" \
  --final-ft-lr "${FINAL_FT_LR}" \
  --final-ft-epochs "${FINAL_FT_EPOCHS}" \
  --final-ft-micro-batch "${FINAL_FT_MICRO_BATCH}" \
  --final-ft-grad-accum "${FINAL_FT_GRAD_ACCUM}" \
  --max-length "${MAX_LENGTH}" \
  --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
  --gsm8k-eval-size "${GSM8K_EVAL_SIZE}" \
  --math-eval-size "${MATH_EVAL_SIZE}" \
  --reference-ft-steps "${REFERENCE_FT_STEPS}" \
  --reference-batch-size "${REFERENCE_BATCH_SIZE}" \
  --reference-ft-lr "${REFERENCE_FT_LR}" \
  --baseline-score-samples-per-source "${BASELINE_SCORE_SAMPLES_PER_SOURCE}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --bf16 \
  "$([[ "${REUSE_SELECTION_BATCHES_FOR_HYPERGRAD}" == "1" ]] && echo --reuse-selection-batches-for-hypergrad || echo --no-reuse-selection-batches-for-hypergrad)" \
  "${EXTRA_ARGS[@]}"
