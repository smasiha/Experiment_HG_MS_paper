#!/bin/bash
#SBATCH --job-name=llm_instr
#SBATCH --output=/scratch/%u/slurm/llm_instr_%j.out
#SBATCH --error=/scratch/%u/slurm/llm_instr_%j.err
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
ENV_DIR="${ENV_DIR:-/home/masiha/venvs/hypergrad_llm_env}"
METHOD="${METHOD:-hg_ms}"
SEED="${SEED:-2424}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-7B}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2-1.5B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/hypergrad_instruction_outputs/${SLURM_JOB_ID}}"
RUN_NAME="${RUN_NAME:-${METHOD}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-3000}"
REWEIGHT_TRAIN_RAW_OVERSAMPLE="${REWEIGHT_TRAIN_RAW_OVERSAMPLE:-1.5}"
REWEIGHT_EPOCHS="${REWEIGHT_EPOCHS:-3}"
REWEIGHT_STEPS="${REWEIGHT_STEPS:-0}"
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS:-0}"
PER_SOURCE_BATCH="${PER_SOURCE_BATCH:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
SELECTION_VAL_BATCH_SIZE="${SELECTION_VAL_BATCH_SIZE:-16}"
SELECTION_VAL_BATCHES="${SELECTION_VAL_BATCHES:-1}"
HYPERGRAD_VAL_BATCHES="${HYPERGRAD_VAL_BATCHES:-1}"
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD="${REUSE_SELECTION_BATCHES_FOR_HYPERGRAD:-1}"
SELECTION_TRAIN_TOLERANCE="${SELECTION_TRAIN_TOLERANCE:-0.05}"
SELECTION_TRAIN_PENALTY="${SELECTION_TRAIN_PENALTY:-0.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
INNER_STEPS="${INNER_STEPS:-1}"
M_BRANCHES="${M_BRANCHES:-2}"
BRANCH_NOISE_STD="${BRANCH_NOISE_STD:-5e-4}"
REWEIGHT_MODEL_LR="${REWEIGHT_MODEL_LR:-1e-5}"
REWEIGHT_OUTER_LR="${REWEIGHT_OUTER_LR:-1e-2}"
THETA_OPTIMIZER="${THETA_OPTIMIZER:-sgd}"
THETA_ADAM_BETA1="${THETA_ADAM_BETA1:-0.9}"
THETA_ADAM_BETA2="${THETA_ADAM_BETA2:-0.999}"
THETA_ADAM_EPS="${THETA_ADAM_EPS:-1e-8}"
REWEIGHT_WEIGHT_DECAY="${REWEIGHT_WEIGHT_DECAY:-0.0}"
THETA_GRAD_CLIP="${THETA_GRAD_CLIP:-0}"
SCALEBIO_ALPHA="${SCALEBIO_ALPHA:-100.0}"
TUNE_EVERY="${TUNE_EVERY:-100}"
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-10000}"
FINAL_FT_LR="${FINAL_FT_LR:-8e-6}"
FINAL_FT_EPOCHS="${FINAL_FT_EPOCHS:-1}"
FINAL_FT_MICRO_BATCH="${FINAL_FT_MICRO_BATCH:-1}"
FINAL_FT_GRAD_ACCUM="${FINAL_FT_GRAD_ACCUM:-64}"
FINAL_TRAIN_RAW_OVERSAMPLE="${FINAL_TRAIN_RAW_OVERSAMPLE:-1.1}"
REFERENCE_FT_STEPS="${REFERENCE_FT_STEPS:-200}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-8}"
REFERENCE_FT_LR="${REFERENCE_FT_LR:-2e-4}"
BASELINE_SCORE_SAMPLES_PER_SOURCE="${BASELINE_SCORE_SAMPLES_PER_SOURCE:-256}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
STOP_AFTER_REWEIGHT="${STOP_AFTER_REWEIGHT:-0}"
USE_BEST_REWEIGHT_CHECKPOINT="${USE_BEST_REWEIGHT_CHECKPOINT:-0}"
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY:-1}"
CACHE_PATH="${CACHE_PATH:-/scratch/${USER}/hypergrad_instruction_cache/instruction_splits_seed${SEED}_train${REWEIGHT_TRAIN_PER_SOURCE}_rov${REWEIGHT_TRAIN_RAW_OVERSAMPLE}.json}"
ALLOW_TRAIN_SHORTFALL="${ALLOW_TRAIN_SHORTFALL:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "/scratch/${USER}/slurm"
cd "${ROOT_DIR}"

source "${ENV_DIR}/bin/activate"

export HF_DATASETS_CACHE_DIR="${HF_DATASETS_CACHE_DIR:-/scratch/${USER}/hypergrad_instruction_cache/hf_datasets}"
mkdir -p "${HF_DATASETS_CACHE_DIR}"
mkdir -p "$(dirname "${CACHE_PATH}")"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export LOCAL_MODEL_ONLY
python -V
nvidia-smi -L

EXTRA_ARGS=()
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

python -u real_world_instruction.py \
  --method "${METHOD}" \
  --model-name "${MODEL_NAME}" \
  --reference-model-name "${REFERENCE_MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --cache-path "${CACHE_PATH}" \
  --reweight-train-per-source "${REWEIGHT_TRAIN_PER_SOURCE}" \
  --reweight-train-raw-oversample "${REWEIGHT_TRAIN_RAW_OVERSAMPLE}" \
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
  --m-branches "${M_BRANCHES}" \
  --branch-noise-std "${BRANCH_NOISE_STD}" \
  --reweight-model-lr "${REWEIGHT_MODEL_LR}" \
  --reweight-outer-lr "${REWEIGHT_OUTER_LR}" \
  --theta-optimizer "${THETA_OPTIMIZER}" \
  --theta-adam-beta1 "${THETA_ADAM_BETA1}" \
  --theta-adam-beta2 "${THETA_ADAM_BETA2}" \
  --theta-adam-eps "${THETA_ADAM_EPS}" \
  --reweight-weight-decay "${REWEIGHT_WEIGHT_DECAY}" \
  --theta-grad-clip "${THETA_GRAD_CLIP}" \
  --scalebio-alpha "${SCALEBIO_ALPHA}" \
  --tune-every "${TUNE_EVERY}" \
  --final-train-size "${FINAL_TRAIN_SIZE}" \
  --final-ft-lr "${FINAL_FT_LR}" \
  --final-ft-epochs "${FINAL_FT_EPOCHS}" \
  --final-ft-micro-batch "${FINAL_FT_MICRO_BATCH}" \
  --final-ft-grad-accum "${FINAL_FT_GRAD_ACCUM}" \
  --final-train-raw-oversample "${FINAL_TRAIN_RAW_OVERSAMPLE}" \
  --reference-ft-steps "${REFERENCE_FT_STEPS}" \
  --reference-batch-size "${REFERENCE_BATCH_SIZE}" \
  --reference-ft-lr "${REFERENCE_FT_LR}" \
  --baseline-score-samples-per-source "${BASELINE_SCORE_SAMPLES_PER_SOURCE}" \
  --max-length "${MAX_LENGTH}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --bf16 \
  "$([[ "${REUSE_SELECTION_BATCHES_FOR_HYPERGRAD}" == "1" ]] && echo --reuse-selection-batches-for-hypergrad || echo --no-reuse-selection-batches-for-hypergrad)" \
  "${EXTRA_ARGS[@]}"
