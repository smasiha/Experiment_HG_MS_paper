#!/bin/bash
#SBATCH --job-name=llm_small
#SBATCH --output=/scratch/masiha/slurm/llm_small_%j.out
#SBATCH --error=/scratch/masiha/slurm/llm_small_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${PACKAGE_ROOT_DEFAULT}}"
ENV_DIR="${ENV_DIR:-/home/masiha/venvs/hypergrad_llm_env}"
TASK="${TASK:-denoising}"
METHOD="${METHOD:-hg_ms}"
SEED="${SEED:-2424}"
MODEL_NAME="${MODEL_NAME:-gpt2}"
FIXED_LAMBDA="${FIXED_LAMBDA:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/controlled/${SLURM_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${TASK}/${METHOD}_seed${SEED}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/cache/controlled}"
RAW_CACHE_PATH="${RAW_CACHE_PATH:-${CACHE_ROOT}/small_${TASK}_seed${SEED}.json}"
TOKENIZED_CACHE_PATH="${TOKENIZED_CACHE_PATH:-${CACHE_ROOT}/small_${TASK}_seed${SEED}_${MODEL_NAME//\//_}_ml${MAX_LENGTH:-256}.pt}"

BATCH_SIZE="${BATCH_SIZE:-64}"
SELECTION_VAL_BATCH_SIZE="${SELECTION_VAL_BATCH_SIZE:-128}"
SELECTION_VAL_MODE="${SELECTION_VAL_MODE:-target_mixture}"
SELECTION_VAL_BATCHES="${SELECTION_VAL_BATCHES:-1}"
HYPERGRAD_VAL_MODE="${HYPERGRAD_VAL_MODE:-same}"
HYPERGRAD_VAL_BATCHES="${HYPERGRAD_VAL_BATCHES:-1}"
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD="${REUSE_SELECTION_BATCHES_FOR_HYPERGRAD:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
REWEIGHT_EPOCHS="${REWEIGHT_EPOCHS:-3}"
REWEIGHT_STEPS="${REWEIGHT_STEPS:-0}"
INNER_STEPS="${INNER_STEPS:-1}"
M_BRANCHES="${M_BRANCHES:-2}"
BRANCH_NOISE_STD="${BRANCH_NOISE_STD:-1e-3}"
SELECTION_TRAIN_TOLERANCE="${SELECTION_TRAIN_TOLERANCE:-inf}"
SELECTION_TRAIN_PENALTY="${SELECTION_TRAIN_PENALTY:-0.0}"
OPTIMIZER="${OPTIMIZER:-sgd}"
REWEIGHT_MODEL_LR="${REWEIGHT_MODEL_LR:-1e-5}"
REWEIGHT_OUTER_LR="${REWEIGHT_OUTER_LR:-1e-2}"
REWEIGHT_WEIGHT_DECAY="${REWEIGHT_WEIGHT_DECAY:-0.01}"
THETA_GRAD_CLIP="${THETA_GRAD_CLIP:-10.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
SCALEBIO_ALPHA="${SCALEBIO_ALPHA:-10.0}"
EVAL_EVERY="${EVAL_EVERY:-25}"
MAX_LENGTH="${MAX_LENGTH:-256}"
BF16="${BF16:-0}"
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY:-0}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}/controlled"

if [[ ! -d "${ENV_DIR}" ]]; then
  python3 -m venv "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"
if [[ ! -f "${ENV_DIR}/.llm_minima_ready" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  touch "${ENV_DIR}/.llm_minima_ready"
fi

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
python -V
nvidia-smi -L

EXTRA_ARGS=()
if [[ "${BF16}" == "1" ]]; then
  EXTRA_ARGS+=(--bf16)
fi
if [[ "${REUSE_SELECTION_BATCHES_FOR_HYPERGRAD}" == "1" ]]; then
  EXTRA_ARGS+=(--reuse-selection-batches-for-hypergrad)
fi
if [[ -n "${FIXED_LAMBDA}" ]]; then
  EXTRA_ARGS+=(--fixed-lambda "${FIXED_LAMBDA}")
fi

export LOCAL_MODEL_ONLY
python -u small_scale_reweight.py \
  --task "${TASK}" \
  --method "${METHOD}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --max-length "${MAX_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --selection-val-batch-size "${SELECTION_VAL_BATCH_SIZE}" \
  --selection-val-mode "${SELECTION_VAL_MODE}" \
  --selection-val-batches "${SELECTION_VAL_BATCHES}" \
  --hypergrad-val-mode "${HYPERGRAD_VAL_MODE}" \
  --hypergrad-val-batches "${HYPERGRAD_VAL_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --reweight-epochs "${REWEIGHT_EPOCHS}" \
  --reweight-steps "${REWEIGHT_STEPS}" \
  --inner-steps "${INNER_STEPS}" \
  --m-branches "${M_BRANCHES}" \
  --branch-noise-std "${BRANCH_NOISE_STD}" \
  --selection-train-tolerance "${SELECTION_TRAIN_TOLERANCE}" \
  --selection-train-penalty "${SELECTION_TRAIN_PENALTY}" \
  --optimizer "${OPTIMIZER}" \
  --reweight-model-lr "${REWEIGHT_MODEL_LR}" \
  --reweight-outer-lr "${REWEIGHT_OUTER_LR}" \
  --reweight-weight-decay "${REWEIGHT_WEIGHT_DECAY}" \
  --theta-grad-clip "${THETA_GRAD_CLIP}" \
  --adam-beta1 "${ADAM_BETA1}" \
  --adam-beta2 "${ADAM_BETA2}" \
  --adam-eps "${ADAM_EPS}" \
  --scalebio-alpha "${SCALEBIO_ALPHA}" \
  --eval-every "${EVAL_EVERY}" \
  --cache-path "${RAW_CACHE_PATH}" \
  --tokenized-cache-path "${TOKENIZED_CACHE_PATH}" \
  "${EXTRA_ARGS[@]}"
