#!/bin/bash
#SBATCH --job-name=llm_zero_shot_gsm8k
#SBATCH --output=/scratch/%u/slurm/llm_zero_shot_gsm8k_%j.out
#SBATCH --error=/scratch/%u/slurm/llm_zero_shot_gsm8k_%j.err
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
MODEL_PRESET="${MODEL_PRESET:-}"
if [[ -z "${HF_CACHE_ROOT:-}" && -n "${MODEL_PRESET}" ]]; then
  TRAINING_CACHE_PARENT="/scratch/${USER}/hf_cache_jobs/${MODEL_PRESET}"
  HF_CACHE_ROOT="$(
    if [[ -d "${TRAINING_CACHE_PARENT}" ]]; then
      find "${TRAINING_CACHE_PARENT}" -mindepth 1 -maxdepth 1 -type d | sort | while read -r candidate; do
        if find "${candidate}/transformers" -path '*/model.safetensors.index.json' -print -quit | grep -q .; then
          echo "${candidate}"
          break
        fi
      done
    fi
  )"
  if [[ -z "${HF_CACHE_ROOT}" ]]; then
    HF_CACHE_ROOT="/scratch/${USER}/hf_cache_prewarm/${MODEL_PRESET}"
  fi
fi
HF_CACHE_ROOT_DEFAULT="/scratch/${USER}/hf_cache_zero_shot/${SLURM_JOB_ID}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${HF_CACHE_ROOT_DEFAULT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_CACHE_ROOT}/assets}"
mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_ASSETS_CACHE}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
ENV_DIR="${ENV_DIR:-/home/masiha/venvs/hypergrad_llm_env}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME must be set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
SEED="${SEED:-2424}"
MAX_LENGTH="${MAX_LENGTH:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE:-0}"
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY:-1}"
BF16="${BF16:-1}"

mkdir -p "/scratch/${USER}/slurm"
cd "${ROOT_DIR}"
source "${ENV_DIR}/bin/activate"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export LOCAL_MODEL_ONLY
python -V
nvidia-smi -L

EXTRA_ARGS=()
if [[ "${BF16}" == "1" ]]; then
  EXTRA_ARGS+=(--bf16)
fi

python -u evaluate_zero_shot_gsm8k.py \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --max-length "${MAX_LENGTH}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
  --gsm8k-eval-size "${GSM8K_EVAL_SIZE}" \
  "${EXTRA_ARGS[@]}"
