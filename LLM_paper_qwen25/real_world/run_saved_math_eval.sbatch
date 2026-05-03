#!/bin/bash
#SBATCH --job-name=llm_math_eval
#SBATCH --output=/scratch/%u/slurm/llm_math_eval_%j.out
#SBATCH --error=/scratch/%u/slurm/llm_math_eval_%j.err
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
HF_CACHE_ROOT_DEFAULT="/scratch/${USER}/hf_cache_math_eval/${SLURM_JOB_ID}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${HF_CACHE_ROOT_DEFAULT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_CACHE_ROOT}/assets}"
mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_ASSETS_CACHE}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
ENV_DIR="${ENV_DIR:-/home/masiha/venvs/hypergrad_llm_env}"
RUNS_ROOT="${RUNS_ROOT:-}"
RUN_DIR="${RUN_DIR:-}"
MATH_EVAL_SIZE="${MATH_EVAL_SIZE:-0}"
EVAL_BATCH_SIZE_OVERRIDE="${EVAL_BATCH_SIZE_OVERRIDE:-0}"
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY:-1}"
FORCE_EVAL="${FORCE_EVAL:-0}"

if [[ -n "${RUNS_ROOT}" && -n "${RUN_DIR}" ]]; then
  echo "Set only one of RUNS_ROOT or RUN_DIR" >&2
  exit 2
fi
if [[ -z "${RUNS_ROOT}" && -z "${RUN_DIR}" ]]; then
  echo "Either RUNS_ROOT or RUN_DIR must be set" >&2
  exit 2
fi

mkdir -p "/scratch/${USER}/slurm"
cd "${ROOT_DIR}"
source "${ENV_DIR}/bin/activate"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export LOCAL_MODEL_ONLY
python -V
nvidia-smi -L

EXTRA_ARGS=()
if [[ "${FORCE_EVAL}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

CMD=(python -u evaluate_saved_math_runs.py --math-eval-size "${MATH_EVAL_SIZE}" --eval-batch-size "${EVAL_BATCH_SIZE_OVERRIDE}")
if [[ -n "${RUN_DIR}" ]]; then
  CMD+=(--run-dir "${RUN_DIR}")
else
  CMD+=(--runs-root "${RUNS_ROOT}")
fi
CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
