#!/bin/bash
#SBATCH --job-name=instr_saved_mt
#SBATCH --output=/scratch/%u/slurm/instr_saved_mt_%j.out
#SBATCH --error=/scratch/%u/slurm/instr_saved_mt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must point to a completed instruction run." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_DIR="${ENV_DIR:-/home/masiha/venvs/hypergrad_llm_env}"
HF_CACHE_ROOT_DEFAULT="/scratch/${USER}/hf_cache_jobs/${SLURM_JOB_ID}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${HF_CACHE_ROOT_DEFAULT}}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_CACHE_ROOT}/assets}"
export LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_ASSETS_CACHE}" "/scratch/${USER}/slurm"

OPENAI_ENV_FILE="${OPENAI_ENV_FILE:-${HOME}/.openai_env}"
if [[ -f "${OPENAI_ENV_FILE}" ]]; then
  set -a
  source "${OPENAI_ENV_FILE}"
  set +a
fi

source "${ENV_DIR}/bin/activate"
cd "${SCRIPT_DIR}"

echo "Host: $(hostname)"
echo "RUN_DIR=${RUN_DIR}"
echo "OPENAI_API_KEY_SET=$([[ -n "${OPENAI_API_KEY:-}" ]] && echo 1 || echo 0)"
python -V
nvidia-smi -L

ARGS=(--run-dir "${RUN_DIR}" --adapter-file "${ADAPTER_FILE:-final_ft_best_adapter_state.pt}")
if [[ -n "${MTBENCH_QUESTION_LIMIT:-}" ]]; then
  ARGS+=(--question-limit "${MTBENCH_QUESTION_LIMIT}")
fi
if [[ -n "${MTBENCH_QUESTION_FILE:-}" ]]; then
  ARGS+=(--question-file "${MTBENCH_QUESTION_FILE}")
fi
if [[ -n "${MTBENCH_MODEL_ID:-}" ]]; then
  ARGS+=(--model-id "${MTBENCH_MODEL_ID}")
fi
if [[ -n "${MTBENCH_JUDGE_MODEL:-}" ]]; then
  ARGS+=(--judge-model "${MTBENCH_JUDGE_MODEL}")
fi
if [[ -n "${MTBENCH_API_BASE:-}" ]]; then
  ARGS+=(--api-base "${MTBENCH_API_BASE}")
fi
if [[ -n "${MTBENCH_API_KEY_ENV:-}" ]]; then
  ARGS+=(--api-key-env "${MTBENCH_API_KEY_ENV}")
fi
if [[ "${JUDGE:-1}" == "1" ]]; then
  ARGS+=(--judge)
else
  ARGS+=(--no-judge)
fi
if [[ "${FORCE_GENERATE:-0}" == "1" ]]; then
  ARGS+=(--force-generate)
fi

python -u evaluate_saved_instruction_mtbench.py "${ARGS[@]}"
