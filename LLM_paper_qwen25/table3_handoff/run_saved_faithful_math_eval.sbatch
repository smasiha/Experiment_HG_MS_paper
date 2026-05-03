#!/bin/bash
#SBATCH --job-name=faithful_math_eval
#SBATCH --output=/scratch/%u/slurm/faithful_math_eval_%j.out
#SBATCH --error=/scratch/%u/slurm/faithful_math_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
ENV_DIR="${ENV_DIR:-/home/masiha/venvs/hypergrad_llm_env}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
MATH_EVAL_SIZE="${MATH_EVAL_SIZE:-0}"
EVAL_BATCH_SIZE_OVERRIDE="${EVAL_BATCH_SIZE_OVERRIDE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY:-0}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-}"

if [[ -z "${HF_CACHE_ROOT}" ]]; then
  MODEL_CACHE_TAG="$(
    python - <<'PY'
import json
import os
import re
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
config = json.loads((run_dir / "config.json").read_text())
model_name = str(config["model_name"]).lower()
print(re.sub(r"[^a-z0-9]+", "_", model_name).strip("_"))
PY
  )"
  HF_CACHE_ROOT="/scratch/${USER}/hf_cache_faithful_math_eval/${MODEL_CACHE_TAG}"
fi

export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_CACHE_ROOT}/assets}"
mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_ASSETS_CACHE}" "/scratch/${USER}/slurm"

cd "${ROOT_DIR}"
source "${ENV_DIR}/bin/activate"

echo "Host: $(hostname)"
echo "RUN_DIR=${RUN_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
python -V
nvidia-smi -L

export LOCAL_MODEL_ONLY
EXTRA_ARGS=()
if [[ "${FORCE_EVAL}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

python -u evaluate_saved_faithful_math_runs.py \
  --run-dir "${RUN_DIR}" \
  --math-eval-size "${MATH_EVAL_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE_OVERRIDE}" \
  "${EXTRA_ARGS[@]}"
