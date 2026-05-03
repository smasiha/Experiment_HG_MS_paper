#!/bin/bash
#SBATCH --job-name=lora_manifold
#SBATCH --output=/scratch/%u/slurm/lora_manifold_%j.out
#SBATCH --error=/scratch/%u/slurm/lora_manifold_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
ENV_DIR="${ENV_DIR:-/home/masiha/venvs/hypergrad_llm_env}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a completed instruction-following run directory.}"
CHECKPOINT="${CHECKPOINT:-best}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/lora_manifold_${CHECKPOINT}}"
BATCH_COUNT="${BATCH_COUNT:-4}"
PER_SOURCE_BATCH="${PER_SOURCE_BATCH:-0}"
NUM_PATH_POINTS="${NUM_PATH_POINTS:-9}"
MAX_ANGLE="${MAX_ANGLE:-6.283185307179586}"
PATH_PLANE_COUNT="${PATH_PLANE_COUNT:-1}"
NUM_GENERATORS="${NUM_GENERATORS:-4}"
GENERATOR_EPS="${GENERATOR_EPS:-0.01}"
EVAL_BF16="${EVAL_BF16:-0}"

mkdir -p "/scratch/${USER}/slurm"
cd "${ROOT_DIR}"

source "${ENV_DIR}/bin/activate"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
python -V
nvidia-smi -L

EXTRA_ARGS=()
if [[ "${EVAL_BF16}" == "1" ]]; then
  EXTRA_ARGS+=(--eval-bf16)
fi

python -u lora_manifold_diagnostic.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-count "${BATCH_COUNT}" \
  --per-source-batch "${PER_SOURCE_BATCH}" \
  --num-path-points "${NUM_PATH_POINTS}" \
  --max-angle "${MAX_ANGLE}" \
  --path-plane-count "${PATH_PLANE_COUNT}" \
  --num-generators "${NUM_GENERATORS}" \
  --generator-eps "${GENERATOR_EPS}" \
  "${EXTRA_ARGS[@]}"
