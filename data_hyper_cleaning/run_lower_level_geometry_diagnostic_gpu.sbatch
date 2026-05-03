#!/bin/bash

# Example:
#   sbatch paper_experiments/data_hyper_cleaning/run_lower_level_geometry_diagnostic_gpu.sbatch
#   sbatch --export=ALL,RUN_NAME=rho0p8_geom,POLLUTE_RATE=0.8,NUM_SAMPLES=16,OUTER_ITR=20 \
#     paper_experiments/data_hyper_cleaning/run_lower_level_geometry_diagnostic_gpu.sbatch

#SBATCH --job-name=dhc_geometry
#SBATCH --output=dhc_geometry_%j.out
#SBATCH --error=dhc_geometry_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=32G

# KUMA-specific settings
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ROOT_DIR:-}" ]]; then
  ROOT_DIR="${ROOT_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/code/lower_level_geometry_diagnostic.py" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/paper_experiments/data_hyper_cleaning/code/lower_level_geometry_diagnostic.py" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}/paper_experiments/data_hyper_cleaning"
else
  ROOT_DIR="${SCRIPT_DIR}"
fi

source /home/masiha/venvs/hypergrad_env/bin/activate
export HYPERGRAD_REQUIRE_CUDA=1

echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V
nvidia-smi -L || true

OUT_DIR="${OUTPUT_ROOT:-${ROOT_DIR}/geometry_outputs}/${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

RUN_NAME="${RUN_NAME:-rho0p8_geom_job${SLURM_JOB_ID}}"
SEED="${SEED:-2424}"
POLLUTE_RATE="${POLLUTE_RATE:-0.8}"
HIDDEN="${HIDDEN:-300}"
REG="${REG:-1e-4}"

ANCHOR_SOURCE="${ANCHOR_SOURCE:-hgms}"

OUTER_ITR="${OUTER_ITR:-20}"
OUTER_TIME_BUDGET_S="${OUTER_TIME_BUDGET_S:-300}"
OUTER_M_CANDIDATES="${OUTER_M_CANDIDATES:-4}"
OUTER_D_INNER_SGLD="${OUTER_D_INNER_SGLD:-4}"
OUTER_LR_INNER_SGLD="${OUTER_LR_INNER_SGLD:-0.02}"
OUTER_SGLD_BATCH_SIZE="${OUTER_SGLD_BATCH_SIZE:-256}"
OUTER_CG_MAX_IT="${OUTER_CG_MAX_IT:-8}"

NUM_SAMPLES="${NUM_SAMPLES:-12}"
SAMPLE_STEPS="${SAMPLE_STEPS:-12}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-256}"
SAMPLE_INIT_NOISE_STD="${SAMPLE_INIT_NOISE_STD:-1e-3}"
NEAR_TOL_ABS="${NEAR_TOL_ABS:-5e-4}"
NEAR_TOL_REL="${NEAR_TOL_REL:-1e-3}"
MIN_NEAR_COUNT="${MIN_NEAR_COUNT:-8}"
MAX_TANGENT_DIM="${MAX_TANGENT_DIM:-6}"
MAX_BARRIER_PATHS="${MAX_BARRIER_PATHS:-6}"
INTERP_POINTS="${INTERP_POINTS:-9}"

python -u "${ROOT_DIR}/code/lower_level_geometry_diagnostic.py" \
  --output-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --seed "${SEED}" \
  --anchor-source "${ANCHOR_SOURCE}" \
  --pollute-rate "${POLLUTE_RATE}" \
  --hidden "${HIDDEN}" \
  --reg "${REG}" \
  --outer-itr "${OUTER_ITR}" \
  --outer-time-budget-s "${OUTER_TIME_BUDGET_S}" \
  --outer-M-candidates "${OUTER_M_CANDIDATES}" \
  --outer-D-inner-sgld "${OUTER_D_INNER_SGLD}" \
  --outer-lr-inner-sgld "${OUTER_LR_INNER_SGLD}" \
  --outer-sgld-batch-size "${OUTER_SGLD_BATCH_SIZE}" \
  --outer-cg-max-it "${OUTER_CG_MAX_IT}" \
  --num-samples "${NUM_SAMPLES}" \
  --sample-steps "${SAMPLE_STEPS}" \
  --sample-batch-size "${SAMPLE_BATCH_SIZE}" \
  --sample-init-noise-std "${SAMPLE_INIT_NOISE_STD}" \
  --near-tol-abs "${NEAR_TOL_ABS}" \
  --near-tol-rel "${NEAR_TOL_REL}" \
  --min-near-count "${MIN_NEAR_COUNT}" \
  --max-tangent-dim "${MAX_TANGENT_DIM}" \
  --max-barrier-paths "${MAX_BARRIER_PATHS}" \
  --interp-points "${INTERP_POINTS}"

echo "Results written under: ${OUT_DIR}/${RUN_NAME}"
