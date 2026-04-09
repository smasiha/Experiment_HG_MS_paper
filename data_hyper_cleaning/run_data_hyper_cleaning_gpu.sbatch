#!/bin/bash

# Job name
#SBATCH --job-name=data_hyper_cleaning

# Output and error files
#SBATCH --output=data_hyper_cleaning_%j.out
#SBATCH --error=data_hyper_cleaning_%j.err

# Resource requests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32G

# KUMA-specific settings
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"

source /home/masiha/venvs/hypergrad_env/bin/activate

# Fail fast if the job doesn't actually see a CUDA device.
export HYPERGRAD_REQUIRE_CUDA=1

echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V

nvidia-smi -L || true

# Run the full Data_hyper_cleaning comparison runner.
# You can override these via env vars at submission time, e.g.:
#   sbatch --export=ALL,PRESET=minsel-gap run_data_hyper_cleaning_gpu.sbatch
#   sbatch --export=ALL,PRESET=minsel-gap-all run_data_hyper_cleaning_gpu.sbatch
#   sbatch --export=ALL,PRESET=minsel-gap,POLLUTE_RATE=0.7 run_data_hyper_cleaning_gpu.sbatch
#   sbatch --export=ALL,PRESET=minsel-gap,OUTER_ITERS=100,RUNS=5 run_data_hyper_cleaning_gpu.sbatch
OUT_DIR="${OUTPUT_ROOT:-${ROOT_DIR}/outputs}/${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

PRESET="${PRESET:-}"
POLLUTE_RATE="${POLLUTE_RATE:-0.8}"
ONLY_HG="${ONLY_HG:-0}"
HG_INNER_STEPS="${HG_INNER_STEPS:-20}"
HG_INNER_LR="${HG_INNER_LR:-0.005}"
HG_LR_X="${HG_LR_X:-0.1}"

if [[ -n "${PRESET}" ]]; then
  # Let the Python preset define defaults; only pass explicit overrides if provided.
  OUTER_ITERS="${OUTER_ITERS:-}"
  RUNS="${RUNS:-}"
else
  OUTER_ITERS="${OUTER_ITERS:-1000}"
  RUNS="${RUNS:-1}"
fi
TIME_BUDGET_S="${TIME_BUDGET_S:-}"

ARGS=(--plots-dir "${OUT_DIR}" --pollute-rate "${POLLUTE_RATE}" --hg-inner-steps "${HG_INNER_STEPS}" --hg-inner-lr "${HG_INNER_LR}" --hg-lr-x "${HG_LR_X}")
if [[ -n "${PRESET}" ]]; then
  ARGS+=(--preset "${PRESET}")
fi
if [[ -n "${OUTER_ITERS}" ]]; then
  ARGS+=(--outer-iterations "${OUTER_ITERS}")
fi
if [[ -n "${RUNS}" ]]; then
  ARGS+=(--runs "${RUNS}")
fi
if [[ -n "${TIME_BUDGET_S}" ]]; then
  ARGS+=(--time-budget-seconds "${TIME_BUDGET_S}")
fi
if [[ "${ONLY_HG}" == "1" ]]; then
  ARGS+=(--only-hg)
fi

python -u "${ROOT_DIR}/code/run_all_algorithms.py" \
  "${ARGS[@]}"
