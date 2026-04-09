#!/bin/bash

#SBATCH --job-name=dhc_rho0p8_120s_inv50_r5
#SBATCH --output=data_hyper_rho0p8_120s_inv50_r5_%j.out
#SBATCH --error=data_hyper_rho0p8_120s_inv50_r5_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --mem=32G

#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"

source /home/masiha/venvs/hypergrad_env/bin/activate
export HYPERGRAD_REQUIRE_CUDA=1

OUT_DIR="${OUTPUT_ROOT:-${ROOT_DIR}/outputs}/${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

python -u "${ROOT_DIR}/code/run_all_algorithms.py" \
  --plots-dir "${OUT_DIR}" \
  --pollute-rate 0.8 \
  --time-budget-seconds 120 \
  --time-grid-seconds 1.0 \
  --outer-iterations 30000 \
  --runs 5 \
  --hg-inner-steps 20 \
  --hg-inner-lr 0.005 \
  --hg-lr-x 0.1 \
  --hgms-lr-x 0.1 \
  --hgms-lr-x-schedule inv \
  --hgms-lr-x-decay-steps 50
