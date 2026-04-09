#!/bin/bash
#
# Run the "Imbalance data" benchmark runner on an H100 GPU node.
#
# Usage:
#   sbatch run_imbalance_all_algorithms_gpu.sbatch
#
# Common overrides:
#   sbatch --export=ALL,OUTER_ITERS=200,REPEATS=3,METHODS=all run_imbalance_all_algorithms_gpu.sbatch
# NOTE: Slurm's `--export` uses commas to separate VAR=VALUE pairs, so you can't safely pass a
# comma-separated METHODS list directly inside `--export`. Instead, set it in the environment:
#   METHODS=pbgd,iaptt,hg_acc OUTER_ITERS=200 REPEATS=1 sbatch --export=ALL run_imbalance_all_algorithms_gpu.sbatch
#

#SBATCH --job-name=imbalance_all
#SBATCH --output=imbalance_all_%j.out
#SBATCH --error=imbalance_all_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00

#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"

source /home/masiha/venvs/hypergrad_env/bin/activate

export PYTHONPATH="${ROOT_DIR}/code:${PYTHONPATH:-}"
export HYPERGRAD_REQUIRE_CUDA=1

# Experiment preset: make minima selection more impactful by default (larger validation split + more branch diversity).
# Override with: sbatch --export=ALL,IMB_PRESET=default,...
export IMB_PRESET="${IMB_PRESET:-minsel_effective}"
# For a "minima selection showcase" setting (adds a train-only spurious shortcut patch), run:
#   IMB_PRESET=minsel_showcase METHODS=all OUTER_ITERS=200 sbatch --export=ALL run_imbalance_all_algorithms_gpu.sbatch

# Safer defaults for the preset (override at submit-time as needed).
# `imbalance_base` too small can remove entire classes from the LL train data (keep_prob = base^label),
# which makes balanced accuracy collapse no matter what HG/minsel does.
export IMB_IMBALANCE_BASE="${IMB_IMBALANCE_BASE:-0.5}"
# Use a fixed validation-set size (MNIST train has 60,000 examples).
# This overrides `IMB_TRAIN_FRACTION` inside the Python runner when set.
# - Default/minsel_effective: 600 -> 1% validation split.
# - minsel_showcase: 3000 -> 5% validation split (higher-SNR f(val) for candidate ranking).
if [[ "${IMB_PRESET}" == "minsel_showcase" ]]; then
  export IMB_VAL_SIZE="${IMB_VAL_SIZE:-3000}"
  export IMB_TRAIN_FRACTION="${IMB_TRAIN_FRACTION:-0.95}"
else
  export IMB_VAL_SIZE="${IMB_VAL_SIZE:-600}"
  # Keep this consistent with `IMB_VAL_SIZE` for any scripts that read `IMB_TRAIN_FRACTION` directly.
  export IMB_TRAIN_FRACTION="${IMB_TRAIN_FRACTION:-0.99}"
fi

# HG_acc minima selection: use raw f(val) each selection step (no EMA ranking, no min-improve gate).
# You can still override these at submit-time via: sbatch --export=ALL,HG_ACC_SELECTION_USE_EMA=1,...
export HG_ACC_SELECTION_USE_EMA="${HG_ACC_SELECTION_USE_EMA:-0}"
export HG_ACC_SELECTION_RANK_BY_EMA="${HG_ACC_SELECTION_RANK_BY_EMA:-0}"
export HG_ACC_SELECTION_MIN_IMPROVE="${HG_ACC_SELECTION_MIN_IMPROVE:-0}"
# In showcase settings selection can otherwise be overly conservative.
export HG_ACC_SELECTION_USE_SIGNIFICANCE="${HG_ACC_SELECTION_USE_SIGNIFICANCE:-0}"

# HG-MS (HG_acc_sel) time-budget defaults:
# Keep branches *time-efficient* so HG-MS can make progress under wall-clock budgets.
export HG_ACC_MINSEL_BACKEND="${HG_ACC_MINSEL_BACKEND:-adamw}"
export HG_ACC_EXTRA_BRANCHES="${HG_ACC_EXTRA_BRANCHES:-4}"
export HG_ACC_BRANCH_STEPS="${HG_ACC_BRANCH_STEPS:-1}"
# Avoid one-time burn-in overhead for deterministic branches in time-budget mode.
export HG_ACC_SGLD_BURN_IN="${HG_ACC_SGLD_BURN_IN:-0}"
export HG_ACC_SAMPLE_EVERY="${HG_ACC_SAMPLE_EVERY:-1}"
export HG_ACC_SEL_POLISH_Y_STEPS="${HG_ACC_SEL_POLISH_Y_STEPS:-0}"
export HG_ACC_Y_STEPS_PER_OUTER="${HG_ACC_Y_STEPS_PER_OUTER:-30}"
export HG_ACC_BRANCH_INIT_NOISE_STD="${HG_ACC_BRANCH_INIT_NOISE_STD:-0.01}"
export HG_ACC_BRANCH_LR_MULT_MIN="${HG_ACC_BRANCH_LR_MULT_MIN:-0.25}"
export HG_ACC_BRANCH_LR_MULT_MAX="${HG_ACC_BRANCH_LR_MULT_MAX:-4.0}"
export HG_ACC_BRANCH_WD_MULT_MIN="${HG_ACC_BRANCH_WD_MULT_MIN:-0.25}"
export HG_ACC_BRANCH_WD_MULT_MAX="${HG_ACC_BRANCH_WD_MULT_MAX:-4.0}"

echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V
nvidia-smi -L || true

OUTER_ITERS="${OUTER_ITERS:-200}"
TIME_BUDGET_S="${TIME_BUDGET_S:-}"
TIME_GRID_S="${TIME_GRID_S:-1}"
REPEATS="${REPEATS:-1}"
SEED="${SEED:-2424}"
METHODS="${METHODS:-all}"

OUT_DIR="${OUTPUT_ROOT:-${ROOT_DIR}/outputs}/${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

python -u "${ROOT_DIR}/code/run_all_algorithms.py" \
  --outer-iters "${OUTER_ITERS}" \
  $( [[ -n "${TIME_BUDGET_S}" ]] && echo "--time-budget-seconds ${TIME_BUDGET_S}" ) \
  $( [[ -n "${TIME_BUDGET_S}" ]] && echo "--time-grid-seconds ${TIME_GRID_S}" ) \
  --repeats "${REPEATS}" \
  --seed "${SEED}" \
  --methods "${METHODS}" \
  --save-dir "${OUT_DIR}" \
  --no-show
