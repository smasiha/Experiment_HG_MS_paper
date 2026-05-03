#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
TAG="${TAG:-math_wallclock_2h_multiseed_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}/math}"

# Two-hour reweight/search budget. The common downstream fine-tuning stage is the
# same for every method and is not wall-clock capped here.
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS:-7200}"

# Space-separated list of seeds and methods. One-shot baselines are allowed to
# finish early under this maximum budget.
SEEDS="${SEEDS:-2424 2525 2626}"
METHODS="${METHODS:-uniform less rho_loss scalebio hg_single hg_ms}"

cd "${ROOT_DIR}"

submit_job() {
  local method="$1"
  local seed="$2"
  local run_name="${method}_seed${seed}_wc2h"

  local outer_lr inner_steps m_branches selection_batches hypergrad_batches
  local selection_batch_size branch_noise train_tolerance train_penalty reweight_steps
  local theta_optimizer="sgd"

  case "${method}" in
    uniform)
      outer_lr="0.1";   inner_steps="1"; m_branches="1"; selection_batches="1"; hypergrad_batches="1"; selection_batch_size="1"; branch_noise="0"; train_tolerance="0.05"; train_penalty="0.0"; reweight_steps="1"
      ;;
    less)
      outer_lr="0.1";   inner_steps="1"; m_branches="1"; selection_batches="1"; hypergrad_batches="1"; selection_batch_size="1"; branch_noise="0"; train_tolerance="0.05"; train_penalty="0.0"; reweight_steps="1"
      ;;
    rho_loss)
      outer_lr="0.1";   inner_steps="1"; m_branches="1"; selection_batches="1"; hypergrad_batches="1"; selection_batch_size="1"; branch_noise="0"; train_tolerance="0.05"; train_penalty="0.0"; reweight_steps="1"
      ;;
    scalebio)
      outer_lr="1.0";   inner_steps="1"; m_branches="1"; selection_batches="1"; hypergrad_batches="1"; selection_batch_size="1"; branch_noise="0"; train_tolerance="0.05"; train_penalty="0.0"; reweight_steps="10000"
      ;;
    hg_single)
      outer_lr="5.0";   inner_steps="1"; m_branches="1"; selection_batches="4"; hypergrad_batches="4"; selection_batch_size="2"; branch_noise="0"; train_tolerance="0.05"; train_penalty="0.0"; reweight_steps="5000"
      ;;
    hg_ms)
      outer_lr="10.0";  inner_steps="2"; m_branches="4"; selection_batches="6"; hypergrad_batches="6"; selection_batch_size="2"; branch_noise="0"; train_tolerance="0.30"; train_penalty="0.25"; reweight_steps="3000"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 1
      ;;
  esac

  sbatch \
    --job-name="wc2h_${method}_${seed}" \
    --export=ALL,\
METHOD="${method}",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
SEED="${seed}",\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
REWEIGHT_WALL_CLOCK_SECONDS="${REWEIGHT_WALL_CLOCK_SECONDS}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
MAX_LENGTH=512,\
VAL_BATCH_SIZE=1,\
SELECTION_VAL_BATCH_SIZE="${selection_batch_size}",\
EVAL_BATCH_SIZE=1,\
REWEIGHT_MODEL_LR=1e-4,\
REWEIGHT_OUTER_LR="${outer_lr}",\
THETA_OPTIMIZER="${theta_optimizer}",\
BRANCH_NOISE_STD="${branch_noise}",\
REWEIGHT_TRAIN_PER_SOURCE=500,\
ALLOW_TRAIN_SHORTFALL=1,\
PER_SOURCE_BATCH=1,\
REWEIGHT_STEPS="${reweight_steps}",\
SELECTION_VAL_BATCHES="${selection_batches}",\
HYPERGRAD_VAL_BATCHES="${hypergrad_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${train_tolerance}",\
SELECTION_TRAIN_PENALTY="${train_penalty}",\
INNER_STEPS="${inner_steps}",\
M_BRANCHES="${m_branches}",\
FINAL_TRAIN_SIZE=20000,\
FINAL_FT_MICRO_BATCH=1,\
FINAL_FT_GRAD_ACCUM=32,\
GSM8K_EVAL_SIZE=0,\
REFERENCE_BATCH_SIZE=2,\
REFERENCE_FT_STEPS=100 \
    run_exact_math_reasoning.sbatch
}

echo "Math 2h wall-clock multi-seed output root: ${OUTPUT_ROOT}"
echo "Reweight/search budget (sec): ${REWEIGHT_WALL_CLOCK_SECONDS}"
echo "Seeds: ${SEEDS}"
echo "Methods: ${METHODS}"

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    submit_job "${method}" "${seed}"
  done
done
