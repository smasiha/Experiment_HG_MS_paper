#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${PACKAGE_ROOT_DEFAULT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/controlled/default_runs}"
SEED="${SEED:-2424}"
TASKS="${TASKS:-denoising multilingual}"

cd "${ROOT_DIR}/controlled"

submit_method() {
  local task="$1"
  local method="$2"

  local batch_size=16
  local selection_val_batch_size=64
  local selection_val_mode="target_mixture"
  local eval_batch_size=32
  local optimizer="sgd"
  local reweight_model_lr="0.0005"
  local reweight_outer_lr="0.1"
  local inner_steps=4
  local m_branches=2
  local reweight_steps="469"
  local eval_every="25"

  if [[ "${task}" == "multilingual" ]]; then
    reweight_steps="200"
    eval_every="50"
    selection_val_batch_size=64
  fi

  if [[ "${method}" == "uniform" ]]; then
    selection_val_batch_size=16
    inner_steps=1
    m_branches=1
  elif [[ "${method}" == "scalebio" ]]; then
    reweight_model_lr="0.0001"
    reweight_outer_lr="0.05"
    inner_steps=1
    m_branches=1
  elif [[ "${method}" == "hg_single" ]]; then
    m_branches=1
  fi

  if [[ "${task}" == "multilingual" && "${method}" == hg_* ]]; then
    reweight_outer_lr="1.0"
  fi

  sbatch --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",TASK="${task}",METHOD="${method}",SEED="${SEED}",MAX_LENGTH=128,BATCH_SIZE="${batch_size}",SELECTION_VAL_BATCH_SIZE="${selection_val_batch_size}",SELECTION_VAL_MODE="${selection_val_mode}",EVAL_BATCH_SIZE="${eval_batch_size}",OPTIMIZER="${optimizer}",REWEIGHT_MODEL_LR="${reweight_model_lr}",REWEIGHT_OUTER_LR="${reweight_outer_lr}",INNER_STEPS="${inner_steps}",M_BRANCHES="${m_branches}",REWEIGHT_STEPS="${reweight_steps}",EVAL_EVERY="${eval_every}" ./run_small_scale_reweight.sbatch
}

for task in ${TASKS}; do
  for method in uniform scalebio hg_single hg_random hg_ms; do
    submit_method "${task}" "${method}"
  done
done
