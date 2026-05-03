#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SEED="${SEED:-2424}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
METHODS="${METHODS:-uniform less rho_loss scalebio hg_single hg_random hg_ms}"

MATH_OUTPUT_ROOT="${MATH_OUTPUT_ROOT:-/scratch/${USER}/realworld_math_full_small_${TAG}}"
INSTRUCTION_OUTPUT_ROOT="${INSTRUCTION_OUTPUT_ROOT:-/scratch/${USER}/realworld_instruction_full_small_${TAG}}"

cd "${ROOT_DIR}"

submit_math() {
  local method="$1"
  local outer_lr="0.1"
  if [[ "${method}" == "scalebio" ]]; then
    outer_lr="1.0"
  elif [[ "${method}" == hg_* ]]; then
    outer_lr="5.0"
  fi

  sbatch \
    --export=ALL,METHOD="${method}",MODEL_NAME="${MODEL_NAME}",REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",SEED="${SEED}",STOP_AFTER_REWEIGHT=0,OUTPUT_ROOT="${MATH_OUTPUT_ROOT}",MAX_LENGTH=512,VAL_BATCH_SIZE=1,SELECTION_VAL_BATCH_SIZE=1,EVAL_BATCH_SIZE=1,REWEIGHT_MODEL_LR=1e-4,REWEIGHT_OUTER_LR="${outer_lr}",BRANCH_NOISE_STD=0,REWEIGHT_TRAIN_PER_SOURCE=500,ALLOW_TRAIN_SHORTFALL=1,PER_SOURCE_BATCH=1,REWEIGHT_STEPS=50,SELECTION_VAL_BATCHES=1,HYPERGRAD_VAL_BATCHES=1,FINAL_TRAIN_SIZE=20000,FINAL_FT_MICRO_BATCH=1,FINAL_FT_GRAD_ACCUM=64,REFERENCE_BATCH_SIZE=2,REFERENCE_FT_STEPS=100 \
    run_exact_math_reasoning.sbatch
}

submit_instruction() {
  local method="$1"
  local outer_lr="0.1"
  if [[ "${method}" == "scalebio" ]]; then
    outer_lr="1.0"
  elif [[ "${method}" == hg_* ]]; then
    outer_lr="10.0"
  fi

  sbatch \
    --export=ALL,METHOD="${method}",MODEL_NAME="${MODEL_NAME}",REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",SEED="${SEED}",STOP_AFTER_REWEIGHT=0,OUTPUT_ROOT="${INSTRUCTION_OUTPUT_ROOT}",MAX_LENGTH=256,VAL_BATCH_SIZE=1,SELECTION_VAL_BATCH_SIZE=1,EVAL_BATCH_SIZE=1,REWEIGHT_MODEL_LR=1e-4,REWEIGHT_OUTER_LR="${outer_lr}",BRANCH_NOISE_STD=0,ALLOW_TRAIN_SHORTFALL=1,PER_SOURCE_BATCH=1,REWEIGHT_STEPS=50,SELECTION_VAL_BATCHES=1,HYPERGRAD_VAL_BATCHES=1,FINAL_TRAIN_SIZE=10000,FINAL_FT_MICRO_BATCH=1,FINAL_FT_GRAD_ACCUM=64,REFERENCE_BATCH_SIZE=2,REFERENCE_FT_STEPS=100 \
    run_real_world_instruction.sbatch
}

echo "Math output root: ${MATH_OUTPUT_ROOT}"
echo "Instruction output root: ${INSTRUCTION_OUTPUT_ROOT}"
for method in ${METHODS}; do
  submit_math "${method}"
done
for method in ${METHODS}; do
  submit_instruction "${method}"
done
