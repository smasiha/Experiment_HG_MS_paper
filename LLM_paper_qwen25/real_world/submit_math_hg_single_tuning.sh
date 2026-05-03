#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-Qwen/Qwen2-1.5B}"
SEED="${SEED:-2424}"
TAG="${TAG:-math_hg_single_tuning_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"

# Short, cheap tuning sweep (reweight-only) for HG-single
# Goal: pick settings that minimize held-out tune loss.
REWEIGHT_STEPS="${REWEIGHT_STEPS:-120}"
REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-500}"
VAL_PER_SOURCE="${VAL_PER_SOURCE:-140}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
PER_SOURCE_BATCH="${PER_SOURCE_BATCH:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
SELECTION_VAL_BATCH_SIZE="${SELECTION_VAL_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

# Grid (reduced, pragmatic)
OUTER_LRS=("5.0" "10.0")
MODEL_LRS=("1e-5")
INNER_STEPS=("1" "2")
SELECTION_VAL_BATCHES=("4")
HYPERGRAD_VAL_BATCHES=("4")
TRAIN_TOLERANCES=("0.05")
TRAIN_PENALTIES=("0.0")
BRANCH_NOISES=("0")

cd "${ROOT_DIR}"

submit_job() {
  local outer_lr="$1"
  local model_lr="$2"
  local inner_steps="$3"
  local sel_batches="$4"
  local hyp_batches="$5"
  local tol="$6"
  local pen="$7"
  local noise="$8"

  local run_name="hgs_tune_olr${outer_lr}_mlr${model_lr}_in${inner_steps}_sb${sel_batches}_hb${hyp_batches}_tol${tol}_pen${pen}_n${noise}"

  sbatch \
    --job-name="hg_single_tune" \
    --export=ALL,\
METHOD="hg_single",\
MODEL_NAME="${MODEL_NAME}",\
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME}",\
LOCAL_MODEL_ONLY=0,\
HF_CACHE_ROOT="/scratch/${USER}/hf_cache_jobs/llama3_8b",\
SEED="${SEED}",\
STOP_AFTER_REWEIGHT=1,\
USE_BEST_REWEIGHT_CHECKPOINT=1,\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
RUN_NAME="${run_name}",\
MAX_LENGTH="${MAX_LENGTH}",\
REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE}",\
VAL_PER_SOURCE="${VAL_PER_SOURCE}",\
TRAIN_RAW_OVERSAMPLE=4.0,\
VAL_RAW_OVERSAMPLE=1.2,\
PER_SOURCE_BATCH="${PER_SOURCE_BATCH}",\
VAL_BATCH_SIZE="${VAL_BATCH_SIZE}",\
SELECTION_VAL_BATCH_SIZE="${SELECTION_VAL_BATCH_SIZE}",\
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}",\
REWEIGHT_MODEL_LR="${model_lr}",\
REWEIGHT_OUTER_LR="${outer_lr}",\
BRANCH_NOISE_STD="${noise}",\
REWEIGHT_STEPS="${REWEIGHT_STEPS}",\
SELECTION_VAL_BATCHES="${sel_batches}",\
HYPERGRAD_VAL_BATCHES="${hyp_batches}",\
REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1,\
SELECTION_TRAIN_TOLERANCE="${tol}",\
SELECTION_TRAIN_PENALTY="${pen}",\
INNER_STEPS="${inner_steps}",\
M_BRANCHES=1,\
LORA_R=128,\
LORA_ALPHA=512,\
LORA_DROPOUT=0.1,\
GRADIENT_CHECKPOINTING=1,\
BF16=1 \
    run_exact_math_reasoning.sbatch
}

echo "HG-single tuning output root: ${OUTPUT_ROOT}"

total=0
for outer_lr in "${OUTER_LRS[@]}"; do
  for model_lr in "${MODEL_LRS[@]}"; do
    for inner_steps in "${INNER_STEPS[@]}"; do
      for sel_batches in "${SELECTION_VAL_BATCHES[@]}"; do
        for hyp_batches in "${HYPERGRAD_VAL_BATCHES[@]}"; do
          for tol in "${TRAIN_TOLERANCES[@]}"; do
            for pen in "${TRAIN_PENALTIES[@]}"; do
              for noise in "${BRANCH_NOISES[@]}"; do
                submit_job "${outer_lr}" "${model_lr}" "${inner_steps}" "${sel_batches}" "${hyp_batches}" "${tol}" "${pen}" "${noise}"
                total=$((total+1))
              done
            done
          done
        done
      done
    done
  done
done

echo "Submitted ${total} HG-single tuning jobs."
