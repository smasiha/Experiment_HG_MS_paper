#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/masiha/multilingual_fixed_grid_20260315}"
SEED="${SEED:-2424}"
MAX_LENGTH="${MAX_LENGTH:-128}"

cd "${ROOT_DIR}"

submit_run() {
  local tag="$1"
  local fixed_lambda="$2"
  sbatch --export=ALL,ROOT_DIR="${ROOT_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}/${tag}",SEED="${SEED}",MAX_LENGTH="${MAX_LENGTH}",LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY:-0}",TASK=multilingual,METHOD=uniform,MODEL_NAME=gpt2,FIXED_LAMBDA="${fixed_lambda}",BATCH_SIZE=8,EVAL_BATCH_SIZE=32,REWEIGHT_STEPS=100,REWEIGHT_MODEL_LR=0.001,REWEIGHT_OUTER_LR=0.0,REWEIGHT_WEIGHT_DECAY=0.0,EVAL_EVERY=25 ./run_small_scale_reweight.sbatch
}

submit_run zh040 "0.40:0.60"
submit_run zh045 "0.45:0.55"
submit_run zh050 "0.50:0.50"
submit_run zh055 "0.55:0.45"
submit_run zh060 "0.60:0.40"
submit_run zh065 "0.65:0.35"
submit_run zh070 "0.70:0.30"
