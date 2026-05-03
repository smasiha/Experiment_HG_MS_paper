#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-20}"
SEED="${SEED:-2424}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"

for method in uniform hg_single hg_random hg_ms; do
  sbatch \
    --export=ALL,ROOT_DIR="${ROOT_DIR}",METHOD="${method}",TIME_BUDGET_MIN="${TIME_BUDGET_MIN}",SEED="${SEED}",MODEL_NAME="${MODEL_NAME}" \
    "${ROOT_DIR}/LLM_minima_selection/run_llm_minima_selection.sbatch"
done
