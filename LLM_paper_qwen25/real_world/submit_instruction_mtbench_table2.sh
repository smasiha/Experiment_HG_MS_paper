#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_DIR}/run_real_world_instruction.sbatch}"

TAG="${TAG:-instruction_mtbench_table2_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/${TAG}}"
JOB_TIME="${JOB_TIME:-}"

MODEL_PRESETS="${MODEL_PRESETS:-llama3_8b qwen2_7b gemma2_9b}"
METHODS="${METHODS:-uniform less rho_loss scalebio hg_single hg_ms}"
SEEDS="${SEEDS:-2424}"

RUN_MTBENCH_EVAL="${RUN_MTBENCH_EVAL:-1}"
RUN_MTBENCH_JUDGE="${RUN_MTBENCH_JUDGE:-1}"
MTBENCH_QUESTION_SOURCE="${MTBENCH_QUESTION_SOURCE:-fastchat}"
MTBENCH_QUESTION_FILE="${MTBENCH_QUESTION_FILE:-}"
MTBENCH_QUESTION_URL="${MTBENCH_QUESTION_URL:-https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl}"
MTBENCH_QUESTION_LIMIT="${MTBENCH_QUESTION_LIMIT:-0}"
MTBENCH_PROMPT_STYLE="${MTBENCH_PROMPT_STYLE:-training_text}"
MTBENCH_JUDGE_MODEL="${MTBENCH_JUDGE_MODEL:-gpt-4o}"
MTBENCH_API_BASE="${MTBENCH_API_BASE:-${OPENAI_API_BASE:-https://api.openai.com/v1}}"
MTBENCH_API_KEY_ENV="${MTBENCH_API_KEY_ENV:-OPENAI_API_KEY}"
MTBENCH_USE_REFERENCE="${MTBENCH_USE_REFERENCE:-1}"
MTBENCH_MAX_INPUT_LENGTH="${MTBENCH_MAX_INPUT_LENGTH:-1024}"
MTBENCH_MAX_NEW_TOKENS="${MTBENCH_MAX_NEW_TOKENS:-512}"

USE_BEST_REWEIGHT_CHECKPOINT="${USE_BEST_REWEIGHT_CHECKPOINT:-1}"
INSTRUCTION_BUDGET_SECONDS="${INSTRUCTION_BUDGET_SECONDS:-1004}"
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-10000}"
REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-3000}"
REWEIGHT_TRAIN_RAW_OVERSAMPLE="${REWEIGHT_TRAIN_RAW_OVERSAMPLE:-1.5}"
FINAL_TRAIN_RAW_OVERSAMPLE="${FINAL_TRAIN_RAW_OVERSAMPLE:-1.1}"
FINAL_FT_EPOCHS="${FINAL_FT_EPOCHS:-1}"
FINAL_FT_LR="${FINAL_FT_LR:-8e-6}"
ALLOW_TRAIN_SHORTFALL="${ALLOW_TRAIN_SHORTFALL:-1}"
STOP_AFTER_REWEIGHT="${STOP_AFTER_REWEIGHT:-0}"
REWEIGHT_EPOCHS="${REWEIGHT_EPOCHS:-3}"
REFERENCE_FT_STEPS="${REFERENCE_FT_STEPS:-200}"
BASELINE_SCORE_SAMPLES_PER_SOURCE="${BASELINE_SCORE_SAMPLES_PER_SOURCE:-256}"
TUNE_EVERY="${TUNE_EVERY:-100}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY:-0}"
ITERATIVE_GRADIENT_CHECKPOINTING="${ITERATIVE_GRADIENT_CHECKPOINTING:-1}"
ITERATIVE_VAL_BATCH_SIZE="${ITERATIVE_VAL_BATCH_SIZE:-1}"
ITERATIVE_SELECTION_VAL_BATCH_SIZE="${ITERATIVE_SELECTION_VAL_BATCH_SIZE:-1}"
ITERATIVE_EVAL_BATCH_SIZE="${ITERATIVE_EVAL_BATCH_SIZE:-1}"
ITERATIVE_REFERENCE_BATCH_SIZE="${ITERATIVE_REFERENCE_BATCH_SIZE:-1}"
ITERATIVE_QWEN_MAX_LENGTH="${ITERATIVE_QWEN_MAX_LENGTH:-256}"
ITERATIVE_QWEN_LORA_R="${ITERATIVE_QWEN_LORA_R:-4}"
ITERATIVE_QWEN_LORA_ALPHA="${ITERATIVE_QWEN_LORA_ALPHA:-8}"
ITERATIVE_LARGE_MAX_LENGTH="${ITERATIVE_LARGE_MAX_LENGTH:-128}"
ITERATIVE_LARGE_LORA_R="${ITERATIVE_LARGE_LORA_R:-2}"
ITERATIVE_LARGE_LORA_ALPHA="${ITERATIVE_LARGE_LORA_ALPHA:-8}"

UNIFORM_OUTER_LR="${UNIFORM_OUTER_LR:-0.01}"
BASELINE_REWEIGHT_MODEL_LR="${BASELINE_REWEIGHT_MODEL_LR:-1e-5}"
SCALEBIO_OUTER_LR="${SCALEBIO_OUTER_LR:-0.01}"
SCALEBIO_REWEIGHT_MODEL_LR="${SCALEBIO_REWEIGHT_MODEL_LR:-1e-5}"
HG_SINGLE_OUTER_LR="${HG_SINGLE_OUTER_LR:-10.0}"
HG_SINGLE_REWEIGHT_MODEL_LR="${HG_SINGLE_REWEIGHT_MODEL_LR:-2e-4}"
HG_MS_OUTER_LR="${HG_MS_OUTER_LR:-0.05}"
HG_MS_REWEIGHT_MODEL_LR="${HG_MS_REWEIGHT_MODEL_LR:-2e-4}"

SCALEBIO_STEP_CAP="${SCALEBIO_STEP_CAP:-100000}"
HG_SINGLE_STEP_CAP="${HG_SINGLE_STEP_CAP:-100000}"
HG_MS_STEP_CAP="${HG_MS_STEP_CAP:-100000}"

HG_SELECTION_BATCHES="${HG_SELECTION_BATCHES:-1}"
HG_HYPERGRAD_BATCHES="${HG_HYPERGRAD_BATCHES:-1}"
HG_SINGLE_TOLERANCE="${HG_SINGLE_TOLERANCE:-0.50}"
HG_MS_TOLERANCE="${HG_MS_TOLERANCE:-0.50}"
HG_SINGLE_PENALTY="${HG_SINGLE_PENALTY:-0.0}"
HG_MS_PENALTY="${HG_MS_PENALTY:-0.0}"
HG_BRANCH_NOISE_STD="${HG_BRANCH_NOISE_STD:-5e-4}"
HG_MS_BRANCHES="${HG_MS_BRANCHES:-8}"
HG_PER_SOURCE_BATCH="${HG_PER_SOURCE_BATCH:-1}"
INITIAL_LAMBDA_METRICS_PATH="${INITIAL_LAMBDA_METRICS_PATH:-}"
INITIAL_LAMBDA_BLEND="${INITIAL_LAMBDA_BLEND:-1.0}"

DRY_RUN="${DRY_RUN:-0}"
PREFER_LOCAL_GATED_MODELS="${PREFER_LOCAL_GATED_MODELS:-1}"
PREFER_LOCAL_MODEL_SNAPSHOTS="${PREFER_LOCAL_MODEL_SNAPSHOTS:-1}"

find_local_model_snapshot() {
  python - "$1" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

repo_id = sys.argv[1]
cache_key = "models--" + repo_id.replace("/", "--")
roots = [
    Path(f"/scratch/{Path.home().name}/hf_cache"),
    Path(f"/scratch/{Path.home().name}/hf_cache_jobs"),
    Path(f"/home/{Path.home().name}/.cache/huggingface/hub"),
]

best: tuple[int, float, Path] | None = None
for root in roots:
    if not root.exists():
        continue
    for config_path in root.glob(f"**/{cache_key}/snapshots/*/config.json"):
        snap = config_path.parent
        files = {child.name for child in snap.iterdir() if child.is_file()}
        if "config.json" not in files:
            continue
        if "model.safetensors.index.json" in files:
            try:
                index = json.loads((snap / "model.safetensors.index.json").read_text())
                weight_map = index.get("weight_map", {})
                shard_names = sorted(set(weight_map.values()))
            except Exception:
                continue
            if not shard_names or any(not (snap / name).exists() for name in shard_names):
                continue
            completeness = 1000 + len(shard_names)
        elif "model.safetensors" in files or "pytorch_model.bin" in files:
            completeness = 100
        else:
            continue
        if "tokenizer.json" in files:
            completeness += 10
        elif "tokenizer_config.json" in files:
            completeness += 5
        score = (completeness, config_path.stat().st_mtime, snap)
        if best is None or score > best:
            best = score

if best is None:
    raise SystemExit(1)
print(best[2])
PY
}

apply_instruction_model_preset() {
  local preset="$1"
  unset MODEL_NAME REFERENCE_MODEL_NAME MAX_LENGTH LORA_R LORA_ALPHA LORA_DROPOUT
  unset FINAL_FT_MICRO_BATCH FINAL_FT_GRAD_ACCUM VAL_BATCH_SIZE SELECTION_VAL_BATCH_SIZE
  unset EVAL_BATCH_SIZE REFERENCE_BATCH_SIZE MTBENCH_GENERATION_BATCH_SIZE
  local target_repo_id=""
  MODEL_LOCAL_ONLY="${LOCAL_MODEL_ONLY:-0}"

  case "${preset}" in
    llama3_8b)
      target_repo_id="meta-llama/Meta-Llama-3-8B-Instruct"
      MODEL_NAME="${MODEL_NAME:-${target_repo_id}}"
      REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-${MODEL_NAME}}"
      MAX_LENGTH="${MAX_LENGTH:-256}"
      LORA_R="${LORA_R:-4}"
      LORA_ALPHA="${LORA_ALPHA:-8}"
      LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
      FINAL_FT_MICRO_BATCH="${FINAL_FT_MICRO_BATCH:-1}"
      FINAL_FT_GRAD_ACCUM="${FINAL_FT_GRAD_ACCUM:-64}"
      VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
      SELECTION_VAL_BATCH_SIZE="${SELECTION_VAL_BATCH_SIZE:-2}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
      REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-2}"
      MTBENCH_GENERATION_BATCH_SIZE="${MTBENCH_GENERATION_BATCH_SIZE:-1}"
      ;;
    qwen2_7b)
      target_repo_id="Qwen/Qwen2-7B-Instruct"
      MODEL_NAME="${MODEL_NAME:-${target_repo_id}}"
      REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-${MODEL_NAME}}"
      MAX_LENGTH="${MAX_LENGTH:-256}"
      LORA_R="${LORA_R:-4}"
      LORA_ALPHA="${LORA_ALPHA:-8}"
      LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
      FINAL_FT_MICRO_BATCH="${FINAL_FT_MICRO_BATCH:-1}"
      FINAL_FT_GRAD_ACCUM="${FINAL_FT_GRAD_ACCUM:-64}"
      VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
      SELECTION_VAL_BATCH_SIZE="${SELECTION_VAL_BATCH_SIZE:-2}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
      REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-2}"
      MTBENCH_GENERATION_BATCH_SIZE="${MTBENCH_GENERATION_BATCH_SIZE:-1}"
      ;;
    gemma2_9b)
      target_repo_id="google/gemma-2-9b-it"
      MODEL_NAME="${MODEL_NAME:-${target_repo_id}}"
      REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-${MODEL_NAME}}"
      MAX_LENGTH="${MAX_LENGTH:-256}"
      LORA_R="${LORA_R:-4}"
      LORA_ALPHA="${LORA_ALPHA:-8}"
      LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
      FINAL_FT_MICRO_BATCH="${FINAL_FT_MICRO_BATCH:-1}"
      FINAL_FT_GRAD_ACCUM="${FINAL_FT_GRAD_ACCUM:-64}"
      VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
      SELECTION_VAL_BATCH_SIZE="${SELECTION_VAL_BATCH_SIZE:-1}"
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
      REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-1}"
      MTBENCH_GENERATION_BATCH_SIZE="${MTBENCH_GENERATION_BATCH_SIZE:-1}"
      ;;
    *)
      echo "Unknown instruction model preset: ${preset}" >&2
      return 1
      ;;
  esac

  if [[ "${PREFER_LOCAL_MODEL_SNAPSHOTS}" == "1" ]] || [[ "${PREFER_LOCAL_GATED_MODELS}" == "1" && "${target_repo_id}" =~ ^(meta-llama/Meta-Llama-3-8B-Instruct|google/gemma-2-9b-it)$ ]]; then
    local snapshot_path=""
    if snapshot_path="$(find_local_model_snapshot "${target_repo_id}")"; then
      MODEL_NAME="${snapshot_path}"
      REFERENCE_MODEL_NAME="${snapshot_path}"
      MODEL_LOCAL_ONLY="1"
    fi
  fi
}

configure_method() {
  local method="$1"
  case "${method}" in
    uniform|less|rho_loss)
      METHOD_REWEIGHT_WALLCLOCK="0"
      METHOD_REWEIGHT_STEPS="1"
      METHOD_THETA_OPTIMIZER="sgd"
      METHOD_REWEIGHT_MODEL_LR="${BASELINE_REWEIGHT_MODEL_LR}"
      METHOD_REWEIGHT_OUTER_LR="${UNIFORM_OUTER_LR}"
      METHOD_M_BRANCHES="1"
      METHOD_PER_SOURCE_BATCH="1"
      METHOD_SELECTION_BATCHES="1"
      METHOD_HYPERGRAD_BATCHES="1"
      METHOD_TRAIN_TOLERANCE="0.05"
      METHOD_TRAIN_PENALTY="0.0"
      METHOD_BRANCH_NOISE_STD="0.0"
      ;;
    scalebio)
      METHOD_REWEIGHT_WALLCLOCK="${INSTRUCTION_BUDGET_SECONDS}"
      METHOD_REWEIGHT_STEPS="${SCALEBIO_STEP_CAP}"
      METHOD_THETA_OPTIMIZER="sgd"
      METHOD_REWEIGHT_MODEL_LR="${SCALEBIO_REWEIGHT_MODEL_LR}"
      METHOD_REWEIGHT_OUTER_LR="${SCALEBIO_OUTER_LR}"
      METHOD_M_BRANCHES="1"
      METHOD_PER_SOURCE_BATCH="1"
      METHOD_SELECTION_BATCHES="1"
      METHOD_HYPERGRAD_BATCHES="1"
      METHOD_TRAIN_TOLERANCE="0.05"
      METHOD_TRAIN_PENALTY="0.0"
      METHOD_BRANCH_NOISE_STD="0.0"
      ;;
    hg_single)
      METHOD_REWEIGHT_WALLCLOCK="${INSTRUCTION_BUDGET_SECONDS}"
      METHOD_REWEIGHT_STEPS="${HG_SINGLE_STEP_CAP}"
      METHOD_THETA_OPTIMIZER="adam"
      METHOD_REWEIGHT_MODEL_LR="${HG_SINGLE_REWEIGHT_MODEL_LR}"
      METHOD_REWEIGHT_OUTER_LR="${HG_SINGLE_OUTER_LR}"
      METHOD_M_BRANCHES="1"
      METHOD_PER_SOURCE_BATCH="${HG_PER_SOURCE_BATCH}"
      METHOD_SELECTION_BATCHES="${HG_SELECTION_BATCHES}"
      METHOD_HYPERGRAD_BATCHES="${HG_HYPERGRAD_BATCHES}"
      METHOD_TRAIN_TOLERANCE="${HG_SINGLE_TOLERANCE}"
      METHOD_TRAIN_PENALTY="${HG_SINGLE_PENALTY}"
      METHOD_BRANCH_NOISE_STD="${HG_BRANCH_NOISE_STD}"
      ;;
    hg_ms)
      METHOD_REWEIGHT_WALLCLOCK="${INSTRUCTION_BUDGET_SECONDS}"
      METHOD_REWEIGHT_STEPS="${HG_MS_STEP_CAP}"
      METHOD_THETA_OPTIMIZER="adam"
      METHOD_REWEIGHT_MODEL_LR="${HG_MS_REWEIGHT_MODEL_LR}"
      METHOD_REWEIGHT_OUTER_LR="${HG_MS_OUTER_LR}"
      METHOD_M_BRANCHES="${HG_MS_BRANCHES}"
      METHOD_PER_SOURCE_BATCH="${HG_PER_SOURCE_BATCH}"
      METHOD_SELECTION_BATCHES="${HG_SELECTION_BATCHES}"
      METHOD_HYPERGRAD_BATCHES="${HG_HYPERGRAD_BATCHES}"
      METHOD_TRAIN_TOLERANCE="${HG_MS_TOLERANCE}"
      METHOD_TRAIN_PENALTY="${HG_MS_PENALTY}"
      METHOD_BRANCH_NOISE_STD="${HG_BRANCH_NOISE_STD}"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 1
      ;;
  esac
}

submit_job() {
  local preset="$1"
  local seed="$2"
  local method="$3"

  configure_method "${method}"

  local job_max_length="${MAX_LENGTH}"
  local job_val_batch_size="${VAL_BATCH_SIZE}"
  local job_selection_val_batch_size="${SELECTION_VAL_BATCH_SIZE}"
  local job_eval_batch_size="${EVAL_BATCH_SIZE}"
  local job_reference_batch_size="${REFERENCE_BATCH_SIZE}"
  local job_lora_r="${LORA_R}"
  local job_lora_alpha="${LORA_ALPHA}"
  local job_lora_dropout="${LORA_DROPOUT}"
  local job_gradient_checkpointing="${GRADIENT_CHECKPOINTING}"

  if [[ "${method}" != "uniform" && "${method}" != "less" && "${method}" != "rho_loss" ]]; then
    job_gradient_checkpointing="${ITERATIVE_GRADIENT_CHECKPOINTING}"
    job_val_batch_size="${ITERATIVE_VAL_BATCH_SIZE}"
    job_selection_val_batch_size="${ITERATIVE_SELECTION_VAL_BATCH_SIZE}"
    job_eval_batch_size="${ITERATIVE_EVAL_BATCH_SIZE}"
    job_reference_batch_size="${ITERATIVE_REFERENCE_BATCH_SIZE}"
    case "${preset}" in
      qwen2_7b)
        job_max_length="${ITERATIVE_QWEN_MAX_LENGTH}"
        job_lora_r="${ITERATIVE_QWEN_LORA_R}"
        job_lora_alpha="${ITERATIVE_QWEN_LORA_ALPHA}"
        ;;
      llama3_8b|gemma2_9b)
        job_max_length="${ITERATIVE_LARGE_MAX_LENGTH}"
        job_lora_r="${ITERATIVE_LARGE_LORA_R}"
        job_lora_alpha="${ITERATIVE_LARGE_LORA_ALPHA}"
        ;;
    esac
  fi

  local run_name="${method}_seed${seed}_${preset}_instr_mtbench"
  local job_name="t2instr_${preset}_${method}_${seed}"
  local model_output_root="${OUTPUT_ROOT}/${preset}"

  local export_items=(
    "METHOD=${method}"
    "ROOT_DIR=${PACKAGE_ROOT}"
    "MODEL_NAME=${MODEL_NAME}"
    "REFERENCE_MODEL_NAME=${REFERENCE_MODEL_NAME}"
    "SEED=${seed}"
    "USE_BEST_REWEIGHT_CHECKPOINT=${USE_BEST_REWEIGHT_CHECKPOINT}"
    "REWEIGHT_WALL_CLOCK_SECONDS=${METHOD_REWEIGHT_WALLCLOCK}"
    "OUTPUT_ROOT=${model_output_root}"
    "RUN_NAME=${run_name}"
    "MAX_LENGTH=${job_max_length}"
    "VAL_BATCH_SIZE=${job_val_batch_size}"
    "SELECTION_VAL_BATCH_SIZE=${job_selection_val_batch_size}"
    "EVAL_BATCH_SIZE=${job_eval_batch_size}"
    "THETA_OPTIMIZER=${METHOD_THETA_OPTIMIZER}"
    "REWEIGHT_MODEL_LR=${METHOD_REWEIGHT_MODEL_LR}"
    "REWEIGHT_OUTER_LR=${METHOD_REWEIGHT_OUTER_LR}"
    "INITIAL_LAMBDA_METRICS_PATH=${INITIAL_LAMBDA_METRICS_PATH}"
    "INITIAL_LAMBDA_BLEND=${INITIAL_LAMBDA_BLEND}"
    "BRANCH_NOISE_STD=${METHOD_BRANCH_NOISE_STD}"
    "REWEIGHT_TRAIN_PER_SOURCE=${REWEIGHT_TRAIN_PER_SOURCE}"
    "REWEIGHT_TRAIN_RAW_OVERSAMPLE=${REWEIGHT_TRAIN_RAW_OVERSAMPLE}"
    "REWEIGHT_EPOCHS=${REWEIGHT_EPOCHS}"
    "ALLOW_TRAIN_SHORTFALL=${ALLOW_TRAIN_SHORTFALL}"
    "STOP_AFTER_REWEIGHT=${STOP_AFTER_REWEIGHT}"
    "PER_SOURCE_BATCH=${METHOD_PER_SOURCE_BATCH}"
    "REWEIGHT_STEPS=${METHOD_REWEIGHT_STEPS}"
    "SELECTION_VAL_BATCHES=${METHOD_SELECTION_BATCHES}"
    "HYPERGRAD_VAL_BATCHES=${METHOD_HYPERGRAD_BATCHES}"
    "REUSE_SELECTION_BATCHES_FOR_HYPERGRAD=1"
    "SELECTION_TRAIN_TOLERANCE=${METHOD_TRAIN_TOLERANCE}"
    "SELECTION_TRAIN_PENALTY=${METHOD_TRAIN_PENALTY}"
    "INNER_STEPS=1"
    "M_BRANCHES=${METHOD_M_BRANCHES}"
    "TUNE_EVERY=${TUNE_EVERY}"
    "FINAL_TRAIN_SIZE=${FINAL_TRAIN_SIZE}"
    "FINAL_FT_LR=${FINAL_FT_LR}"
    "FINAL_FT_EPOCHS=${FINAL_FT_EPOCHS}"
    "FINAL_FT_MICRO_BATCH=${FINAL_FT_MICRO_BATCH}"
    "FINAL_FT_GRAD_ACCUM=${FINAL_FT_GRAD_ACCUM}"
    "FINAL_TRAIN_RAW_OVERSAMPLE=${FINAL_TRAIN_RAW_OVERSAMPLE}"
    "REFERENCE_BATCH_SIZE=${job_reference_batch_size}"
    "REFERENCE_FT_STEPS=${REFERENCE_FT_STEPS}"
    "BASELINE_SCORE_SAMPLES_PER_SOURCE=${BASELINE_SCORE_SAMPLES_PER_SOURCE}"
    "LORA_R=${job_lora_r}"
    "LORA_ALPHA=${job_lora_alpha}"
    "LORA_DROPOUT=${job_lora_dropout}"
    "GRADIENT_CHECKPOINTING=${job_gradient_checkpointing}"
    "LOCAL_MODEL_ONLY=${MODEL_LOCAL_ONLY}"
    "RUN_MTBENCH_EVAL=${RUN_MTBENCH_EVAL}"
    "RUN_MTBENCH_JUDGE=${RUN_MTBENCH_JUDGE}"
    "MTBENCH_QUESTION_SOURCE=${MTBENCH_QUESTION_SOURCE}"
    "MTBENCH_QUESTION_FILE=${MTBENCH_QUESTION_FILE}"
    "MTBENCH_QUESTION_URL=${MTBENCH_QUESTION_URL}"
    "MTBENCH_QUESTION_LIMIT=${MTBENCH_QUESTION_LIMIT}"
    "MTBENCH_PROMPT_STYLE=${MTBENCH_PROMPT_STYLE}"
    "MTBENCH_JUDGE_MODEL=${MTBENCH_JUDGE_MODEL}"
    "MTBENCH_API_BASE=${MTBENCH_API_BASE}"
    "MTBENCH_API_KEY_ENV=${MTBENCH_API_KEY_ENV}"
    "MTBENCH_USE_REFERENCE=${MTBENCH_USE_REFERENCE}"
    "MTBENCH_MAX_INPUT_LENGTH=${MTBENCH_MAX_INPUT_LENGTH}"
    "MTBENCH_MAX_NEW_TOKENS=${MTBENCH_MAX_NEW_TOKENS}"
    "MTBENCH_GENERATION_BATCH_SIZE=${MTBENCH_GENERATION_BATCH_SIZE}"
  )

  local export_csv="ALL"
  local item
  for item in "${export_items[@]}"; do
    export_csv+=",${item}"
  done

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ -n "${JOB_TIME}" ]]; then
      printf 'sbatch --time=%q --job-name=%q --export=%q %q\n' "${JOB_TIME}" "${job_name}" "${export_csv}" "${RUN_SCRIPT}"
    else
      printf 'sbatch --job-name=%q --export=%q %q\n' "${job_name}" "${export_csv}" "${RUN_SCRIPT}"
    fi
  else
    if [[ -n "${JOB_TIME}" ]]; then
      sbatch --time="${JOB_TIME}" --job-name="${job_name}" --export="${export_csv}" "${RUN_SCRIPT}"
    else
      sbatch --job-name="${job_name}" --export="${export_csv}" "${RUN_SCRIPT}"
    fi
  fi
}

mkdir -p "${OUTPUT_ROOT}"

if [[ "${RUN_MTBENCH_JUDGE}" == "1" ]]; then
  if [[ -z "${!MTBENCH_API_KEY_ENV:-}" ]]; then
    echo "Warning: ${MTBENCH_API_KEY_ENV} is not set in the current shell. Jobs will generate answers but judge calls will be skipped." >&2
  fi
fi

echo "Instruction Table-2 launcher"
echo "  output root: ${OUTPUT_ROOT}"
echo "  model presets: ${MODEL_PRESETS}"
echo "  methods: ${METHODS}"
echo "  seeds: ${SEEDS}"
echo "  MT-Bench eval: ${RUN_MTBENCH_EVAL}"
echo "  MT-Bench judge: ${RUN_MTBENCH_JUDGE}"
echo "  question source: ${MTBENCH_QUESTION_SOURCE}"
echo "  iterative reweight budget (sec): ${INSTRUCTION_BUDGET_SECONDS}"
if [[ -n "${JOB_TIME}" ]]; then
  echo "  slurm time override: ${JOB_TIME}"
fi

for preset in ${MODEL_PRESETS}; do
  apply_instruction_model_preset "${preset}"
  echo "Preset ${preset}: MODEL_NAME=${MODEL_NAME} REF=${REFERENCE_MODEL_NAME} MAX_LENGTH=${MAX_LENGTH} LORA_R=${LORA_R}"
  for seed in ${SEEDS}; do
    for method in ${METHODS}; do
      submit_job "${preset}" "${seed}" "${method}"
    done
  done
done
