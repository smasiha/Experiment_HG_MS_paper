#!/bin/bash

set -euo pipefail

apply_math_model_preset() {
  local preset="${1:-}"
  case "${preset}" in
    llama3_8b)
      : "${MODEL_NAME:=meta-llama/Meta-Llama-3-8B-Instruct}"
      : "${REFERENCE_MODEL_NAME:=${MODEL_NAME}}"
      : "${MAX_LENGTH:=256}"
      : "${LORA_R:=4}"
      : "${LORA_ALPHA:=8}"
      : "${LORA_DROPOUT:=0.05}"
      : "${FINAL_FT_MICRO_BATCH:=1}"
      : "${FINAL_FT_GRAD_ACCUM:=32}"
      : "${VAL_BATCH_SIZE:=1}"
      : "${SELECTION_VAL_BATCH_SIZE:=1}"
      : "${EVAL_BATCH_SIZE:=1}"
      : "${PER_SOURCE_BATCH:=1}"
      : "${REFERENCE_BATCH_SIZE:=1}"
      : "${LOCAL_MODEL_ONLY:=0}"
      ;;
    qwen2_7b)
      : "${MODEL_NAME:=Qwen/Qwen2-7B-Instruct}"
      : "${REFERENCE_MODEL_NAME:=${MODEL_NAME}}"
      : "${MAX_LENGTH:=256}"
      : "${LORA_R:=4}"
      : "${LORA_ALPHA:=8}"
      : "${LORA_DROPOUT:=0.05}"
      : "${FINAL_FT_MICRO_BATCH:=1}"
      : "${FINAL_FT_GRAD_ACCUM:=32}"
      : "${VAL_BATCH_SIZE:=1}"
      : "${SELECTION_VAL_BATCH_SIZE:=1}"
      : "${EVAL_BATCH_SIZE:=1}"
      : "${PER_SOURCE_BATCH:=1}"
      : "${REFERENCE_BATCH_SIZE:=1}"
      : "${LOCAL_MODEL_ONLY:=0}"
      ;;
    gemma2_9b)
      : "${MODEL_NAME:=google/gemma-2-9b-it}"
      : "${REFERENCE_MODEL_NAME:=${MODEL_NAME}}"
      : "${MAX_LENGTH:=256}"
      : "${LORA_R:=4}"
      : "${LORA_ALPHA:=8}"
      : "${LORA_DROPOUT:=0.05}"
      : "${FINAL_FT_MICRO_BATCH:=1}"
      : "${FINAL_FT_GRAD_ACCUM:=32}"
      : "${VAL_BATCH_SIZE:=1}"
      : "${SELECTION_VAL_BATCH_SIZE:=1}"
      : "${EVAL_BATCH_SIZE:=1}"
      : "${PER_SOURCE_BATCH:=1}"
      : "${REFERENCE_BATCH_SIZE:=1}"
      : "${LOCAL_MODEL_ONLY:=0}"
      ;;
    qwen25_32b)
      : "${MODEL_NAME:=Qwen/Qwen2.5-32B-Instruct}"
      : "${REFERENCE_MODEL_NAME:=Qwen/Qwen2.5-0.5B-Instruct}"
      : "${MAX_LENGTH:=512}"
      : "${LORA_R:=16}"
      : "${LORA_ALPHA:=32}"
      : "${LORA_DROPOUT:=0.05}"
      : "${FINAL_FT_MICRO_BATCH:=1}"
      : "${FINAL_FT_GRAD_ACCUM:=64}"
      : "${VAL_BATCH_SIZE:=1}"
      : "${SELECTION_VAL_BATCH_SIZE:=1}"
      : "${EVAL_BATCH_SIZE:=1}"
      : "${PER_SOURCE_BATCH:=1}"
      : "${REFERENCE_BATCH_SIZE:=1}"
      : "${LOCAL_MODEL_ONLY:=0}"
      ;;
    *)
      echo "Unknown model preset: ${preset}" >&2
      return 1
      ;;
  esac
}
