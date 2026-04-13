#!/bin/bash
#SBATCH --job-name=faithful_scalebio
#SBATCH --output=/scratch/%u/slurm/faithful_scalebio_%j.out
#SBATCH --error=/scratch/%u/slurm/faithful_scalebio_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
ENV_DIR="${ENV_DIR:-/home/masiha/venvs/hypergrad_llm_env}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B}"
SEED="${SEED:-2424}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/faithful_table4_scalebio_${SEED}_${SLURM_JOB_ID}}"
RUN_NAME="${RUN_NAME:-scalebio_seed${SEED}_llama3_8b_faithful}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/scratch/${USER}/hf_cache_jobs/faithful_llama3_8b_scalebio/${RUN_NAME}}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-3000}"
TRAIN_RAW_OVERSAMPLE="${TRAIN_RAW_OVERSAMPLE:-4.0}"
VAL_PER_SOURCE="${VAL_PER_SOURCE:-280}"
VAL_RAW_OVERSAMPLE="${VAL_RAW_OVERSAMPLE:-1.2}"
ALLOW_TRAIN_SHORTFALL="${ALLOW_TRAIN_SHORTFALL:-0}"
REWEIGHT_EPOCHS="${REWEIGHT_EPOCHS:-3}"
PER_SOURCE_BATCH="${PER_SOURCE_BATCH:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
REWEIGHT_MODEL_LR="${REWEIGHT_MODEL_LR:-8e-6}"
REWEIGHT_WEIGHT_DECAY="${REWEIGHT_WEIGHT_DECAY:-0.0}"
REWEIGHT_OUTER_LR="${REWEIGHT_OUTER_LR:-1e-2}"
SCALEBIO_ALPHA="${SCALEBIO_ALPHA:-100.0}"
TUNE_EVERY="${TUNE_EVERY:-50}"
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-20000}"
FINAL_FT_LR="${FINAL_FT_LR:-8e-6}"
FINAL_FT_EPOCHS="${FINAL_FT_EPOCHS:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
GSM8K_EVAL_SIZE="${GSM8K_EVAL_SIZE:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
THETA_OPTIMIZER="${THETA_OPTIMIZER:-sgd}"
THETA_GRAD_CLIP="${THETA_GRAD_CLIP:-0.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "/scratch/${USER}/slurm"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/transformers" "${HF_CACHE_ROOT}/assets"
cd "${ROOT_DIR}"

source "${ENV_DIR}/bin/activate"

export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
export HF_ASSETS_CACHE="${HF_CACHE_ROOT}/assets"
export MODEL_NAME
export LOCAL_MODEL_ONLY=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
GC_FLAG=""
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  GC_FLAG="--gradient-checkpointing"
fi
TRAIN_SHORTFALL_FLAG=""
if [[ "${ALLOW_TRAIN_SHORTFALL}" == "1" ]]; then
  TRAIN_SHORTFALL_FLAG="--allow-train-shortfall"
fi

echo "Prewarming model cache for ${MODEL_NAME}"
python - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = os.environ["MODEL_NAME"]
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager",
    local_files_only=False,
    low_cpu_mem_usage=True,
)
print(f"PREWARM_OK {model_name}")
PY

export LOCAL_MODEL_ONLY=1
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"

echo "Host list: ${SLURM_JOB_NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
python -V

srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  bash -lc '
    set -euo pipefail
    cd "'"${ROOT_DIR}"'"
    source "'"${ENV_DIR}"'/bin/activate"
    export HF_HUB_CACHE="'"${HF_HUB_CACHE}"'"
    export HUGGINGFACE_HUB_CACHE="'"${HUGGINGFACE_HUB_CACHE}"'"
    export TRANSFORMERS_CACHE="'"${TRANSFORMERS_CACHE}"'"
    export HF_ASSETS_CACHE="'"${HF_ASSETS_CACHE}"'"
    export LOCAL_MODEL_ONLY=1
    export MASTER_ADDR="'"${MASTER_ADDR}"'"
    export MASTER_PORT="'"${MASTER_PORT}"'"
    export OMP_NUM_THREADS="'"${OMP_NUM_THREADS}"'"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG="'"${NCCL_DEBUG}"'"
    echo "node_rank=${SLURM_NODEID} host=$(hostname)"
    torchrun \
      --nnodes="'"${SLURM_NNODES}"'" \
      --nproc-per-node="'"${NPROC_PER_NODE}"'" \
      --node-rank="${SLURM_NODEID}" \
      --master-addr="'"${MASTER_ADDR}"'" \
      --master-port="'"${MASTER_PORT}"'" \
      faithful_scalebio_math_fsdp.py \
      --model-name "'"${MODEL_NAME}"'" \
      --output-dir "'"${OUTPUT_DIR}"'" \
      --seed "'"${SEED}"'" \
      --max-length "'"${MAX_LENGTH}"'" \
      --reweight-train-per-source "'"${REWEIGHT_TRAIN_PER_SOURCE}"'" \
      --train-raw-oversample "'"${TRAIN_RAW_OVERSAMPLE}"'" \
      --val-per-source "'"${VAL_PER_SOURCE}"'" \
      --val-raw-oversample "'"${VAL_RAW_OVERSAMPLE}"'" \
      '"${TRAIN_SHORTFALL_FLAG}"' \
      --reweight-epochs "'"${REWEIGHT_EPOCHS}"'" \
      --per-source-batch "'"${PER_SOURCE_BATCH}"'" \
      --val-batch-size "'"${VAL_BATCH_SIZE}"'" \
      --reweight-model-lr "'"${REWEIGHT_MODEL_LR}"'" \
      --reweight-weight-decay "'"${REWEIGHT_WEIGHT_DECAY}"'" \
      --reweight-outer-lr "'"${REWEIGHT_OUTER_LR}"'" \
      --scalebio-alpha "'"${SCALEBIO_ALPHA}"'" \
      --tune-every "'"${TUNE_EVERY}"'" \
      --theta-optimizer "'"${THETA_OPTIMIZER}"'" \
      --theta-grad-clip "'"${THETA_GRAD_CLIP}"'" \
      --final-train-size "'"${FINAL_TRAIN_SIZE}"'" \
      --final-ft-lr "'"${FINAL_FT_LR}"'" \
      --final-ft-epochs "'"${FINAL_FT_EPOCHS}"'" \
      --per-device-batch-size "'"${PER_DEVICE_BATCH_SIZE}"'" \
      --grad-accum "'"${GRAD_ACCUM}"'" \
      --eval-batch-size "'"${EVAL_BATCH_SIZE}"'" \
      --eval-max-new-tokens "'"${EVAL_MAX_NEW_TOKENS}"'" \
      --gsm8k-eval-size "'"${GSM8K_EVAL_SIZE}"'" \
      --bf16 \
      '"${GC_FLAG}"'
  '
