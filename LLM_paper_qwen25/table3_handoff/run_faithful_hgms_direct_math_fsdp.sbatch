#!/bin/bash
#SBATCH --job-name=hgms_direct
#SBATCH --output=/scratch/%u/slurm/hgms_direct_%j.out
#SBATCH --error=/scratch/%u/slurm/hgms_direct_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
ENV_DIR="${ENV_DIR:-/home/masiha/venvs/hypergrad_llm_env}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct}"
SEED="${SEED:-2424}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/hgms_direct_${SEED}_${SLURM_JOB_ID}}"
RUN_NAME="${RUN_NAME:-hgms_direct_seed${SEED}_qwen25_32b}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/scratch/${USER}/hf_cache_jobs/qwen25_32b_hgms_direct/${RUN_NAME}}"
CACHE_PATH="${CACHE_PATH:-${OUTPUT_DIR}/hgms_direct_splits.json}"
MAX_LENGTH="${MAX_LENGTH:-512}"
REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-80}"
TRAIN_RAW_OVERSAMPLE="${TRAIN_RAW_OVERSAMPLE:-4.0}"
VAL_PER_SOURCE="${VAL_PER_SOURCE:-30}"
VAL_RAW_OVERSAMPLE="${VAL_RAW_OVERSAMPLE:-1.2}"
ALLOW_TRAIN_SHORTFALL="${ALLOW_TRAIN_SHORTFALL:-1}"
REWEIGHT_STEPS="${REWEIGHT_STEPS:-3}"
M_BRANCHES="${M_BRANCHES:-2}"
PER_SOURCE_BATCH="${PER_SOURCE_BATCH:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
BRANCH_NOISE_STD="${BRANCH_NOISE_STD:-5e-4}"
SELECTION_TRAIN_TOLERANCE="${SELECTION_TRAIN_TOLERANCE:-0.05}"
SELECTION_TRAIN_PENALTY="${SELECTION_TRAIN_PENALTY:-0.0}"
REWEIGHT_MODEL_LR="${REWEIGHT_MODEL_LR:-8e-6}"
REWEIGHT_WEIGHT_DECAY="${REWEIGHT_WEIGHT_DECAY:-0.0}"
REWEIGHT_OUTER_LR="${REWEIGHT_OUTER_LR:-0.05}"
THETA_OPTIMIZER="${THETA_OPTIMIZER:-adam}"
THETA_GRAD_CLIP="${THETA_GRAD_CLIP:-0.0}"
TUNE_EVERY="${TUNE_EVERY:-1}"
MIN_LAMBDA_MOVEMENT="${MIN_LAMBDA_MOVEMENT:-1e-6}"
MIN_CENTERED_GRAD_NORM="${MIN_CENTERED_GRAD_NORM:-1e-12}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-10000000}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
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
export LOCAL_MODEL_ONLY="${LOCAL_MODEL_ONLY:-0}"
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

if [[ "${LOCAL_MODEL_ONLY}" != "1" ]]; then
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
fi

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
    export PYTORCH_CUDA_ALLOC_CONF="'"${PYTORCH_CUDA_ALLOC_CONF}"'"
    echo "node_rank=${SLURM_NODEID} host=$(hostname)"
    torchrun \
      --nnodes="'"${SLURM_NNODES}"'" \
      --nproc-per-node="'"${NPROC_PER_NODE}"'" \
      --node-rank="${SLURM_NODEID}" \
      --master-addr="'"${MASTER_ADDR}"'" \
      --master-port="'"${MASTER_PORT}"'" \
      faithful_hgms_direct_math_fsdp.py \
      --model-name "'"${MODEL_NAME}"'" \
      --output-dir "'"${OUTPUT_DIR}"'" \
      --seed "'"${SEED}"'" \
      --cache-path "'"${CACHE_PATH}"'" \
      --max-length "'"${MAX_LENGTH}"'" \
      --reweight-train-per-source "'"${REWEIGHT_TRAIN_PER_SOURCE}"'" \
      --train-raw-oversample "'"${TRAIN_RAW_OVERSAMPLE}"'" \
      --val-per-source "'"${VAL_PER_SOURCE}"'" \
      --val-raw-oversample "'"${VAL_RAW_OVERSAMPLE}"'" \
      '"${TRAIN_SHORTFALL_FLAG}"' \
      --reweight-steps "'"${REWEIGHT_STEPS}"'" \
      --m-branches "'"${M_BRANCHES}"'" \
      --per-source-batch "'"${PER_SOURCE_BATCH}"'" \
      --val-batch-size "'"${VAL_BATCH_SIZE}"'" \
      --branch-noise-std "'"${BRANCH_NOISE_STD}"'" \
      --selection-train-tolerance "'"${SELECTION_TRAIN_TOLERANCE}"'" \
      --selection-train-penalty "'"${SELECTION_TRAIN_PENALTY}"'" \
      --reweight-model-lr "'"${REWEIGHT_MODEL_LR}"'" \
      --reweight-weight-decay "'"${REWEIGHT_WEIGHT_DECAY}"'" \
      --reweight-outer-lr "'"${REWEIGHT_OUTER_LR}"'" \
      --theta-optimizer "'"${THETA_OPTIMIZER}"'" \
      --theta-grad-clip "'"${THETA_GRAD_CLIP}"'" \
      --tune-every "'"${TUNE_EVERY}"'" \
      --min-lambda-movement "'"${MIN_LAMBDA_MOVEMENT}"'" \
      --min-centered-grad-norm "'"${MIN_CENTERED_GRAD_NORM}"'" \
      --fsdp-min-num-params "'"${FSDP_MIN_NUM_PARAMS}"'" \
      --attn-implementation "'"${ATTN_IMPLEMENTATION}"'" \
      --lora-r "'"${LORA_R}"'" \
      --lora-alpha "'"${LORA_ALPHA}"'" \
      --lora-dropout "'"${LORA_DROPOUT}"'" \
      --bf16 \
      '"${GC_FLAG}"'
  '
