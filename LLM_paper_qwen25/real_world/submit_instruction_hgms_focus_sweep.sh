#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SUBMIT="${BASE_SUBMIT:-${SCRIPT_DIR}/submit_instruction_mtbench_table2.sh}"

TAG="${TAG:-instruction_hgms_focus_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/scratch/${USER}/${TAG}}"

MODEL_PRESETS="${MODEL_PRESETS:-qwen2_7b}"
SEEDS="${SEEDS:-2424}"
PROFILES="${PROFILES:-conservative_lr low_noise_more_val tiny_step}"

RUN_MTBENCH_EVAL="${RUN_MTBENCH_EVAL:-0}"
RUN_MTBENCH_JUDGE="${RUN_MTBENCH_JUDGE:-0}"
DRY_RUN="${DRY_RUN:-0}"
JOB_TIME="${JOB_TIME:-04:00:00}"

INSTRUCTION_BUDGET_SECONDS="${INSTRUCTION_BUDGET_SECONDS:-600}"
REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE:-1500}"
FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE:-4000}"
REFERENCE_FT_STEPS="${REFERENCE_FT_STEPS:-100}"
BASELINE_SCORE_SAMPLES_PER_SOURCE="${BASELINE_SCORE_SAMPLES_PER_SOURCE:-128}"
REWEIGHT_EPOCHS="${REWEIGHT_EPOCHS:-2}"
STOP_AFTER_REWEIGHT="${STOP_AFTER_REWEIGHT:-0}"
TUNE_EVERY="${TUNE_EVERY:-5}"
LESS_INIT_METRICS_PATH="${LESS_INIT_METRICS_PATH:-/scratch/${USER}/instruction_mtbench_table2_20260421_195509/qwen2_7b/less_seed2424_qwen2_7b_instr_mtbench/metrics.json}"

profile_hg_ms_branches=""
profile_selection_batches=""
profile_hypergrad_batches=""
profile_hg_ms_outer_lr=""
profile_hg_ms_tolerance=""
profile_hg_ms_penalty=""
profile_hg_branch_noise_std=""
profile_initial_lambda_metrics_path=""
profile_initial_lambda_blend="1.0"
profile_note=""

configure_profile() {
  local profile="$1"
  profile_initial_lambda_metrics_path=""
  profile_initial_lambda_blend="1.0"
  case "${profile}" in
    stable_selector)
      profile_hg_ms_branches="4"
      profile_selection_batches="2"
      profile_hypergrad_batches="2"
      profile_hg_ms_outer_lr="0.05"
      profile_hg_ms_tolerance="0.50"
      profile_hg_ms_penalty="0.00"
      profile_hg_branch_noise_std="5e-4"
      profile_note="Fewer branches, more stable selector and hypergrad batches."
      ;;
    conservative_lr)
      profile_hg_ms_branches="4"
      profile_selection_batches="2"
      profile_hypergrad_batches="2"
      profile_hg_ms_outer_lr="0.02"
      profile_hg_ms_tolerance="0.50"
      profile_hg_ms_penalty="0.00"
      profile_hg_branch_noise_std="1e-4"
      profile_note="Keep the stable-selector structure but reduce outer-step size and branch noise."
      ;;
    low_noise_more_val)
      profile_hg_ms_branches="4"
      profile_selection_batches="4"
      profile_hypergrad_batches="4"
      profile_hg_ms_outer_lr="0.02"
      profile_hg_ms_tolerance="0.50"
      profile_hg_ms_penalty="0.00"
      profile_hg_branch_noise_std="1e-4"
      profile_note="More conservative updates plus cleaner branch scoring from more held-out batches."
      ;;
    tiny_step)
      profile_hg_ms_branches="4"
      profile_selection_batches="2"
      profile_hypergrad_batches="2"
      profile_hg_ms_outer_lr="0.01"
      profile_hg_ms_tolerance="0.50"
      profile_hg_ms_penalty="0.00"
      profile_hg_branch_noise_std="1e-4"
      profile_note="Very small outer updates to test whether HG-MS is currently overshooting immediately."
      ;;
    balanced_push)
      profile_hg_ms_branches="4"
      profile_selection_batches="2"
      profile_hypergrad_batches="2"
      profile_hg_ms_outer_lr="0.10"
      profile_hg_ms_tolerance="0.75"
      profile_hg_ms_penalty="0.02"
      profile_hg_branch_noise_std="1e-3"
      profile_note="Stronger outer updates with a softer train-loss gate."
      ;;
    diverse_branches)
      profile_hg_ms_branches="6"
      profile_selection_batches="1"
      profile_hypergrad_batches="2"
      profile_hg_ms_outer_lr="0.10"
      profile_hg_ms_tolerance="0.75"
      profile_hg_ms_penalty="0.00"
      profile_hg_branch_noise_std="1e-3"
      profile_note="More branch diversity without paying for the full 8-branch setup."
      ;;
    less_init_soft)
      profile_hg_ms_branches="4"
      profile_selection_batches="2"
      profile_hypergrad_batches="2"
      profile_hg_ms_outer_lr="0.02"
      profile_hg_ms_tolerance="0.50"
      profile_hg_ms_penalty="0.00"
      profile_hg_branch_noise_std="5e-4"
      profile_initial_lambda_metrics_path="${LESS_INIT_METRICS_PATH}"
      profile_initial_lambda_blend="0.50"
      profile_note="Start halfway between uniform and LESS, then refine conservatively."
      ;;
    less_init_anchor)
      profile_hg_ms_branches="4"
      profile_selection_batches="2"
      profile_hypergrad_batches="2"
      profile_hg_ms_outer_lr="0.01"
      profile_hg_ms_tolerance="0.50"
      profile_hg_ms_penalty="0.00"
      profile_hg_branch_noise_std="5e-4"
      profile_initial_lambda_metrics_path="${LESS_INIT_METRICS_PATH}"
      profile_initial_lambda_blend="0.80"
      profile_note="Use a strong LESS prior and only allow mild HG-MS refinement."
      ;;
    *)
      echo "Unknown HG-MS profile: ${profile}" >&2
      return 1
      ;;
  esac
}

echo "HG-MS focus sweep launcher"
echo "  base submit: ${BASE_SUBMIT}"
echo "  output root base: ${OUTPUT_ROOT_BASE}"
echo "  model presets: ${MODEL_PRESETS}"
echo "  seeds: ${SEEDS}"
echo "  profiles: ${PROFILES}"
echo "  MT-Bench eval: ${RUN_MTBENCH_EVAL}"
echo "  MT-Bench judge: ${RUN_MTBENCH_JUDGE}"
echo "  iterative reweight budget (sec): ${INSTRUCTION_BUDGET_SECONDS}"
echo "  reweight train/source: ${REWEIGHT_TRAIN_PER_SOURCE}"
echo "  final train size: ${FINAL_TRAIN_SIZE}"
echo "  reference ft steps: ${REFERENCE_FT_STEPS}"
echo "  reweight epochs: ${REWEIGHT_EPOCHS}"
echo "  stop after reweight: ${STOP_AFTER_REWEIGHT}"
echo "  tune every: ${TUNE_EVERY}"
echo "  slurm time override: ${JOB_TIME}"

for profile in ${PROFILES}; do
  configure_profile "${profile}"
  profile_output_root="${OUTPUT_ROOT_BASE}/${profile}"
  echo "Profile ${profile}: ${profile_note}"
  echo "  output root: ${profile_output_root}"
  echo "  hg_ms_branches=${profile_hg_ms_branches} selection_batches=${profile_selection_batches} hypergrad_batches=${profile_hypergrad_batches}"
  echo "  outer_lr=${profile_hg_ms_outer_lr} tolerance=${profile_hg_ms_tolerance} penalty=${profile_hg_ms_penalty} branch_noise_std=${profile_hg_branch_noise_std}"
  if [[ -n "${profile_initial_lambda_metrics_path}" ]]; then
    echo "  initial_lambda=${profile_initial_lambda_metrics_path} blend=${profile_initial_lambda_blend}"
  fi

  env \
    MODEL_PRESETS="${MODEL_PRESETS}" \
    METHODS="hg_ms" \
    SEEDS="${SEEDS}" \
    OUTPUT_ROOT="${profile_output_root}" \
    RUN_MTBENCH_EVAL="${RUN_MTBENCH_EVAL}" \
    RUN_MTBENCH_JUDGE="${RUN_MTBENCH_JUDGE}" \
    INSTRUCTION_BUDGET_SECONDS="${INSTRUCTION_BUDGET_SECONDS}" \
    REWEIGHT_TRAIN_PER_SOURCE="${REWEIGHT_TRAIN_PER_SOURCE}" \
    FINAL_TRAIN_SIZE="${FINAL_TRAIN_SIZE}" \
    REFERENCE_FT_STEPS="${REFERENCE_FT_STEPS}" \
    BASELINE_SCORE_SAMPLES_PER_SOURCE="${BASELINE_SCORE_SAMPLES_PER_SOURCE}" \
    REWEIGHT_EPOCHS="${REWEIGHT_EPOCHS}" \
    STOP_AFTER_REWEIGHT="${STOP_AFTER_REWEIGHT}" \
    TUNE_EVERY="${TUNE_EVERY}" \
    JOB_TIME="${JOB_TIME}" \
    HG_MS_BRANCHES="${profile_hg_ms_branches}" \
    HG_SELECTION_BATCHES="${profile_selection_batches}" \
    HG_HYPERGRAD_BATCHES="${profile_hypergrad_batches}" \
    HG_MS_OUTER_LR="${profile_hg_ms_outer_lr}" \
    HG_MS_TOLERANCE="${profile_hg_ms_tolerance}" \
    HG_MS_PENALTY="${profile_hg_ms_penalty}" \
    HG_BRANCH_NOISE_STD="${profile_hg_branch_noise_std}" \
    INITIAL_LAMBDA_METRICS_PATH="${profile_initial_lambda_metrics_path}" \
    INITIAL_LAMBDA_BLEND="${profile_initial_lambda_blend}" \
    DRY_RUN="${DRY_RUN}" \
    bash "${BASE_SUBMIT}"
done
