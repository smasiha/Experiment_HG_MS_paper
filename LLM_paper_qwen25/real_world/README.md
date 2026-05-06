# Real-World LLM Source-Reweighting Experiments

This directory contains the adapted LoRA pipelines used for the paper's real-world LLM experiments.

## Covered Benchmarks

- **Mathematical reasoning:** ten-source ScaleBiO-style portfolio, final evaluation on GSM8K and MATH.
- **Instruction following:** eighteen-source instruction/chat portfolio, final evaluation on MT-Bench with a GPT-4o judge.
- **Diagnostics:** Qwen-2-7B budget sensitivity, HG-MS branch-count sensitivity, zero-shot checks, and the large-model Qwen2.5-32B path.

## Main Entry Points

| File | Purpose |
|---|---|
| `exact_math_reasoning.py` | Main GSM8K/MATH LoRA source-reweighting and final fine-tuning pipeline. |
| `exact_math_reasoning_data.py` | Math portfolio construction, prompt formatting, and answer extraction helpers. |
| `real_world_instruction.py` | Instruction-following source reweighting, final fine-tuning, generation, and optional MT-Bench judging. |
| `real_world_instruction_data.py` | Instruction portfolio and MT-Bench-style data helpers. |
| `evaluate_saved_math_runs.py` | Re-evaluate saved math checkpoints. |
| `evaluate_zero_shot_gsm8k.py` | Zero-shot GSM8K sanity checks. |
| `model_presets.sh` | Model-name and memory preset definitions for Llama, Qwen, Gemma, and Qwen2.5-32B variants. |

## Main Launchers

| File | Purpose |
|---|---|
| `submit_real_world_wallclock_full.sh` | Main matched-budget adapted math grid. |
| `submit_instruction_mtbench_table2.sh` | Main instruction-following / MT-Bench grid. |
| `submit_iterative_longbudget_seed2424.sh` | One-shot long-budget iterative math runs. |
| `submit_llama3_hgms_performance_sweep.sh` | Llama HG-MS tuning/performance runs. |
| `submit_qwen2_hgms_performance_sweep.sh` | Qwen HG-MS tuning/performance runs. |
| `submit_qwen2_hgms_branch_ablation.sh` | Qwen branch-count sensitivity runs. |
| `submit_qwen2_hgms_branch_smoke.sh` | Cheap branch-count smoke tests. |
| `submit_large_qwen25_32b_lambda_diag.sh` | Large-model lambda/branch diagnostic path. |
| `submit_large_qwen25_32b_math_table5.sh` | Qwen2.5-32B final table runs. |

Slurm wrappers such as `run_exact_math_reasoning.sbatch`, `run_real_world_instruction.sbatch`, and `run_saved_math_eval.sbatch` define the exact command-line flags used by the launchers.

## Main Protocol

Mathematical reasoning:

- Backbones: `Llama-3-8B`, `Qwen-2-7B`, `Gemma-2-9B`.
- Training portfolio: ten sources matching the ScaleBiO math-reasoning layout.
- Methods: `Uniform`, `RHO-LOSS`, `LESS`, `ScaleBiO`, `HG-baseline`, `HG-MS`.
- Reweighting: single-H100 LoRA path; iterative methods use matched per-backbone wall-clock budgets.
- Final fine-tuning: one epoch on a sampled `20K`-example SFT set.
- Evaluation: full GSM8K and MATH test sets with greedy generation.

Instruction following:

- Backbones: `Llama-3-8B`, `Qwen-2-7B`, `Gemma-2-9B`.
- Training portfolio: eighteen instruction/chat sources.
- Final fine-tuning: one epoch on a sampled `10K`-example SFT set.
- Evaluation: standard FastChat MT-Bench questions, one generated answer per prompt, GPT-4o judge.

Large-model Qwen2.5-32B:

- Methods reported in the paper-facing ledger: `Uniform`, `RHO-LOSS`, `LESS`, `ScaleBiO`, `HG-baseline`, and `HG-MS`.
- `RHO-LOSS` and `LESS` are marked `OOM` in the current paper table under the same large-model memory budget.
- Completed non-OOM rows use the corresponding 32B source mixture, the full `20K`-example final fine-tuning set, and full GSM8K/MATH evaluation.

Appendix F protocol details reflected in `results_summary.md`:

- Final SFT is one fixed epoch with AdamW learning rate `8e-6`; math uses `20K` examples and instruction following uses `10K`.
- Main math `HG-MS` uses `N=4` branches for Llama-3-8B, Qwen-2-7B, and Gemma-2-9B; instruction `HG-MS` uses `N=8`; `HG-baseline` uses `N=1`.
- The source-mixture audit tables report the final source proportions used by the paper-facing rows.
- The uncertainty table reports finite-evaluation-set intervals only; it does not estimate training-seed variance.

## Example Commands

Dry-run a main math launcher:

```bash
DRY_RUN=1 bash submit_real_world_wallclock_full.sh
```

Launch the Qwen branch-count ablation:

```bash
MODEL_PRESET=qwen2_7b M_BRANCHES="2 8" bash submit_qwen2_hgms_branch_ablation.sh
```

Run a zero-shot GSM8K check:

```bash
sbatch --export=ALL,MODEL_PRESET=qwen2_7b run_zero_shot_gsm8k.sbatch
```

Re-evaluate saved math checkpoints:

```bash
python submit_missing_math_eval_parallel.py --runs-root /scratch/$USER/<runs-root>
```

## Results

The current paper-facing numbers, Appendix F source-mixture audit, uncertainty ranges, and key implementation hyperparameters are in `results_summary.md`. Generated run directories, model checkpoints, and raw logs are deliberately excluded from git.
