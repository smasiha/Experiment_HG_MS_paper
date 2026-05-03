# Experiment_HG_MS_paper

This repository is the submission-facing experiment package for the paper on **Hyper-Gradient with Minima Selection (HG-MS)**.

The goal is reproducibility and auditability: the repository keeps the code, launch scripts, configuration notes, and paper-facing result ledgers used for the current draft. It intentionally does **not** track large model checkpoints, raw downloaded datasets, Hugging Face caches, Slurm logs, or generated output directories.

## Repository Layout

| Path | Purpose |
|---|---|
| `LLM_paper_qwen25/` | Controlled GPT-2 source-reweighting experiments and real-world LLM data-composition experiments. |
| `LLM_paper_qwen25/controlled/` | Two-source controlled LLM tasks: denoising and multilingual source reweighting. |
| `LLM_paper_qwen25/real_world/` | Main single-H100 LoRA pipelines for GSM8K/MATH and MT-Bench experiments. |
| `LLM_paper_qwen25/table3_handoff/` | Faithful/large-run handoff bundle with FSDP launchers and non-Slurm guidance. |
| `data_hyper_cleaning/` | MNIST label-noise hyper-cleaning code and result summary. |
| `imbalance_data/` | MNIST imbalanced-loss tuning code and result summary. |

## Paper-Facing Result Ledgers

The result summaries are frozen snapshots of the numbers currently used in the draft:

- LLM controlled tasks: `LLM_paper_qwen25/controlled/results_summary.md`
- LLM GSM8K/MATH, Qwen2.5-32B, MT-Bench, budget, and branch-count tables: `LLM_paper_qwen25/real_world/results_summary.md`
- Data hyper-cleaning: `data_hyper_cleaning/results_summary.md`
- Imbalanced loss tuning: `imbalance_data/results_summary.md`

The LaTeX source remains the final source of truth for paper formatting, but these ledgers are meant to make the GitHub artifact self-contained and easy to inspect.

## Environment

The LLM experiments were run with Python plus PyTorch/Transformers/PEFT. The pinned Python package list for the LLM code is:

```bash
cd LLM_paper_qwen25
python -m pip install -r requirements.txt
```

The large LLM runs require access to gated/public Hugging Face model checkpoints and sufficient GPU memory. Set the usual Hugging Face cache variables before running large jobs, for example:

```bash
export HF_HUB_CACHE=/scratch/$USER/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache/transformers
export HF_ASSETS_CACHE=/scratch/$USER/hf_cache/assets
```

Some instruction-following evaluation scripts can call a GPT judge for MT-Bench scoring and require `OPENAI_API_KEY`.

## Running Experiments

Most scripts support Slurm through `*.sbatch` wrappers, but the Python entrypoints can also be run directly. Each subdirectory README gives the relevant commands.

Typical entrypoints:

```bash
# Controlled GPT-2 tasks
cd LLM_paper_qwen25/controlled
bash submit_small_scale_tasks.sh

# Main adapted GSM8K/MATH and MT-Bench LLM pipelines
cd ../real_world
bash submit_real_world_wallclock_full.sh
bash submit_instruction_mtbench_table2.sh

# Faithful large-run handoff path
cd ../table3_handoff
MODEL_PRESET=qwen2_7b SEEDS="2424" bash submit_faithful_uniform_canary.sh
```

For non-Slurm systems, read the corresponding `run_*.sbatch` file and run the Python module directly with the same flags. The `table3_handoff/README.md` includes explicit `torchrun` examples for an 8-GPU machine.

## Submission Hygiene

The `.gitignore` excludes generated artifacts such as:

- `outputs/`, `cache/`, `hf_cache/`, `slurm/`, `wandb/`
- model/checkpoint files such as `*.pt`, `*.safetensors`, `*.bin`
- local datasets under `data/`
- Slurm logs `*.out` and `*.err`
- Python caches

If you regenerate experiments, commit only code/configuration changes and small summary files. Do not commit model checkpoints, dataset mirrors, or raw run directories.
