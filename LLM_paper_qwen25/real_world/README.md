# Real-World Qwen2.5 Experiments

This subfolder contains the existing real-world LLM reproduction code used in the paper's Qwen2.5 setup.

Scope:
- backbone: `Qwen2.5-0.5B-Instruct`
- lower level: LoRA adapters
- tasks:
  - math reasoning
  - instruction following

Files:
- `exact_math_reasoning.py`, `exact_math_reasoning_data.py`
- `real_world_instruction.py`, `real_world_instruction_data.py`
- `run_exact_math_reasoning.sbatch`, `run_real_world_instruction.sbatch`
- `submit_real_world_wallclock_full.sh`
- `summarize_wallclock_metrics.py`, `summarize_real_world_full_small.py`
- `check_model_access.py`

This package intentionally excludes the newer larger-model GSM8K expansion work. The goal here is to keep the current paper-facing Qwen2.5 code path clean for git.

