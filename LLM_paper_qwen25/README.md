# LLM Paper Experiments

This directory contains the LLM experiment code used by the current HG-MS paper draft.

The package covers three tracks:

1. `controlled/`: small GPT-2 two-source source-reweighting tasks adapted from ScaleBiO.
2. `real_world/`: main adapted LoRA source-reweighting experiments for GSM8K/MATH and MT-Bench.
3. `table3_handoff/`: faithful/large-run handoff bundle with FSDP launchers and direct `torchrun` guidance.

Large generated artifacts are intentionally excluded. The repository keeps runnable code, launch scripts, and paper-facing result ledgers.

## Current Paper Tables Covered

| Paper component | Location |
|---|---|
| Controlled LLM source reweighting | `controlled/` |
| Mathematical reasoning on Llama-3-8B, Qwen-2-7B, Gemma-2-9B | `real_world/` |
| Large-model Qwen2.5-32B extension | `real_world/` and `table3_handoff/` |
| Instruction following / MT-Bench | `real_world/` |
| Qwen-2-7B budget sensitivity | `real_world/` |
| Qwen-2-7B HG-MS branch-count sensitivity | `real_world/` |

## Dependencies

```bash
python -m pip install -r requirements.txt
```

For MT-Bench judging, export `OPENAI_API_KEY`. For Hugging Face models and datasets, configure cache paths and make sure gated model access has been accepted where required.

## Result Snapshots

- `controlled/results_summary.md`: current controlled-task table values.
- `real_world/results_summary.md`: current main LLM tables and diagnostics.

## Important Protocol Notes

- The main real-world LLM tables use LoRA fine-tuning and a matched wall-clock reweighting budget for iterative methods.
- The mathematical-reasoning final fine-tuning set has `20K` examples; instruction following uses `10K` examples.
- Final GSM8K/MATH evaluation uses greedy generation and strict answer extraction.
- MT-Bench generations are judged with a GPT-4o judge at temperature zero.
- `HG-MS` branch count is internal to source reweighting; it is not best-of-N inference.

## Where To Start

```bash
# Controlled tasks
cd controlled
bash submit_small_scale_tasks.sh

# Main LLM math/instruction runs
cd ../real_world
less README.md

# Large-run/collaborator handoff path
cd ../table3_handoff
less README.md
```
