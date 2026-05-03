# Faithful Reproduction Checklist for ScaleBiO Table 4

## Goal

Reproduce the mathematical reasoning setting of Table 4 in `Hyper_grad_paper/Minima_Selection/ScaleBiO.pdf` closely enough that:

1. `Uniform Weighting` lands in the same regime as the paper.
2. `ScaleBiO`, `LESS`, and `RHO-LOSS` are compared under the same protocol.
3. `HG-MS` is only compared after the reproduction baseline is credible.

This is a separate track from the current single-node adapted Table 3 pipeline.

## Why the current pipeline is not faithful

The current `LLM_minima_selection` math pipeline is an adapted benchmark, not a Table 4 reproduction.

Main mismatches:

- **Backbones**: current presets use instruction-tuned checkpoints (`Meta-Llama-3-8B-Instruct`, `Qwen2-7B-Instruct`, `gemma-2-9b-it`).
- **Lower-level training**: current code fine-tunes **LoRA adapters** only.
- **Compute regime**: current launcher is **single GPU** and not distributed.
- **Memory settings**: current presets use `MAX_LENGTH=256`, `LoRA r=4`, `LoRA alpha=8`.
- **Validation**: current main Table 3 runs used **GSM8K-only validation**.
- **Baselines**: current `LESS` / `RHO-LOSS` are in-house proxies, not the original large-scale implementations.

Strong evidence that the current setup is not Table 4:

- Current zero-shot GSM8K:
  - `Llama-3-8B-Instruct`: `72.86`
  - `Gemma-2-9b-it`: `72.86`
- ScaleBiO Table 4 `Uniform Weighting` on GSM8K:
  - `Llama-3-8B`: `53.6`
  - `Gemma-2-9B`: `56.3`

If zero-shot is already far above the paper's `Uniform`, the protocol is not aligned.

## Target protocol from ScaleBiO Appendix

The reproduction target for the math task should match the following:

- **Models**: `Llama-3-8B`, `Qwen-2-7B`, `Gemma-2-9B`
- **Training portfolio**: 10-source math portfolio, 2.4M total samples
- **Validation portfolio**: 6 validation sources
- **Validation sampling**: `280` examples per validation source, total `1680`
- **ScaleBiO val split**: `1400` for reweighting objective + `280` for tuning
- **Reweighting pool**: `3000` examples per training source
- **Final train size**: `20K`
- **Final fine-tuning**: `1` epoch, lr `8e-6`, global batch size `64`
- **ScaleBiO reweighting**:
  - `3` epochs
  - `alpha = 100`
  - lr for source weights `lambda`: `1e-2`
  - model learning rate searched in `{1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 8e-6, 1e-5}`
- **Precision**: `bf16`
- **System**: `8 x H100 80GB`
- **Distributed training**: `FSDP`
- **Optimization style**: randomized coordinate descent / LISA-style large-scale training

Appendix-specific baseline details to match:

- **Uniform Weighting**: direct equal-source sampling for the final train set.
- **LESS**: original-paper settings, not the current cosine-similarity proxy.
- **RHO-LOSS**: original setup; appendix notes a shared `Qwen-2-1.5B` reference model.

## Current code blockers

These items block a faithful Table 4 reproduction in the current codebase:

1. **No distributed training path**
   - `run_exact_math_reasoning.sbatch` requests `1` GPU.
   - No FSDP / `torchrun` / multi-process training path exists in the math trainer.

2. **LoRA-only lower level**
   - `exact_math_reasoning.py` wraps the model with `get_peft_model(...)`.
   - There is no full-parameter or LISA-style block-coordinate mode.

3. **Baseline mismatch**
   - `LESS` and `RHO-LOSS` are simplified internal approximations.
   - They are not faithful reproductions of the baseline training recipes from the ScaleBiO appendix.

4. **Checkpoint mismatch**
   - Current presets default to instruction-tuned checkpoints.
   - Table 4 likely requires a different checkpoint choice; this must be verified before launch.

5. **Validation mismatch in current Table 3 path**
   - Current finished Table 3 runs used simplified GSM8K-only validation and aggressive memory reductions.

## Reproduction plan

### Phase 0 — Freeze the target protocol

Before launching anything:

- Confirm exact model checkpoint ids for the Table 4 reproduction target.
- Decide whether the target is:
  - exact official backbone checkpoints, or
  - the closest public equivalents if the original code/checkpoints are not accessible.
- Lock the math prompt and answer extraction format for the reproduction track.

### Phase 1 — Build the missing infrastructure

Required engineering work:

- Add a **distributed math trainer** with `8` GPUs.
- Add **FSDP** support.
- Add a **non-LoRA mode** suitable for full-parameter or LISA-style updates.
- Separate the current adapted single-GPU pipeline from the faithful-reproduction pipeline.

### Phase 2 — Reproduce paper baselines before HG-MS

Run only these first:

1. `Uniform Weighting`
2. `ScaleBiO`
3. `LESS`
4. `RHO-LOSS`

Start with **one model only**:

- Recommended canary: `Llama-3-8B`

### Phase 3 — Canary acceptance rule

Do **not** run `HG-MS` yet.

First require:

- `Uniform Weighting` on GSM8K is near the Table 4 regime.
- `ScaleBiO` beats `Uniform` by a similar margin directionally.

Suggested stop/go rule for the canary:

- If `Uniform` is not within roughly `±3` GSM8K points of the paper target, stop and debug protocol mismatch.
- If `Uniform` is still abnormally strong, do not compare `HG-MS` in that setting.

### Phase 4 — Add HG-MS fairly

Only after the canary passes:

- run `HG-MS` under the same:
  - checkpoints
  - data splits
  - validation protocol
  - final fine-tuning setup
  - compute budget

If `HG-MS` cannot be run under the same lower-level training regime, then it is **not** a faithful comparison to Table 4 and should be labeled as an adapted comparison.

## Recommended immediate next actions

1. Verify the exact target checkpoints for Table 4.
2. Decide whether faithful reproduction requires full-parameter training rather than LoRA.
3. Implement the distributed/FSDP training path.
4. Reproduce `Uniform` and `ScaleBiO` on `Llama-3-8B` first.
5. Only after that, add `LESS`, `RHO-LOSS`, and `HG-MS`.

## Concrete implementation scope in this repo

The minimum engineering scope for a faithful reproduction track is:

1. **Separate configs from the current adapted path**
   - Keep the current single-GPU LoRA pipeline unchanged for adapted ablations.
   - Add a separate faithful-reproduction config path rather than overloading the current presets.
   - Expected touch points:
     - `model_presets.sh`
     - `submit_math_gsm8k_multiseed.sh`
     - a new faithful-specific launcher or preset file

2. **Add an 8-GPU math launch path**
   - Create a new Slurm wrapper for multi-GPU jobs.
   - Launch the trainer with `torchrun` and explicit world-size / rank wiring.
   - Expected touch points:
     - `run_exact_math_reasoning.sbatch` (reference only)
     - a new `run_exact_math_reasoning_fsdp.sbatch`

3. **Add a non-LoRA lower-level training mode**
   - The faithful path should not force `get_peft_model(...)`.
   - Add a mode switch so the trainer can run either:
     - the current LoRA path, or
     - a full-parameter / block-coordinate faithful path.
   - Expected touch points:
     - `exact_math_reasoning.py`
     - training-argument parsing and model construction

4. **Isolate faithful baseline implementations**
   - Move the current proxy logic for `LESS` and `RHO-LOSS` out of the faithful path.
   - Add baseline-specific modules or functions that match the paper settings.
   - Expected touch points:
     - `exact_math_reasoning.py`
     - new helper modules for `less` and `rho_loss`

5. **Freeze the evaluation contract**
   - Keep GSM8K and MATH answer extraction fixed for the faithful track.
   - Record the prompt template and extraction logic used for every run.
   - Expected touch points:
     - `exact_math_reasoning_data.py`
     - run manifests / config dumps

6. **Add a canary report script**
   - Aggregate the first faithful `Uniform` and `ScaleBiO` runs against the paper targets.
   - Fail fast if `Uniform` is outside the acceptance window.
   - Expected touch points:
     - a new reporting script under `LLM_minima_selection/`

## Minimal canary acceptance checks

Before running `HG-MS`, require all of the following on `Llama-3-8B`:

- checkpoint choice is frozen and documented;
- training mode is the faithful one, not the LoRA fallback;
- `Uniform` is within the target regime of Table 4;
- `ScaleBiO` improves over `Uniform` directionally.

## Go / No-Go summary

- **Go** if we are willing to build a separate large-scale reproduction pipeline.
- **No-Go** if we expect the current single-H100 LoRA codepath to become faithful through hyperparameter tuning alone.

The current code can still support adapted ablations, but it should not be presented as a faithful Table 4 reproduction without the steps above.
