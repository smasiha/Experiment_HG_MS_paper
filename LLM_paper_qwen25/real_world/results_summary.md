# Real-World LLM Results Snapshot

These are the current paper-facing numbers from `Hyper_grad_paper/Minima_Selection/example_paper.tex`.

## Mathematical Reasoning: GSM8K / MATH

Final test accuracy in percent after the shared one-epoch final fine-tuning stage. Iterative methods use matched reweighting budgets of approximately `12h` for Llama-3-8B, `9h` for Qwen-2-7B, and `15h` for Gemma-2-9B.

| Method | Llama GSM8K | Qwen GSM8K | Gemma GSM8K | Llama MATH | Qwen MATH | Gemma MATH |
|---|---:|---:|---:|---:|---:|---:|
| Uniform Weighting | `60.06` | `60.03` | `78.08` | `30.24` | `30.26` | `37.29` |
| RHO-LOSS | `65.43` | `63.15` | `78.47` | `25.89` | `33.40` | `37.35` |
| LESS | `74.45` | `65.28` | `79.28` | `29.93` | `29.24` | `38.00` |
| ScaleBiO | `66.19` | `75.74` | `79.62` | `30.58` | `36.93` | `37.98` |
| HG-baseline | `73.77` | `68.46` | `80.14` | `30.49` | `34.27` | `38.23` |
| HG-MS | `74.85` | `82.03` | `80.86` | `30.86` | `38.98` | `38.48` |

## Large-Model Extension: Qwen2.5-32B

Final GSM8K/MATH accuracy in percent. The transfer rows use source mixtures learned on Qwen-2-7B before final fine-tuning Qwen2.5-32B. The direct-diagnostic HG-MS row uses the memory-safe FSDP branch-selection diagnostic mixture.

| Method | GSM8K | MATH |
|---|---:|---:|
| Uniform Weighting | `87.19` | `43.30` |
| RHO-LOSS | `pending` | `pending` |
| LESS | `pending` | `pending` |
| ScaleBiO transfer | `88.40` | `51.29` |
| ScaleBiO direct | `84.99` | `41.99` |
| HG-MS transfer | `92.58` | `46.82` |
| HG-MS direct diagnostic | `86.81` | `44.32` |

## Instruction Following: MT-Bench

Final MT-Bench scores on the standard 80 FastChat questions. Higher is better.

| Method | Llama-3-8B | Qwen-2-7B | Gemma-2-9B |
|---|---:|---:|---:|
| Uniform Weighting | `7.07` | `6.54` | `8.04` |
| RHO-LOSS | `7.09` | `6.37` | `8.29` |
| LESS | `7.06` | `6.80` | `8.19` |
| ScaleBiO | `7.36` | `6.58` | `8.11` |
| HG-baseline | `7.32` | `6.58` | `8.09` |
| HG-MS | `7.33` | `6.92` | `8.30` |

## Qwen-2-7B Budget-Sensitivity Ledger

Each row uses the same Qwen-2-7B math pipeline and changes only the iterative reweighting budget before the same final fine-tuning/evaluation stage.

| Method | Budget | Outer steps | Top-3 learned source weights | GSM8K / MATH |
|---|---:|---:|---|---:|
| ScaleBiO | `3h` | `10527` | Magicoder `0.848`; GPTeacher `0.037`; WizardLM `0.026` | `66.19 / 30.70` |
| HG-baseline | `3h` | `1950` | GPT4-LLM-Cleaned `0.565`; MathInstruct `0.370`; SlimOrca `0.025` | `61.41 / 29.97` |
| HG-MS | `3h` | `1233` | MathInstruct `0.993`; GPT4-LLM-Cleaned `0.002`; SlimOrca `0.002` | `68.61 / 33.75` |
| ScaleBiO | `9h` | `13804` | MathInstruct `0.996`; SlimOrca `0.001`; GPT4-LLM-Cleaned `0.001` | `75.74 / 36.93` |
| HG-baseline | `9h` | `6129` | MathInstruct `0.996`; GPT4-LLM-Cleaned `0.003`; SlimOrca `0.000` | `68.46 / 34.27` |
| HG-MS | `9h` | `3754` | DART-Math `0.940`; MathInstruct `0.056`; Orca-Math `0.004` | `82.03 / 38.98` |
| ScaleBiO | `15h` | `23039` | Magicoder `0.980`; DART-Math `0.010`; MathInstruct `0.003` | `70.66 / 29.65` |
| HG-baseline | `15h` | `10150` | MathInstruct `1.000`; GPT4-LLM-Cleaned `0.000`; SlimOrca `0.000` | `70.05 / 34.14` |
| HG-MS | `15h` | `6283` | DART-Math `0.828`; Orca-Math `0.156`; MathInstruct `0.016` | `81.96 / 37.38` |
| ScaleBiO | `21h` | `32208` | Magicoder `0.978`; DART-Math `0.010`; MathInstruct `0.005` | `69.96 / 30.73` |
| HG-baseline | `21h` | `14220` | MathInstruct `0.992`; GPT4-LLM-Cleaned `0.002`; SlimOrca `0.001` | `69.73 / 34.09` |
| HG-MS | `21h` | `8813` | DART-Math `0.830`; Orca-Math `0.142`; MathInstruct `0.026` | `81.98 / 37.28` |

## Qwen-2-7B Fixed-Budget Branch-Count Sensitivity

All rows use the same nine-hour reweighting budget and one-epoch final fine-tuning protocol. `N` is the number of candidate branches per HG-MS outer update; larger `N` trades richer branch selection against fewer source-weight updates.

| Method | Branches N | Top-3 learned source weights | Outer steps | GSM8K / MATH |
|---|---:|---|---:|---:|
| HG-baseline | `1` | MathInstruct `0.996`; GPT4-LLM-Cleaned `0.003`; SlimOrca `0.000` | `6129` | `68.46 / 34.27` |
| HG-MS | `2` | Orca-Math `0.492`; DART-Math `0.423`; MathInstruct `0.085` | `5211` | `76.65 / 31.96` |
| HG-MS | `4` | DART-Math `0.940`; MathInstruct `0.056`; Orca-Math `0.004` | `3754` | `82.03 / 38.98` |
| HG-MS | `8` | DART-Math `0.760`; Magicoder `0.190`; Orca-Math `0.008` | `2516` | `66.19 / 29.37` |

## Notes

- `HG-baseline` is the one-branch hyper-gradient baseline; older notes may call it `HG-single`.
- `HG-MS` branch count is internal to the source-reweighting algorithm and is not best-of-N decoding.
- The current main LLM comparisons are single-seed matched-budget comparisons.
- Large model checkpoints, final adapters, and raw run directories are not tracked in git.
