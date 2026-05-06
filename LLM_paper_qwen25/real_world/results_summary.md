# Real-World LLM Results Snapshot

These are the current paper-facing numbers from `Hyper_grad_paper/Minima_Selection/example_paper.tex` and its included appendix files.

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

Final GSM8K/MATH accuracy in percent. Each completed non-OOM row applies the corresponding source-reweighting method to Qwen2.5-32B, forms the full `20K`-example final fine-tuning set from the resulting source mixture, and evaluates on the full GSM8K/MATH test sets.

| Method | GSM8K | MATH |
|---|---:|---:|
| Uniform Weighting | `78.19` | `43.30` |
| RHO-LOSS | `OOM` | `OOM` |
| LESS | `OOM` | `OOM` |
| ScaleBiO | `88.40` | `51.29` |
| HG-baseline | `86.81` | `46.82` |
| HG-MS | `92.58` | `53.32` |

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

## Evaluation-Set Uncertainty

These intervals reproduce the Appendix F lightweight uncertainty check. They quantify finite-evaluation-set uncertainty only, not training-seed or source-reweighting variance.

| Benchmark | CI construction | Evaluation units | 95% half-width range |
|---|---|---:|---:|
| GSM8K, 7B/9B math table | Wilson binomial, `n=1319` | accuracy points | `2.07-2.64` |
| MATH, 7B/9B math table | Wilson binomial, `n=4989` | accuracy points | `1.22-1.35` |
| GSM8K, 32B panel | Wilson binomial, `n=1319` | accuracy points | `1.42-2.23` |
| MATH, 32B panel | Wilson binomial, `n=4989` | accuracy points | `1.37-1.39` |
| MT-Bench instruction table | Question-level bootstrap, `n=80` | MT-Bench points | `0.30-0.50` |

## Source-Mixture Audit

Rows below reproduce the final source proportions reported in Appendix F. `Uniform Weighting` is fixed by construction; `RHO-LOSS` and `LESS` are empirical source proportions in the selected final SFT set; `ScaleBiO`, `HG-baseline`, and `HG-MS` report the largest components of the source distribution used to sample the final SFT set.

### Mathematical Reasoning: Llama-3-8B

| Method | Final source mixture |
|---|---|
| Uniform Weighting | uniform over ten sources, each `0.100` |
| RHO-LOSS | SlimOrca `0.352`; GPT4-LLM-Cleaned `0.202`; WizardLM `0.142` |
| LESS | Orca-Math `0.119`; MathInstruct `0.117`; DART-Math `0.104` |
| ScaleBiO | Magicoder `0.829`; SlimOrca `0.096`; GPTeacher `0.044` |
| HG-baseline | MathInstruct `0.809`; SlimOrca `0.091`; GPT4-LLM-Cleaned `0.005` |
| HG-MS | MathInstruct `0.902`; DART-Math `0.036`; Orca-Math `0.010` |

### Mathematical Reasoning: Qwen-2-7B

| Method | Final source mixture |
|---|---|
| Uniform Weighting | uniform over ten sources, each `0.100` |
| RHO-LOSS | MathInstruct `0.299`; SlimOrca `0.215`; GPT4-LLM-Cleaned `0.133` |
| LESS | MathInstruct `0.140`; DART-Math `0.128`; Orca-Math `0.125` |
| ScaleBiO | MathInstruct `0.996`; SlimOrca `0.001`; GPT4-LLM-Cleaned `0.001` |
| HG-baseline | MathInstruct `0.996`; GPT4-LLM-Cleaned `0.003`; SlimOrca `0.000` |
| HG-MS | DART-Math `0.940`; MathInstruct `0.056`; Orca-Math `0.004` |

### Mathematical Reasoning: Gemma-2-9B

| Method | Final source mixture |
|---|---|
| Uniform Weighting | uniform over ten sources, each `0.100` |
| RHO-LOSS | SlimOrca `0.267`; GPT4-LLM-Cleaned `0.178`; WizardLM `0.144` |
| LESS | Orca-Math `0.110`; MathInstruct `0.109`; DART-Math `0.103` |
| ScaleBiO | UltraInteract `0.859`; Magicoder `0.083`; SlimOrca `0.033` |
| HG-baseline | MathInstruct `0.975`; SlimOrca `0.004`; GPT4-LLM-Cleaned `0.004` |
| HG-MS | MathInstruct `0.941`; DART-Math `0.016`; Orca-Math `0.010` |

### Large-Model Mathematical Reasoning: Qwen2.5-32B

| Method | Final source mixture |
|---|---|
| Uniform Weighting | uniform over ten sources, each `0.100` |
| RHO-LOSS | `OOM`; no completed final source mixture |
| LESS | `OOM`; no completed final source mixture |
| ScaleBiO | MathInstruct `0.996`; SlimOrca `0.001`; GPT4-LLM-Cleaned `0.001` |
| HG-baseline | Orca-Math `0.159`; UltraInteract `0.155`; DART-Math `0.153` |
| HG-MS | DART-Math `0.940`; MathInstruct `0.056`; Orca-Math `0.004` |

### Instruction Following: Llama-3-8B

| Method | Final source mixture |
|---|---|
| Uniform Weighting | uniform over eighteen sources, each `0.0556` |
| RHO-LOSS | Alpaca-es `0.096`; WildChat `0.079`; Alpaca-it `0.078` |
| LESS | Airoboros `0.088`; AlpacaChat `0.086`; AlpacaGPT4 `0.084` |
| ScaleBiO | OpenOrcaGPT4 `0.056`; LMSYS-Chat `0.056`; Alpaca-ru `0.056` |
| HG-baseline | Alpaca-it `0.135`; WildChat `0.134`; Airoboros `0.130` |
| HG-MS | WildChat `0.059`; Alpaca-ja `0.059`; OpenOrcaGPT4 `0.059` |

### Instruction Following: Qwen-2-7B

| Method | Final source mixture |
|---|---|
| Uniform Weighting | uniform over eighteen sources, each `0.0556` |
| RHO-LOSS | WildChat `0.130`; Alpaca-es `0.127`; Alpaca-de `0.117` |
| LESS | AlpacaChat `0.108`; SlimOrca `0.106`; Airoboros `0.105` |
| ScaleBiO | OpenOrcaGPT4 `0.057`; Alpaca-zh `0.056`; AlpacaGPT4 `0.056` |
| HG-baseline | Alpaca-it `0.210`; Alpaca-pt `0.209`; SlimOrca `0.206` |
| HG-MS | AlpacaGPT4 `0.060`; OpenOrcaGPT4 `0.060`; Alpaca-ko `0.060` |

### Instruction Following: Gemma-2-9B

| Method | Final source mixture |
|---|---|
| Uniform Weighting | uniform over eighteen sources, each `0.0556` |
| RHO-LOSS | uniform over eighteen sources, each `0.0556` |
| LESS | Airoboros `0.069`; AlpacaChat `0.069`; AlpacaGPT4 `0.069` |
| ScaleBiO | uniform over eighteen sources, each `0.0556` |
| HG-baseline | WildChat `0.211`; ShareGPT4 `0.210`; Alpaca-de `0.209` |
| HG-MS | Alpaca-ko `0.059`; OpenOrcaGPT4 `0.059`; WildChat `0.059` |

## Appendix F Protocol Details

- Final SFT uses AdamW with learning rate `8e-6`, no scheduler or warmup, default AdamW weight decay `0.01`, and one fixed epoch.
- Math final SFT uses a `20K`-example sampled training set, micro-batch size `1`, gradient accumulation `32`, LoRA rank `16`, scaling `32`, dropout `0.05`, maximum sequence length `512`, and greedy GSM8K/MATH evaluation.
- Instruction final SFT uses a `10K`-example sampled training set, micro-batch size `1`, gradient accumulation `64`, greedy MT-Bench generation, and GPT-4o judging at temperature zero.
- Main math `HG-MS` uses `N=4` branches for Llama-3-8B, Qwen-2-7B, and Gemma-2-9B; instruction `HG-MS` uses `N=8`; `HG-baseline` uses `N=1`.
- Math Qwen-2-7B `ScaleBiO` uses lower-model learning rate `1e-4`, outer learning rate `1.0`, SGD source-logit updates, weight decay `0`, and `scalebio_alpha=10`.
- Instruction `ScaleBiO` uses lower-model learning rate `1e-5`, outer learning rate `0.01`, weight decay `0`, and `scalebio_alpha=100`.
- Qwen2.5-32B `ScaleBiO` uses lower-model learning rate `5e-5`, outer learning rate `0.05`, SGD source-logit updates, weight decay `0`, and `scalebio_alpha=100`.

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
