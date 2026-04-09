# Real-World Qwen2.5 Results Snapshot

## Instruction Following

These are the current paper numbers from `Hyper_grad_paper/Minima_Selection/example_paper.tex`.

| Method | Max Dev. | Held-out Loss |
|---|---:|---:|
| Uniform | `0.0000` | `0.5470` |
| LESS | `0.0461` | `0.5368` |
| RHO-LOSS | `0.0556` | `0.5501` |
| ScaleBiO | `0.9416` | `0.5529` |
| HG-single | `0.0007` | `0.5401` |
| HG-MS | `0.2120` | `0.5299` |

## Math Reasoning

The real-world math code path is included in this package, but the main-text math tables in the current draft are being reworked around the larger multi-model benchmark layout.

For that reason, this package keeps the existing Qwen2.5 math runner and launch scripts, but does **not** pin a frozen paper table here.

Source of truth:
- `Hyper_grad_paper/Minima_Selection/example_paper.tex`
