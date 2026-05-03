# Controlled LLM Results Snapshot

These are the current paper numbers from `Hyper_grad_paper/Minima_Selection/example_paper.tex` and its included appendix files.

## Denoising

| Method | Learned omega | L1 error | Val loss | Test loss |
|---|---:|---:|---:|---:|
| Uniform | `(0.500, 0.500)` | `1.000` | `2.571` | `2.610` |
| ScaleBiO | `(0.922, 0.078)` | `0.156` | `2.336` | `2.443` |
| HG-baseline | `(0.930, 0.070)` | `0.140` | `2.427` | `2.454` |
| HG-MS | `(0.962, 0.038)` | `0.077` | `2.403` | `2.437` |

## Multilingual

| Method | Learned omega | L1 error | Val loss | Test loss |
|---|---:|---:|---:|---:|
| Uniform | `(0.500, 0.500)` | `0.200` | `2.430` | `2.407` |
| ScaleBiO | `(0.623, 0.377)` | `0.047` | `2.339` | `2.371` |
| HG-baseline | `(0.540, 0.460)` | `0.119` | `2.327` | `2.307` |
| HG-MS | `(0.514, 0.486)` | `0.172` | `2.323` | `2.298` |

Source of truth:
- `Hyper_grad_paper/Minima_Selection/example_paper.tex`
- `Hyper_grad_paper/Minima_Selection/Appendix_neurips.tex`
