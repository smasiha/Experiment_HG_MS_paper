# Imbalanced Loss-Tuning Results Snapshot

Current paper numbers from `Hyper_grad_paper/Minima_Selection/example_paper.tex` and its included appendix files.

## MNIST balanced accuracy at 120s per algorithm

| Method | Train BalAcc | Test BalAcc |
|---|---:|---:|
| HG-baseline | `99.46±0.20` | `95.48±0.77` |
| HG-MS | `99.51±0.26` | `96.45±0.11` |
| V-PBGD | `91.31±2.19` | `90.54±1.95` |
| G-PBGD | `75.88±2.17` | `77.40±1.41` |
| IAPTT-GM | `82.74±2.62` | `82.13±2.30` |

Figure source in the draft:
- `Hyper_grad_paper/Minima_Selection/figures/imbalance/compare_acc_vs_time_mean_ci_job2551942.png`

Source of truth:
- `Hyper_grad_paper/Minima_Selection/Appendix_neurips.tex`
