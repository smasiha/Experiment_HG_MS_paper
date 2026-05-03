# Data Hyper-Cleaning Results Snapshot

Current paper numbers from `Hyper_grad_paper/Minima_Selection/example_paper.tex` and its included appendix files.

## Fixed-budget MNIST label-noise results

| Method | rho=0.4 Train | rho=0.4 Test | rho=0.6 Train | rho=0.6 Test | rho=0.8 Train | rho=0.8 Test |
|---|---:|---:|---:|---:|---:|---:|
| HG-baseline | `79.69±1.78` | `81.22±1.45` | `70.19±7.56` | `73.09±4.86` | `63.48±2.48` | `64.54±2.62` |
| HG-MS | `89.00±1.60` | `86.98±1.02` | `81.12±2.71` | `80.08±2.52` | `66.47±2.69` | `66.08±3.51` |
| V-PBGD | `61.62±4.56` | `60.43±4.69` | `62.12±2.65` | `61.23±3.44` | `60.29±2.16` | `60.07±2.08` |
| G-PBGD | `70.73±4.48` | `69.21±4.93` | `67.40±7.94` | `66.26±7.86` | `58.84±3.87` | `58.77±4.93` |
| IAPTT-GM | `71.46±2.76` | `71.04±3.45` | `69.86±2.74` | `69.68±3.29` | `63.84±2.95` | `63.48±2.14` |

Figure source in the draft:
- `Hyper_grad_paper/Minima_Selection/figures/data_hyper_cleaning/`

Source of truth:
- `Hyper_grad_paper/Minima_Selection/Appendix_neurips.tex`
