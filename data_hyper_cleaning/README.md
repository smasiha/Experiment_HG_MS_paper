# Data Hyper-Cleaning

This subfolder contains the MNIST label-noise hyper-cleaning experiment used in Section 6 of the paper.

Files:
- `code/`: experiment implementation
- `run_data_hyper_cleaning_gpu.sbatch`: generic launcher
- `run_data_hyper_cleaning_rho0p8_time120_inv50_all_gpu_r5.sbatch`: tuned paper-style `rho=0.8` launcher
- `results_summary.md`: current paper numbers

The copied Slurm scripts are package-relative and write outputs under `outputs/` by default.

