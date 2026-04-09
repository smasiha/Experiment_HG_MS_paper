# Imbalance Data

This subfolder contains the MNIST imbalanced-loss-tuning experiment currently reported in the appendix.

Files:
- `code/`: experiment implementation
- `run_imbalance_all_algorithms_gpu.sbatch`: generic launcher
- `tools/imbalance_scan_job.py`: helper to inspect a completed sweep
- `results_summary.md`: current paper numbers

The copied Slurm script is package-relative and writes outputs under `outputs/` by default.

