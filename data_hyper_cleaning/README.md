# Data Hyper-Cleaning

This subfolder contains the MNIST label-noise hyper-cleaning experiment used in Section 6 of the paper.

Files:
- `code/`: experiment implementation
- `run_data_hyper_cleaning_gpu.sbatch`: generic launcher
- `run_data_hyper_cleaning_rho0p8_time120_inv50_all_gpu_r5.sbatch`: tuned paper-style `rho=0.8` launcher
- `run_lower_level_geometry_diagnostic_gpu.sbatch`: geometry-diagnostic launcher for the lower-level near-minimizer probe
- `results_summary.md`: current paper numbers

The copied Slurm scripts are package-relative and write outputs under `outputs/` by default.

Geometry diagnostic:
- `code/lower_level_geometry_diagnostic.py` freezes a representative outer iterate and probes whether the lower-level near-minimizer set looks non-singleton, low-dimensional, locally connected, and validation-sensitive.
- Example launch:
  `/home/masiha/venvs/hypergrad_env/bin/python paper_experiments/data_hyper_cleaning/code/lower_level_geometry_diagnostic.py --run-name rho0p8_geom`
- Outputs:
  `summary.json`, `candidate_metrics.csv`, `train_vs_val_scatter.png`, `pca_spectrum.png`, and `path_barriers.png`.
