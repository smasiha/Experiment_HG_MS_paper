# Controlled LLM Source Reweighting

This subfolder contains the controlled two-source LLM experiments already used in the paper:
- `denoising`
- `multilingual`

Files:
- `small_scale_data.py`: dataset construction
- `small_scale_reweight.py`: training / bilevel reweighting runner
- `run_small_scale_reweight.sbatch`: Slurm entrypoint
- `submit_small_scale_tasks.sh`: baseline grid launcher
- `submit_controlled_hg_tuning.sh`: tuned HG sweep launcher
- `summarize_small_scale_controlled.py`: result summarizer

This copy has been adjusted so the default paths stay inside `LLM_paper_qwen25/`.

