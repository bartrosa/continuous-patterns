# Agate CH simulation outputs

Runs from `python -m continuous_patterns.agate_ch.run` write here by default:

- **`run_<YYYYMMDD_HHMMSS>/`** — single baseline or `--quick` jobs
- **`stage_seq_run_a_<UTC>/`**, **`stage_seq_run_b_<UTC>/`** — sequential Stage I→II (`python -m continuous_patterns.agate_ch.run_sequence`), each with optional `final_state.npz` when `save_final_state: true`
- **`stage_sequence_latest.json`**, **`stage_sequence_comparison.png`** — written by `run_sequence` / `plot_stage_sequence` when you run the pipeline
- **`sweep_<YYYYMMDD_HHMMSS>/`** — `--sweep` batches (subfolder per run `id`). A **gamma** sweep (`gamma_scan.yaml`) also writes **`paper_figures/`** and updates the repo-root **`RESULTS.md`** when the run finishes (if a main six-config sweep with `no_pinning` already exists, figures are placed there; otherwise under the new gamma sweep).

These trees are **gitignored** (large HDF5, videos, PNG grids). Keep this file so the directory is tracked; attach sweep paths when documenting a paper or report.

Historical outputs may still live under the repo root `results/` from before this layout; new jobs use **`results/agate_ch/`** only.
