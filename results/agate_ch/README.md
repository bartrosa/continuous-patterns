# Agate CH simulation outputs

Runs from `python -m continuous_patterns.agate_ch.run` write here by default:

- **`run_<YYYYMMDD_HHMMSS>/`** — single baseline or `--quick` jobs
- **`sweep_<YYYYMMDD_HHMMSS>/`** — `--sweep` batches (subfolder per run `id`). A **gamma** sweep (`gamma_scan.yaml`) also writes **`paper_figures/`** and updates the repo-root **`RESULTS.md`** when the run finishes (if a main six-config sweep with `no_pinning` already exists, figures are placed there; otherwise under the new gamma sweep).

These trees are **gitignored** (large HDF5, videos, PNG grids). Keep this file so the directory is tracked; attach sweep paths when documenting a paper or report.

Historical outputs may still live under the repo root `results/` from before this layout; new jobs use **`results/agate_ch/`** only.
