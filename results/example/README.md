# Example run layout (for reviewers)

A real baseline run writes a **timestamped** folder under **`results/agate_ch/`**, e.g. `results/agate_ch/run_20260422_083900/`, with:

| File | Role |
|------|------|
| `fields_final.png` | 2×2 heatmaps: φ_m, φ_c, φ_m+φ_c, c with cavity outline |
| `radial_profile.png` | Azimuthally averaged ⟨φ⟩(r) for moganite, chalcedony, and total |
| `jablczynski.png` | Liesegang-style diagnostics: log d vs log r, q_n, d_n |
| `evolution.mp4` | Time-lapse of φ_m+φ_c (not stored here to keep the repo small) |
| `snapshots.h5` | Field stacks (not stored here) |
| `summary.json` | Parameters, band count, mean/std of q, classification, wall time |

`summary.json` in this folder is a **schema example** only; numbers are illustrative. After `uv run python -m continuous_patterns.agate_ch.run --config configs/agate_ch/baseline.yaml` you will get a full output tree under `results/agate_ch/run_<timestamp>/`.
