# `agate_stage2`: pure Stage II Cahn–Hilliard relaxation

## Physical meaning

**Stage II** treats long geological aging as gradient flow on the immiscibility double well: γ-driven segregation of model “moganite/chalcedony” order parameters in rock that is already crystallised. There is **no silica influx** and **no cavity wall**.

## Why not `agate_ch`?

The main `agate_ch` experiment couples **reactive precipitation** + **rim Dirichlet silica supply** with γ–CH dynamics. Stage II asks whether **χ–free, periodic bulk CH alone** yields phase separation starting from an almost uniform mixture — a separate falsification lever.

## What differs here

| Aspect | `agate_ch` | `agate_stage2` |
|--------|------------|----------------|
| Geometry | Cavities, χ-mask, rim ring | Full torus: χ=1, ring=0 |
| Reaction | On (configurable) | Off (`enable_reaction: false`) |
| Rim BC | Dirichlet c (configurable) | Off (`enable_dirichlet: false`) |
| IC | Cavity noise / blob | Default **homogeneous** mixture + noise |

## Run baseline

From repo root (after installing optional deps, e.g. `agate` extra):

```bash
python -m continuous_patterns.agate_stage2.run \
  --config configs/agate_stage2/baseline.yaml
```

Nested YAML is flattened automatically; overrides: `--T`, `--snapshot-every`. All YAML for this experiment lives under **`configs/agate_stage2/`** (baseline + `gamma_*.yaml`). Outputs: `results/agate_stage2/stage2_<timestamp>/`.

## Mass balance

With periodic bulk runs, **Option B** (dense rim flux vs dissolved disk) is **not defined**. Use **Option D** via `mass_balance_mode: spectral_only` and `spectral_mass_conservation` in `summary.json`.

## γ scan (labyrinth / no front)

Files `configs/agate_stage2/gamma_2.yaml` … `gamma_8.yaml` differ only in `physics.gamma`, `experiment.name`, and `noise_amplitude: 0.05`. From repo root:

- `uv run python -m continuous_patterns.agate_stage2.sweep_gamma` — writes `results/agate_stage2/stage2_gamma_<n>_0/` per γ.
- `uv run python -m continuous_patterns.agate_stage2.aggregate_sweep` — reads those dirs, writes `gamma_scan_summary.{json,csv}` and `gamma_scan_lambda_comparison.png`, plus `labyrinth_analysis.json` in each run via **spectral** λ_peak (not slice-based CV(q)).
