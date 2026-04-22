# Agate Cahn–Hilliard runner — reviewer notes

This document summarizes the numerical model, diagnostics, and known limitations so the workflow can be reproduced and critiqued without reverse-engineering the codebase.

## Numerical method

- **Spatial discretisation:** Periodic spectral grid on a square domain `[0,L]^2`; a smooth cavity mask `χ` restricts fields to an approximate circular cavity of geometric radius `R`, with a diffuse interface of width controlled by `eps_scale·dx` in `build_geometry`.
- **Temporal integration:** Pseudo-spectral **Model C** Cahn–Hilliard-style IMEX splitting: diffusion of `c` treated implicitly (factor `1 + dt·D_c·k²`), reaction `G` implicit in `c`; phase fields `(φ_m, φ_c)` use implicit diffusion of auxiliary chemical potentials constructed from `dfdphi_total` plus explicit reaction/ratchet sources; each sub-step advances with `jax.lax.scan`.
- **Why FFT:** Periodic boundaries allow exact Laplacians and biharmonic operators via `−k²` / `k⁴` multipliers with minimal viscosity; cavity geometry enters only through masking and Dirichlet replenishment near the rim, not through body forces.

## Mass balance

- **Physical flux (rim):** At each saved snapshot, dissolved `c` is sampled and a Fick-style influx rate is integrated in time (trapezoidal on snapshot times). This is the **primary** physical mass-balance check; `summary.json` reports the residual as `mass_balance_percent_physical` and a front-aware **surface flux budget** as `mass_balance_surface_flux` (fixed measurement radius, optional early stop when the precip front reaches that radius).
- **Direct / tautological check:** The solver can also report per-chunk changes in integrated silica `(c + ρ_m φ_m + ρ_c φ_c)·χ dx²`; telescoping gives a near-zero residual vs `(final − initial)` by construction.
- **Spectral mass diagnostic:** A short auxiliary run (default 1.0 time units at `dt=0.01`, 100 steps) with `disable_dirichlet: true`, a χ-windowed off-centre Gaussian in `c` (phases zero), and **no χ projection** after each step, so the update is periodic on the full torus. **Total silica** is the full-grid integral of `c + ρ_m φ_m + ρ_c φ_c` at each snapshot. **`leak_pct`** is the relative drift `(final − initial) / |initial|`. Expectation: `|leak_pct| ≪ 0.1%` (near roundoff). Results are stored in `summary.json` as `spectral_mass_conservation`. Enable with **`record_spectral_mass_diagnostic`** in YAML (the baseline config turns it on). Optional: `spectral_mass_T`, `spectral_mass_dt`, `spectral_mass_snapshot_every`.

## Interpolation at the rim

Sampling uses `scipy.ndimage.map_coordinates(..., order=1)` at fractional indices `(x/dx − ½, y/dx − ½)` corresponding to cell-centred grid values.

## Soft clip on phase fields

After each IMEX step, `φ_m` and `φ_c` are clipped to **`[-0.05, 1.05]`** before masking with `χ`. This keeps order parameters near physical `[0,1]` while leaving a narrow numerical buffer for spectral ringing; barrier terms in `dfdphi_barrier` penalise excursions beyond `[0,1]` in chemical potential space.

## Model C formulation — limitations

- Two phase fields share a single precipitation driver `G`; polymorph-specific kinetic prefactors enter through `ψ_m`, `ψ_c` (Ostwald + optional ratchet).
- No elastic energy, no fluid pressure, no explicit grain boundaries beyond diffuse interfaces.
- **Ratchet:** optional smooth dependence of `ψ_m` on `φ_m` between low/high thresholds to bias moganite near intermediate precipitate fractions — a phenomenological proxy for asymmetric nucleation/growth barriers, **not** a calibrated mineral kinetic law.

## Polymorph discrimination

Relative precipitation rates switch between “moganite-preferred” and “chalcedony-preferred” via sigmoid Ostwald partitioning in `gamma_sigma_ratchet`; the ratchet optionally tilts preference when `φ_m` enters a prescribed band.

## Diagnostics & classification

- Band counts use **multi-slice** radial sampling (see `diagnostics.py`), not azimuthally averaged radial profiles alone.
- Jabłczyński metrics use a canonical horizontal slice; classifications include **RATCHET-BANDED** etc. (`classify_jab_banding`).
- **Labyrinth heuristic:** flagged if median multislice band count `< 10` OR azimuthal variance along an intermediate-radius ring dominates variance of the azimuthally averaged radial profile (`labyrinth_heuristic`).

## Dimensionless parameters

All lengths, times, diffusivities, and reaction constants are dimensionless unless you introduce a calibration map to SI units (not attempted here).

## Outputs & scripts

- Main sweep: `--config configs/agate_ch/baseline.yaml --sweep configs/agate_ch/sweep.yaml`
- Gamma scan: same with `configs/agate_ch/gamma_scan.yaml`
- Publication assets (after runs):
  `--generate-paper <main_sweep_dir> <gamma_sweep_dir>`
  `--write-results <main_sweep_dir> <gamma_sweep_dir|`none`

## References cited in docs

Szymczak-style reviews of silica self-organisation (see project narrative in `RESULTS.md`).
