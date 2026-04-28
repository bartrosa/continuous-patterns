# Configuration Legend (English)

This document explains all parameters from:

`results/paper_v1/gamma/sweeps/v1_gamma_scan_20260426T183117Z/v1_gamma_scan_run_0001/20260426T183415Z/config.yaml`

It includes:
- what each parameter does,
- which parameters are discrete (enum/boolean),
- and allowed choices for those discrete fields.

---

## 1) `experiment`

- `name`: Run identifier used for output directory naming.
- `model` (**discrete**): simulation operating mode.
  - Allowed values:
    - `cavity_reactive`: Stage I style run with active rim-fed reaction dynamics.
    - `bulk_relaxation`: Stage II style run with bulk relaxation behavior.
  - Legacy aliases accepted by loader:
    - `agate_ch` -> `cavity_reactive`
    - `agate_stage2` -> `bulk_relaxation`
- `seed`: RNG seed for reproducible stochastic initialization/noise.
- `description`: Human-readable run notes.
- `scenario` (**discrete/optional string**): optional scenario tag; `null` means no explicit scenario tag stored in this field.

### Scenario presets available in the loader (`initial.scenario`)

Even if `experiment.scenario` is `null`, the config system supports `initial.scenario` presets:
- `open_inflow`
- `closed_supersaturated`
- `closed_aging`
- `open_aging`
- `bulk_relaxation`

These presets apply bundled overrides to `physics` and `initial`.

---

## 2) `geometry`

- `type` (**discrete enum**): cavity geometry family.
  - Allowed values:
    - `circular_cavity`
    - `elliptic_cavity`
    - `polygon_cavity`
    - `wedge_cavity`
    - `rectangular_slot`
  - Current run: `circular_cavity`.
- `L`: physical side length of the square computational domain.
- `n`: grid resolution (`n x n` cells).
- `eps_scale`: interface transition width scaling used in smooth cavity masks.

### Geometry-specific fields

- `R`: radius (used by `circular_cavity`).
- `a`, `b`, `theta`: ellipse semi-axes and rotation (used by `elliptic_cavity`).
- `width`, `height`: slot size (used by `rectangular_slot`).
- `n_sides`, `vertices`, `theta_offset`: polygon definition fields.
- `R_inner`, `R_outer`, `opening_angle`, `theta_center`: wedge/annular sector fields.

In this run, only `R` is used; other geometry-specific fields are `null`.

---

## 3) `physics` (core model controls)

- `W`: double-well potential barrier scale.
- `gamma`: immiscibility coupling strength between polymorph fields.
- `kappa_x`, `kappa_y`: gradient-energy coefficients (x/y).
- `M_m`, `M_c`: mobilities for moganite/chalcedony fields.
- `D_c`: diffusion coefficient for dissolved silica field `c`.
- `k_rxn`: reaction/precipitation rate scale.
- `c_sat`: saturation concentration threshold.
- `c_0`: initial dissolved concentration baseline.
- `c_ostwald`, `w_ostwald`: Ostwald-like term parameters.
- `lambda_bar`: coupling scale used by kinetic/front terms.

### Ratchet controls

- `use_ratchet` (**boolean**): enables/disables ratchet window logic.
- `phi_m_ratchet_low`: lower `phi_m` ratchet bound.
- `phi_m_ratchet_high`: upper `phi_m` ratchet bound.

Current run: ratchet is enabled (`true`) in `[0.3, 0.5]`.

---

## 4) `physics.phases`

Per-phase block defines energetic and transport behavior.

### Common fields per phase

- `potential` (**discrete enum**):
  - Allowed values:
    - `double_well`
    - `tilted_well`
    - `asymmetric_well`
    - `zero`
- `potential_kwargs`: parameters for selected potential.
- `mobility`: phase mobility.
- `rho`: phase density/weight factor.
- `psi_sign`: sign convention in split-order parameter contribution.
- `active` (**boolean**): includes/excludes phase from active evolution.

### Phases present

- `moganite`: active.
- `chalcedony`: active.
- `alpha_quartz`: `null` (not used in this run).
- `impurity`: `null` (not used in this run).

---

## 5) `physics.aging`

- `active` (**boolean**): turns kinetic aging path on/off.
- `k_age`: aging rate.
- `q_to_quartz`: optional conversion fraction toward quartz channel.

Current run: aging disabled (`active: false`).

---

## 6) `stress`

- `mode` (**discrete enum**): stress field builder.
  - Allowed values:
    - `none`
    - `uniform_uniaxial`
    - `uniform_biaxial`
    - `pure_shear`
    - `flamant_two_point`
    - `pressure_gradient`
    - `kirsch`
    - `lithostatic`
    - `tectonic_far_field`
    - `inglis`
  - Current run: `none` (stress disabled).
- `sigma_0`: main amplitude for simple stress modes.
- `stress_coupling_B`: coupling strength from stress tensor to chemical potential.
- `stress_eps_factor`: smoothing/regularization factor used by some stress builders.

### Mode-specific stress fields (used depending on `stress.mode`)

- `rho_g_dim`, `lateral_K`: lithostatic mode parameters.
- `S_H`, `S_h`, `S_V`, `theta_SH`: tectonic far-field parameters.
- `R`: circular cavity radius for Kirsch/related coupling.
- `a`, `b`, `theta`: elliptic geometry linkage for Inglis.
- `S_xx_far`, `S_yy_far`, `S_xy_far`: remote stress tensor components.
- `pore_pressure` (optional nested block):
  - `field` (**discrete enum**): `uniform` or `hydrostatic`
  - `p0`, `biot_alpha`: pore-pressure magnitude and Biot factor.

In this run these mode-specific fields are `null` because `mode: none`.

---

## 7) `gravity`

- `rim_alpha`: rim weighting/ramp parameter.
- `g_c`: body-force coefficient on dissolved field `c`.
- `g_phi_m`, `g_phi_c`, `g_phi_q`, `g_phi_imp`: body-force coefficients on phase fields.

Current run: all are zero (gravity off).

---

## 8) `time`

- `dt`: simulation time step.
- `T`: final simulation time horizon.
- `snapshot_every`: interval (in steps) for snapshot sampling.

---

## 9) `output`

- `save_final_state` (**boolean**): write final `npz` state.
- `flux_sample_dt`: diagnostic sampling interval for flux/mass-balance metrics.
- `record_spectral_mass_diagnostic` (**boolean**): enable spectral mass drift diagnostics.
- `record_evolution_gif` (**boolean**): enable evolution GIF export.
- `save_snapshots_h5` (**boolean**): enable HDF5 snapshot export.
- `save_jablczynski_plot` (**boolean**): enable Jabłczyński plot export.
- `gif_max_frames`: frame cap when GIF export is enabled.
- `gif_fps`: GIF frame rate.
- `include_params_panel` (**boolean**): include parameter panel in figure export.
- `log_level` (**discrete practical set**): runtime logging level.
  - Typical values used by runner:
    - `DEBUG`
    - `INFO`
    - `WARNING`
    - `ERROR`

---

## 10) `initial`

- `initial: {}` means no explicit per-field initialization overrides in this YAML.
- Initialization then follows default/model/scenario logic from the config loader and model implementation.

---

## Quick profile of this specific run

- Geometry: circular cavity (`R=80`, `L=200`, `n=512`)
- Model mode: `cavity_reactive`
- Stress: `none`
- Gravity: off
- Gamma value: `3.0` (this is the gamma-scan run `0001`)
- Ratchet: on (`phi_m` window `0.3` to `0.5`)
- Aging: off
