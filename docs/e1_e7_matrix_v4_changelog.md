# E1–E7 matrix v4 changelog

Evidence from the v3 run (20260605T185225Z) that motivated v4:

- `comparison_progress.json` showed only E3 and E7 were computed in the final invocation — other rows were **resumed** from an earlier attempt (fresh compute silently violated).
- E6 still open-like: mass_residual 91.7%, deposit 95 from a 50-unit budget. `initial.c_init: 0.5` applied but a **stale resumed run** kept `bc_right: dirichlet`.
- E7 numerically invalid: ~2.6M projection events, min_phi_w=0, ~493 bands at ~1 cell spacing — CH used O(dx²) dt and arithmetic-mean face mobility leaked demixing into the dry bulk.

## (A) No-resume enforcement

- `run_banding_comparison.py` no longer accepts `--resume` / `--finalize`.
- Every invocation creates a new timestamp directory; all seven rows simulate fresh.
- Each per-run `summary.json` stores `invocation_id` and `config_hash`.
- Before figure export, the runner asserts every row summary matches the current `invocation_id`.

## (B) bc_right pipeline

- `_verify_physics_overrides()` asserts every `physics.*` comparison override is present in the merged config.
- Integration test: `tests/integration/test_comparison_overrides.py`.
- Runtime: `max_abs_right_flux` accumulated when `bc_right == "noflux"`; must be 0.0 for valid closed rows.

## (C) c-update diff summary (v2 → v3, adopted in v4)

| v2 (legacy) | v3/v4 (adopted) |
|---|---|
| Updated `c` directly after transport with full `−ρ·R` debit | Transport **`wc = φ_w·c`**, then debit at most **`R_eff = min(R, wc_avail/(dt·ρ))`**, then **`c = wc/φ_w_new`** (`_wc_after_transport_and_reaction`, `_c_from_wc` in `slab_twophase.py`) |
| Could over-debit solute when `φ_w → 0`, producing fake rejection spikes and mass ledger drift | Discrete conservation of dissolved silica per cell; no solute debit where no water volume |

**Verdict: adopted.** The v3 form is the correct discrete conservation update. The v2 solute-rejection spike at the wall disappeared because transport is on `wc` with capped reaction debit, not because physics was removed. The faster E4 front is consistent with not artificially locking solute at the seed.

E4 figure caption states: `c-update=wc-transport v3 (R_eff capped)`.

## (D5) Front-advance mechanism (file + line)

**AC mode (`_simulate_ac`, `slab_twophase.py`):**

1. **Nucleation:** seed cells start with `φ_tot = 0.5` (`_build_initial_ac`).
2. **Reaction:** `R = k·relu(c−c_sat)·φ_tot·(1−φ_tot)` (`_reaction_rate`) — zero where `φ_tot=0`, so only existing solid catalyses precipitation.
3. **Growth:** `φ_tot_new = φ_tot_old + dt·R_eff` (line ~920) — deposit thickness increases where solid already exists.
4. **Packing advance:** when `φ_tot > φ_cap=0.95`, `_propagate_phi_overflow_ac` clips and pushes excess to the **right neighbor** (`_propagate_phi_overflow_ac`, called from `_apply_phi_overflow_ac` ~924), advancing the front without creating new solid in empty cells from reaction alone.

**CH mode (after v4 fixes):**

- Same **`R_eff` deposit** with **`phi_gate=0.02`** — no reaction where `φ_tot < 0.02`, preventing bulk trace ignition.
- Demixing flux uses **geometric-mean face mobility** (`_mobility_face_geom_mean`) — zero flux into `φ_tot=0` cells.
- Front advance still requires local `φ_tot > phi_gate` for precipitation; overflow packing same as AC when cap exceeded.
- If CH demixing does not couple to deposit advance, report `front_frozen=true` honestly.

## (D6) lambda_CH

Implemented free energy: `μ_α = W g'(φ_α) + γ φ_{β≠α} − κ ∂xx φ_α`.

Linear stability about symmetric mixed state:

\[
\lambda_{\mathrm{CH}} = 2\pi\sqrt{\frac{2\kappa}{\gamma}}
\]

For E7 (`gamma=4`, `kappa=0.5`):

\[
\lambda_{\mathrm{CH}} = 2\pi\sqrt{\frac{2 \cdot 0.5}{4}} = \pi \approx 3.14 \text{ domain units}
\]

At `L=100`, `n=512`: `dx ≈ 0.195`, so **~16 grid cells per spinodal wavelength**.

Compare measured mean band spacing `⟨d_n⟩` on E7 against this value after the v4 run (expect `⟨d_n⟩/λ_CH ~ O(1)` if demixing bands resolved; not ~0.05 as in v3).

## (D) CH numerics (v4 code)

1. `dt = dt_safety · min(dx²/(2D), dx⁴/(8·M0·κ))` — printed at CH start and stored in summary.
2. Face mobility: geometric mean `M_face = √(M_i M_j)`, `M_i = M0·φ_tot,i²`.
3. Nucleation gate: `R = 0` where `φ_tot < PHI_GATE_CH = 0.02` (CH only).
4. Rows with `n_projection_events > 0` → `NUMERICALLY_INVALID` (red cells, no band metrics).
5. E7 `time.T: 6000` only (documented above) if full T=20k exceeds ~2 h with correct dt.
