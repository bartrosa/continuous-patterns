# Future work and known scope debt

## Aging reaction (kinetic trap)

The current aging implementation (`_G_aging` in `core/imex.py`,
PHYSICS.md §3.5) is a phenomenologically correct **kinetic rate law**
but exhibits a numerical/physical kinetic trap when applied to uniform
moganite. The double-well bulk dynamics suppresses small chalcedony
seed perturbations faster than aging can grow them, so homogeneous
moganite-to-chalcedony conversion is not observed in scenarios where
`phi_c_init = 0` exactly.

The `closed_aging` and `open_aging` scenario presets initialise with
small chalcedony seeds (`phi_c_init = 0.05`, `phi_c_noise = 0.02`) to
provide nucleation sites. With seeds, individual nuclei in cells where
`phi_c > 0.5` grow toward 1, while cells where `phi_c < 0.5` are
absorbed back to 0 by bulk well dynamics. **The result is
nucleation-and-growth conversion, not uniform conversion.** This is
geologically realistic (moganite → chalcedony in real agates also
proceeds via nucleation) but the canonical aging scenarios do **not**
demonstrate clean homogeneous aging conversion.

### Roadmap options for future iterations

1. **Thermodynamic tilt alternative**: replace explicit aging rate law
   with a `tilted_well` potential on phi_m (already implemented in
   `core/potentials.py`). Tilt shifts the moganite minimum below 1,
   so CH gradient flow naturally drives conversion. Mass-conserving
   by construction. Cost: changes physical interpretation from
   "kinetic rate law" to "thermodynamic preference"; would need a
   matching update to PHYSICS.md.

2. **Coupled aging-CH formulation**: introduce aging as a contribution
   to chemical potential `mu_alpha` rather than as an explicit
   post-CH increment. Mass-conserving by variational structure. Cost:
   reformulates aging mathematics; requires careful re-derivation of
   the rate law as a free-energy gradient.

3. **Explicit nucleation seeding**: extend `build_initial_state` to
   place discrete chalcedony nuclei (Voronoi-style) rather than
   uniform `phi_c_init` + noise. Most geologically faithful. Cost:
   new IC builder logic; trade between deterministic seed placement
   and reproducibility.

The kinetic-trap behaviour is documented in PHYSICS.md §3.5 and
§10.10. Reviewers and downstream users should be aware that aging
canonical YAMLs are scope debt, not finished demonstrations.

## Other deferred items

- Full Inglis elliptic field for `a ≠ b` (currently `sqrt(a·b)`
  Kirsch surrogate; see PHYSICS.md §7.6).
- Spectral dealiasing (Orszag 2/3 rule) — currently no explicit
  dealiasing applied; see PHYSICS.md §11 summary.
- Performance: `lax.scan` superscan over chunk loop; deferred from
  optimisation session because steady-state SM utilisation already
  ~80%.
