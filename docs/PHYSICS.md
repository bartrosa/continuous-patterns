# Physics and numerics specification — agate Cahn–Hilliard (Model C) with stress

This document describes the **continuum model**, **free-energy structure** (including ψ-split stress coupling), **kinetic selectors**, **boundary treatment**, and **numerical discretization** implemented in this repository. It is written as a specification for code and as draft material for a paper supplementary appendix.

**Notation:** $\phi_m$, $\phi_c$ are volume fractions of two crystalline polymorphs (moganite-like and chalcedony-like); $c$ is dissolved silica concentration. The domain is a periodic square $[0,L)^2$ with a **circular cavity** embedded via smooth masks (generalizations to other geometries are planned). Spatial coordinates are $(x,y)$, time is $t$.

---

## 1. Continuum model (Model C)

We evolve three fields on the periodic torus:

1. **Dissolved silica** $c(\mathbf{x},t)$ — reaction–diffusion with diffusivity $D_c$.
2. **Phase fields** $\phi_m(\mathbf{x},t)$, $\phi_c(\mathbf{x},t)$ — coupled Cahn–Hilliard equations with mobilities $M_m$, $M_c$ and a common immiscibility strength $\gamma$ between the two phases.

Let $f_{\mathrm{bulk}}(\phi_m) + f_{\mathrm{bulk}}(\phi_c)$ denote symmetric double-well energies confining each $\phi_\alpha$ toward $0$ and $1$, and let $f_{\mathrm{bar}}(\phi_\alpha)$ be an **outer barrier** that penalizes $\phi_\alpha < 0$ and $\phi_\alpha > 1$ (Section 4). The **local chemical driving** uses

$$
f_\alpha^{\mathrm{tot}}(\phi_\alpha) = f_{\mathrm{bulk}}(\phi_\alpha) + f_{\mathrm{bar}}(\phi_\alpha), \qquad \alpha \in \{m,c\}.
$$

Define **chemical potentials** (variational derivatives of the **local** free-energy density, plus linear cross-coupling $\gamma \phi_m \phi_c$ and optional stress, Section 5):

$$
\mu_m = \frac{\partial f_m^{\mathrm{tot}}}{\partial \phi_m} + \gamma \phi_c + \delta\mu_m^{\mathrm{stress}}, \qquad
\mu_c = \frac{\partial f_c^{\mathrm{tot}}}{\partial \phi_c} + \gamma \phi_m + \delta\mu_c^{\mathrm{stress}}.
$$

**Gradient energy** is anisotropic-diagonal in Fourier space: stiffnesses $\kappa_x$, $\kappa_y$ (defaults $\kappa_x = \kappa_y = \kappa$) multiply $|\partial_x \phi|^2$ and $|\partial_y \phi|^2$ in the energy; in the semi-discrete implementation this appears as a symbol $\kappa_x k_x^2 + \kappa_y k_y^2$ acting on each phase.

**Transport equations** (strong form on the torus):

$$
\partial_t c = D_c \nabla^2 c - G(c,\phi_m,\phi_c),
$$

$$
\partial_t \phi_m = M_m \nabla^2 \mu_m + \psi_m\, G(c,\phi_m,\phi_c), \qquad
\partial_t \phi_c = M_c \nabla^2 \mu_c + \psi_c\, G(c,\phi_m,\phi_c),
$$

where $\psi_m$, $\psi_c$ are **Ostwald partition factors** (Section 3) satisfying $\psi_m + \psi_c = 1$, and $G \ge 0$ is the **precipitation rate** (Section 3). Thus silica leaves the solution ($-G$ in $c$) and partitions between the two solid phases according to $(\psi_m,\psi_c)$.

**Silica bookkeeping:** bulk silica density is taken as $c + \rho_m \phi_m + \rho_c \phi_c$ with stoichiometric weights $\rho_m$, $\rho_c$ (often unity). Diagnostics integrate this combination over the cavity indicator $\chi$ (Section 6) to monitor mass balance (Section **10.2** Option C — Dirichlet injection; **10.3** Option B — surface flux budget; **10.1** Option D — spectral aux).

---

## 2. Free-energy viewpoint

The total free energy splits schematically as

$$
\mathcal{F} = \mathcal{F}_{\mathrm{grad}}[\phi_m,\phi_c] + \int \bigl( f_m^{\mathrm{tot}} + f_c^{\mathrm{tot}} + \gamma \phi_m \phi_c \bigr)\,\mathrm{d}\mathbf{x} + \mathcal{F}_{\mathrm{stress}}[\psi],
$$

with $\psi = \phi_m - \phi_c$ and

$$
\mathcal{F}_{\mathrm{stress}}[\psi] = \frac{B}{2} \int \sum_{i,j} \sigma_{ij}(\mathbf{x})\, \partial_i \psi\, \partial_j \psi \,\mathrm{d}\mathbf{x}.
$$

Here $\sigma_{ij}$ is a **prescribed** Cauchy stress tensor (Section 7), and $B \equiv$ `stress_coupling_B` controls coupling strength. When $B=0$ or $\sigma=0$, the legacy model is recovered.

The implementation adds $\delta\mu_m^{\mathrm{stress}}$, $\delta\mu_c^{\mathrm{stress}}$ such that the **variational derivative with respect to $\psi$** matches the divergence form (Section 5). This **ψ-split** is critical: the stress contribution enters the dynamics in a way consistent with using $\psi$ as the order parameter in $\mathcal{F}_{\mathrm{stress}}$, and the discrete operators preserve the intended **flux structure** for mass accounting in the coupled CH step.

---

## 3. Reaction term and Ostwald selector

**Precipitation rate** (implemented form):

$$
G(c,\{\phi_\alpha\}) = k_{\mathrm{rxn}}\,\max(c - c_{\mathrm{sat}}, 0)\,\max(1 - \textstyle\sum_\alpha \phi_\alpha,\, 0),
$$

where the sum runs over **all** solid phase fields tracked in the IMEX step (moganite, chalcedony, optional α-quartz, optional impurity placeholder). The factor $\max(c - c_{\mathrm{sat}}, 0)$ is a **first-order, supersaturation-gated precipitation rate** of the kind used in **reactive transport** models. The complementary packing factor is a **numerical / phenomenological device**; it is **not** derived from a detailed surface-nucleation law. Throughout, **temperature and pressure** are treated as **fixed**; any thermal or pressure dependence is absorbed into the constants $k_{\mathrm{rxn}}$ and $c_{\mathrm{sat}}$ (**isothermal, isobaric** effective medium).

**Ostwald channel:** the partition $(\psi_m,\psi_c)$ still uses **pre-step** $(c,\phi_m)$ and the same ratchet as before; **α-quartz is not fed by $G$** (only by optional **aging**, below).

So precipitation only occurs in supersaturated regions and only while the **total** solid packing tracked in $G$ has not saturated the local ceiling at $1$.

**Aging (optional):** when enabled, an explicit kinetic term
$G_{\mathrm{age}} = k_{\mathrm{age}}\,\phi_m\,\max(c_{\mathrm{sat}} - c,\, 0)$
reduces $\phi_m$ and increases $\phi_c$ and (optionally) $\phi_q$ with fractions $(1 - q_{\mathrm{quartz}})$ and $q_{\mathrm{quartz}}$. This conserves silica in the solid sum $(\rho_m\phi_m + \rho_c\phi_c + \rho_q\phi_q)$ when $\rho_\alpha \equiv 1$; **$c$ is not sourced** by aging. The split is gated in YAML so that $q_{\mathrm{quartz}} > 0$ requires an **active** α-quartz phase block.

**Base Ostwald factors** (smooth sigmoid in $c$):

$$
\psi_m^{(0)} = \sigma_{\mathrm{logit}}\!\left(\frac{c - c_{\mathrm{ostwald}}}{w_{\mathrm{ostwald}}}\right), \qquad \psi_c^{(0)} = 1 - \psi_m^{(0)},
$$

with $\sigma_{\mathrm{logit}}$ the logistic sigmoid.

**Ratchet (kinetic bias for slow moganite):** when the ratchet is enabled, a **smoothstep** ramp $S(\phi_m)$ between `phi_m_ratchet_low` and `phi_m_ratchet_high` modulates moganite uptake:

$$
\psi_m = \mathrm{clip}\Bigl( \psi_m^{(0)} + r\, (1-\psi_m^{(0)})\, S(\phi_m),\, 0,\, 1 \Bigr), \qquad \psi_c = 1 - \psi_m,
$$

where $r \in \{0,1\}$ is the ratchet on/off flag in simulation parameters (`use_ratchet`). Intuitively, for intermediate $\phi_m$ the effective partition shifts toward moganite relative to the pure Ostwald curve — a **phenomenological kinetic asymmetry**, not derived from a single equilibrium double well.

---

## 4. Double well and barrier confinement

**Symmetric double well** (same $W$ for both phases):

$$
f_{\mathrm{bulk}}(\phi) = W\,\phi^2 (1-\phi)^2 \quad\Rightarrow\quad
\frac{\partial f_{\mathrm{bulk}}}{\partial \phi} = 2W\,\phi(1-\phi)(1-2\phi).
$$

**Outer barrier** (quadratic penalty outside $[0,1]$):

Let $\phi^- = \max(-\phi, 0)$ and $\phi^+ = \max(\phi - 1, 0)$. Then

$$
\frac{\partial f_{\mathrm{bar}}}{\partial \phi} = -2\lambda_{\mathrm{bar}}\,\phi^- + 2\lambda_{\mathrm{bar}}\,\phi^+.
$$

This is an **ad hoc confinement** to keep $\phi$ numerically inside (or near) $[0,1]$; it is **not** a smooth double-well derived from a single thermodynamic potential on $\mathbb{R}$.

**Implementation note:** bulk $\partial f/\partial\phi$ is evaluated in ``core/potentials.py`` via plain-function dispatch (:data:`~continuous_patterns.core.potentials.POTENTIAL_BUILDERS`); optional variants include a tilted well and an asymmetric double well. The barrier derivative above is always added separately (:func:`~continuous_patterns.core.potentials.barrier_prime`). Per-phase static coefficients (including mobility $M_\alpha$, stoichiometric weight $\rho_\alpha$, and the potential kind) live in ``PhasePotentialParams`` on ``SimParams`` as ``phi_m_potential`` and ``phi_c_potential``. Run cards may set ``physics.phases`` or rely on legacy flat keys ``W``, ``M_m``, ``M_c``, ``rho_m``, ``rho_c`` (normalized to ``phases`` at YAML load in ``core/io.py``).

**Hard clip after every IMEX step (implementation):** after each step, both phase fields are **pointwise clamped** in real space to

$$
\phi_\alpha \in [-0.05,\, 1.05], \qquad \alpha \in \{m,c\},
$$

immediately following the inverse FFT that recovers $\phi_\alpha$ from the implicit update. This is **not** a smooth projection: it is a **numerical guardrail** that can **inject or remove** a tiny amount of “phase volume” if a field would otherwise leave the interval (mass within the strict variational formulation is therefore not exact at machine level). In practice the **barrier forces** and typical amplitudes keep $\phi$ near $[0,1]$, so the clip rarely differs from a smooth confinement; **global silica budgets** are still checked with **Option C** (Section 10.2) and related diagnostics rather than by assuming exact invariance under clipping.

---

## 5. Stress coupling — ψ-split divergence form

Define $\psi = \phi_m - \phi_c$. The stress contribution to the chemical potentials is implemented in Fourier space as

$$
\mu_{\mathrm{stress}} = -B\, \nabla\cdot(\sigma \nabla \psi),
$$

computed **pseudospectrally**: gradients of $\psi$ via $ik$, products $\sigma \nabla \psi$ in real space, divergence via $ik$, then $-B$ scaling in Fourier space, inverse FFT to real space.

The **ψ-split** assignment is

$$
\delta\mu_m^{\mathrm{stress}} = +\tfrac{1}{2}\,\mu_{\mathrm{stress}}, \qquad
\delta\mu_c^{\mathrm{stress}} = -\tfrac{1}{2}\,\mu_{\mathrm{stress}},
$$

so that $\delta\mu_m^{\mathrm{stress}} - \delta\mu_c^{\mathrm{stress}} = \mu_{\mathrm{stress}}$, matching $\delta \mathcal{F}_{\mathrm{stress}} / \delta \psi$ for the free-energy density quadratic in $\nabla \psi$. The Cahn–Hilliard fluxes then apply $M_\alpha \nabla^2 \mu_\alpha$ to these full $\mu_\alpha$.

**Mass conservation (structural):** the divergence form $-B\nabla\cdot(\sigma\nabla\psi)$ is the variational derivative of a **gradient–gradient** coupling in $\psi$; when $\sigma$ is symmetric and boundary / mask handling is consistent, this is the standard route to avoid spurious source terms in the **pair** $(\phi_m,\phi_c)$ relative to the chosen $\psi$-energy (discrete conservation additionally depends on periodicity, mask projection of $\phi$ inside the cavity, rim silica exchange, and the hard clip in Section 4 — monitored numerically via **Option C** in Section 10.2).

**Numerical branch:** if `stress_coupling_B` is zero or the tensor is numerically zero, the code takes the **legacy** Laplacian path without recomputing stress derivatives (JIT-friendly `cond`).

### 5.1 Why $\psi$ (and not independent per-field coupling)

**Physical:** $\psi = \phi_m - \phi_c$ is the natural **contrast** between the two polymorph fractions. Stress in this phenomenological model is interpreted as biasing **how** silica partitions between polymorphs — the **difference** in crystal content — rather than driving total solid mass as an independent copy of the same field equation for each $\phi_\alpha$.

**Mathematical:** coupling through a single $\mathcal{F}_{\mathrm{stress}}[\psi]$ with $\mu_{\mathrm{stress}} = -B\nabla\cdot(\sigma\nabla\psi)$ and the **ψ-split** $(\pm\tfrac12)$ is **variational**: it is the gradient flow of a well-defined contribution to the free energy with respect to $\psi$. By contrast, **applying the same stress operator independently to $\phi_m$ and $\phi_c$** (or copying $\sigma|\nabla\phi_\alpha|^2$ for each field) is **not** generically derivable from a single free energy in $(\phi_m,\phi_c)$ with the intended mass split between phases; it breaks the **divergence structure** that keeps the CH step consistent with a conserved or budgeted silica density when combined with reaction.

**Historical (Phase 2 bugfix):** an early implementation used **per-field** stress-like terms. That path allowed **pathological growth of $\phi_m + \phi_c$** (observed up to $\approx 2$ in the cavity), i.e. a **non-physical** breakdown of the two-phase packing interpretation. The fix was to adopt the **ψ-split divergence form** above, which restored physically plausible totals and stable coupling in calibration scans.

---

## 6. Geometry, masks, and boundary conditions

### 6.1 Stage I — cavity, rim, and silica bookkeeping

**Cell-centred periodic grid:** $n \times n$ points, spacing $\Delta x = L/n$. Default production setup in this project: $n=512$, $L=200$, hence $\Delta x = 200/512$.

**Radial distance** from cavity centre $(L/2, L/2)$: $r = \|\mathbf{x} - \mathbf{x}_c\|$.

**Cavity indicator $\chi$:** smooth tanh transition at $r = R$,

$$
\chi(\mathbf{x}) = \tfrac{1}{2}\Bigl(1 - \tanh\bigl(\tfrac{r - R}{\varepsilon_\chi \Delta x}\bigr)\Bigr),
$$

with default $\varepsilon_\chi = 2$ (`eps_scale` in code). Thus $\chi \approx 1$ inside the cavity and $\approx 0$ outside.

**Implementation detail (χ transition width):** the effective denominator is $\max(\varepsilon_\chi \Delta x,\, \Delta x)$ — i.e. `max(eps_scale·Δx, Δx)` — so the transition width is never narrower than one cell. This floor prevents pathological aliasing when `eps_scale` is sent close to zero. For standard `eps_scale ≥ 2` the floor is inactive.

**Dirichlet ring for $c$:** a **normalized Gaussian** mask centred on $r \approx R$, with **unit peak**, enforces rim values of $c$ (scalar $c_0$ or vertical ramp with `c0_alpha` for gravity-rim experiments). **Implementation:** `ring = exp(-½((r-R)/σ)²) / max(...)` with $\sigma = \texttt{eps\_scale} \cdot \Delta x$, again floored as $\sigma = \max(\texttt{eps\_scale} \cdot \Delta x,\, \Delta x)$ in code. **No hard threshold** is applied — the smooth form is JIT-friendly and the numerical tail beyond $\sim 5\sigma$ is indistinguishable from zero. **Not** a sharp variational Dirichlet line — a **smooth ring mask** in the discrete equations.

**Initial conditions for $c$:** the cavity interior is initialised at $c = c_{\mathrm{sat}}$ (no uniform supersaturation at $t=0$); the rim Dirichlet ring then actively enforces $c \approx c_0$ in the narrow annulus each step, supporting a diffusive gradient rim→interior that drives front propagation. Override the interior level via the YAML key `initial.c_init` for special tests (e.g. uniform supersaturation).

**Outside cavity ($\chi \approx 0$):** $c$ is initialised at **zero**. Physically, this represents solid rock with no dissolved silica reservoir. The rim Dirichlet ring models the single silica delivery channel from hydrothermal flow through wall fissures. Setting $c_{\mathrm{outside}} = 0$ instead of $c_0$ eliminates an artifact of the periodic-FFT solver, where a large outside reservoir would leak diffusively into the cavity across periodic boundaries and complicate mass balance accounting (see §10.2).

**Accounting annulus:** a thin shell $R - 2\Delta x \le r < R$ is reserved for rim-adjacent mask bookkeeping in the geometry builders.

**Option B (§10.3) sampling:** $c$ on circles at fixed radius uses **bilinear** interpolation of the cell-centred grid with **periodic wrap** (``bilinear_sample_field``), then an azimuthal mean over **360** uniformly spaced angles. The radial derivative at $r_{\mathrm{fix}}$ uses a **$2\Delta x$** central difference between circles at $r_{\mathrm{fix}} \pm \Delta x$ (same bilinear pipeline) — **not** thin shell bin-averages on the grid, which alias narrow agate bands ($\sim 2\Delta x$) into azimuthal noise.

### 6.2 Stage II — pure Cahn–Hilliard relaxation (separate model)

Stage II is a **second, distinct** continuum model used for **long-time post-formation** scenarios (e.g. sequential Experiment 2: Stage I run A → Stage II run B with extended horizon $T \sim 10^5$ in code configurations).

**Equations:** the same **Cahn–Hilliard structure** for $(\phi_m,\phi_c)$ on the **periodic torus**, but with **$G \equiv 0$** (no reactive transfer between $c$ and solids in this mode) and **no cavity mask / no rim Dirichlet** in the Stage II package: the domain is the **full** periodic square with uniform or loaded initial phases.

**Initial conditions:** either a **mixed bulk** state (e.g. $\phi_m \approx \phi_c \approx 0.5$ with noise, depending on YAML) or **continuation** from a Stage I `final_state` snapshot, depending on experiment configuration.

**Physical interpretation (paper narrative):** **isothermal relaxation** of polymorph texture on **geological** time scales after primary pattern formation — an “aging” or **ordering** stage without ongoing precipitation chemistry in this idealization.

**Key empirical result (Experiment 2):** concentric **Stage I** banding patterns were reported **stable** under a **longer-horizon** Stage II integration. The repository run cards use **`configs/agate_ch/stage_sequence/run_a_stage1.yaml`** with `time.T_total: 10000.0` and **`configs/agate_ch/stage_sequence/run_b_long.yaml`** with `time.T_total: 100000.0` at the same `dt` — i.e. a **10×** longer configured end time for Run B than Run A (not 100×). Cite those paths and your archived `summary.json` / figures when stating the factor in the paper; if you compare against a different Stage I horizon, recompute the ratio explicitly.

---

## 7. Stress field catalog (prescribed $\sigma_{ij}$)

All tensors are precomputed on the grid (typically `float32`). They are **prescribed fields**, not self-consistently solved elasticity.

### 7.1 Uniform uniaxial

$$
\sigma_{xx} = \sigma_0, \qquad \sigma_{yy} = \sigma_{xy} = 0.
$$

**ψ-coupling vs $\kappa$-anisotropy (distinct mechanisms):** diagonal gradient anisotropy adds terms proportional to $\kappa_x (\partial_x \phi_m)^2 + \kappa_x (\partial_x \phi_c)^2 + \kappa_y (\partial_y \phi_m)^2 + \cdots$. That **penalizes both symmetric and antisymmetric** combinations of $(\phi_m,\phi_c)$ gradients at each $x$ or $y$. The ψ-stress energy depends only on

$$
(\partial_i \psi)^2 = \bigl(\partial_i(\phi_m - \phi_c)\bigr)^2,
$$

i.e. on the **antisymmetric (polymorph contrast) mode** alone. Heuristically, for comparable field amplitudes, **only the antisymmetric part** of $(\partial_x \phi_m, \partial_x \phi_c)$ contributes to ψ-stress, whereas **anisotropic $\kappa$** also damps or costs **symmetric** gradients; as a rule of thumb one therefore expects **ψ-coupling to be substantially weaker per unit control parameter** than raw $\kappa_x \neq \kappa_y$ — order **$\sim 6\times$** in effective orientation bias is a useful mental scaling when scanning $\sigma_0$ against $\kappa_x/\kappa_y$ ratios (not a rigorous linear stability coefficient).

**Order-of-magnitude calibration (informal):** **$\sigma_0 = 0.5$** in ψ-coupling can behave similarly in **morphology class** to an **effectively much larger** anisotropic-$\kappa$ control — past internal calibration notes used a **κ-equivalent factor $\sim 3$** (i.e. “$\sigma_0=0.5$” loosely maps to “$\Delta\kappa$ / anisotropy strength $\sim 3$” in loose lab language). This is **not** a universal dimensionless identity; it is a **workflow mnemonic** only.

**Calibration narrative:** in production calibration, **$\sigma_0 = 0.5$** under ψ-split uniaxial coupling produced **horizontal-band “onyx-like”** textures comparable in **morphology class** to strong gradient anisotropy (e.g. `aniso_10x` with $\kappa_x = 5.0$, $\kappa_y = 0.5$ at $T=10000$), yet the two runs are **not** pixel-identical: **Pearson correlation** of $\psi = \phi_m - \phi_c$ on the **cavity mask** $r < R$ between validation uniaxial ($\sigma_0=0.5$, ψ-stress) and the aniso\_10x reference was **$\approx 0.23$** — low because **phase offsets, defect positions, and sub-band texture** differ even when **band orientation and spacing class** match. For the paper: treat them as **qualitatively analogous orientation selectors**, **not** interchangeable reparameterizations of the same physics.

#### 7.1.1 Sign convention for $\sigma_0$ (effective coupling, not Cauchy magnitude)

In this codebase, **`$\sigma_0$` is an effective coupling strength** in the ψ-stress free energy, **not** a literal Cauchy traction amplitude with standard solid-mechanics sign (compression negative, etc.). Confusion during calibration (e.g. large negative $\sigma_0$ triggering checkerboard-like instability, large positive $\sigma_0$ amplifying numerical noise before settling on moderate positive $\sigma_0 \approx 0.5$ for clean onyx) motivated documenting the **morphological** association:

| $\sigma_{xx}$ value | Effective $\kappa_x$ (heuristic) | Preferred bands (qualitative) |
|---------------------|----------------------------------|-------------------------------|
| $> 0$ | $> \kappa_0$ (stiffer along $x$ in ψ-energy sense) | **Perpendicular to $x$** — stripes **along $y$** (“horizontal” bands in usual $x$–$y$ plot frames) |
| $< 0$ | $<\kappa_0$ (softer / inverted along $x$) | **Parallel to $x$** — vertical stripes; often **unstable** or very noisy at large $|\sigma_0|$ in empirical scans |

**Reproducibility note:** when porting numbers into other codes, **do not** map $\sigma_0$ to laboratory stress without an explicit calibration layer — reproduce **our** $\sigma_{xx}$ definition and ψ-split first, then reinterpret.

### 7.2 Uniform biaxial (hydrostatic control in plane)

$$
\sigma_{xx} = \sigma_{yy} = \sigma_0, \qquad \sigma_{xy} = 0.
$$

Deviatoric part vanishes; used as a **symmetric** control on ψ-split coupling.

### 7.3 Pure shear

$$
\sigma_{xy} = \sigma_0, \qquad \sigma_{xx} = \sigma_{yy} = 0.
$$

### 7.4 Flamant two-point squeeze

Two **Flamant line-load** half-space solutions (normal point loads) are placed on the vertical diameter at $(L/2, L/2 \pm R)$, one pushing “down”, one “up”, and superposed. Near-load **regularization** multiplies each singular contribution by

$$
\frac{r^2}{r^2 + \varepsilon^2},
$$

with default $\varepsilon = (\texttt{stress\_eps\_factor})\,\Delta x$ and **factor 3** in configs (`stress_eps_factor: 3`). The pair is then **rescaled globally** so that $\max |\sigma_{xx} - \sigma_{yy}| = \sigma_0$.

### 7.5 Pressure gradient (isotropic linear $y$-pressure)

Define

$$
p(y) = \sigma_0\,\frac{y - L/2}{L/2}, \qquad \sigma_{xx} = \sigma_{yy} = -p(y), \qquad \sigma_{xy} = 0.
$$

So this is an **isotropic** pressure field varying linearly in $y$ (signs as implemented).

### 7.6 Kirsch (classical hole in plate — **simplification**)

An **analytic Kirsch** solution for a circular hole in an infinite plate under remote biaxial/uniaxial loading is implemented in polar coordinates about the cavity centre. **Inside $r < R$** the code **does not** use the singular origin value; it evaluates the tensor at

$$
r_{\mathrm{eff}} = \max(r, R),
$$

i.e. an **effective rim** stress is applied throughout the cavity interior. This is a **deliberate modeling shortcut** for exploratory coupling — **not** a full inclusion problem with Eshelby-type interior fields or fitted remote boundary conditions.

**Planned future work:** **inclusion-theory** stress for inclusions, elliptical holes, and flint-relevant geometries — beyond this closed-form shortcut.

---

## 8. Numerical method

### 8.1 Spatial discretization

**Pseudospectral FFT** on the torus: spatial derivatives use $ik$ factors; the Laplacian symbol is $-k^2 = -(k_x^2 + k_y^2)$.

**Nonlinear terms** ($G$, double-well derivatives, barrier, products with $\sigma$ for stress) are evaluated in **real space**; **linear stiff operators** are treated in Fourier space.

**Dealiasing:** the current codebase **does not** apply an explicit Orszag 2/3-rule (or other) dealiasing filter on quadratic nonlinearities. In practice the combination of **barrier forces**, **hard clip**, cavity masks, and moderate gradients keeps runs stable for the calibrated parameter sets; **high-$k$ fidelity** should be interpreted with this limitation in mind.

### 8.2 Time integration (IMEX)

One IMEX step (schematically):

- **Implicit** treatment of **linear** diffusive / stiff Cahn–Hilliard pieces in Fourier space (including biharmonic-type stiff symbol from $\nabla^2 \mu$ with $\kappa_x k_x^2 + \kappa_y k_y^2$ weighting).
- **Explicit** treatment of nonlinear chemical driving and reaction coupling.

Dissolved silica is updated with implicit Laplacian $D_c \nabla^2 c$ and explicit reaction $-G$. Phase fields use implicit denominators involving $ \Delta t\, M_\alpha\, k^2\, (\kappa_x k_x^2 + \kappa_y k_y^2)$ (isotropic limit reduces to $\kappa k^4$ structure).

### 8.3 Precision

The GPU-oriented integrator uses **`float32`** for the main JAX state and many intermediate terms; some diagnostics and file I/O use **`float64`** (e.g. time series for mass flux). **CPU-only** runs may promote or keep higher precision depending on JAX defaults — for publication runs, **document the JAX platform and dtype policy** alongside figures.

**Default grid:** $512 \times 512$, $L=200$, $\Delta x = 200/512$.

---

## 9. Empirical stability calibration (ψ-split scans)

These are **empirical** stability boundaries from short scans (e.g. `stability_scan_20260424_071500`), not linear stability analysis of the full nonlinear coupled system. They should be quoted as **operational** guidance when choosing $\sigma_0$:

| Mode | Stable up to (approx.) | Unstable at (reported step) |
|------|------------------------|-----------------------------|
| Uniform uniaxial | $\sigma_0 \approx 0.75$ | $1.0$ |
| Pure shear | $\sigma_0 \approx 0.25$ | $0.5$ |
| Uniform biaxial | $\sigma_0 \approx 0.75$ | $1.0$ |

**Phase 3 production calibration** (long runs, validated morphology and mass diagnostics):

| Mode | $\sigma_0$ |
|------|------------|
| Uniform uniaxial | $0.5$ |
| Pure shear | $0.25$ |
| Uniform biaxial | $0.5$ |
| Flamant two-point | $0.25$ |
| Pressure gradient | $0.25$ |

---

## 10. Numerical validation

Three independent mass-balance diagnostics are computed for every production Stage I run and reported in `summary.json`. Additional figure-facing metrics follow in §10.4+. All are **post-processed** from saved fields unless noted.

### 10.1 Spectral kernel conservation (Option D)

**Goal:** isolate **FFT + IMEX** drift on a **simplified** periodic problem: $\chi \equiv 1$, **no** rim Dirichlet, **no** reaction ($G \equiv 0$), and an **off-centre Gaussian** in $c$ only with $\phi_m = \phi_c = 0$.

**Method:** when `output.record_spectral_mass_diagnostic` is **true**, the driver runs a **short** auxiliary trajectory (defaults: `spectral_mass_T = 1.0`, `spectral_mass_dt = 0.01` ⇒ **100** steps; overridable via `output`). The same `imex_step` kernel advances $(\phi_m,\phi_c,c)$ on the full $L\times L$ torus.

**Output:** `spectral_mass_drift` with **`leak_pct`** $= 100\,|M_{\mathrm{final}} - M_{\mathrm{initial}}| / \max(|M_{\mathrm{initial}}|,\varepsilon)$ for $M = \iint (c + \rho_m\phi_m + \rho_c\phi_c)\,\mathrm{d}A$.

**Expected:** $|\texttt{leak\_pct}| \ll 0.1\%$ in **fp64**; **$\sim 10^{-5}$–$10^{-7}\%$** order in **fp32**; v1 reported $\sim 1.7\times 10^{-7}\%$ in fp64.

### 10.2 Full-scheme Dirichlet accounting (Option C)

**Goal:** track per-step Dirichlet injection (**χ-weighted** change in $c$ from the rim blend toward $c_0$) and compare cumulative injection to $\Delta M_{\mathrm{tot}}$ for $M_{\mathrm{tot}} = \iint \chi\,(c + \rho_m\phi_m + \rho_c\phi_c)\,\mathrm{d}A$.

**Construction:** before the rim blend, $c_{\mathrm{pre}}$ is the field after diffusion and cavity masking; after, $c_{\mathrm{post}} = (1-\texttt{ring})\,c_{\mathrm{pre}} + \texttt{ring}\,c_0$. The injected scalar that step is $\Delta m_{\mathrm{inject}} = \sum \chi\,(c_{\mathrm{post}} - c_{\mathrm{pre}})\,\Delta x^2$.

**Output:** `dirichlet_mass_balance` (`residual_pct`, `ratio` $=\Delta M/\sum\Delta m_{\mathrm{inject}}$, …).

**Caveat:** smooth $\chi$ projection and the hard $\phi$ clip $[-0.05,\,1.05]$ introduce **bounded** residuals that are **structural** to the discrete scheme. On live agate runs, `residual_pct` can sit in a **wide advisory band** (often tens of percent) while morphology remains acceptable — treat Option C as **diagnostic**, not a strict pass/fail gate, unless you establish a baseline on a controlled configuration. **Stage II** uses `dirichlet_active=False`, so injection is **zero** and $\Delta M_{\mathrm{tot}}$ should be $\approx 0$ up to fp drift.

### 10.3 Surface flux balance (Option B — v1 paper §3.2 method)

**Goal:** at $r_{\mathrm{fix}} = \texttt{option\_b\_r\_fix\_frac}\cdot R$ (default **0.75**), compare $\int F\,\mathrm{d}t$ to $\Delta M_{\mathrm{dissolved}}$ over samples with $t \le t_{\mathrm{front}}$, where $t_{\mathrm{front}}$ is the first sample with azimuthal mean $(\phi_m + \phi_c) > 0.3$ on the $r_{\mathrm{fix}}$ circle (v1 default **0.3**).

**Sampling (critical):** **bilinear** interpolation of $c$ on the cell-centred periodic grid at **360** angles on each circle — **not** bin-masking grid shells (which alias narrow $\sim 2\Delta x$ bands).

**Gradient stencil:** **$2\Delta x$** central difference between azimuthal means at $r_{\mathrm{fix}} \pm \Delta x$; **flux rate** $F(t) = D_c \cdot \partial c/\partial r \cdot 2\pi r_{\mathrm{fix}}$ (v1 inflow sign).

**Dissolved mass:** hard disk $r < r_{\mathrm{fix}}$, **no** $\chi$: $M_{\mathrm{dissolved}} = \iint_{r < r_{\mathrm{fix}}} c\,\mathrm{d}A$.

**Residual:** **signed** `leak_pct` $= 100 \cdot (\Delta M_{\mathrm{dissolved}} - \int F\,\mathrm{d}t) / \max(|\text{initial}|,\,|\text{final}|,\,|\int F\,\mathrm{d}t|,\,10^{-30})$.

**Output:** `surface_flux_balance` (`leak_pct`, `flux_integrated`, `dissolved_change`, `residual`, `n_samples`, `front_reached`, `front_arrival_t`, `r_fix`, …).

**Expected:** $|\texttt{leak\_pct}| < 1\%$ typical, $< 5\%$ worst-case; v1 §3.2 worst-case **0.58%** over 13 configurations.

### 10.4 χ-weighted cavity silica windows (`main_silica_window_drifts`)

**Goal:** track **transient vs late** mass drift of **χ-weighted** cavity silica total $\int (c + \rho_m\phi_m + \rho_c\phi_c)\,\chi\,\mathrm{d}A$ over three **100-step** windows: early, mid-run, and final.

**Interpretation:** complements **§10.1–10.3** mass headlines by summarizing **bulk silica inside the soft cavity** over the **main** run, including solid phases.

### 10.5 FFT ψ-anisotropy ratio

**Goal:** classify **horizontal vs vertical** dominance of $\psi = \phi_m - \phi_c$ in the cavity.

**Method:** compute $|FFT(\psi \cdot \chi_{\mathrm{disk}})|^2$ with a **hard** disk mask $r < R$, then form the ratio of low-$|k_x|$ band power to low-$|k_y|$ band power (threshold tied to domain size). Values $>1$ favour **horizontal** structure, $<1$ **vertical**, $\approx 1$ **isotropic** in this coarse metric.

### 10.6 Pixel noise metric (stability scans)

**Goal:** scalar **texture noise** on $\phi_m$ inside $r < R$: RMS of $\phi_m - \mathrm{uniform\_filter}_{3\times 3}(\phi_m)$ with **periodic** wrapping on the grid.

**Interpretation:** used together with $\max(\phi_m + \phi_c)$ in empirical **STABLE / MARGINAL / UNSTABLE** classification for ψ-coupling calibration sweeps.

### 10.7 Canonical-slice Jabłczyński metrics

**Geometry of the measurement:** **horizontal** centreline $y = L/2$, **right half** of the cavity ($x \ge L/2$), $\theta \approx 0$ ray in the radial sense used for **peak finding** in $\phi_c$ (fallback to $\phi_m$ if too few peaks).

**Outputs:** band count $N_b$, outer-in peak positions, spacing list $\{d_n\}$, ratios $q_n = d_n/d_{n-1}$, CV$(q)$, Spearman correlation of $\{d_n\}$ vs band index, and a discrete **classification** label (e.g. ratchet-banded vs insufficient bands — see code thresholds).

**Critical caveat:** this pipeline is **horizontal-slice-centric**. **Vertical** banding, **radial** Liesegang structure, or **labyrinth** dominance may yield **misleading $N_b$ or class labels** relative to a human eye on the 2D pattern — a **diagnostic limitation**, not a claim about what the PDE forbids.

### 10.8 Multislice band count and persistence

**Multislice count:** median over **8 angles** and both $(\phi_m,\phi_c)$ of peak counts along **right-half rays** through the cavity (see `count_bands_multislice` in code) — more isotropic than the single canonical slice, but still **ray-based**.

**Persistence check:** compares **peak** multislice count time vs **final** multislice count to flag **band loss** vs **persistent** banding in post-processing.

### 10.9 Additional figure-facing metrics

Examples: **moganite–chalcedony anticorrelation** on the horizontal centreline; **overshoot fraction** above physical packing; **radial profiles** of $\phi_m + \phi_c$; **kymograph** $(t,r_{\mathrm{peak}})$ from horizontal-line peaks. These support morphology narrative but are secondary to Sections 10.4–10.8 for mass and stability claims.

**Labyrinth heuristic (morphology tag):** a run may be flagged as **labyrinth-like** when the **median multislice** band count (Section 10.8) is **below a fixed threshold** (e.g. $< 10$ in the legacy implementation) **or** when **azimuthal variance** of the pattern on an intermediate-radius ring **dominates** the variance of the **azimuthally averaged** radial profile — i.e. tangential structure wins over purely radial banding in that diagnostic. This is a **post-process classifier**, not a separate PDE term.

### 10.10 Known model / implementation limitations (summary)

- **Barrier + hard clip** (Section 4): not a strict variational $\phi \in [0,1]$ surface-phase model.
- **Kirsch mode** (Section 7.6): toy σ-field, not inclusion theory in the cavity.
- **Canonical Jabłczyński** (Section 10.7): horizontal bias — do not over-interpret vertical textures through this scalar alone.
- **No elasticity / fluid pressure:** the model omits elastic strain energy and explicit fluid pressure beyond what is encoded in prescribed $\sigma_{ij}$ and $c$; there are **no** explicit sharp grain-boundary interfaces beyond diffuse interfaces.
- **Ratchet** (Section 3): a **phenomenological** kinetic tilt of Ostwald partitioning in a band of $\phi_m$ — **not** a calibrated multi-mineral rate law.
- **Dimensionless formulation:** all lengths, times, diffusivities, and reaction constants are **dimensionless** unless an explicit map to SI units is introduced elsewhere.

---

## 11. Summary

We solve a **reaction-coupled anisotropic Model C** system for $(c,\phi_m,\phi_c)$ in a periodic square with a **smooth circular cavity**, **rim Dirichlet** control of $c$, **double-well + outer barrier** for $\phi_\alpha$, optional **ratcheted Ostwald partitioning**, and optional **ψ-split stress coupling** $\mathcal{F}_{\mathrm{stress}} \propto (\nabla\psi)^\top \sigma (\nabla\psi)$ implemented as $\mu_{\mathrm{stress}} = -B\nabla\cdot(\sigma\nabla\psi)$ with the $(\pm\tfrac12)$ split between $(\phi_m,\phi_c)$. A separate **Stage II** bulk CH model removes reaction and cavity physics for long-time relaxation studies. **Pseudospectral IMEX** time stepping is used; **no explicit spectral dealiasing** is currently applied. **Empirical σ-limits**, **Phase 3 calibration**, **mass diagnostics (§10.1–10.3)**, and **ψ vs κ-anisotropy** distinctions should accompany any figure caption comparing mechanisms.

---

*Document version: post-NOTES merge — includes Option D/C/B mass validation (§10.1–10.3; Option B = v1 bilinear circle + $2\Delta x$ stencil), optional `map_coordinates` elsewhere, labyrinth heuristic, and dimensionless-scope notes ported from legacy reviewer notes.*
