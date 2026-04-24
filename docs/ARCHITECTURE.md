# Architecture — continuous-patterns (rewrite target)

This document defines the **target** layout and contracts for a clean-slate, publication-grade package. It complements **`docs/PHYSICS.md`** (equations and methods).

After cleanup, the **`continuous_patterns.agate_ch`** / **`agate_stage2`** packages and their CLIs **do not exist**; all simulation entry is through **`continuous_patterns.experiments`**. Historical **`results/agate_ch/...`** trees may remain on disk as **read-only archives**; **new** runs use the **single** `results/` schema below (no `results/v2/` parallel).

**Design rules**

- **No framework:** no plugin registries, no ABCs with one implementation. Use **plain functions** and **`dict` dispatch** (`MASK_BUILDERS`, `STRESS_BUILDERS`).
- **Acyclic imports:** `core` must not import `models` or `experiments`. `models` may import `core` only. `experiments` may import `models` and `core`. **`core.plotting`** must not import `io`, `models`, or `experiments`.
- **Size discipline (targets):** soft cap **400** lines / module, hard **600**; soft cap **80** lines / public function, hard **150**. Split modules when approaching caps.
- **Style:** **Ruff** lint + **Ruff format** (project standard); import order via Ruff **isort** rules. **No mypy** gate in CI unless added later. **Type hints** on all **public** function and method signatures; internals may stay untyped if clarity suffers.
- **Documentation:** every package/module has a module docstring; public symbols use **NumPy-style** docstrings (`Parameters`, `Returns`, `Raises` as needed).

---

## 1. Repository layout (target)

```text
src/continuous_patterns/
  core/
    spectral.py              # FFT symbols, laplacian, gradients (pseudospectral)
    masks.py                 # geometry: χ, ring, ring_accounting + metadata; MASK_BUILDERS
    stress.py                # σ field builders + ψ-split μ_stress; STRESS_BUILDERS
    imex.py                  # single IMEX step for Stage I and II (flagged; see §3.4)
    diagnostics_stage1.py    # cavity / rim / Option B / Jabłczyński / … (NumPy)
    diagnostics_stage2.py    # bulk: S(k,t), domain stats, coarsening, … (NumPy)
    plotting.py              # fields → PNG; NumPy in, no io/models/experiments imports
    io.py                    # nested YAML only → validate → paths + summary writers

  models/
    agate_ch.py              # Stage I: build Geometry + SimParams; integrate
    agate_stage2.py          # Stage II: bulk setup; same imex; different diagnostics

  experiments/
    run.py                   # canonical CLI: YAML → model → results tree
    sweep.py                 # sweep YAML → combinations → run API
    templates/               # nested canonical YAML only

tests/
  unit/
  integration/
  regression/

docs/
  PHYSICS.md
  ARCHITECTURE.md
  CONTRIBUTING.md            # may lag until Phase 5

examples/
  reproduce_canonical.py

README.md
pyproject.toml
```

**Canonical entry point:** `python -m continuous_patterns.experiments.run --config <path/to/nested.yaml> [--out-dir ...]`. There is **no** `python -m continuous_patterns.agate_ch.run` in the new tree.

---

## 2. Data contracts

### 2.1 `Geometry` (dataclass)

Immutable **per-cell** masks, spectral symbols, and prescribed $\sigma$ on the $n\times n$ grid. **Both** stages use the **same** `Geometry` type so `core.imex` has one signature.

| Field | Type | Meaning |
|-------|------|---------|
| `chi` | `jax.Array` | Cavity / domain indicator. Stage I: smoothed $\chi$. Stage II: **ones** $(n,n)$ (entire torus active for phase fields as configured). |
| `ring` | `jax.Array` | Rim mask for Dirichlet on $c$. Stage II: **zeros**. |
| `ring_accounting` | `jax.Array` | Annulus for rim flux bookkeeping. Stage II: **zeros** or unused mask. |
| `sigma_*` | `jax.Array` | Prescribed stress; Stage II typically **zero** unless an experiment explicitly adds bulk stress. |
| `k_sq`, `kx_sq`, `ky_sq`, `kx_wave`, `ky_wave`, `k_four` | `jax.Array` | From `core.spectral`. |
| `rv` | `jax.Array` | Stage I: radius from centre. Stage II: may be dummy or omitted in builders; diagnostics that need a cavity must **not** run for Stage II. |
| `dx`, `L`, `R`, `n`, `xc`, `yc` | scalars | Metadata; `R` may be unused for pure bulk. |

**Construction:** `models.agate_ch.build_geometry` vs `models.agate_stage2.build_geometry` (bulk builder). **Numerical time stepping** does not switch modules — only **`SimParams` flags** (§2.2) and **diagnostics modules** differ.

### 2.2 `SimParams` / physics knob bundle

Frozen dataclass or `NamedTuple` mirroring **`docs/PHYSICS.md`**, plus **stage routing for one IMEX implementation**:

| Flag | Type | Meaning |
|------|------|---------|
| `reaction_active` | `bool` | If `True`, precipitation $G$ and Ostwald/ratchet sources are applied (Stage I). If `False`, $G \equiv 0$ and phase sources from reaction vanish (**Stage II**). |
| `dirichlet_active` | `bool` | If `True`, rim Dirichlet handling for $c$ as in PHYSICS. If `False`, no rim overwrite of $c$ (**Stage II**). |

Stage I: `reaction_active=True`, `dirichlet_active=True` (subject to YAML overrides such as explicit reaction off). Stage II: **`reaction_active=False`**, **`dirichlet_active=False`**.

All remaining knobs ($W$, $\gamma$, $\kappa_x$, $\kappa_y$, barrier, mobilities, `stress_coupling_B`, ratchet parameters, `c0_alpha`, `apply_cavity_mask`, etc.) are **shared fields**; IMEX branches read the booleans **first** so Stage II never executes rim-only code paths.

**Single compute path:** **`core.imex.imex_step(state, geom, prm, dt)`** — no `imex_step_stage2`. Optional JIT-friendly `lax.cond` on `prm.reaction_active` / `prm.dirichlet_active` as today’s code does for stress.

### 2.3 Diagnostics split (conceptual)

- **`core.diagnostics_stage1`:** Option B (rim flux vs dissolved disk), χ-weighted silica windows, Jabłczyński / canonical slice / multislice **inside cavity**, FFT ψ-anisotropy with disk mask, stability pixel noise, etc. **Physically meaningful only when** `dirichlet_active` / cavity semantics apply.
- **`core.diagnostics_stage2`:** **Different** observables: e.g. **structure factor** $S(\mathbf{k},t)$ from $\phi_m,\phi_c$, domain-mean order parameters, **coarsening** metrics / exponents, correlation lengths — no Option B “leak %” as a primary headline (avoid “Option B = 0 because there is no rim to integrate” confusion). Stage II `summary.json` is populated **only** from Stage II helpers.

`experiments.run` (or `models.*`) dispatches post-process to **`diagnostics_stage1`** vs **`diagnostics_stage2`** based on `experiment.model` (or equivalent YAML field).

### 2.4 `SimState`

```python
@dataclass(frozen=True)
class SimState:
    """Instantaneous fields on the grid."""

    phi_m: jax.Array  # (n, n)
    phi_c: jax.Array  # (n, n)
    c: jax.Array      # (n, n)
    t: float          # physical time
```

### 2.5 `SimResult`

```python
@dataclass
class SimResult:
    """Provenance + final fields + diagnostics handles."""

    state_final: SimState
    meta: dict[str, Any]             # step-loop bookkeeping (flux samples, mass series, …)
    diagnostics: dict[str, Any]     # summary.json payload (stage-appropriate keys only)
    config_resolved: dict[str, Any]  # nested as-run dict (canonical copy for config.yaml)
    paths: ResultPaths
```

`ResultPaths`: `root`, `summary_json`, `config_yaml`, `final_state_npz`, optional `h5_path`, `fields_png`.

### 2.6 Config YAML (**nested only — canonical**)

The **only** format accepted by **`core.io`** for new code is **nested** YAML matching §2.6 schema (same structure as former §2.5: `experiment`, `geometry`, `physics`, `stress`, `time`, `output`). **Flat** legacy configs are **not** parsed by production `io.py`.

**One-time migration (Phase 4):** archival flat YAMLs are converted to nested templates using a **throwaway script** (e.g. `scripts/flatten_to_nested_once.py` or a notebook) — **not** part of the stable `core` API. After conversion, templates live under **`experiments/templates/`**.

**Validation:** **`pydantic` v2** models in `core.io` (or a dedicated `core/config_schema.py`) — nested YAML loads into typed settings; use **`Literal`** / unions for `stress.mode` and `geometry.type` dispatch. Fail fast on unknown keys / missing required sections per `experiment.model`.

```yaml
experiment:
  name: str
  model: agate_ch | agate_stage2
  seed: int

geometry:
  type: circular_cavity   # stage2 bulk may use periodic_bulk or circular with full chi
  L: float
  R: float
  n: int

physics:
  # …

stress:
  mode: none | ...
  sigma_0: float
  stress_coupling_B: float

time:
  dt: float
  T: float
  snapshot_every: int

output:
  save_final_state: bool
  record_spectral_mass_diagnostic: bool
  flux_sample_dt: float | null
```

### 2.7 Reproducibility (seeds and numerics)

- **RNG:** `experiment.seed` must flow into **`jax.random.PRNGKey(seed)`** (and split keys for IC noise). Document the **splitting strategy** in `models` docstrings so auxiliary draws remain ordered.
- **Determinism:** same seed + same platform + same XLA flags should reproduce **bitwise** or near-bitwise results where JAX guarantees it. **GPU** reductions can be **nondeterministic** at float32 unless `jax_threefry_partitionable` / deterministic flags are set — document platform and any `JAX_*` env vars used for paper runs.
- **`jax_enable_x64`:** **off** for the main production trajectory (default `float32` state). **On** only inside the **Option D / spectral mass** auxiliary diagnostic path (short blob run), and **on demand** for selected **regression** tests that compare tiny drifts. Do not enable x64 globally in the hot loop without an explicit reason (performance and GPU behavior).

### 2.8 Type discipline — Pydantic at boundaries, dataclass inside

Two separate concerns, two separate tools:

**External input validation (Pydantic v2)** — used in `core/io.py` only. Purpose: validate user-supplied YAML configs, produce clear error messages for malformed input, reject flat legacy formats. This is the **only** place in the codebase where Pydantic models appear.

- `ExperimentSpec`, `RunConfigValidated` in `core/io.py`
- Permissive by default (`extra="allow"` on nested dicts) to avoid blocking new experiment types during development
- Will be tightened in Phase 3l: `Literal` types for `experiment.model` and `stress.mode`, required field validation per model, field constraints (e.g. `dt > 0`)

**Runtime types (frozen dataclasses)** — used everywhere else:

- `Geometry`, `SimParams` in `core/imex.py`
- `SimState`, `SimResult` in `core/types.py`
- `ResultPaths` in `core/io.py`

Reasons runtime types are **not** Pydantic:

1. **JAX pytree compatibility:** dataclasses register as JAX pytrees via `jax.tree_util` helpers with minimal effort; Pydantic models require custom pytree glue that is fragile across Pydantic versions.
2. **Performance:** `SimState` is constructed inside `jax.lax.scan` / `fori_loop` hot paths. Pydantic validation on each step adds measurable overhead (milliseconds per step × many steps).
3. **Trust boundary:** runtime types are built by our own functions (`build_geometry`, `build_sim_params`) from already-validated config. Re-validating inside is redundant.
4. **Type hint clarity:** `phi_m: jax.Array` in a dataclass is a documentation hint, not a runtime check — which matches how JAX arrays are actually used.

**Rule for new code:** if the object receives data from outside Python (YAML, JSON, CLI args, HTTP), validate with Pydantic at the boundary. If it is constructed internally from already-validated sources, use dataclass. Do not mix.

---

## 3. Module reference (`core/`)

### 3.1 `core.spectral`

**Depends on:** `jax`, `jax.numpy` only.

**Public API (illustrative):**

```python
def k_vectors(*, L: float, n: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return (k_sq, kx_sq, ky_sq, kx_wave, ky_wave) for cell-centred periodic grid."""

def laplacian_real(u: jax.Array, k_sq: jax.Array) -> jax.Array:
    """∇² u via FFT, real output."""

def grad_real(u: jax.Array, kx_wave: jax.Array, ky_wave: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(∂_x u, ∂_y u) pseudospectral, real output."""
```

---

### 3.2 `core.masks`

**Depends on:** `jax.numpy` only (no `spectral` import).

**Public API:** `circular_cavity_masks(...) -> dict[str, jax.Array | float]`; `MASK_BUILDERS: dict[str, Callable[...]]`.

---

### 3.3 `core.stress`

**Depends on:** `jax.numpy`.

**Public API:** `stress_mu_hat`, `stress_contribution_to_mu`, per-mode builders, `STRESS_BUILDERS`.

---

### 3.4 `core.imex`

**Depends on:** `core.spectral`, `core.stress`.

**Responsibility:** exactly **`imex_step(state, geom, prm, dt)`**. Implementation uses **`prm.reaction_active`** and **`prm.dirichlet_active`** (and existing stress / clip logic) so **Stage I and Stage II share one kernel** — no second `imex_step_stage2` file.

```python
def imex_step(
    state: tuple[jax.Array, jax.Array, jax.Array],
    geom: Geometry,
    prm: SimParams,
    dt: float,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Single step; rim accounting delta_pair may be zeros when dirichlet_active is False."""
```

---

### 3.5 `core.diagnostics_stage1`

**Depends on:** `numpy`, `scipy` optional, `h5py` optional; **no JAX**.

**Responsibility:** cavity-centric post-processing described in **`docs/PHYSICS.md` §10** (Option B, χ-window drifts, Jabłczyński canonical slice, multislice counts, FFT ψ-anisotropy on $r<R$, pixel noise for stability scans, etc.).

```python
def option_b_leak_pct_from_meta(meta: dict[str, Any], cfg: dict[str, Any]) -> float: ...
def jab_metrics_canonical_slice(phi_m: np.ndarray, phi_c: np.ndarray, L: float, R: float, cavity_R: float) -> dict[str, Any]: ...
# …
```

---

### 3.6 `core.diagnostics_stage2`

**Depends on:** `numpy`, `scipy` FFT for $S(k)$; **no JAX** for post-run.

**Responsibility:** **bulk** observables — structure factor pipelines, spatial means / variances, coarsening-length fits, two-point correlations — **not** rim flux or cavity-only Jabłczyński defaults. `summary.json` keys must be **disjoint** in meaning from Stage I headline metrics (no fake Option B as primary).

---

### 3.7 `core.plotting`

**Depends on:** `numpy`, `matplotlib` only.

**Responsibility:** **`plot_fields_final(phi_m, phi_c, c, *, L, R, path, ...)`** and other figure helpers taking **NumPy** arrays. **Must not** import `core.io`, `models`, or `experiments`.

**Callers:** `core.io` and/or `experiments.run` after device-get of final fields.

---

### 3.8 `core.io`

**Depends on:** `pathlib`, **PyYAML**, `json`, `numpy`; optional `h5py`; may import **`core.plotting`** for default PNG output.

**Responsibility:**

- **`load_run_config(path: Path) -> dict[str, Any]`** — **nested YAML only**; validate; reject flat files at the door with a clear error pointing to the migration script.
- **Internal:** build `Geometry`, `SimParams` (including `reaction_active` / `dirichlet_active` from `experiment.model` + `physics` blocks) — implementation may use a private normalized dict, but **on-disk** `config.yaml` is the **nested** canonical copy.
- **`allocate_run_dir`**, **`save_summary`**, **`save_final_state_npz`**, optional H5 snapshot writer.

**No** `flatten_run_config` in the public API for ingesting alternate legacy shapes.

```python
@dataclass(frozen=True)
class ResultPaths:
    root: Path
    summary_json: Path
    config_yaml: Path
    final_state_npz: Path

def load_run_config(path: Path) -> dict[str, Any]: ...
def allocate_run_dir(*, experiment_name: str, results_root: Path) -> ResultPaths: ...
def save_summary(path: Path, payload: dict[str, Any]) -> None: ...
```

---

## 4. Module reference (`models/`)

### 4.1 `models.agate_ch`

**Depends on:** `core` only.

Builds cavity `Geometry`, sets `SimParams(reaction_active=True, dirichlet_active=True)` unless YAML disables reaction/Dirichlet explicitly, calls **`core.imex`** integration loop, then **`core.diagnostics_stage1`** for `summary.json` / figures.

```python
def build_geometry(cfg: dict[str, Any]) -> Geometry: ...
def build_sim_params(cfg: dict[str, Any]) -> SimParams: ...
def build_initial_state(cfg: dict[str, Any], geom: Geometry, prm: SimParams, key: jax.Array) -> SimState: ...
def simulate(cfg: dict[str, Any], *, chunk_size: int = 2000) -> SimResult: ...
```

### 4.2 `models.agate_stage2`

**Depends on:** `core` only.

Builds bulk `Geometry` (`chi \equiv 1`, zero `ring`), sets **`SimParams(reaction_active=False, dirichlet_active=False)`** (and `enable_reaction`-like knobs consistent with PHYSICS), uses the **same** **`core.imex.imex_step`**, then **`core.diagnostics_stage2`** for summaries.

```python
def simulate(cfg: dict[str, Any], *, chunk_size: int = 2000) -> SimResult: ...
```

---

## 5. Results directory schema (target)

**New** runs only (archived `results/agate_ch/...` from old code stay untouched):

```text
results/
  <experiment_name>/
    <timestamp>/
      config.yaml          # nested, as-run (canonical copy)
      summary.json         # stage-appropriate diagnostics only
      final_state.npz
      snapshots.h5         # optional; **HDF5 only** for time series (no per-step NPZ shard format in this refactor)
      fields_final.png
      log.txt              # optional

  sweeps/
    <sweep_name>_<timestamp>/
      manifest.json
      <run_id>/
      report.md
```

**Searchability:** `manifest.json` lists `run_id`, relative path, nested parameter subset, optional git hash.

---

## 6. `experiments/` runners

### 6.1 `experiments.run` (**canonical**)

**CLI:** `python -m continuous_patterns.experiments.run --config path/to/nested.yaml [--out-dir ...]`.

**Flow:** `core.io.load_run_config` → validate → `allocate_run_dir` → dispatch `models.agate_ch.simulate` vs `models.agate_stage2.simulate` → stage-specific diagnostics → write `config.yaml` / `summary.json` / NPZ / PNG.

**Depends on:** `models`, `core.io`, `core.diagnostics_stage1` or `core.diagnostics_stage2`, `core.plotting`.

### 6.2 `experiments.sweep`

Merges sweep YAML into nested per-run dicts; calls **`experiments.run`** programmatically (no `subprocess`).

---

## 7. Tests

| Tier | Scope |
|------|--------|
| `tests/unit/` | `spectral`, `masks`, `stress`, `imex` flags (Stage II branch), `io` nested-only rejects |
| `tests/integration/` | Short-$T$ **CPU** smoke per model; **`jax_enable_x64=False`** (production default); assert **correct diagnostics module** keys in `summary.json` |
| `tests/regression/` | Optional refs; **`jax_enable_x64=True` only where a test explicitly needs it** (§2.7) |

**CI policy (resolved):** GitLab / local CI runs **CPU-only** integration smoke; **no** GPU runner configuration. Long GPU reproduction is **user-managed** (local or HPC), not a CI gate.

---

## 8. `examples/reproduce_canonical.py`

Calls **`experiments.run`** with eight nested template paths (post Phase 4 conversion).

---

## 9. Extension protocols (summary)

| Goal | Files to touch | Typical LOC |
|------|----------------|-------------|
| New **stress mode** | `core/stress.py`, template YAML | < 50 |
| New **geometry** | `core/masks.py`, template YAML | < 50 |
| New **Stage I metric** | `core/diagnostics_stage1.py`, `summary` schema in `io` | varies |
| New **Stage II metric** | `core/diagnostics_stage2.py` | varies |
| New **figure** | `core/plotting.py` | varies |
| New **model** | `models/…`, `experiments/run` dispatch | — |

---

## 10. Migration (clean slate — no shim)

1. **Delete** old packages **`agate_ch/`**, **`agate_stage2/`**, and their **`python -m …run`** entry points as part of **Phase 2 cleanup** (see **`CLEANUP_PLAN.md`**). **No** compatibility shim.
2. **Configs:** convert archived flat YAML → **`experiments/templates/*.yaml`** nested form **once** (throwaway converter); **`core.io`** accepts **nested only**.
3. **Results:** do **not** move or delete historical directories; **new** runs write only the **§5** layout under `results/`.
4. **README** documents **`experiments.run`** as the sole simulation CLI.

---

## 11. Resolved implementation choices (formerly open)

### §11.1 — YAML validation: **Pydantic v2**

Nested run configs are validated with **`pydantic` v2** (typed models, discriminated unions / `Literal` for `stress.mode` and `geometry.type`, clear validation errors). This is **required** for `core.io` **only** at the load/save boundary; see **§2.8** for why `Geometry` / `SimParams` / `SimState` remain dataclasses. Hand-rolled ad hoc dict checks are avoided for **I/O** load paths.

### §11.2 — Snapshot format: **HDF5 only**

Trajectory storage remains **`snapshots.h5`** (existing chunk/group convention or a documented successor within HDF5). **No** introduction of per-step NPZ shards in this refactor; changing snapshot format is **out of scope**.

### §11.3 — CI scope: **CPU-only smoke**

Integration tests run on **CPU** with **`jax_enable_x64=False`** to match the default production integrator policy. **GPU** regression and long-$T$ figure regeneration are **manual / user-managed**, not CI-managed.

---

*Document version: Phase 1 + §11 closure — aligned with `CLEANUP_PLAN.md` Phase 2.*
