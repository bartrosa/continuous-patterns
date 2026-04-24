# Contributing to continuous-patterns

## Coding standards

- **Ruff** for format and lint (`pyproject.toml`); import order follows Ruff isort rules.
- **Type hints** on all public functions and methods.
- **NumPy-style docstrings** where non-trivial: `Parameters`, `Returns`, `Raises` as needed.
- **Size targets** (ARCHITECTURE): soft cap ~400 lines per module / ~80 per public function; hard caps ~600 / 150. Split modules when approaching limits.

## Type discipline

See [ARCHITECTURE.md §2.8](ARCHITECTURE.md). In short:

- **Pydantic v2** only in `core/io.py` at the YAML boundary (`load_run_config` / `save_run_config`).
- **Dataclasses** for runtime objects (`Geometry`, `SimParams`, `SimState`, …) so JAX pytrees stay simple and hot loops avoid per-step validation.

## Extending the code

### New stress mode

1. Add `def my_stress(*, L: float, n: int, sigma_0: float, ...) -> tuple[Array, Array, Array]:` in `core/stress.py` (same return layout as `none`).
2. `STRESS_BUILDERS["my_stress"] = my_stress`.
3. Append `"my_stress"` to `StressSpec.mode`’s `Literal[...]` in `core/io.py`.
4. Add or copy a template YAML with `stress: { mode: my_stress, ... }`.
5. Add a small unit test under `tests/unit/` if the mode has non-trivial kwargs.

### New geometry

1. Add `def my_masks(*, L, R, n, ...) -> dict[str, Array | float | int]:` in `core/masks.py` (same keys as `circular_cavity_masks`).
2. `MASK_BUILDERS["my_geometry"] = my_masks`.
3. Extend `GeometrySpec.type` in `core/io.py`.
4. Template YAML under `experiments/templates/`.

### New diagnostic

- **Stage I (cavity / rim):** `core/diagnostics_stage1.py` — NumPy (and SciPy) only, no JAX.
- **Stage II (bulk):** `core/diagnostics_stage2.py` — same rule.
- Prefer keys that do not collide with the other stage’s primary `summary.json` headlines (ARCHITECTURE §2.3).

### New model driver

1. New module under `src/continuous_patterns/models/`, thin composition of `core/` + `imex_step` + the right diagnostics module.
2. Register `simulate` in `experiments/run.py` `MODEL_DISPATCH`.
3. Add `experiment.model` to `ExperimentSpec.model` `Literal` in `core/io.py`.
4. One integration smoke test (`tests/integration/`) with small `n` and short `T`.

## Testing

- **Unit:** `tests/unit/` — fast, CPU, isolated modules (target a few seconds total for the default suite).
- **Integration:** short end-to-end runs per model driver and CLI paths.
- **Regression:** reserved for Phase 4 (bitwise or metric comparisons to archived results); not CI-gated yet.

**Progress / logs in tests:** pass **`show_progress=False`** to **`simulate`** / **`run_one`** so tqdm stays off. **`run_one`** with **`write_artifacts=True`** writes **`run.log`** under the run root (`ResultPaths.log_file`) — assert on file contents or attach a **`logging`** handler to the **`continuous_patterns`** logger if you need in-memory capture (the package logger uses **`propagate=False`** once configured by **`run_one`**).

## Known permissive areas (future tightening)

- **`RunConfigValidated.physics`:** still a `dict[str, Any]`. Model-specific knobs vary; a full `PhysicsSpec` per `experiment.model` is deferred until validation pain justifies it.
- **`RunConfigValidated.initial`:** optional IC block, interpreted by each `models.*.build_initial_state`.

## Known design trade-offs

- **Cavity CH update** `(1−χ)φ + χ·φ_new` is not a mass-conserving projection on the full torus; cavity-weighted silica integrals are the intended accounting (PHYSICS §4, ARCHITECTURE §3.4).
- **Hard clip** on `φ` is a numerical guardrail; rim flux diagnostics (Option B) monitor consistency with diffusion, not variational exactness of the clip.
- **IMEX split:** biharmonic part implicit, nonlinear + reaction + stress explicit; stability is empirical for the calibrated σ ranges (PHYSICS §9).
