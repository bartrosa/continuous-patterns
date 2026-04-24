# continuous-patterns

Pseudospectral Cahn–Hilliard simulation of agate and related pattern formation. The code implements **Model C** (coupled phase fields with reaction–diffusion), optional **ψ-split stress** coupling, and cavity geometry via smooth masks.

## Status

Research codebase, pre-publication. Equations and diagnostics are specified in [docs/PHYSICS.md](docs/PHYSICS.md). Package layout and contracts are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

### Install

Python **3.12** (see `pyproject.toml`). With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Optional GPU (CUDA 12 JAX wheels):

```bash
uv sync --extra cuda
```

Developer tools (pytest, ruff, pre-commit):

```bash
uv sync --extra dev
```

### Run one experiment

```bash
uv run python -m continuous_patterns.experiments.run \
  --config src/continuous_patterns/experiments/templates/agate_ch_baseline.yaml \
  --out-dir results \
  --no-write
```

Omit `--no-write` to write `config.yaml`, `summary.json`, `final_state.npz`, `figures_final.png`, and **`run.log`** (Python logging at DEBUG) under `results/<experiment_name>/<timestamp>/`.

Optional flags:

- **`--log-level`** — console only: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default `INFO`). The file log stays DEBUG when artifacts are written.
- **`--no-progress`** — disable **tqdm** chunk progress (models still print nothing per step; progress is per JIT chunk).

### Run a parameter sweep

```bash
uv run python -m continuous_patterns.experiments.sweep \
  --sweep src/continuous_patterns/experiments/templates/sweeps/gamma_scan.yaml \
  --out-dir results
```

Each sweep creates `results/sweeps/<name>_<timestamp>/` with a `manifest.json`, `report.md`, and one subdirectory per grid point. The same **`--log-level`** and **`--no-progress`** flags apply (outer tqdm over combinations plus per-run bars unless disabled).

### Programmatic baseline smoke

[`examples/reproduce_canonical.py`](examples/reproduce_canonical.py) loads shipped templates and calls `run_one` for all eight Phase 4 canonical baselines (production `n`/`T` by default). Set `CP_REPRODUCE_MINI=1` for a short local smoke.

```bash
uv run python examples/reproduce_canonical.py
```

Quick smoke (small ``n`` / short ``T``): `CP_REPRODUCE_MINI=1 uv run python examples/reproduce_canonical.py`.

Environment variables for that script: **`CP_LOG_LEVEL`** (default `INFO`), **`CP_NO_PROGRESS=1`** to turn off tqdm, same semantics as the CLI flags above.

## Results directory layout

New runs follow **ARCHITECTURE §5**: under the chosen results root, single runs use `results_root / experiment_name / UTC_timestamp /`, sweeps use `results_root / sweeps / sweep_name_timestamp /`. Large trees should stay **gitignored**; templates and docs live in the repo.

## Extending

### Add a new stress mode

1. Implement a builder `my_mode(*, L, n, sigma_0, **kwargs)` in `src/continuous_patterns/core/stress.py` returning `(sigma_xx, sigma_yy, sigma_xy)`.
2. Register it in `STRESS_BUILDERS`.
3. Add the mode string to `StressSpec.mode` in `src/continuous_patterns/core/io.py` (`Literal[...]`).
4. Copy an existing YAML under `src/continuous_patterns/experiments/templates/` and set `stress.mode`.

### Add a new geometry

1. Implement a mask builder in `src/continuous_patterns/core/masks.py` returning the standard dict (`chi`, `ring`, `ring_accounting`, `rv`, plus `dx`, `L`, `R`, `n`, `xc`, `yc`).
2. Register it in `MASK_BUILDERS`.
3. Extend `GeometrySpec.type` in `core/io.py`.
4. Add a template YAML under `experiments/templates/`.

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for coding standards, testing, and known permissive areas (`physics` / `initial` dicts).

## Development

Runtime deps include **`tqdm`** (chunk progress in **`models.*.simulate`**) and standard-library **`logging`** (console + **`run.log`** via **`experiments.run`**).

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

## License

Licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE).
