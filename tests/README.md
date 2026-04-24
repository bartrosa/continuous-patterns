# Tests

Pytest discovers everything under **`tests/`** (see **`pyproject.toml`** → `[tool.pytest.ini_options]` → `testpaths`).

## Layout

- **`unit/`** — Fast, isolated checks of `continuous_patterns.core` primitives (spectral symbols, masks, stress builders, IMEX branches, I/O validation) with no full simulation harness.
- **`integration/`** — End-to-end smoke runs: nested YAML → model → a few steps (or tiny grids), asserting invariants and that Stage I / II entry points wire up.
- **`regression/`** — Golden or drift-tolerant comparisons against archived references (HDF5 snapshots, key scalars); intended for longer runs or x64-on-demand paths from **`docs/ARCHITECTURE.md`**.

Add new modules under the tier that matches the failure mode you care about; keep helpers next to the tests that use them, not under `src/`.
