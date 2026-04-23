# Tests

All pytest modules live under **`tests/`** (see **`pyproject.toml`** → `[tool.pytest.ini_options]` → `testpaths`).

| Path | Role |
|------|------|
| **`test_agate_ch_solver_regression.py`** | Flat config / `cfg_to_sim_params` / YAML flatten checks for `agate_ch` |
| **`test_agate_stage2_smoke.py`** | Smoke imports and params for `agate_stage2` |
| **`agate_ch/test_smoke.py`** | Optional heavier `agate_ch` smoke (nested folder mirrors package name) |

Add new files here—**not** under `src/`—so CI and `uv run pytest` discover them without extra wiring.
