# continuous-patterns

Research-oriented experiments with **JAX** (continuous media / pattern formation). Tooling: **uv**, **Ruff**, **pytest**, **pre-commit** (including **Conventional Commits** on `commit-msg`).

## Requirements

- Python 3.11+ (see `.python-version` for the local default)
- [uv](https://docs.astral.sh/uv/) for environments and lockfile
- **NVIDIA driver** new enough for the JAX+CUDA wheels you install (for GPU; see [JAX installation](https://jax.readthedocs.io/en/latest/installation.html))

## Quick start

**CPU (CI / laptops / no GPU):**

```bash
uv sync --extra cpu
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

**GPU (this machine or any host with CUDA 12–compatible JAX wheels):**

```bash
uv sync --extra cuda
```

Check that JAX sees the GPU:

```bash
uv run python -c "import jax; print(jax.devices())"
```

## Conventional Commits (enforced)

Install hook types so **commit messages** are validated (in addition to pre-commit checks on files):

```bash
uv run pre-commit install --install-hooks
```

Use messages like: `feat: add reaction-diffusion core`, `fix: correct boundary condition`, `ci: add quality job`, `docs: document uv sync`. Types include `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`, `build`, `perf` (see [Conventional Commits](https://www.conventionalcommits.org/)).

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files
```

`--all-files` only lints **tracked** files; add new code with `git add` first (or use the hooks on commit).

## Layout

- `src/continuous_patterns/` — package code
- `tests/` — pytest
- `uv.lock` — lockfile (commit it for reproducible installs)
