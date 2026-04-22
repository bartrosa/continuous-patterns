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

## Agate Cahn–Hilliard falsification (2D)

Model C for **moganite / chalcedony** with a reactive source, in a circular cavity. This is a *falsification* run: if banding appears and the **Jabłczyński** spacing ratios `q_n = d_n/d_{n-1}` are **nearly constant** and `q̄ > 1.05`, the pattern is **Liesegang-like** (classical radial geometry); if bands exist but ratios **vary**, you may have a **non-Liesegang** mechanism; **fewer than three** resolved bands ⇒ **INSUFFICIENT BANDS** in the diagnostic (v1.5+).

**v1.5** adds a **barrier** on φ outside [0,1], **soft clip** to [−0.05,1.05], **ratcheting** moganite nucleation, time-resolved **kymograph** / **band count** plots, and a **mass balance** that uses total silica in (c, φ_m, φ_c) plus an estimated **boundary flux** of c.

Install simulation extras (JAX CUDA stack + HDF5/plotting):

```bash
uv sync --extra agate --extra cuda   # or --extra cpu on machines without GPU
uv run python -m continuous_patterns.agate_ch.run --config configs/baseline_v15.yaml
```

Quick smoke (small grid, short time):

```bash
uv run python -m continuous_patterns.agate_ch.run --config configs/baseline_v15.yaml --quick
```

Parameter sweep (four pinning / overshoot variants, radial + kymograph grid):

```bash
uv run python -m continuous_patterns.agate_ch.run --config configs/baseline_v15.yaml --sweep configs/sweep.yaml
```

Outputs under `results/run_<timestamp>/` include `band_count_evolution.png`, `kymograph.png`, `jablczynski.png` (final), and `jablczynski_timeresolved.png` (at peak band count). Those folders are **gitignored** by default; see **`results/example/`** for a file checklist.

During integration, **`tqdm` prints a step progress bar on stderr** (skipped when stderr is not a terminal, e.g. pytest). Use **`--no-progress`** or YAML **`progress: false`** to silence it.

The printed **mass balance error** uses (Δ total silica − ∫ flux_in dt) / initial total silica (%); **≤2%** on a tuned run indicates the boundary bookkeeping is consistent.

**Reading the Jabłczyński figure:** subplot (b) shows `q_n` vs band index — a **flat** curve means geometric progression of spacings (Liesegang-like when CV is low and mean `q > 1.05`). Subplot (a) is log–log `d_n` vs `r_n`; slope near 1 is consistent with classical Liesegang scaling.

## Layout

- `src/continuous_patterns/` — package code (`agate_ch/` = CH falsification experiment)
- `configs/` — YAML for runs
- `tests/` — pytest
- `uv.lock` — lockfile (commit it for reproducible installs)
