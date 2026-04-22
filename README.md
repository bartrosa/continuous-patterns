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

Model C for **moganite / chalcedony** with a reactive source, in a circular cavity. This is a *falsification* setup: if banding appears and the **Jabłczyński** spacing ratios `q_n = d_n/d_{n-1}` are **nearly constant** and `q̄ > 1.05`, the pattern is **Liesegang-like** (classical radial geometry); if bands exist but ratios **vary**, you may have a **non-Liesegang** mechanism; **fewer than three** resolved bands ⇒ **INSUFFICIENT BANDS** in the diagnostic.

The stack includes a **barrier** on φ outside [0,1], **soft clip** to [−0.05,1.05], optional **ratcheting** for moganite nucleation, time-resolved **kymograph** / **band count** plots, and **mass balance** checks (bulk silica plus estimated **boundary flux** of dissolved silica).

Defaults live in **`configs/agate_ch/baseline.yaml`** (referenced by `--config`; you can omit `--config` to use it).

### Install simulation extras

```bash
uv sync --extra agate --extra cuda   # or --extra cpu on machines without GPU
```

### Workflow (step by step)

Each sweep creates a **new** directory **`results/agate_ch/sweep_<YYYYMMDD_HHMMSS>/`**. Note the paths printed at the end (or list `results/agate_ch/` afterwards); you need them for publication commands.

1. **Optional — cheap sanity check** (small grid, short horizon):

   ```bash
   uv run python -m continuous_patterns.agate_ch.run --config configs/agate_ch/baseline.yaml --quick
   ```

   Writes under `results/agate_ch/run_<timestamp>/` with PNG diagnostics.

2. **Single full baseline run** (uses `configs/agate_ch/baseline.yaml` parameters):

   ```bash
   uv run python -m continuous_patterns.agate_ch.run --config configs/agate_ch/baseline.yaml
   ```

   Same layout as step 1, full grid and `T` from YAML.

3. **Main falsification sweep** (pinning / ratchet / seeds — see `configs/agate_ch/sweep.yaml`):

   ```bash
   uv run python -m continuous_patterns.agate_ch.run --config configs/agate_ch/baseline.yaml --sweep configs/agate_ch/sweep.yaml
   ```

   Produces comparison PNGs at the sweep root (`sweep_comparison.png`, `comparison_grid.png`, …) plus one subdirectory per run ID. Stdout ends with `Outputs: …/results/agate_ch/sweep_<timestamp>`.

4. **Optional — γ scan** (immiscibility ladder — `configs/agate_ch/gamma_scan.yaml`, long):

   ```bash
   uv run python -m continuous_patterns.agate_ch.run --config configs/agate_ch/baseline.yaml --sweep configs/agate_ch/gamma_scan.yaml
   ```

   Adds `gamma_phase_diagram.png`, `gamma_phase_diagram.csv`, `gamma_scan_fields.png`, etc.

5. **Publication bundle** — after you have **both** a main sweep directory and a gamma-scan directory (from steps 3–4), build figures under `paper_figures/` inside the **main** sweep folder:

   ```bash
   uv run python -m continuous_patterns.agate_ch.run \
     --generate-paper results/agate_ch/sweep_<main_ts> results/agate_ch/sweep_<gamma_ts>
   ```

6. **Results markdown** — write **`RESULTS.md`** at the repo root from sweep CSV/summaries (use `none` if you did not run a gamma sweep):

   ```bash
   uv run python -m continuous_patterns.agate_ch.run \
     --write-results results/agate_ch/sweep_<main_ts> results/agate_ch/sweep_<gamma_ts>
   # or without gamma assets:
   uv run python -m continuous_patterns.agate_ch.run \
     --write-results results/agate_ch/sweep_<main_ts> none
   ```

During integration, **`tqdm` prints a step progress bar on stderr** (skipped when stderr is not a terminal). Use **`--no-progress`** or YAML **`progress: false`** to silence it.

The printed **mass balance** line reports **Option B** (`mass_balance_surface_flux.leak_pct`), independent of snapshot cadence; **`sweep_summary.csv`** column **`mass_balance_leak_pct`** matches it.

**Reading the Jabłczyński figure:** subplot (b) shows `q_n` vs band index — a **flat** curve means geometric progression of spacings (Liesegang-like when CV is low and mean `q > 1.05`). Subplot (a) is log–log `d_n` vs `r_n`; slope near 1 is consistent with classical Liesegang scaling.

See **`NOTES.md`** for methodology detail. Run outputs are **gitignored** by default; **`results/example/`** is a schema checklist; **`results/agate_ch/README.md`** describes where Agate CH writes sweeps.

## Layout

- `src/continuous_patterns/` — package code (add sibling packages for new experiments)
- `configs/<experiment>/` — YAML per domain (e.g. **`configs/agate_ch/`** for Agate CH)
- `tests/<experiment>/` — pytest mirroring package layout
- `results/<experiment>/` — default output roots (gitignored except README placeholders)
- `uv.lock` — lockfile (commit it for reproducible installs)
