"""Smoke tests for Agate CH (environment, mass balance, IMEX rim)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from continuous_patterns.agate_ch.mass_balance import (
    plot_mass_balance_comparison,
    print_mass_balance_smoke_stdout,
)
from continuous_patterns.agate_ch.run import enrich_meta_physical_flux
from continuous_patterns.agate_ch.solver import (
    cfg_to_sim_params,
    imex_step,
    initial_state,
    integrate_chunks,
    simulate,
    simulate_to_host,
    total_silica_full_domain_jnp,
)


def test_fp32_default_after_import() -> None:
    """Fresh interpreter: importing agate_ch must not enable global jax x64."""
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    env = os.environ.copy()
    sep = os.pathsep
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{sep}{prev}" if prev else str(src)
    code = (
        "import jax\n"
        "assert not jax.config.jax_enable_x64, 'expected fp32 default'\n"
        "import continuous_patterns.agate_ch.model  # noqa: F401\n"
        "import continuous_patterns.agate_ch.solver  # noqa: F401\n"
        "assert not jax.config.jax_enable_x64, 'import must not enable x64'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=str(root))


@pytest.mark.skipif(
    not os.environ.get("AGATE_CH_LONG_SMOKE"),
    reason="optional T=10000-style check; run with AGATE_CH_LONG_SMOKE=1",
)
@pytest.mark.slow
def test_surface_flux_long_horizon_baseline_style() -> None:
    """Spot-check Option B with full baseline (production) parameters."""
    cfg = baseline_cfg()
    cfg["progress"] = False
    cfg["print_mass_balance"] = False
    _, _, _, meta, snaps = simulate_to_host(cfg, chunk_size=2000)
    geom = meta["geom"]
    key = jax.random.PRNGKey(int(cfg["seed"]))
    prm = cfg_to_sim_params(cfg)
    ic = initial_state(
        geom,
        key,
        c_sat=prm.c_sat,
        c_0=prm.c_0,
        noise=0.01,
        uniform_supersaturation=prm.uniform_supersaturation,
    )
    snaps_full = [
        (
            0,
            np.asarray(jax.device_get(ic[0])),
            np.asarray(jax.device_get(ic[1])),
            np.asarray(jax.device_get(ic[2])),
        )
    ]
    snaps_full.extend(snaps)
    enrich_meta_physical_flux(cfg, meta, snaps_full)
    surface = meta["mass_balance_surface_flux"]
    assert surface.get("budget_source") == "integrate_chunks_dense"
    assert abs(float(surface["leak_pct"])) < 5.0


@pytest.mark.slow
def test_fp32_mass_balance_still_acceptable() -> None:
    """Option B (surface flux residual) and periodic spectral drift at fp32 defaults."""
    cfg = {
        "grid": 512,
        "L": 200.0,
        "R": 80.0,
        "W": 1.0,
        "gamma": 2.0,
        "kappa": 0.5,
        "lambda_barrier": 10.0,
        "D_c": 1.0,
        "k_reaction": 0.5,
        "M_m": 0.1,
        "M_c": 1.0,
        "c_sat": 0.2,
        "c_0": 1.0,
        "c_ostwald": 0.6,
        "w_ostwald": 0.1,
        "phi_m_ratchet_low": 0.3,
        "phi_m_ratchet_high": 0.5,
        "use_ratchet": False,
        "dt": 0.01,
        "T": 500.0,
        "snapshot_every": 400,
        "flux_sample_dt": 2.0,
        "seed": 42,
        "uniform_supersaturation": False,
        "progress": False,
        "print_mass_balance": False,
        "rho_m": 1.0,
        "rho_c": 1.0,
        "mass_balance_r_measure_fixed_fraction": 0.75,
    }
    _, _, _, meta, snaps = simulate_to_host(cfg, chunk_size=2000)
    geom = meta["geom"]
    key = jax.random.PRNGKey(int(cfg["seed"]))
    prm = cfg_to_sim_params(cfg)
    ic = initial_state(
        geom,
        key,
        c_sat=prm.c_sat,
        c_0=prm.c_0,
        noise=0.01,
        uniform_supersaturation=prm.uniform_supersaturation,
    )
    snaps_full = [
        (
            0,
            np.asarray(jax.device_get(ic[0])),
            np.asarray(jax.device_get(ic[1])),
            np.asarray(jax.device_get(ic[2])),
        )
    ]
    snaps_full.extend(snaps)

    enrich_meta_physical_flux(cfg, meta, snaps_full)
    surface = meta["mass_balance_surface_flux"]
    assert surface.get("budget_source") == "integrate_chunks_dense"
    assert abs(float(surface["leak_pct"])) < 5.0

    scfg = baseline_cfg()
    scfg["T"] = 1.0
    scfg["dt"] = 0.01
    scfg["disable_dirichlet"] = True
    scfg["initial_condition"] = "blob"
    scfg["snapshot_every"] = 10
    scfg["grid"] = 256

    periodic = simulate(scfg)
    m0 = periodic["total_mass_series"][0]
    m1 = periodic["total_mass_series"][-1]
    leak_pct = 100.0 * (m1 - m0) / m0
    assert abs(leak_pct) < 0.1, f"spectral mass drift too large: {leak_pct}%"


def baseline_cfg() -> dict:
    """Mutable copy of ``configs/agate_ch/baseline.yaml``."""
    root = Path(__file__).resolve().parents[2]
    with (root / "configs" / "agate_ch" / "baseline.yaml").open() as f:
        return yaml.safe_load(f)


@pytest.mark.slow
def test_flux_closure_ring_vs_bulk() -> None:
    """Dense ring influx integral vs cavity silica gain (flux_closure_ratio_dense).

    ``pytest -s`` prints milestone stdout. The closure ratio is not 1 because the
    finite-difference gradient at fixed radii does not match the bulk budget
    once the front and the probe geometry diverge; this pins a loose band only.
    """
    cfg = {
        "grid": 512,
        "L": 200.0,
        "R": 80.0,
        "W": 1.0,
        "gamma": 2.0,
        "kappa": 0.5,
        "lambda_barrier": 10.0,
        "D_c": 1.0,
        "k_reaction": 0.5,
        "M_m": 0.1,
        "M_c": 1.0,
        "c_sat": 0.2,
        "c_0": 1.0,
        "c_ostwald": 0.6,
        "w_ostwald": 0.1,
        "phi_m_ratchet_low": 0.3,
        "phi_m_ratchet_high": 0.5,
        "use_ratchet": False,
        "dt": 0.01,
        "T": 500.0,
        "snapshot_every": 2500,
        "seed": 42,
        "uniform_supersaturation": False,
        "progress": False,
        "print_mass_balance": True,
        "diagnose_flux_detail": True,
        "rho_m": 1.0,
        "rho_c": 1.0,
    }
    _, _, _, meta = integrate_chunks(cfg, chunk_size=2000, on_snapshot=None)
    ratio = float(meta["flux_closure_ratio_dense"])
    assert 4.0 < ratio < 9.0, (
        f"unexpected flux closure ratio={ratio:.4f} "
        "(expected rough band for R-3/R-5 FD)"
    )


@pytest.mark.slow
def test_mass_balance_matches_production(tmp_path: Path) -> None:
    """Baseline ``T``, ``snapshot_every``, ``flux_sample_dt`` (production path).

    Uses grid ``256``; ``128`` fails the coarse finite-difference 5% gate.
    """
    cfg = baseline_cfg()
    cfg["grid"] = 256
    cfg["progress"] = False
    cfg["print_mass_balance"] = False

    _, _, _, meta, snaps = simulate_to_host(cfg, chunk_size=2000)
    budget = meta["mass_balance_surface_flux"]
    assert budget.get("budget_source") == "integrate_chunks_dense"

    leak = abs(float(budget["leak_pct"]))
    assert leak < 5.0, (
        f"Option B leak {leak:.2f}% exceeds 5% threshold. "
        f"Front reached r_fixed: {budget['front_reached_r_measure']}. "
        f"N samples: {budget['n_flux_samples']}."
    )
    assert int(budget["n_flux_samples"]) >= 20, (
        f"Too few flux samples ({budget['n_flux_samples']}); "
        "flux_sample_dt likely too large or horizon too short."
    )

    geom = meta["geom"]
    key = jax.random.PRNGKey(int(cfg["seed"]))
    prm = cfg_to_sim_params(cfg)
    ic = initial_state(
        geom,
        key,
        c_sat=prm.c_sat,
        c_0=prm.c_0,
        noise=0.01,
        uniform_supersaturation=prm.uniform_supersaturation,
    )
    snaps_full = [
        (
            0,
            np.asarray(jax.device_get(ic[0])),
            np.asarray(jax.device_get(ic[1])),
            np.asarray(jax.device_get(ic[2])),
        )
    ]
    snaps_full.extend(snaps)
    enrich_meta_physical_flux(cfg, meta, snaps_full)
    surface = meta["mass_balance_surface_flux"]
    assert isinstance(surface, dict) and "flux_integrated_to_stop" in surface

    scfg = baseline_cfg()
    scfg["T"] = 1.0
    scfg["dt"] = 0.01
    scfg["disable_dirichlet"] = True
    scfg["initial_condition"] = "blob"
    scfg["snapshot_every"] = 10
    scfg["grid"] = 256

    periodic = simulate(scfg)
    m0 = periodic["total_mass_series"][0]
    m1 = periodic["total_mass_series"][-1]
    leak_pct = 100.0 * (m1 - m0) / m0
    assert abs(leak_pct) < 0.1, f"spectral mass drift too large: {leak_pct}%"

    plot_mass_balance_comparison(
        meta=meta,
        surface_flux_budget=surface,
        path=tmp_path / "mass_balance_comparison.png",
    )
    assert (tmp_path / "mass_balance_comparison.png").is_file()

    print_mass_balance_smoke_stdout(
        label="baseline params, grid=256",
        meta=meta,
        surface_flux_budget=surface,
        spectral_kernel_check={
            "total_mass_initial": m0,
            "total_mass_final": m1,
            "leak_pct": leak_pct,
            "grid": scfg["grid"],
            "T": scfg["T"],
            "dt": scfg["dt"],
            "n_steps": int(round(float(scfg["T"]) / float(scfg["dt"]))),
        },
    )


def test_imex_rim_replenishment_positive() -> None:
    """With rim BC, IMEX injects a positive thin-grid and full-grid mass increment."""
    from continuous_patterns.agate_ch.model import build_geometry

    cfg = {
        "grid": 96,
        "L": 200.0,
        "R": 80.0,
        "W": 1.0,
        "gamma": 2.0,
        "kappa": 0.5,
        "lambda_barrier": 10.0,
        "D_c": 1.0,
        "k_reaction": 0.5,
        "M_m": 0.1,
        "M_c": 1.0,
        "c_sat": 0.2,
        "c_0": 1.0,
        "c_ostwald": 0.6,
        "w_ostwald": 0.1,
        "phi_m_ratchet_low": 0.3,
        "phi_m_ratchet_high": 0.5,
        "use_ratchet": False,
        "seed": 1,
        "uniform_supersaturation": False,
        "rho_m": 1.0,
        "rho_c": 1.0,
    }
    geom = build_geometry(cfg["L"], cfg["R"], cfg["grid"])
    prm = cfg_to_sim_params(cfg)
    key = jax.random.PRNGKey(1)
    state = initial_state(
        geom,
        key,
        c_sat=prm.c_sat,
        c_0=prm.c_0,
        noise=0.01,
        uniform_supersaturation=prm.uniform_supersaturation,
    )
    new_s, dd = imex_step(state, None, geom, prm, 0.01)
    assert float(dd[0]) > 0.0 and float(dd[1]) > 0.0
    assert (
        float(
            total_silica_full_domain_jnp(
                new_s[0], new_s[1], new_s[2], geom.dx, prm.rho_m, prm.rho_c
            )
        )
        > 0.0
    )


def test_package_version() -> None:
    import continuous_patterns as cp

    assert cp.__version__


def test_jax_import() -> None:
    import jax

    assert jax.__version__
