"""Shape, FFT, and mass-balance checks for agate_ch."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from continuous_patterns.agate_ch.diagnostics import band_metrics, radial_profile
from continuous_patterns.agate_ch.model import build_geometry, dfdphi_total
from continuous_patterns.agate_ch.solver import integrate_chunks, laplacian


def test_laplacian_eigenmode_roundtrip() -> None:
    n = 32
    L = 1.0
    geom = build_geometry(L, 0.4, n)
    k0 = 2 * np.pi * 2 / L
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.sin(k0 * X) * np.sin(k0 * Y)
    uj = jnp.asarray(u)
    lap = np.asarray(laplacian(uj, geom.k_sq))
    expected = -(k0**2) * 2 * u  # two sin factors
    err = np.max(np.abs(lap - expected))
    assert err < 0.02


def test_radial_and_peaks() -> None:
    n = 64
    L = 10.0
    R = 3.0
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt((X - L / 2) ** 2 + (Y - L / 2) ** 2)
    f = np.exp(-(((r - 1.0) / 0.2) ** 2)) + 0.3 * np.exp(-(((r - 2.0) / 0.2) ** 2))
    rc, prof = radial_profile(f, L=L, R=R, nbins=40)
    m = band_metrics(rc, prof, R)
    assert m["N_b"] >= 1


def test_fft_roundtrip_real() -> None:
    key = jax.random.PRNGKey(3)
    u = jax.random.normal(key, (24, 24))
    v = jnp.fft.ifft2(jnp.fft.fft2(u)).real
    assert float(jnp.max(jnp.abs(u - v))) < 1e-5  # float32 FFT roundtrip


def test_mass_balance_bounded_short_run() -> None:
    """Smoke: closed-form mass balance residual stays moderate on short run."""
    cfg = {
        "grid": 48,
        "L": 100.0,
        "R": 35.0,
        "W": 1.0,
        "gamma": 2.0,
        "kappa": 0.5,
        "lambda_barrier": 10.0,
        "D_c": 0.01,
        "k_reaction": 0.0,
        "M_m": 0.05,
        "M_c": 0.05,
        "c_sat": 0.2,
        "c_0": 0.4,
        "c_ostwald": 0.6,
        "w_ostwald": 0.1,
        "phi_m_ratchet_low": 0.3,
        "phi_m_ratchet_high": 0.5,
        "use_ratchet": True,
        "dt": 0.001,
        "T": 0.05,
        "snapshot_every": 99999,
        "seed": 0,
        "uniform_supersaturation": False,
        "progress": False,
        "print_mass_balance": False,
    }
    _, _, _, meta = integrate_chunks(cfg, chunk_size=25, on_snapshot=None)
    mb = float(meta["mass_balance_percent_direct"])
    assert mb < 1.0, f"mass balance {mb}% should be tiny with chunk silica deltas"


def test_shapes_dfdphi() -> None:
    phi = jnp.ones((4, 4)) * 0.2
    g = dfdphi_total(phi, 1.0, 10.0)
    assert g.shape == phi.shape
