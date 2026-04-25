"""Unit tests for :mod:`continuous_patterns.core.gravity`."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import pytest

from continuous_patterns.core.gravity import (
    body_force_advection_y,
    body_force_potential,
    rim_ramp_field,
)
from continuous_patterns.core.imex import Geometry, imex_step
from continuous_patterns.core.masks import circular_cavity_masks
from continuous_patterns.core.spectral import k_vectors
from continuous_patterns.core.stress import none as stress_none
from continuous_patterns.models.cavity_reactive import build_sim_params


def _geom_stage1(*, L: float, R: float, n: int) -> Geometry:
    m = circular_cavity_masks(L=L, R=R, n=n, eps_scale=2.0)
    k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)
    sxx, syy, sxy = stress_none(L=L, n=n)
    z = jnp.asarray(m["chi"], dtype=jnp.float64)
    return Geometry(
        chi=z,
        ring=jnp.asarray(m["ring"], dtype=jnp.float64),
        ring_accounting=jnp.asarray(m["ring_accounting"], dtype=jnp.float64),
        sigma_xx=jnp.asarray(sxx, dtype=jnp.float64),
        sigma_yy=jnp.asarray(syy, dtype=jnp.float64),
        sigma_xy=jnp.asarray(sxy, dtype=jnp.float64),
        k_sq=jnp.asarray(k_sq, dtype=jnp.float64),
        kx_sq=jnp.asarray(kx_sq, dtype=jnp.float64),
        ky_sq=jnp.asarray(ky_sq, dtype=jnp.float64),
        kx_wave=jnp.asarray(kx_wave, dtype=jnp.float64),
        ky_wave=jnp.asarray(ky_wave, dtype=jnp.float64),
        k_four=jnp.asarray(k_four, dtype=jnp.float64),
        rv=jnp.asarray(m["rv"], dtype=jnp.float64),
        dx=float(m["dx"]),
        L=float(m["L"]),
        R=float(m["R"]),
        n=int(m["n"]),
        xc=float(m["xc"]),
        yc=float(m["yc"]),
    )


def test_rim_ramp_field_endpoints() -> None:
    L, n, c0, ra = 10.0, 100, 2.0, 0.5
    f = rim_ramp_field(L=L, n=n, c0=c0, rim_alpha=ra, dtype=jnp.float64)
    dx = L / n
    jj = jnp.arange(n, dtype=jnp.float64)
    y1d = (jj + 0.5) * dx
    j_mid = int(jnp.argmin(jnp.abs(y1d - 0.5 * L)))
    j_lo, j_hi = 0, n - 1
    assert float(f[0, j_mid]) == pytest.approx(c0, rel=0, abs=0.02 * c0)
    assert float(f[0, j_hi]) == pytest.approx(c0 * (1.0 + ra), rel=0, abs=0.03 * c0)
    assert float(f[0, j_lo]) == pytest.approx(c0 * (1.0 - ra), rel=0, abs=0.03 * c0)


def test_body_force_potential_midplane_zero() -> None:
    L, n, g = 8.0, 64, 0.3
    mu = body_force_potential(L=L, n=n, g_value=g, dtype=jnp.float64)
    jj = jnp.arange(n, dtype=jnp.float64)
    y1d = (jj + 0.5) * (L / n)
    j_mid = int(jnp.argmin(jnp.abs(y1d - 0.5 * L)))
    assert float(mu[0, j_mid]) == pytest.approx(0.0, abs=0.02)


def test_gravity_zero_matches_absent_block() -> None:
    """Explicit zero gravity matches omitting the ``gravity`` block in ``SimParams``."""
    base = {
        "experiment": {"name": "g0", "model": "cavity_reactive", "seed": 0},
        "geometry": {"type": "circular_cavity", "L": 12.0, "R": 3.0, "n": 32},
        "physics": {
            "W": 1.0,
            "gamma": 2.0,
            "kappa_x": 0.5,
            "kappa_y": 0.5,
            "M_m": 0.1,
            "M_c": 1.0,
            "D_c": 0.2,
            "k_rxn": 0.5,
            "c_sat": 0.2,
            "c_0": 0.5,
            "lambda_bar": 10.0,
            "c_ostwald": 0.5,
            "w_ostwald": 0.1,
            "use_ratchet": True,
        },
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
        "time": {"dt": 0.01, "T": 0.2, "snapshot_every": 1000},
        "output": {"save_final_state": False, "record_spectral_mass_diagnostic": True},
        "initial": {},
    }
    with_g = copy.deepcopy(base)
    with_g["gravity"] = {
        "rim_alpha": 0.0,
        "g_c": 0.0,
        "g_phi_m": 0.0,
        "g_phi_c": 0.0,
        "g_phi_q": 0.0,
        "g_phi_imp": 0.0,
    }
    pr0 = build_sim_params(base)
    pr1 = build_sim_params(with_g)
    assert pr0 == pr1
    geom = _geom_stage1(L=12.0, R=3.0, n=32)
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    phi_m = 0.1 * jax.random.normal(k1, (32, 32))
    phi_c = 0.1 * jax.random.normal(k2, (32, 32))
    phi_q = jnp.zeros((32, 32))
    phi_imp = jnp.zeros((32, 32))
    c = 0.15 * jax.random.normal(k5, (32, 32))
    state0 = (phi_m, phi_c, phi_q, phi_imp, c)
    s0, _ = imex_step(state0, geom, pr0, 0.01)
    s1, _ = imex_step(state0, geom, pr1, 0.01)
    for a, b in zip(s0, s1, strict=True):
        assert jnp.allclose(a, b)


def test_body_force_advection_y_sine() -> None:
    n = 32
    L = 2.0 * jnp.pi
    _, _, _, _, ky_wave, _ = k_vectors(L=L, n=n)
    jj = jnp.arange(n, dtype=jnp.float64)[jnp.newaxis, :]
    y = jnp.broadcast_to((jj + 0.5) * (L / n), (n, n))
    u = jnp.sin(y)
    u_hat = jnp.fft.fft2(u)
    du = body_force_advection_y(u_hat, jnp.asarray(ky_wave, dtype=jnp.float64))
    assert jnp.max(jnp.abs(du - jnp.cos(y))) < 1e-3
