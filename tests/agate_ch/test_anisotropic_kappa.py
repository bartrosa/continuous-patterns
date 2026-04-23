"""Experiment 5a: anisotropic κ defaults and spectral stiff symbol."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytest.importorskip("jax")

from continuous_patterns.agate_ch.model import build_geometry
from continuous_patterns.agate_ch.solver import cfg_to_sim_params


def test_cfg_defaults_kappa_xy_from_kappa() -> None:
    cfg = {
        "W": 1.0,
        "gamma": 3.0,
        "kappa": 0.5,
        "lambda_barrier": 10.0,
        "D_c": 1.0,
        "k_reaction": 0.5,
        "M_m": 0.01,
        "M_c": 1.0,
        "c_sat": 0.2,
        "c_0": 1.0,
        "c_ostwald": 0.6,
        "w_ostwald": 0.1,
        "uniform_supersaturation": False,
        "rho_m": 1.0,
        "rho_c": 1.0,
    }
    prm = cfg_to_sim_params(cfg)
    assert prm.kappa_x == prm.kappa_y == prm.kappa == 0.5


def test_isotropic_stiff_symbol_matches_legacy_kappa_kfour() -> None:
    """``k_sq * (κ_x kx² + κ_y ky²)`` reduces to ``κ k_sq²`` when κ_x=κ_y=κ."""
    geom = build_geometry(200.0, 80.0, 64)
    kappa = jnp.float32(0.5)
    legacy = kappa * geom.k_four
    aniso_sym = kappa * geom.kx_sq + kappa * geom.ky_sq
    assert jnp.all(aniso_sym == kappa * geom.k_sq)
    stiff = geom.k_sq * aniso_sym
    assert jnp.all(stiff == legacy)


def test_explicit_equal_kappa_matches_legacy_symbol() -> None:
    geom = build_geometry(200.0, 80.0, 128)
    kappa = jnp.float32(0.5)
    legacy = kappa * geom.k_four
    aniso_sym = jnp.float32(0.5) * geom.kx_sq + jnp.float32(0.5) * geom.ky_sq
    stiff = geom.k_sq * aniso_sym
    assert jnp.all(stiff == legacy)
