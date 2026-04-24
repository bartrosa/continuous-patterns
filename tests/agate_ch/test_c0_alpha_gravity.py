"""Experiment 4: rim Dirichlet ``c_0(y)`` gradient (``c0_alpha``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax")

from continuous_patterns.agate_ch.model import build_geometry
from continuous_patterns.agate_ch.solver import (
    cfg_to_sim_params,
    imex_step,
    initial_state,
    rim_dirichlet_c_targets,
)


def _minimal_cfg(**kwargs: object) -> dict:
    base = {
        "grid": 64,
        "L": 200.0,
        "R": 80.0,
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
        "phi_m_ratchet_low": 0.3,
        "phi_m_ratchet_high": 0.5,
        "use_ratchet": True,
        "seed": 42,
        "uniform_supersaturation": False,
        "rho_m": 1.0,
        "rho_c": 1.0,
    }
    base.update(kwargs)
    return base


def test_c0_alpha_defaults_to_zero_in_cfg() -> None:
    prm = cfg_to_sim_params(_minimal_cfg())
    assert prm.c0_alpha == 0.0


def test_rim_targets_constant_when_alpha_zero() -> None:
    cfg = _minimal_cfg()
    prm = cfg_to_sim_params(cfg)
    geom = build_geometry(cfg["L"], cfg["R"], cfg["grid"])
    rim = rim_dirichlet_c_targets(float(cfg["L"]), geom, prm)
    assert bool(jnp.all(jnp.abs(rim - jnp.float32(prm.c_0)) < 1e-6))


def test_imex_step_identical_for_zero_alpha_vs_scalar_path() -> None:
    """With c0_alpha=0, rim field equals c_0; one step matches explicit scalar BC."""
    cfg = _minimal_cfg()
    prm = cfg_to_sim_params(cfg)
    geom = build_geometry(cfg["L"], cfg["R"], cfg["grid"])
    key = jax.random.PRNGKey(0)
    state = initial_state(geom, key, prm=prm, L=float(cfg["L"]), noise=0.01)
    out0, dd0 = imex_step(state, None, geom, prm, 0.01)

    c, pm, pc = state
    c2 = jnp.where(geom.ring > 0.5, jnp.float32(prm.c_0), c)
    state_alt = (c2, pm, pc)
    out1, dd1 = imex_step(state_alt, None, geom, prm, 0.01)

    max_c = float(jnp.max(jnp.abs(out0[0] - out1[0])))
    max_m = float(jnp.max(jnp.abs(out0[1] - out1[1])))
    max_c_ph = float(jnp.max(jnp.abs(out0[2] - out1[2])))
    assert max_c == 0.0 and max_m == 0.0 and max_c_ph == 0.0
    assert float(jnp.max(jnp.abs(dd0 - dd1))) == 0.0


def test_uniform_supersaturation_legacy_matches_chi_when_alpha_zero() -> None:
    """``uniform_supersaturation`` + ``c0_alpha==0`` keeps historical ``c = c_0 * χ``."""
    cfg = _minimal_cfg(uniform_supersaturation=True)
    prm = cfg_to_sim_params(cfg)
    assert prm.uniform_supersaturation is True
    geom = build_geometry(cfg["L"], cfg["R"], cfg["grid"])
    key = jax.random.PRNGKey(1)
    state = initial_state(geom, key, prm=prm, L=float(cfg["L"]), noise=0.01)
    chi = geom.chi
    expected_c = jnp.where(chi > 0.5, prm.c_0, 0.0) * chi
    assert float(jnp.max(jnp.abs(state[0] - expected_c))) < 1e-5


def test_rim_gradient_top_bottom_bracket_c0() -> None:
    cfg = _minimal_cfg(c0_alpha=0.2)
    prm = cfg_to_sim_params(cfg)
    geom = build_geometry(cfg["L"], cfg["R"], cfg["grid"])
    rim = rim_dirichlet_c_targets(float(cfg["L"]), geom, prm)
    ring = geom.ring > 0.5
    if not bool(jnp.any(ring)):
        pytest.skip("no ring cells at this resolution")
    y1d = (np.arange(cfg["grid"]) + 0.5) * (cfg["L"] / cfg["grid"])
    Y = np.broadcast_to(y1d[np.newaxis, :], rim.shape)
    vals = np.asarray(jax.device_get(rim))
    mask = np.asarray(jax.device_get(ring))
    yr = Y[mask]
    vr = vals[mask]
    lo = float(np.min(yr))
    hi = float(np.max(yr))
    top = vr[np.argmax(yr)]
    bot = vr[np.argmin(yr)]
    c0 = float(prm.c_0)
    alpha = float(prm.c0_alpha)
    spread = float(np.max(vr) - np.min(vr))
    assert spread > 0.15 * c0 * alpha
    assert top > bot + 1e-3
    assert lo < 0.51 * cfg["L"] < hi
