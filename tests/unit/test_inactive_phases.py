"""Inactive phases must stay bit-exact zero across IMEX steps."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from continuous_patterns.core.imex import Geometry, SimParams, imex_step
from continuous_patterns.core.masks import circular_cavity_masks
from continuous_patterns.core.spectral import k_vectors
from continuous_patterns.core.stress import none as stress_none
from continuous_patterns.core.stress import uniform_biaxial


def _stage1_geometry_f32(*, L: float = 200.0, R: float = 80.0, n: int = 64) -> Geometry:
    """Small Stage I geometry in float32, matching production precision."""
    m = circular_cavity_masks(L=L, R=R, n=n, eps_scale=2.0, dtype=jnp.float32)
    k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)
    sxx, syy, sxy = stress_none(L=L, n=n, dtype=jnp.float32)
    f32 = jnp.float32
    return Geometry(
        chi=m["chi"],
        ring=m["ring"],
        ring_accounting=m["ring_accounting"],
        sigma_xx=sxx,
        sigma_yy=syy,
        sigma_xy=sxy,
        k_sq=jnp.asarray(k_sq, dtype=f32),
        kx_sq=jnp.asarray(kx_sq, dtype=f32),
        ky_sq=jnp.asarray(ky_sq, dtype=f32),
        kx_wave=jnp.asarray(kx_wave, dtype=f32),
        ky_wave=jnp.asarray(ky_wave, dtype=f32),
        k_four=jnp.asarray(k_four, dtype=f32),
        rv=jnp.asarray(m["rv"], dtype=f32),
        dx=float(m["dx"]),
        L=float(m["L"]),
        R=float(m["R"]),
        n=int(m["n"]),
        xc=float(m["xc"]),
        yc=float(m["yc"]),
    )


def _stage1_geometry_f32_stress(
    *, L: float = 200.0, R: float = 80.0, n: int = 64, sigma_0: float = 0.1
) -> Geometry:
    """Stage I float32 geometry with uniform biaxial stress (ψ-split path on)."""
    m = circular_cavity_masks(L=L, R=R, n=n, eps_scale=2.0, dtype=jnp.float32)
    k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)
    sxx, syy, sxy = uniform_biaxial(L=L, n=n, sigma_0=sigma_0, dtype=jnp.float32)
    f32 = jnp.float32
    return Geometry(
        chi=m["chi"],
        ring=m["ring"],
        ring_accounting=m["ring_accounting"],
        sigma_xx=sxx,
        sigma_yy=syy,
        sigma_xy=sxy,
        k_sq=jnp.asarray(k_sq, dtype=f32),
        kx_sq=jnp.asarray(kx_sq, dtype=f32),
        ky_sq=jnp.asarray(ky_sq, dtype=f32),
        kx_wave=jnp.asarray(kx_wave, dtype=f32),
        ky_wave=jnp.asarray(ky_wave, dtype=f32),
        k_four=jnp.asarray(k_four, dtype=f32),
        rv=jnp.asarray(m["rv"], dtype=f32),
        dx=float(m["dx"]),
        L=float(m["L"]),
        R=float(m["R"]),
        n=int(m["n"]),
        xc=float(m["xc"]),
        yc=float(m["yc"]),
    )


def _default_simparams_two_phase_active() -> SimParams:
    """Defaults: moganite and chalcedony active; α-quartz and impurity inactive (zeros)."""
    return SimParams()


def _simparams_stage1_full() -> SimParams:
    """Stage I defaults used in production (reaction + rim Dirichlet on)."""
    return SimParams(
        reaction_active=True,
        dirichlet_active=True,
        aging_active=False,
    )


def test_phi_q_stays_bit_exact_zero_after_many_steps() -> None:
    """100 IMEX steps with α-quartz inactive → phi_q must remain exactly zero."""
    geom = _stage1_geometry_f32()
    prm = _default_simparams_two_phase_active()
    n = geom.n
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    phi_m = (0.0 + 0.01 * jax.random.normal(k1, (n, n), dtype=jnp.float32)) * geom.chi
    phi_c = (0.0 + 0.01 * jax.random.normal(k2, (n, n), dtype=jnp.float32)) * geom.chi
    phi_q = jnp.zeros((n, n), dtype=jnp.float32)
    phi_imp = jnp.zeros((n, n), dtype=jnp.float32)
    c = prm.c_sat * geom.chi

    state = (phi_m, phi_c, phi_q, phi_imp, c)
    for _ in range(100):
        state, _aux = imex_step(state, geom, prm, dt=0.01)

    phi_q_final = state[2]
    phi_imp_final = state[3]
    zeros = jnp.zeros((n, n), dtype=jnp.float32)

    assert jnp.array_equal(phi_q_final, zeros), (
        f"phi_q drifted from zero: max abs = {float(jnp.max(jnp.abs(phi_q_final))):.3e}"
    )
    assert jnp.array_equal(phi_imp_final, zeros), (
        f"phi_imp drifted from zero: max abs = {float(jnp.max(jnp.abs(phi_imp_final))):.3e}"
    )


def test_phi_q_stays_zero_with_stress_active() -> None:
    """Stress coupling on, phi_q inactive → phi_q must still stay exactly zero."""
    geom = _stage1_geometry_f32_stress()
    prm = SimParams(stress_coupling_B=0.5)
    n = geom.n
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    phi_m = (0.0 + 0.01 * jax.random.normal(k1, (n, n), dtype=jnp.float32)) * geom.chi
    phi_c = (0.0 + 0.01 * jax.random.normal(k2, (n, n), dtype=jnp.float32)) * geom.chi
    phi_q = jnp.zeros((n, n), dtype=jnp.float32)
    phi_imp = jnp.zeros((n, n), dtype=jnp.float32)
    c = prm.c_sat * geom.chi

    state = (phi_m, phi_c, phi_q, phi_imp, c)
    for _ in range(100):
        state, _aux = imex_step(state, geom, prm, dt=0.01)

    zeros = jnp.zeros((n, n), dtype=jnp.float32)
    assert jnp.array_equal(state[2], zeros), (
        f"phi_q drifted: max abs = {float(jnp.max(jnp.abs(state[2]))):.3e}"
    )
    assert jnp.array_equal(state[3], zeros), (
        f"phi_imp drifted: max abs = {float(jnp.max(jnp.abs(state[3]))):.3e}"
    )


def test_phi_q_stays_zero_with_reaction_and_dirichlet() -> None:
    """Full Stage I (reaction + rim Dirichlet), phi_q inactive → exact zero."""
    geom = _stage1_geometry_f32()
    prm = _simparams_stage1_full()
    n = geom.n
    key = jax.random.PRNGKey(99)
    k1, k2 = jax.random.split(key)
    phi_m = (0.0 + 0.01 * jax.random.normal(k1, (n, n), dtype=jnp.float32)) * geom.chi
    phi_c = (0.0 + 0.01 * jax.random.normal(k2, (n, n), dtype=jnp.float32)) * geom.chi
    phi_q = jnp.zeros((n, n), dtype=jnp.float32)
    phi_imp = jnp.zeros((n, n), dtype=jnp.float32)
    c = prm.c_sat * geom.chi

    state = (phi_m, phi_c, phi_q, phi_imp, c)
    for _ in range(50):
        state, _aux = imex_step(state, geom, prm, dt=0.01)

    zeros = jnp.zeros((n, n), dtype=jnp.float32)
    assert jnp.array_equal(state[2], zeros)
    assert jnp.array_equal(state[3], zeros)
