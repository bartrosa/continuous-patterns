"""Unit tests for :mod:`continuous_patterns.core.imex`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from continuous_patterns.core.imex import Geometry, SimParams, imex_step
from continuous_patterns.core.masks import circular_cavity_masks
from continuous_patterns.core.spectral import k_vectors
from continuous_patterns.core.stress import none as stress_none
from continuous_patterns.core.stress import uniform_biaxial


def _geometry_stage2(*, L: float, n: int) -> Geometry:
    """Bulk Stage II geometry: ``χ ≡ 1``, zero rim, zero stress (override σ as needed)."""
    return _geometry_bulk(L=L, n=n)


def _geometry_bulk(*, L: float, n: int) -> Geometry:
    k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)
    z = jnp.zeros((n, n), dtype=jnp.float64)
    o = jnp.ones((n, n), dtype=jnp.float64)
    return Geometry(
        chi=o,
        ring=z,
        ring_accounting=z,
        sigma_xx=z,
        sigma_yy=z,
        sigma_xy=z,
        k_sq=k_sq,
        kx_sq=kx_sq,
        ky_sq=ky_sq,
        kx_wave=kx_wave,
        ky_wave=ky_wave,
        k_four=k_four,
        rv=z,
        dx=L / n,
        L=L,
        R=0.0,
        n=n,
        xc=0.5 * L,
        yc=0.5 * L,
    )


def _geometry_stage1(*, L: float, R: float, n: int) -> Geometry:
    m = circular_cavity_masks(L=L, R=R, n=n, eps_scale=2.0)
    k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)
    sxx, syy, sxy = stress_none(L=L, n=n)
    return Geometry(
        chi=m["chi"],
        ring=m["ring"],
        ring_accounting=m["ring_accounting"],
        sigma_xx=sxx,
        sigma_yy=syy,
        sigma_xy=sxy,
        k_sq=k_sq,
        kx_sq=kx_sq,
        ky_sq=ky_sq,
        kx_wave=kx_wave,
        ky_wave=ky_wave,
        k_four=k_four,
        rv=m["rv"],
        dx=float(m["dx"]),
        L=float(m["L"]),
        R=float(m["R"]),
        n=int(m["n"]),
        xc=float(m["xc"]),
        yc=float(m["yc"]),
    )


def test_imex_step_preserves_shape_and_dtype() -> None:
    L, n = 6.0, 24
    geom = _geometry_bulk(L=L, n=n)
    prm = SimParams(
        reaction_active=False,
        dirichlet_active=False,
        stress_coupling_B=0.0,
        gamma=0.0,
    )
    phi_m = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    phi_c = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    c = 0.3 * jnp.ones((n, n), dtype=jnp.float64)
    (pm, pc, cc), delta = imex_step((phi_m, phi_c, c), geom, prm, 1e-3)
    assert pm.shape == pc.shape == cc.shape == (n, n)
    assert pm.dtype == jnp.float64
    assert delta.shape == (2,)


def test_stage2_no_reaction_no_dirichlet_near_equilibrium() -> None:
    """Stage II flags: uniform fields with ``γ=0`` stay numerically flat."""
    L, n = 5.0, 32
    geom = _geometry_bulk(L=L, n=n)
    prm = SimParams(
        reaction_active=False,
        dirichlet_active=False,
        gamma=0.0,
        stress_coupling_B=0.0,
        M_m=0.05,
        M_c=0.05,
        D_c=0.05,
    )
    phi_m = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    phi_c = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    c = 0.4 * jnp.ones((n, n), dtype=jnp.float64)
    dt = 1e-4
    (pm, pc, cc), _ = imex_step((phi_m, phi_c, c), geom, prm, dt)
    assert jnp.allclose(pm, phi_m, rtol=0, atol=1e-9)
    assert jnp.allclose(pc, phi_c, rtol=0, atol=1e-9)
    assert jnp.allclose(cc, c, rtol=0, atol=1e-9)


def test_stage1_smoke_one_step_finite() -> None:
    L, R, n = 4.0, 0.9, 20
    geom = _geometry_stage1(L=L, R=R, n=n)
    prm = SimParams(
        reaction_active=True,
        dirichlet_active=True,
        c_sat=0.2,
        c0=0.5,
        k_rxn=0.5,
        stress_coupling_B=0.0,
        M_m=0.02,
        M_c=0.02,
        D_c=0.02,
    )
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    phi_m = 0.2 + 0.05 * jax.random.normal(k1, (n, n), dtype=jnp.float64)
    phi_c = 0.2 + 0.05 * jax.random.normal(k2, (n, n), dtype=jnp.float64)
    c = 0.35 * jnp.ones((n, n), dtype=jnp.float64)
    (pm, pc, cc), _ = imex_step((phi_m, phi_c, c), geom, prm, 1e-4)
    assert jnp.all(jnp.isfinite(pm))
    assert jnp.all(jnp.isfinite(pc))
    assert jnp.all(jnp.isfinite(cc))


def test_silica_mass_weighted_by_chi_conserved_stage2() -> None:
    """With ``G=0`` and uniform equilibrium, ``∫ χ (c + ρ_m φ_m + ρ_c φ_c)`` is unchanged."""
    L, n = 4.0, 28
    geom = _geometry_bulk(L=L, n=n)
    prm = SimParams(
        reaction_active=False,
        dirichlet_active=False,
        gamma=0.0,
        stress_coupling_B=0.0,
        rho_m=1.0,
        rho_c=1.0,
        M_m=0.1,
        M_c=0.1,
        D_c=0.1,
    )
    phi_m = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    phi_c = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    c = 0.25 * jnp.ones((n, n), dtype=jnp.float64)
    dx = L / n
    chi = geom.chi
    before = jnp.sum(chi * (c + prm.rho_m * phi_m + prm.rho_c * phi_c)) * (dx * dx)
    (pm, pc, cc), _ = imex_step((phi_m, phi_c, c), geom, prm, 5e-4)
    after = jnp.sum(chi * (cc + prm.rho_m * pm + prm.rho_c * pc)) * (dx * dx)
    assert float(jnp.abs(after - before)) < 1e-10


def test_mass_conservation_closed_system_stage2() -> None:
    """No reaction / Dirichlet: solid silica integral is preserved (CH + stress path)."""
    L, n = 10.0, 32
    geom = _geometry_stage2(L=L, n=n)
    prm = SimParams(
        reaction_active=False,
        dirichlet_active=False,
        kappa_x=0.5,
        kappa_y=0.5,
        gamma=1.0,
        M_m=0.1,
        M_c=1.0,
        stress_coupling_B=0.0,
    )
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    phi_m = 0.5 + 0.1 * jax.random.normal(k1, (n, n), dtype=jnp.float64)
    phi_c = 0.5 + 0.1 * jax.random.normal(k2, (n, n), dtype=jnp.float64)
    c = jnp.zeros((n, n), dtype=jnp.float64)
    state = (phi_m, phi_c, c)

    def integral(s: tuple) -> jax.Array:
        pm, pc, _cc = s
        dx = L / n
        return jnp.sum(prm.rho_m * pm + prm.rho_c * pc) * (dx * dx)

    mass_0 = integral(state)
    for _ in range(10):
        state, _ = imex_step(state, geom, prm, dt=0.001)
    mass_final = integral(state)
    rel_err = jnp.abs(mass_final - mass_0) / jnp.maximum(jnp.abs(mass_0), 1e-30)
    assert float(rel_err) < 1e-10


def test_mass_conservation_closed_system_stage2_with_psi_stress() -> None:
    """Uniform biaxial ψ-stress must not inject silica when ``G = 0`` (Phase 2 bug class)."""
    L, n = 10.0, 32
    k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)
    z = jnp.zeros((n, n), dtype=jnp.float64)
    sxx, syy, sxy = uniform_biaxial(L=L, n=n, sigma_0=0.1)
    geom = Geometry(
        chi=jnp.ones((n, n), dtype=jnp.float64),
        ring=z,
        ring_accounting=z,
        sigma_xx=sxx,
        sigma_yy=syy,
        sigma_xy=sxy,
        k_sq=k_sq,
        kx_sq=kx_sq,
        ky_sq=ky_sq,
        kx_wave=kx_wave,
        ky_wave=ky_wave,
        k_four=k_four,
        rv=z,
        dx=L / n,
        L=L,
        R=0.0,
        n=n,
        xc=0.5 * L,
        yc=0.5 * L,
    )
    prm = SimParams(
        reaction_active=False,
        dirichlet_active=False,
        kappa_x=0.5,
        kappa_y=0.5,
        gamma=1.0,
        M_m=0.1,
        M_c=1.0,
        stress_coupling_B=0.5,
    )
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    phi_m = 0.5 + 0.08 * jax.random.normal(k1, (n, n), dtype=jnp.float64)
    phi_c = 0.5 + 0.08 * jax.random.normal(k2, (n, n), dtype=jnp.float64)
    c = jnp.zeros((n, n), dtype=jnp.float64)
    state = (phi_m, phi_c, c)

    def integral(s: tuple) -> jax.Array:
        pm, pc, _cc = s
        dx = L / n
        return jnp.sum(prm.rho_m * pm + prm.rho_c * pc) * (dx * dx)

    mass_0 = integral(state)
    for _ in range(10):
        state, _ = imex_step(state, geom, prm, dt=0.001)
    mass_final = integral(state)
    rel_err = jnp.abs(mass_final - mass_0) / jnp.maximum(jnp.abs(mass_0), 1e-30)
    assert float(rel_err) < 1e-10


def test_mass_conservation_closed_system_with_reaction() -> None:
    """Reaction redistributes silica with ``ψ_m + ψ_c = 1``; total ``c + ρφ`` conserved."""
    L, n = 10.0, 32
    geom = _geometry_stage2(L=L, n=n)
    prm = SimParams(
        reaction_active=True,
        dirichlet_active=False,
        kappa_x=0.5,
        kappa_y=0.5,
        gamma=1.0,
        M_m=0.1,
        M_c=1.0,
        D_c=0.1,
        k_rxn=0.5,
        c_sat=0.0,
        stress_coupling_B=0.0,
        use_ratchet=False,
    )
    key = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(key, 3)
    phi_m = 0.3 + 0.05 * jax.random.normal(k1, (n, n), dtype=jnp.float64)
    phi_c = 0.3 + 0.05 * jax.random.normal(k2, (n, n), dtype=jnp.float64)
    c = 0.5 + 0.1 * jax.random.normal(k3, (n, n), dtype=jnp.float64)
    state = (phi_m, phi_c, c)

    def total_silica(s: tuple) -> jax.Array:
        pm, pc, cc = s
        dx = L / n
        return jnp.sum(cc + prm.rho_m * pm + prm.rho_c * pc) * (dx * dx)

    mass_0 = total_silica(state)
    for _ in range(10):
        state, _ = imex_step(state, geom, prm, dt=0.001)
    mass_final = total_silica(state)
    rel_err = jnp.abs(mass_final - mass_0) / jnp.maximum(jnp.abs(mass_0), 1e-30)
    assert float(rel_err) < 1e-10
