"""Unit tests for :mod:`continuous_patterns.core.imex`."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import pytest

from continuous_patterns.core.imex import _G, Geometry, SimParams, imex_step
from continuous_patterns.core.io import apply_physics_phases_legacy_shim
from continuous_patterns.core.masks import circular_cavity_masks
from continuous_patterns.core.spectral import k_vectors
from continuous_patterns.core.stress import none as stress_none
from continuous_patterns.core.stress import uniform_biaxial
from continuous_patterns.core.types import PhasePotentialParams
from continuous_patterns.models.agate_ch import build_sim_params, phase_potential_params_from_spec


def _s5(
    phi_m: jax.Array, phi_c: jax.Array, c: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    z = jnp.zeros_like(phi_m)
    return (phi_m, phi_c, z, z, c)


def _pm(
    *,
    W: float = 1.0,
    mobility: float = 1.0,
    rho: float = 1.0,
    psi_sign: float = 1.0,
    active: bool = True,
) -> PhasePotentialParams:
    return PhasePotentialParams(
        kind="double_well",
        W=W,
        mobility=mobility,
        rho=rho,
        psi_sign=psi_sign,
        active=active,
    )


def _pc(
    *,
    W: float = 1.0,
    mobility: float = 1.0,
    rho: float = 1.0,
    psi_sign: float = -1.0,
    active: bool = True,
) -> PhasePotentialParams:
    return PhasePotentialParams(
        kind="double_well",
        W=W,
        mobility=mobility,
        rho=rho,
        psi_sign=psi_sign,
        active=active,
    )


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
    st0 = _s5(phi_m, phi_c, c)
    (pm, pc, pq, pim, cc), (delta_pair, injection) = imex_step(st0, geom, prm, 1e-3)
    assert pm.shape == pc.shape == pq.shape == pim.shape == cc.shape == (n, n)
    assert pm.dtype == jnp.float64
    assert float(jnp.max(jnp.abs(pq))) == 0.0
    assert float(jnp.max(jnp.abs(pim))) == 0.0
    assert delta_pair.shape == (2,)
    assert injection.shape == ()
    assert float(injection) == 0.0


def test_stage2_no_reaction_no_dirichlet_near_equilibrium() -> None:
    """Stage II flags: uniform fields with ``γ=0`` stay numerically flat."""
    L, n = 5.0, 32
    geom = _geometry_bulk(L=L, n=n)
    prm = SimParams(
        reaction_active=False,
        dirichlet_active=False,
        gamma=0.0,
        stress_coupling_B=0.0,
        phi_m_potential=_pm(mobility=0.05),
        phi_c_potential=_pc(mobility=0.05),
        D_c=0.05,
    )
    phi_m = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    phi_c = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    c = 0.4 * jnp.ones((n, n), dtype=jnp.float64)
    dt = 1e-4
    (pm, pc, pq, pim, cc), _pair = imex_step(_s5(phi_m, phi_c, c), geom, prm, dt)
    assert jnp.allclose(pm, phi_m, rtol=0, atol=1e-9)
    assert jnp.allclose(pc, phi_c, rtol=0, atol=1e-9)
    assert jnp.allclose(cc, c, rtol=0, atol=1e-9)
    assert jnp.allclose(pq, 0.0, atol=0.0)
    assert jnp.allclose(pim, 0.0, atol=0.0)


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
        phi_m_potential=_pm(mobility=0.02),
        phi_c_potential=_pc(mobility=0.02),
        D_c=0.02,
    )
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    phi_m = 0.2 + 0.05 * jax.random.normal(k1, (n, n), dtype=jnp.float64)
    phi_c = 0.2 + 0.05 * jax.random.normal(k2, (n, n), dtype=jnp.float64)
    c = 0.35 * jnp.ones((n, n), dtype=jnp.float64)
    (pm, pc, pq, pim, cc), _pair = imex_step(_s5(phi_m, phi_c, c), geom, prm, 1e-4)
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
        phi_m_potential=_pm(mobility=0.1, rho=1.0),
        phi_c_potential=_pc(mobility=0.1, rho=1.0),
        D_c=0.1,
    )
    phi_m = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    phi_c = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    c = 0.25 * jnp.ones((n, n), dtype=jnp.float64)
    dx = L / n
    chi = geom.chi
    rm = float(prm.phi_m_potential.rho)
    rc = float(prm.phi_c_potential.rho)
    before = jnp.sum(chi * (c + rm * phi_m + rc * phi_c)) * (dx * dx)
    (pm, pc, pq, pim, cc), _pair = imex_step(_s5(phi_m, phi_c, c), geom, prm, 5e-4)
    assert jnp.allclose(pq, 0.0)
    assert jnp.allclose(pim, 0.0)
    after = jnp.sum(chi * (cc + rm * pm + rc * pc)) * (dx * dx)
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
        phi_m_potential=_pm(mobility=0.1),
        phi_c_potential=_pc(mobility=1.0),
        stress_coupling_B=0.0,
    )
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    phi_m = 0.5 + 0.1 * jax.random.normal(k1, (n, n), dtype=jnp.float64)
    phi_c = 0.5 + 0.1 * jax.random.normal(k2, (n, n), dtype=jnp.float64)
    c = jnp.zeros((n, n), dtype=jnp.float64)
    state = _s5(phi_m, phi_c, c)

    def integral(s: tuple) -> jax.Array:
        pm, pc, pq, pim, _cc = s
        dx = L / n
        return jnp.sum(prm.phi_m_potential.rho * pm + prm.phi_c_potential.rho * pc) * (dx * dx)

    mass_0 = integral(state)
    for _ in range(10):
        state, _pair = imex_step(state, geom, prm, dt=0.001)
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
        phi_m_potential=_pm(mobility=0.1),
        phi_c_potential=_pc(mobility=1.0),
        stress_coupling_B=0.5,
    )
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    phi_m = 0.5 + 0.08 * jax.random.normal(k1, (n, n), dtype=jnp.float64)
    phi_c = 0.5 + 0.08 * jax.random.normal(k2, (n, n), dtype=jnp.float64)
    c = jnp.zeros((n, n), dtype=jnp.float64)
    state = _s5(phi_m, phi_c, c)

    def integral(s: tuple) -> jax.Array:
        pm, pc, _pq, _pim, _cc = s
        dx = L / n
        return jnp.sum(prm.phi_m_potential.rho * pm + prm.phi_c_potential.rho * pc) * (dx * dx)

    mass_0 = integral(state)
    for _ in range(10):
        state, _pair = imex_step(state, geom, prm, dt=0.001)
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
        phi_m_potential=_pm(mobility=0.1),
        phi_c_potential=_pc(mobility=1.0),
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
    state = _s5(phi_m, phi_c, c)

    def total_silica(s: tuple) -> jax.Array:
        pm, pc, pq, pim, cc = s
        dx = L / n
        return jnp.sum(
            cc
            + prm.phi_m_potential.rho * pm
            + prm.phi_c_potential.rho * pc
            + prm.phi_q_potential.rho * pq
            + prm.phi_imp_potential.rho * pim
        ) * (dx * dx)

    mass_0 = total_silica(state)
    for _ in range(10):
        state, _pair = imex_step(state, geom, prm, dt=0.001)
    mass_final = total_silica(state)
    rel_err = jnp.abs(mass_final - mass_0) / jnp.maximum(jnp.abs(mass_0), 1e-30)
    assert float(rel_err) < 1e-10


def test_imex_legacy_physics_matches_explicit_phase_params() -> None:
    """``SimParams`` from legacy flat keys vs explicit ``PhasePotentialParams`` (fp64)."""
    L, n = 8.0, 24
    geom = _geometry_bulk(L=L, n=n)
    physics = {
        "gamma": 1.2,
        "kappa_x": 0.4,
        "kappa_y": 0.6,
        "D_c": 0.08,
        "k_rxn": 0.0,
        "c_sat": 0.1,
        "c_0": 0.0,
        "lambda_bar": 9.0,
        "c_ostwald": 0.4,
        "w_ostwald": 0.12,
        "use_ratchet": False,
        "reaction_active": False,
        "dirichlet_active": False,
        "W": 1.25,
        "M_m": 0.07,
        "M_c": 0.11,
        "rho_m": 1.03,
        "rho_c": 0.97,
    }
    cfg = {
        "physics": copy.deepcopy(physics),
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
    }
    apply_physics_phases_legacy_shim(cfg["physics"])
    prm_legacy = build_sim_params(cfg)

    phases = cfg["physics"]["phases"]
    prm_explicit = SimParams(
        reaction_active=False,
        dirichlet_active=False,
        gamma=1.2,
        kappa_x=0.4,
        kappa_y=0.6,
        D_c=0.08,
        k_rxn=0.0,
        c_sat=0.1,
        c0=0.0,
        lambda_bar=9.0,
        c_ostwald=0.4,
        w_ostwald=0.12,
        use_ratchet=False,
        phi_m_potential=phase_potential_params_from_spec(phases["moganite"]),
        phi_c_potential=phase_potential_params_from_spec(phases["chalcedony"]),
        stress_coupling_B=0.0,
    )
    assert prm_legacy == prm_explicit

    key = jax.random.PRNGKey(11)
    k1, k2, k3 = jax.random.split(key, 3)
    phi_m = 0.4 + 0.12 * jax.random.normal(k1, (n, n), dtype=jnp.float64)
    phi_c = 0.35 + 0.12 * jax.random.normal(k2, (n, n), dtype=jnp.float64)
    c = 0.2 + 0.08 * jax.random.normal(k3, (n, n), dtype=jnp.float64)
    state0 = _s5(phi_m, phi_c, c)
    s_leg = state0
    s_exp = state0
    for _ in range(10):
        s_leg, _ = imex_step(s_leg, geom, prm_legacy, 0.002)
        s_exp, _ = imex_step(s_exp, geom, prm_explicit, 0.002)
    assert jnp.allclose(s_leg[0], s_exp[0], rtol=1e-12, atol=1e-12)
    assert jnp.allclose(s_leg[1], s_exp[1], rtol=1e-12, atol=1e-12)
    assert jnp.allclose(s_leg[2], s_exp[2], rtol=1e-12, atol=1e-12)
    assert jnp.allclose(s_leg[3], s_exp[3], rtol=1e-12, atol=1e-12)
    assert jnp.allclose(s_leg[4], s_exp[4], rtol=1e-12, atol=1e-12)


def test_step2_default_slice_matches_duplicate_run() -> None:
    """Inactive ``φ_q``, ``φ_imp`` with zero IC: deterministic duplicate trajectories."""
    L, n = 6.0, 20
    geom = _geometry_bulk(L=L, n=n)
    prm = SimParams(reaction_active=False, dirichlet_active=False, gamma=0.5, stress_coupling_B=0.0)
    key = jax.random.PRNGKey(99)
    k1, k2, k3 = jax.random.split(key, 3)
    pm = 0.1 * jax.random.normal(k1, (n, n), dtype=jnp.float64)
    pc = 0.1 * jax.random.normal(k2, (n, n), dtype=jnp.float64)
    cc = 0.1 * jax.random.normal(k3, (n, n), dtype=jnp.float64)
    s0 = _s5(pm, pc, cc)
    a, b = s0, s0
    for _ in range(10):
        a, _ = imex_step(a, geom, prm, 0.002)
        b, _ = imex_step(b, geom, prm, 0.002)
    for i in range(5):
        assert jnp.allclose(a[i], b[i], rtol=0.0, atol=1e-15)


def test_aging_conserves_phase_mass() -> None:
    L, n = 4.0, 24
    geom = _geometry_bulk(L=L, n=n)
    prm = SimParams(
        reaction_active=False,
        dirichlet_active=False,
        gamma=0.0,
        stress_coupling_B=0.0,
        aging_active=True,
        k_age=0.1,
        q_to_quartz=0.0,
        c_sat=0.5,
        phi_m_potential=_pm(mobility=0.0, rho=1.0),
        phi_c_potential=_pc(mobility=0.0, rho=1.0),
        phi_q_potential=PhasePotentialParams(
            kind="double_well", W=1.0, mobility=0.0, rho=1.0, psi_sign=0.0, active=False
        ),
    )
    phi_m = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    phi_c = jnp.zeros((n, n), dtype=jnp.float64)
    c = jnp.zeros((n, n), dtype=jnp.float64)
    state = _s5(phi_m, phi_c, c)
    dx = L / n

    def solid_mass(s: tuple) -> jax.Array:
        pm, pc, _, _, _cc = s
        return jnp.sum(prm.phi_m_potential.rho * pm + prm.phi_c_potential.rho * pc) * (dx * dx)

    m0 = solid_mass(state)
    for _ in range(10):
        state, _ = imex_step(state, geom, prm, 0.05)
    m1 = solid_mass(state)
    assert float(jnp.abs(m1 - m0) / jnp.maximum(jnp.abs(m0), 1e-30)) < 1e-10
    assert float(jnp.mean(state[0])) < 0.5
    assert float(jnp.mean(state[1])) > 0.0


def test_aging_with_quartz_split() -> None:
    L, n = 4.0, 24
    geom = _geometry_bulk(L=L, n=n)
    prm = SimParams(
        reaction_active=False,
        dirichlet_active=False,
        gamma=0.0,
        stress_coupling_B=0.0,
        aging_active=True,
        k_age=0.1,
        q_to_quartz=0.5,
        c_sat=0.5,
        phi_m_potential=_pm(mobility=0.0, rho=1.0),
        phi_c_potential=_pc(mobility=0.0, rho=1.0),
        phi_q_potential=_pm(W=1.0, mobility=0.0, rho=1.0, psi_sign=0.0, active=True),
    )
    phi_m = 0.5 * jnp.ones((n, n), dtype=jnp.float64)
    phi_c = jnp.zeros((n, n), dtype=jnp.float64)
    z = jnp.zeros((n, n), dtype=jnp.float64)
    c = jnp.zeros((n, n), dtype=jnp.float64)
    state = (phi_m, phi_c, z, z, c)
    dx = L / n

    def split_mass(s: tuple) -> tuple[float, float]:
        pm, pc, pq, _, _ = s
        return float(jnp.sum(pc) * dx * dx), float(jnp.sum(pq) * dx * dx)

    c0, q0 = split_mass(state)
    for _ in range(10):
        state, _ = imex_step(state, geom, prm, 0.05)
    c1, q1 = split_mass(state)
    assert c1 > c0 and q1 > q0
    assert abs((c1 - c0) - (q1 - q0)) < 1e-6 * max(abs(c1 - c0), 1e-9)


def test_aging_requires_active_quartz() -> None:
    cfg = {
        "physics": {
            "kappa_x": 0.5,
            "kappa_y": 0.5,
            "D_c": 1.0,
            "gamma": 1.0,
            "k_rxn": 0.0,
            "c_sat": 0.5,
            "c_0": 0.0,
            "c_ostwald": 0.5,
            "w_ostwald": 0.1,
            "lambda_bar": 1.0,
            "phases": {
                "moganite": {"potential": "double_well", "potential_kwargs": {"W": 1.0}},
                "chalcedony": {"potential": "double_well", "potential_kwargs": {"W": 1.0}},
                "alpha_quartz": {
                    "potential": "double_well",
                    "potential_kwargs": {"W": 1.0},
                    "active": False,
                },
            },
            "aging": {"active": True, "k_age": 0.01, "q_to_quartz": 0.5},
        },
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
    }
    apply_physics_phases_legacy_shim(cfg["physics"])
    with pytest.raises(ValueError, match="alpha_quartz"):
        build_sim_params(cfg)


def test_packing_ceiling_includes_quartz_channel() -> None:
    """When α-quartz is active, its solid fraction enters the packing ceiling."""
    prm = SimParams(
        reaction_active=True,
        k_rxn=1.0,
        c_sat=0.0,
        phi_q_potential=PhasePotentialParams(
            kind="double_well",
            W=2.0,
            mobility=0.1,
            rho=1.0,
            psi_sign=0.0,
            active=True,
        ),
    )
    c = jnp.asarray(1.0)
    pm = jnp.asarray(0.45)
    pc = jnp.asarray(0.45)
    pq = jnp.asarray(0.2)
    pim = jnp.asarray(0.0)
    g = _G(c, pm, pc, pq, pim, prm)
    assert float(g) == 0.0


def test_G_positive_when_packing_allows() -> None:
    prm = SimParams(reaction_active=True, k_rxn=2.0, c_sat=0.0)
    c = jnp.asarray(1.0)
    pm = jnp.asarray(0.2)
    pc = jnp.asarray(0.2)
    pq = jnp.asarray(0.2)
    pim = jnp.asarray(0.0)
    g = _G(c, pm, pc, pq, pim, prm)
    assert float(g) > 0.0
