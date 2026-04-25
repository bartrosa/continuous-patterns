"""Single IMEX time step for Model C CH + reaction + optional stress.

One ``imex_step(state, geom, prm, dt)`` shared by Stage I and Stage II; branches
on ``SimParams.reaction_active`` / ``SimParams.dirichlet_active`` and on whether
stress coupling is active (``stress_coupling_B`` and non-zero ``σ``), using
:func:`jax.lax.cond` for JIT-friendly predicates (``docs/ARCHITECTURE.md`` §3.4).

Semi-discrete scheme (``docs/PHYSICS.md`` §8.2): implicit Laplacian on ``c``
and linear stiff Cahn--Hilliard symbol (anisotropic ``κ``) in Fourier space;
nonlinear bulk, barrier, reaction ``G``, and ψ-split stress are explicit.

Phase updates use a short Python ``for`` loop over the four phase slots
(``φ_m``, ``φ_c``, ``φ_q``, ``φ_imp``); ``pot.active`` and ``pot.kind`` are
Python-level so XLA sees a fixed graph per :class:`SimParams` instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from continuous_patterns.core.potentials import POTENTIAL_BUILDERS, barrier_prime
from continuous_patterns.core.stress import mu_stress_real
from continuous_patterns.core.types import PhasePotentialParams


@dataclass(frozen=True)
class Geometry:
    """Per-cell masks, spectral symbols, and prescribed stress (ARCHITECTURE §2.1)."""

    chi: Array
    ring: Array
    ring_accounting: Array
    sigma_xx: Array
    sigma_yy: Array
    sigma_xy: Array
    k_sq: Array
    kx_sq: Array
    ky_sq: Array
    kx_wave: Array
    ky_wave: Array
    k_four: Array
    rv: Array
    dx: float
    L: float
    R: float
    n: int
    xc: float
    yc: float


@dataclass(frozen=True)
class SimParams:
    """Minimal physics bundle for one IMEX step (extend in Phase 4+)."""

    reaction_active: bool = True
    dirichlet_active: bool = True
    D_c: float = 1.0
    phi_m_potential: PhasePotentialParams = field(
        default_factory=lambda: PhasePotentialParams(
            kind="double_well",
            W=1.0,
            mobility=1.0,
            rho=1.0,
            psi_sign=1.0,
            active=True,
        )
    )
    phi_c_potential: PhasePotentialParams = field(
        default_factory=lambda: PhasePotentialParams(
            kind="double_well",
            W=1.0,
            mobility=1.0,
            rho=1.0,
            psi_sign=-1.0,
            active=True,
        )
    )
    phi_q_potential: PhasePotentialParams = field(
        default_factory=lambda: PhasePotentialParams(
            kind="double_well",
            W=2.0,
            mobility=0.1,
            rho=1.0,
            psi_sign=0.0,
            active=False,
        )
    )
    phi_imp_potential: PhasePotentialParams = field(
        default_factory=lambda: PhasePotentialParams(
            kind="zero",
            W=1.0,
            mobility=0.0,
            rho=0.0,
            psi_sign=0.0,
            active=False,
        )
    )
    aging_active: bool = False
    k_age: float = 0.0
    q_to_quartz: float = 0.0
    gamma: float = 1.0
    kappa_x: float = 1.0
    kappa_y: float = 1.0
    stress_coupling_B: float = 0.0
    k_rxn: float = 1.0
    c_sat: float = 0.0
    c0: float = 1.0
    lambda_bar: float = 10.0
    c_ostwald: float = 0.5
    w_ostwald: float = 0.1
    use_ratchet: bool = True
    phi_m_ratchet_low: float = 0.3
    phi_m_ratchet_high: float = 0.5


def _potential_kwargs_for_kind(pot: PhasePotentialParams) -> dict[str, float]:
    """Kwargs for :data:`~continuous_patterns.core.potentials.POTENTIAL_BUILDERS`[``pot.kind``]."""
    if pot.kind == "double_well":
        return {"W": pot.W}
    if pot.kind == "tilted_well":
        return {"W": pot.W, "tilt": pot.tilt}
    if pot.kind == "asymmetric_well":
        return {"W": pot.W, "phi_left": pot.phi_left, "phi_right": pot.phi_right}
    if pot.kind == "zero":
        return {}
    raise ValueError(f"Unknown potential kind {pot.kind!r}")


def _smoothstep_S(phi_m: Array, low: float, high: float) -> Array:
    """Smoothstep ``S`` on ``[low, high]`` (PHYSICS §3 ratchet)."""
    eps = jnp.asarray(1e-12, dtype=phi_m.dtype)
    span = jnp.maximum(jnp.asarray(high - low, dtype=phi_m.dtype), eps)
    t = (phi_m - low) / span
    t = jnp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _G(
    c: Array,
    phi_m: Array,
    phi_c: Array,
    phi_q: Array,
    phi_imp: Array,
    prm: SimParams,
) -> Array:
    """Intrinsic precipitation rate with packing ceiling on all solid fractions (no ``χ``).

    Ostwald partition still uses only ``(ψ_m, ψ_c)`` vs pre-step ``φ_m`` (PHYSICS §3);
    α-quartz is not fed by this channel — only by aging when configured.
    """
    relu_c = jnp.maximum(c - prm.c_sat, 0.0)
    relu_p = jnp.maximum(1.0 - phi_m - phi_c - phi_q - phi_imp, 0.0)
    return prm.k_rxn * relu_c * relu_p


def _G_aging(phi_m: Array, c: Array, prm: SimParams) -> Array:
    """Aging rate: ``G_age = k_age · φ_m · max(c_sat - c, 0)`` (no ``χ`` here)."""
    undersat = jnp.maximum(prm.c_sat - c, 0.0)
    return prm.k_age * phi_m * undersat


def _psi_ostwald(c: Array, phi_m: Array, prm: SimParams) -> tuple[Array, Array]:
    """Ostwald partition with optional ratchet (PHYSICS §3)."""
    t = (c - prm.c_ostwald) / jnp.maximum(prm.w_ostwald, 1e-12)
    psi_m_base = jax.nn.sigmoid(t)

    def _with_ratchet(_: Array) -> tuple[Array, Array]:
        S = _smoothstep_S(phi_m, prm.phi_m_ratchet_low, prm.phi_m_ratchet_high)
        pm = psi_m_base + (1.0 - psi_m_base) * S
        pm = jnp.clip(pm, 0.0, 1.0)
        return pm, jnp.clip(1.0 - pm, 0.0, 1.0)

    def _no_ratchet(_: Array) -> tuple[Array, Array]:
        return jnp.clip(psi_m_base, 0.0, 1.0), jnp.clip(1.0 - psi_m_base, 0.0, 1.0)

    operand = jnp.asarray(0.0, dtype=phi_m.dtype)
    return jax.lax.cond(jnp.asarray(prm.use_ratchet), _with_ratchet, _no_ratchet, operand)


def _sigma_active(geom: Geometry) -> Array:
    s = (
        jnp.max(jnp.abs(geom.sigma_xx))
        + jnp.max(jnp.abs(geom.sigma_yy))
        + jnp.max(jnp.abs(geom.sigma_xy))
    )
    return s > 1e-14


def _psi_linear_order_parameter(
    phi_m: Array,
    phi_c: Array,
    phi_q: Array,
    phi_imp: Array,
    prm: SimParams,
) -> Array:
    """``ψ = Σ_α ψ_sign_α φ_α`` over **active** phases only (Python loop; static per ``prm``)."""
    acc = jnp.zeros_like(phi_m)
    for phi, pot in (
        (phi_m, prm.phi_m_potential),
        (phi_c, prm.phi_c_potential),
        (phi_q, prm.phi_q_potential),
        (phi_imp, prm.phi_imp_potential),
    ):
        if pot.active:
            acc = acc + jnp.asarray(pot.psi_sign, dtype=phi.dtype) * phi
    return acc


def _gamma_cross_sum_other(
    phi_m: Array,
    phi_c: Array,
    phi_q: Array,
    phi_imp: Array,
    prm: SimParams,
    *,
    skip: str,
) -> Array:
    """``γ Σ_{β≠α, β active} φ_β`` for the phase named ``skip``."""
    pairs: tuple[tuple[str, Array, PhasePotentialParams], ...] = (
        ("phi_m", phi_m, prm.phi_m_potential),
        ("phi_c", phi_c, prm.phi_c_potential),
        ("phi_q", phi_q, prm.phi_q_potential),
        ("phi_imp", phi_imp, prm.phi_imp_potential),
    )
    acc = jnp.zeros_like(phi_m)
    for name, phi, pot in pairs:
        if name == skip:
            continue
        if pot.active:
            acc = acc + phi
    return prm.gamma * acc


def _stress_delta(mu_stress: Array, pot: PhasePotentialParams) -> Array:
    """``δμ^stress_α = ½ ψ_sign_α μ_stress`` (zero if inactive)."""
    if not pot.active:
        return jnp.zeros_like(mu_stress)
    half = jnp.asarray(0.5, dtype=mu_stress.dtype)
    return half * jnp.asarray(pot.psi_sign, dtype=mu_stress.dtype) * mu_stress


def _update_phase(
    phi: Array,
    phi_other_sum: Array,
    chi: Array,
    geom: Geometry,
    prm: SimParams,
    dt: float,
    pot: PhasePotentialParams,
    stress_delta: Array,
) -> Array:
    """Implicit linear CH stiff block in Fourier space with explicit ``μ_nl``."""
    stiff_sym = prm.kappa_x * geom.kx_sq + prm.kappa_y * geom.ky_sq
    builder = POTENTIAL_BUILDERS[pot.kind]
    kwargs = _potential_kwargs_for_kind(pot)
    df = builder(phi, **kwargs)
    bar = barrier_prime(phi, lambda_bar=prm.lambda_bar)
    mu_nl = df + bar + phi_other_sum + stress_delta
    phi_hat = jnp.fft.fft2(phi)
    nl_hat = jnp.fft.fft2(mu_nl)
    den = 1.0 + dt * pot.mobility * geom.k_sq * stiff_sym
    phi_new_hat = (phi_hat - dt * pot.mobility * geom.k_sq * nl_hat) / den
    phi_new = jnp.real(jnp.fft.ifft2(phi_new_hat))
    return (1.0 - chi) * phi + chi * phi_new


def imex_step(
    state: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    geom: Geometry,
    prm: SimParams,
    dt: float,
) -> tuple[tuple[Array, Array, Array, Array, Array], tuple[Array, Array]]:
    r"""Advance ``(φ_m, φ_c, φ_q, φ_\mathrm{imp}, c)`` by one IMEX step of length ``dt``.

    Parameters
    ----------
    state
        ``(phi_m, phi_c, phi_q, phi_imp, c)`` each ``(n, n)``.
    geom
        Masks and spectral symbols.
    prm
        Physics flags and coefficients.
    dt
        Time step (positive).

    Returns
    -------
    tuple
        ``((phi_m', phi_c', phi_q', phi_imp', c'), (delta_pair, injection_this_step))``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    phi_m = jnp.asarray(state[0])
    phi_c = jnp.asarray(state[1])
    phi_q = jnp.asarray(state[2])
    phi_imp = jnp.asarray(state[3])
    c = jnp.asarray(state[4])
    chi = geom.chi

    G = jax.lax.cond(
        jnp.asarray(prm.reaction_active),
        lambda cc: _G(cc[0], cc[1], cc[2], cc[3], cc[4], prm),
        lambda cc: jnp.zeros_like(cc[0]),
        (c, phi_m, phi_c, phi_q, phi_imp),
    )

    c_hat = jnp.fft.fft2(c)
    g_hat = jnp.fft.fft2(chi * G)
    c_den = 1.0 + dt * prm.D_c * geom.k_sq
    c_new_hat = (c_hat - dt * g_hat) / c_den
    c_new = jnp.real(jnp.fft.ifft2(c_new_hat))
    c_new = (1.0 - chi) * c + chi * c_new

    stress_pred = jnp.logical_and(
        jnp.asarray(prm.stress_coupling_B > 1e-20),
        _sigma_active(geom),
    )
    psi_op = _psi_linear_order_parameter(phi_m, phi_c, phi_q, phi_imp, prm)

    def _mu_st_on(_: Array) -> Array:
        return mu_stress_real(
            psi_op,
            geom.sigma_xx,
            geom.sigma_xy,
            geom.sigma_yy,
            geom.kx_wave,
            geom.ky_wave,
            prm.stress_coupling_B,
        )

    def _mu_st_off(_: Array) -> Array:
        return jnp.zeros_like(phi_m)

    zop = jnp.asarray(0.0, dtype=phi_m.dtype)
    mu_stress = jax.lax.cond(stress_pred, _mu_st_on, _mu_st_off, zop)

    phases: tuple[tuple[str, Array, PhasePotentialParams], ...] = (
        ("phi_m", phi_m, prm.phi_m_potential),
        ("phi_c", phi_c, prm.phi_c_potential),
        ("phi_q", phi_q, prm.phi_q_potential),
        ("phi_imp", phi_imp, prm.phi_imp_potential),
    )
    new_phis: dict[str, Array] = {}
    for name, phi, pot in phases:
        if not pot.active:
            new_phis[name] = phi
            continue
        cross = _gamma_cross_sum_other(phi_m, phi_c, phi_q, phi_imp, prm, skip=name)
        sd = _stress_delta(mu_stress, pot)
        new_phis[name] = _update_phase(phi, cross, chi, geom, prm, dt, pot, sd)

    phi_m_new = new_phis["phi_m"]
    phi_c_new = new_phis["phi_c"]
    phi_q_new = new_phis["phi_q"]
    phi_imp_new = new_phis["phi_imp"]

    psi_m, psi_c = _psi_ostwald(c, phi_m, prm)
    phi_m_new = phi_m_new + dt * chi * psi_m * G
    phi_c_new = phi_c_new + dt * chi * psi_c * G

    G_age = jax.lax.cond(
        jnp.asarray(prm.aging_active),
        lambda _: _G_aging(phi_m, c, prm),
        lambda _: jnp.zeros_like(phi_m),
        jnp.asarray(0.0, dtype=phi_m.dtype),
    )
    phi_m_new = phi_m_new - dt * chi * G_age
    phi_c_new = phi_c_new + dt * chi * (1.0 - prm.q_to_quartz) * G_age
    phi_q_new = phi_q_new + dt * chi * prm.q_to_quartz * G_age

    c_before_rim = c_new
    c_new = jax.lax.cond(
        jnp.asarray(prm.dirichlet_active),
        lambda z: (1.0 - geom.ring) * z + geom.ring * jnp.asarray(prm.c0, dtype=z.dtype),
        lambda z: z,
        c_new,
    )
    dx_arr = jnp.asarray(geom.dx, dtype=c_new.dtype)
    injection_this_step = jnp.sum(geom.chi * (c_new - c_before_rim)) * (dx_arr * dx_arr)

    lo = jnp.asarray(-0.05, dtype=phi_m_new.dtype)
    hi = jnp.asarray(1.05, dtype=phi_m_new.dtype)
    phi_m_new = jnp.clip(phi_m_new, lo, hi)
    phi_c_new = jnp.clip(phi_c_new, lo, hi)
    phi_q_new = jnp.clip(phi_q_new, lo, hi)
    phi_imp_new = jnp.clip(phi_imp_new, lo, hi)

    delta_pair = jnp.zeros((2,), dtype=c_new.dtype)
    return (phi_m_new, phi_c_new, phi_q_new, phi_imp_new, c_new), (delta_pair, injection_this_step)
