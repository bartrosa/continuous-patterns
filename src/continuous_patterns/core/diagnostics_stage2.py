"""Post-run diagnostics for Stage II bulk relaxation.

Structure factors, domain statistics, coarsening metrics — distinct from
rim/cavity tooling in ``diagnostics_stage1`` (``docs/ARCHITECTURE.md`` §3.6).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def structure_factor_radial_average(
    field: np.ndarray,
    *,
    L: float,
    n_bins: int | None = None,
) -> dict[str, Any]:
    """Circularly averaged ``S(|k|) ∝ ⟨|FFT(field)|²⟩`` on ``|k|`` shells."""
    n = int(field.shape[0])
    dx = L / n
    nb = n // 2 if n_bins is None else int(n_bins)
    nb = max(nb, 4)
    u = np.fft.fftshift(np.fft.fft2(field.astype(np.float64)))
    P = (np.abs(u) ** 2) / float(n * n)
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kxs = np.fft.fftshift(kx)
    kxm, kym = np.meshgrid(kxs, kxs, indexing="ij")
    kmag = np.sqrt(kxm * kxm + kym * kym).ravel()
    pv = P.ravel()
    kmax = float(np.max(kmag)) + 1e-12
    counts, edges = np.histogram(kmag, bins=nb, range=(0.0, kmax))
    weighted, _ = np.histogram(kmag, bins=nb, range=(0.0, kmax), weights=pv)
    shells = np.maximum(counts, 1e-30)
    S_avg = weighted / shells
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"k_shell_centers": centers, "S_radial_mean": S_avg, "shell_counts": counts}


def bulk_scalar_stats(phi_m: np.ndarray, phi_c: np.ndarray) -> dict[str, float]:
    """Spatial means / variances of phase fields (Stage II summaries)."""
    pm = phi_m.astype(np.float64)
    pc = phi_c.astype(np.float64)
    return {
        "mean_phi_m": float(np.mean(pm)),
        "mean_phi_c": float(np.mean(pc)),
        "var_phi_m": float(np.var(pm)),
        "var_phi_c": float(np.var(pc)),
        "mean_psi": float(np.mean(pm - pc)),
        "var_psi": float(np.var(pm - pc)),
    }


def _periodic_grad_sq_sum(f: np.ndarray, *, L: float) -> np.ndarray:
    """``|∇f|²`` with central differences and periodic wrap."""
    n = f.shape[0]
    dx = L / n
    fx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) * (0.5 / dx)
    fy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) * (0.5 / dx)
    return fx * fx + fy * fy


def interface_density(phi_m: np.ndarray, phi_c: np.ndarray, *, L: float) -> dict[str, float]:
    """Mean ``|∇φ_m| + |∇φ_c|`` proxy for interfacial length per area."""
    gm = np.sqrt(_periodic_grad_sq_sum(phi_m.astype(np.float64), L=L))
    gc = np.sqrt(_periodic_grad_sq_sum(phi_c.astype(np.float64), L=L))
    return {
        "mean_grad_mag_phi_m": float(np.mean(gm)),
        "mean_grad_mag_phi_c": float(np.mean(gc)),
        "mean_grad_mag_sum": float(np.mean(gm + gc)),
    }


def coarsening_metrics(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
) -> dict[str, Any]:
    """Bundle Stage II coarsening-oriented scalars (structure + interfaces)."""
    psi = phi_m.astype(np.float64) - phi_c.astype(np.float64)
    sf = structure_factor_radial_average(psi, L=L)
    iface = interface_density(phi_m, phi_c, L=L)
    return {
        "structure_factor_psi": sf,
        "interfaces": iface,
        "stats": bulk_scalar_stats(phi_m, phi_c),
    }
