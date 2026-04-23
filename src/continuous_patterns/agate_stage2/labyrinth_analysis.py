"""Analysis tools for labyrinth / isotropic spinodal patterns (no radial symmetry)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def spectral_peak_wavelength(
    phi_diff: np.ndarray,
    dx: float,
    *,
    n_bins: int | None = None,
    min_k_fraction: float = 0.02,
) -> dict[str, Any]:
    """Dominant length scale from the 2D power spectrum of ``phi_diff``.

    Radially averages :math:`|FFT|^2` in :math:`|k|`, finds the peak (excluding
    near-DC bins), returns :math:`k_{\\mathrm{peak}}`,
    :math:`\\lambda_{\\mathrm{peak}} = 2\\pi/k_{\\mathrm{peak}}`, and the
    binned radial spectrum.
    """
    phi = np.asarray(phi_diff, dtype=np.float64)
    phi = phi - np.mean(phi)
    n = int(phi.shape[0])
    if phi.shape != (n, n):
        raise ValueError("phi_diff must be square")

    f2 = np.fft.fft2(phi)
    psd = (np.abs(f2) ** 2) / float(n * n)

    k1d = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2)
    km_flat = kmag.ravel()
    ps_flat = psd.ravel()

    k_max = float(np.max(km_flat))
    nb = int(n_bins) if n_bins is not None else max(64, min(200, n * n // 8))
    edges = np.linspace(0.0, k_max, nb + 1)
    sum_p, _ = np.histogram(km_flat, bins=edges, weights=ps_flat)
    cnt, _ = np.histogram(km_flat, bins=edges)
    radial = np.divide(sum_p, np.maximum(cnt.astype(np.float64), 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])

    k_min_sel = float(min_k_fraction * max(np.max(centers), 1e-12))
    mask = centers >= k_min_sel
    if not np.any(mask):
        return {
            "k_peak": float("nan"),
            "lambda_peak": float("nan"),
            "k_centers": centers.tolist(),
            "radial_spectrum": radial.tolist(),
            "peak_index": None,
        }

    valid_idx = np.where(mask)[0]
    local_argmax = int(np.argmax(radial[mask]))
    i_peak = int(valid_idx[local_argmax])
    k_peak = float(centers[i_peak])
    lam = float(2.0 * np.pi / k_peak) if k_peak > 1e-14 else float("nan")

    return {
        "k_peak": k_peak,
        "lambda_peak": lam,
        "k_centers": centers.tolist(),
        "radial_spectrum": radial.tolist(),
        "peak_index": i_peak,
    }


def directional_anisotropy(
    phi_diff: np.ndarray,
    dx: float,
    *,
    k_peak: float | None = None,
    ring_width: float = 0.12,
    n_theta: int = 72,
) -> dict[str, Any]:
    """Azimuthal variation of power on an annulus near ``k_peak`` (isotropy diagnostic)."""
    if k_peak is not None and (not np.isfinite(k_peak) or float(k_peak) <= 0.0):
        k_peak = None

    phi = np.asarray(phi_diff, dtype=np.float64)
    phi = phi - np.mean(phi)
    n = int(phi.shape[0])
    if phi.shape != (n, n):
        raise ValueError("phi_diff must be square")

    spec = spectral_peak_wavelength(phi, dx)
    k0 = float(spec["k_peak"]) if k_peak is None else float(k_peak)
    if k0 != k0 or k0 <= 0.0:
        return {
            "anisotropy_ratio": float("nan"),
            "azimuthal_power": [],
            "theta_centers": [],
            "k_peak_used": k0,
            "note": "invalid k_peak",
        }

    f2 = np.fft.fft2(phi)
    psd = (np.abs(f2) ** 2) / float(n * n)

    k1d = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2)
    lo = k0 * (1.0 - ring_width)
    hi = k0 * (1.0 + ring_width)
    ring = (kmag >= lo) & (kmag <= hi) & (kmag > 1e-14)

    if not np.any(ring):
        return {
            "anisotropy_ratio": float("nan"),
            "azimuthal_power": [],
            "theta_centers": [],
            "k_peak_used": k0,
            "ring_k_lo": lo,
            "ring_k_hi": hi,
            "note": "empty k-ring",
        }

    theta = np.arctan2(ky, kx)
    theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
    powers, _ = np.histogram(
        theta[ring].ravel(),
        bins=theta_bins,
        weights=psd[ring].ravel(),
    )
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    pmax = float(np.max(powers)) if powers.size else float("nan")
    pmin = (
        float(np.min(powers[np.isfinite(powers) & (powers > 0)]))
        if np.any(powers > 0)
        else float("nan")
    )
    ratio = (pmax / pmin) if (pmin > 0 and np.isfinite(pmin)) else float("nan")

    return {
        "anisotropy_ratio": ratio,
        "azimuthal_power": powers.tolist(),
        "theta_centers": theta_centers.tolist(),
        "k_peak_used": k0,
        "ring_k_lo": lo,
        "ring_k_hi": hi,
    }


def contrast_amplitude(phi_diff: np.ndarray, *, edge_trim: int = 0) -> float:
    """Std.~dev. of ``phi_diff`` (optional edge trim for masked domains; periodic = 0)."""
    a = np.asarray(phi_diff, dtype=np.float64)
    if edge_trim > 0:
        e = edge_trim
        a = a[e:-e, e:-e]
    if a.size == 0:
        return float("nan")
    return float(np.std(a.ravel()))


def analyze_stage2_run(run_dir: Path) -> dict[str, Any]:
    """Load the final snapshot and write ``labyrinth_analysis.json`` under ``run_dir``."""
    run_dir = Path(run_dir).resolve()
    h5_path = run_dir / "snapshots.h5"
    summary_path = run_dir / "summary.json"

    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    summ: dict[str, Any] = {}
    if summary_path.is_file():
        summ = json.loads(summary_path.read_text())

    params = summ.get("parameters") or {}
    L = float(params.get("L", 1.0))
    n = int(params.get("grid", 1))
    dx = L / float(n)

    pm, pc = _load_final_snapshot(h5_path)
    phi_diff = pm - pc

    spec = spectral_peak_wavelength(phi_diff, dx)
    _kp = spec.get("k_peak")
    _use_k = isinstance(_kp, (int, float)) and np.isfinite(_kp) and float(_kp) > 0.0
    ani = directional_anisotropy(
        phi_diff,
        dx,
        k_peak=float(_kp) if _use_k else None,
    )
    contrast = contrast_amplitude(phi_diff)

    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "dx": dx,
        "numpy_version": np.__version__,
        "spectral_peak_wavelength": spec,
        "directional_anisotropy": ani,
        "contrast_amplitude": contrast,
        "mass_balance_note": ("Option B not used in stage2 spectral_only runs; ignored here."),
    }
    out_path = run_dir / "labyrinth_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    return out


def _load_final_snapshot(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py  # optional dependency via agate extra

    with h5py.File(h5_path, "r") as h5:
        keys = sorted(h5.keys(), key=lambda x: int(str(x).split("_")[1]))
        if not keys:
            raise ValueError(f"no snapshot groups in {h5_path}")
        g = h5[keys[-1]]
        pm = np.asarray(g["phi_m"])
        pc = np.asarray(g["phi_c"])
    return pm, pc
