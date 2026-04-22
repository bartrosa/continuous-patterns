"""Figures for agate-CH falsification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from continuous_patterns.agate_ch.model import build_geometry


def plot_fields_final(
    c: np.ndarray,
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    R: float,
    path: Path,
) -> None:
    n = c.shape[0]
    geom = build_geometry(L, R, n)
    chi = np.asarray(jnp.asarray(geom.chi))
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    xs = np.linspace(0, L, n)
    ims = [
        (phi_m, r"$\phi_m$"),
        (phi_c, r"$\phi_c$"),
        (phi_m + phi_c, r"$\phi_m+\phi_c$"),
        (c, r"$c$"),
    ]
    for ax, (arr, tit) in zip(np.ravel(axes), ims, strict=True):
        im = ax.imshow(
            np.asarray(arr).T,
            origin="lower",
            extent=[0, L, 0, L],
            aspect="equal",
        )
        ax.contour(xs, xs, chi.T, levels=[0.5], colors="w", linewidths=1.2)
        ax.set_title(tit)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_radial(
    centers: np.ndarray,
    curves: dict[str, np.ndarray],
    path: Path,
    *,
    title: str = "",
) -> None:
    plt.figure(figsize=(7, 5))
    for lab, yy in curves.items():
        plt.plot(centers, yy, label=lab)
    plt.xlabel(r"$r$")
    plt.ylabel("azimuthal average")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_jablczynski(metrics: dict[str, Any], path: Path, klass: str) -> None:
    qs = np.array(metrics.get("q", []), dtype=np.float64)
    dd = np.array(metrics.get("d", []), dtype=np.float64)
    rr = np.array(metrics.get("r_outer_in", []), dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    if dd.size and rr.size and dd.size == rr.size:
        axes[0].loglog(rr, dd, "o-")
        axes[0].set_xlabel(r"$\log r_n$")
        axes[0].set_ylabel(r"$\log d_n$")
    if qs.size:
        axes[1].plot(np.arange(1, len(qs) + 1), qs, "o-")
        axes[1].set_xlabel(r"$n$")
        axes[1].set_ylabel(r"$q_n$")
    if dd.size:
        axes[2].plot(np.arange(1, len(dd) + 1), dd, "o-")
        axes[2].set_xlabel(r"$n$")
        axes[2].set_ylabel(r"$d_n$")
    fig.suptitle(f"Jabłczyński diagnostic — {klass}")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_sweep_compare(
    profiles: list[tuple[str, np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    plt.figure(figsize=(9, 6))
    for lab, centers, yy in profiles:
        plt.plot(centers, yy, label=lab)
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\phi_m+\phi_c$ (mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def write_animation(
    frames: list[np.ndarray],
    path: Path,
    *,
    fps: float = 30.0,
) -> None:
    import imageio.v2 as imageio

    out = [(np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8) for f in frames]
    imageio.mimsave(path, out, fps=fps, macro_block_size=1)
