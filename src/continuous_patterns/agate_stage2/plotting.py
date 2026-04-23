"""Figures for agate-CH falsification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

from continuous_patterns.agate_stage2.diagnostics import horizontal_centerline
from continuous_patterns.agate_stage2.model import build_geometry

_ANTIPHASE_CMAP = "RdBu_r"


def _antiphase_cmap():
    try:
        cmap = plt.colormaps[_ANTIPHASE_CMAP].copy()
    except (AttributeError, KeyError, TypeError):
        cmap = plt.cm.get_cmap(_ANTIPHASE_CMAP).copy()
    cmap.set_bad(color="white")
    return cmap


def _antiphase_masked_crop(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    R: float,
) -> np.ndarray:
    """Contrast φ_m−φ_c: clip [-1,1], crop cavity, mask outside χ."""
    pm = np.asarray(phi_m, dtype=np.float64)
    pc = np.asarray(phi_c, dtype=np.float64)
    diff = np.clip(pm - pc, -1.0, 1.0)
    geom = build_geometry(L, R, diff.shape[0])
    chi = np.asarray(jnp.asarray(geom.chi))
    s0, s1 = _cavity_square_slices(chi)
    chi_c = chi[s0, s1]
    crop = diff[s0, s1]
    out = np.where(chi_c > 0.5, crop, np.nan).T
    return out


def _cavity_square_slices(chi: np.ndarray, *, border_frac: float = 0.05) -> tuple[slice, slice]:
    m = chi > 0.01
    if not np.any(m):
        n = chi.shape[0]
        return slice(0, n), slice(0, n)
    ii, jj = np.where(m)
    i0, i1 = int(ii.min()), int(ii.max()) + 1
    j0, j1 = int(jj.min()), int(jj.max()) + 1
    n = chi.shape[0]
    ci, cj = (i0 + i1) // 2, (j0 + j1) // 2
    span = max(i1 - i0, j1 - j0)
    margin = max(1, int(border_frac * span))
    half = span // 2 + margin
    i_lo = max(0, ci - half)
    i_hi = min(n, ci + half)
    j_lo = max(0, cj - half)
    j_hi = min(n, cj + half)
    return slice(i_lo, i_hi), slice(j_lo, j_hi)


def save_final_pub(
    field: np.ndarray,
    *,
    L: float,
    R: float,
    path: Path,
    min_px: int = 1600,
    cmap: str = "cividis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Single-panel field image: cropped cavity + margin, colormap, no decorations."""
    n = field.shape[0]
    geom = build_geometry(L, R, n)
    chi = np.asarray(jnp.asarray(geom.chi))
    s0, s1 = _cavity_square_slices(chi)
    z = np.clip(np.asarray(field)[s0, s1], 0.0, None)
    if vmin is None:
        vmin = float(np.min(z)) if z.size else 0.0
    if vmax is None:
        vmax = float(np.max(z)) if z.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9
    h0, w0 = z.shape
    scale = min_px / max(h0, w0, 1)
    zh = zoom(z, scale, order=1)
    dpi = min_px / max(zh.shape[0], 1)
    fig = plt.figure(figsize=(zh.shape[1] / dpi, zh.shape[0] / dpi), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(
        zh.T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.axis("off")
    fig.savefig(path, dpi=dpi, pad_inches=0, bbox_inches=None)
    plt.close(fig)


def plot_comparison_grid(
    run_dirs: list[tuple[str, Path]],
    path: Path,
    *,
    cmap: str = "cividis",
    min_px: int = 400,
) -> None:
    """One row: final (phi_m+phi_c) cropped, high-res thumbnails, labels below."""
    import h5py

    ncols = len(run_dirs)
    if ncols == 0:
        return
    fig, axes = plt.subplots(1, ncols, figsize=(2.4 * ncols, 2.8))
    if ncols == 1:
        axes = np.array([axes])
    for ax, (cid, rdir) in zip(np.ravel(axes), run_dirs, strict=False):
        h5p = rdir / "snapshots.h5"
        summ_path = rdir / "summary.json"
        L, R = 200.0, 80.0
        if summ_path.is_file():
            with summ_path.open() as f:
                summ = json.loads(f.read())
            prm = summ.get("parameters") or {}
            L = float(prm.get("L", L))
            R = float(prm.get("R", R))
        if not h5p.is_file():
            ax.axis("off")
            ax.set_title(cid, fontsize=9)
            continue
        with h5py.File(h5p, "r") as h5:
            keys = sorted(h5.keys(), key=lambda x: int(x.split("_")[1]))
            pm = np.asarray(h5[keys[-1]]["phi_m"])
            pc = np.asarray(h5[keys[-1]]["phi_c"])
        tot = pm + pc
        geom = build_geometry(L, R, tot.shape[0])
        chi = np.asarray(jnp.asarray(geom.chi))
        s0, s1 = _cavity_square_slices(chi)
        z = np.clip(tot[s0, s1], 0.0, None)
        scale = min_px / max(z.shape[0], z.shape[1], 1)
        zh = zoom(z, scale, order=1)
        ax.imshow(zh.T, origin="lower", cmap=cmap, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(cid, fontsize=9, pad=6)
    plt.subplots_adjust(bottom=0.08, top=0.92, left=0.02, right=0.98)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_canonical_slice_grid(
    run_dirs: list[tuple[str, Path]],
    path: Path,
) -> None:
    """2×3 grid of canonical slices with titles."""
    import h5py

    n = len(run_dirs)
    if n == 0:
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows))
    axes = np.atleast_2d(np.asarray(axes))
    for idx, (cid, rdir) in enumerate(run_dirs):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        nb, klass, ac = 0, "", float("nan")
        L, R = 200.0, 80.0
        summ_path = rdir / "summary.json"
        if summ_path.is_file():
            with summ_path.open() as f:
                summ = json.loads(f.read())
            nb = int(summ.get("final_band_count", summ.get("N_b", 0)))
            klass = str(summ.get("classification_at_final", summ.get("classification", "")))
            ac = float(summ.get("moganite_chalcedony_anticorrelation", float("nan")))
            prm = summ.get("parameters") or {}
            L = float(prm.get("L", L))
            R = float(prm.get("R", R))
        h5p = rdir / "snapshots.h5"
        if not h5p.is_file():
            ax.axis("off")
            continue
        with h5py.File(h5p, "r") as h5:
            keys = sorted(h5.keys(), key=lambda x: int(x.split("_")[1]))
            pm = np.asarray(h5[keys[-1]]["phi_m"])
            pc = np.asarray(h5[keys[-1]]["phi_c"])
        xm, vm = horizontal_centerline(pm, L, R)
        _, vc = horizontal_centerline(pc, L, R)
        ax.plot(xm, vm, color="tab:blue", lw=1.2)
        ax.plot(xm, vc, color="tab:orange", lw=1.2)
        ax.set_ylim(0.0, 1.1)
        ac_s = f"{ac:.2f}" if ac == ac else "nan"
        ax.set_title(
            f"{cid}: N_bands={nb}, class={klass}, anticorr={ac_s}",
            fontsize=8,
        )
    for idx in range(n, nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def write_evolution_gif_phi_m(
    snaps_full: list[tuple[int, Any, Any, Any]],
    path: Path,
    *,
    L: float = 200.0,
    R: float = 80.0,
    n_frames: int = 48,
    fps: float = 10.0,
    max_side: int = 384,
) -> None:
    """GIF of φ_m−φ_c (anti-phase); cavity interior colored, exterior white."""
    import imageio.v2 as imageio

    if not snaps_full:
        return
    ix = np.linspace(0, len(snaps_full) - 1, num=min(n_frames, len(snaps_full)))
    ix = np.unique(ix.astype(int))
    cmap = _antiphase_cmap()
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    frames_rgb: list[np.ndarray] = []
    for k in ix:
        pm = np.asarray(snaps_full[k][2], dtype=np.float64)
        pc = np.asarray(snaps_full[k][3], dtype=np.float64)
        geom = build_geometry(L, R, pm.shape[0])
        chi = np.asarray(jnp.asarray(geom.chi))
        diff = np.clip(pm - pc, -1.0, 1.0)
        d = np.where(chi > 0.5, diff, np.nan)
        rgba = cmap(norm(d))
        rgb = (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
        sc = max_side / max(rgb.shape[0], rgb.shape[1], 1)
        if sc < 1.0:
            rgb = (np.clip(zoom(rgba[..., :3], (sc, sc, 1), order=1), 0.0, 1.0) * 255.0).astype(
                np.uint8
            )
        frames_rgb.append(rgb)
    imageio.mimsave(path, frames_rgb, fps=float(fps), loop=0)
    mp4_path = path.with_suffix(".mp4")
    try:
        imageio.mimsave(mp4_path, frames_rgb, fps=float(fps), codec="libx264")
    except Exception:
        pass


def choose_pub_field(phi_m: np.ndarray, phi_c: np.ndarray) -> tuple[np.ndarray, str]:
    """Prefer phi_m if it has clearer contrast; else phi_m+phi_c."""
    pm = np.asarray(phi_m)
    pc = np.asarray(phi_c)
    vm = float(np.var(pm))
    vt = float(np.var(pm + pc))
    if vm >= 0.25 * max(vt, 1e-12):
        return pm, "phi_m"
    return pm + pc, "phi_m+phi_c"


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


def plot_jablczynski(
    metrics: dict[str, Any],
    path: Path,
    *,
    title: str,
    radial_centers: np.ndarray | None = None,
    radial_profile_arr: np.ndarray | None = None,
) -> None:
    nb = int(metrics.get("N_b", 0))
    klass = metrics.get("classification", "")
    qs = np.array(metrics.get("q", []), dtype=np.float64)
    dd = np.array(metrics.get("d", []), dtype=np.float64)
    rr = np.array(metrics.get("r_outer_in", []), dtype=np.float64)
    pos_label = r"$x_n$" if metrics.get("canonical_field") else r"$r_n$"

    insufficient = nb < 3 or klass.startswith("INSUFFICIENT")
    if insufficient and radial_centers is not None and radial_profile_arr is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(radial_centers, radial_profile_arr, "k-", lw=1.2)
        plt.xlabel(r"$r$")
        plt.ylabel(r"$\phi_m+\phi_c$ (mean)")
        plt.title(title)
        plt.annotate(
            f"INSUFFICIENT BANDS (N={nb})",
            xy=(0.5, 0.92),
            xycoords="axes fraction",
            ha="center",
            fontsize=11,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.9},
        )
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        return

    if insufficient:
        plt.figure(figsize=(7, 4))
        plt.annotate(
            f"INSUFFICIENT BANDS (N={nb}) — no radial curve cached",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            fontsize=11,
        )
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        return

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    if dd.size and rr.size and dd.size == rr.size:
        axes[0].loglog(rr, dd, "o-")
        axes[0].set_xlabel(pos_label)
        axes[0].set_ylabel(r"$d_n$")
    if qs.size:
        axes[1].plot(np.arange(1, len(qs) + 1), qs, "o-")
        axes[1].set_xlabel(r"$n$")
        axes[1].set_ylabel(r"$q_n$")
    if dd.size:
        axes[2].plot(np.arange(1, len(dd) + 1), dd, "o-")
        axes[2].set_xlabel(r"$n$")
        axes[2].set_ylabel(r"$d_n$")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_band_count_evolution(
    records: list[tuple[int, int, list[float]]],
    dt: float,
    path: Path,
) -> None:
    if not records:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "no snapshots", ha="center")
        plt.savefig(path, dpi=120)
        plt.close()
        return
    steps = [r[0] for r in records]
    counts = [r[1] for r in records]
    times = [s * dt for s in steps]
    plt.figure(figsize=(7, 4))
    plt.plot(times, counts, "b.-")
    if counts:
        plt.axhline(max(counts), color="gray", ls="--", lw=0.8)
    plt.xlabel("time")
    plt.ylabel(r"$N_{\mathrm{peaks}}$")
    plt.title("Band count evolution")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_kymograph(
    t: list[float],
    r: list[float],
    path: Path,
    *,
    title: str = "",
) -> None:
    plt.figure(figsize=(7, 5))
    if t and r:
        plt.scatter(t, r, s=8, alpha=0.5, c="navy")
    plt.xlabel("time")
    plt.ylabel(r"peak $r$")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_canonical_slice(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    R: float,
    path: Path,
    n_bands: int,
    classification: str,
    anticorr: float,
) -> None:
    xm, vm = horizontal_centerline(phi_m, L, R)
    xc, vc = horizontal_centerline(phi_c, L, R)
    plt.figure(figsize=(9, 4.5))
    plt.plot(xm, vm, color="tab:blue", lw=1.6, label=r"$\phi_m$")
    plt.plot(xc, vc, color="tab:orange", lw=1.6, label=r"$\phi_c$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\phi$")
    ac_s = f"{anticorr:.3f}" if anticorr == anticorr else "nan"
    ttl = (
        rf"$N_b={n_bands}$, {classification}, "
        rf"moganite/chalcedony $\rho={ac_s}$"
    )
    plt.title(ttl, fontsize=11)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_sweep_fields_and_slices(
    run_dirs: list[tuple[str, Path]],
    path: Path,
) -> None:
    """Row 1: moganite (φ_m) final field. Row 2: canonical horizontal slice."""
    import h5py

    ncols = len(run_dirs)
    if ncols == 0:
        return
    fig, axes = plt.subplots(2, ncols, figsize=(3.8 * max(ncols, 1), 8.0))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)
    for j, (cid, rdir) in enumerate(run_dirs):
        summ_path = rdir / "summary.json"
        nb, klass = 0, ""
        L, R = 1.0, 0.45
        if summ_path.is_file():
            with summ_path.open() as f:
                summ = json.loads(f.read())
            nb = int(summ.get("final_band_count", summ.get("N_b", 0)))
            klass = str(summ.get("classification_at_final", summ.get("classification", "")))
            prm = summ.get("parameters") or {}
            L = float(prm.get("L", L))
            R = float(prm.get("R", R))
        h5p = rdir / "snapshots.h5"
        if not h5p.is_file():
            continue
        with h5py.File(h5p, "r") as h5:
            keys = sorted(h5.keys(), key=lambda x: int(x.split("_")[1]))
            pm = np.asarray(h5[keys[-1]]["phi_m"])
            pc = np.asarray(h5[keys[-1]]["phi_c"])
        n = pm.shape[0]
        geom = build_geometry(L, R, n)
        chi = np.asarray(jnp.asarray(geom.chi))
        xs = np.linspace(0, L, n)
        ax0 = axes[0, j]
        im = ax0.imshow(pm.T, origin="lower", extent=[0, L, 0, L], aspect="equal")
        ax0.contour(xs, xs, chi.T, levels=[0.5], colors="w", linewidths=1.0)
        ax0.set_title(cid, fontsize=10)
        plt.colorbar(im, ax=ax0, fraction=0.046)
        xm, vm = horizontal_centerline(pm, L, R)
        _, vc = horizontal_centerline(pc, L, R)
        ax1 = axes[1, j]
        ax1.plot(xm, vm, color="tab:blue", lw=1.2, label=r"$\phi_m$")
        ax1.plot(xm, vc, color="tab:orange", lw=1.2, label=r"$\phi_c$")
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$\phi$")
        ax1.legend(fontsize=7, loc="upper right")
        subt = rf"$N_b={nb}$, {klass}"
        ax1.set_title(subt, fontsize=9)
    fig.suptitle("Sweep comparison — fields and canonical slices", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
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


def plot_sweep_kymographs(
    items: list[tuple[str, list[float], list[float]]],
    path: Path,
) -> None:
    n = len(items)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3.6 * max(n, 1), 3.8), sharey=True)
    axes_list = np.atleast_1d(np.asarray(axes)).ravel().tolist()
    for ax, (lab, tt, rr) in zip(axes_list, items, strict=False):
        if tt and rr:
            ax.scatter(tt, rr, s=6, alpha=0.45)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("time")
    axes_list[0].set_ylabel(r"peak $r$")
    plt.suptitle("Kymographs — sweep comparison")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def plot_gamma_scan_fields(
    run_dirs: list[tuple[str, Path]],
    path: Path,
    *,
    cmap: str = "cividis",
    titles_gamma: list[str] | None = None,
) -> None:
    """Single row: (phi_m+phi_c) final, cavity crop."""
    import h5py

    n = len(run_dirs)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.2))
    axes_list = np.atleast_1d(np.asarray(axes)).ravel().tolist()
    for j, (cid, rdir) in enumerate(run_dirs):
        ax = axes_list[j]
        summ_path = rdir / "summary.json"
        L, R = 200.0, 80.0
        if summ_path.is_file():
            with summ_path.open() as f:
                summ = json.loads(f.read())
            prm = summ.get("parameters") or {}
            L = float(prm.get("L", L))
            R = float(prm.get("R", R))
        ttl = titles_gamma[j] if titles_gamma and j < len(titles_gamma) else cid
        h5p = rdir / "snapshots.h5"
        if not h5p.is_file():
            ax.axis("off")
            ax.set_title(ttl, fontsize=9)
            continue
        with h5py.File(h5p, "r") as h5:
            keys = sorted(h5.keys(), key=lambda x: int(x.split("_")[1]))
            pm = np.asarray(h5[keys[-1]]["phi_m"])
            pc = np.asarray(h5[keys[-1]]["phi_c"])
        tot = pm + pc
        geom = build_geometry(L, R, tot.shape[0])
        chi = np.asarray(jnp.asarray(geom.chi))
        s0, s1 = _cavity_square_slices(chi)
        z = np.clip(tot[s0, s1], 0.0, None)
        scale = 480 / max(z.shape[0], z.shape[1], 1)
        zh = zoom(z, scale, order=1)
        ax.imshow(zh.T, origin="lower", cmap=cmap, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(ttl, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_phase_diagram(
    rows: list[dict[str, Any]],
    path_png: Path,
    path_csv: Path,
) -> None:
    """Panels A (CV_q vs γ) and B (N_bands vs γ); writes CSV."""
    import csv

    gammas = [float(r["gamma"]) for r in rows]
    cv_pcts = [float(r["CV_q_pct"]) for r in rows]
    nb = [int(r["N_bands"]) for r in rows]
    labyrinth = [bool(r.get("labyrinth")) for r in rows]

    with path_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "gamma",
                "N_bands",
                "CV_q_pct",
                "std_q_pct_err",
                "spearman_rho",
                "anticorrelation",
                "classification",
                "labyrinth",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["gamma"],
                    r["N_bands"],
                    r["CV_q_pct"],
                    r.get("std_q_pct_err", ""),
                    r.get("spearman_rho", ""),
                    r.get("anticorrelation", ""),
                    r.get("classification", ""),
                    r.get("labyrinth", ""),
                ]
            )

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.5))
    axa.plot(gammas, cv_pcts, "o-", color="tab:blue", lw=1.5, markersize=7)
    axa.axhline(50.0, color="gray", ls="--", lw=1, label="50% CV (regular regime)")
    axa.set_xlabel(r"$\gamma$")
    axa.set_ylabel(r"CV$(q)$ (%)")
    axa.set_title("A: Jabłczyński ratio variability")
    axa.legend(fontsize=8)
    axa.text(
        0.02,
        0.98,
        ("Each point: single run — CV(q) is a pattern statistic,\nnot a measurement uncertainty."),
        transform=axa.transAxes,
        fontsize=7,
        va="top",
        color="0.35",
    )

    axb.plot(gammas, nb, "s-", color="tab:green")
    for g, nn, lab in zip(gammas, nb, labyrinth, strict=False):
        if lab:
            axb.scatter([g], [nn], c="red", s=80, zorder=5, marker="x")
    axb.set_xlabel(r"$\gamma$")
    axb.set_ylabel(r"$N_{\mathrm{bands}}$ (final)")
    axb.set_title("B: Band count vs immiscibility")
    plt.suptitle(r"$\gamma$ scan — phase diagram", fontsize=12)
    plt.tight_layout()
    plt.savefig(path_png, dpi=180)
    plt.close(fig)


def compose_gamma_scan_publication_figure(
    gamma_rows_phase: list[dict[str, Any]],
    run_dirs_ordered: list[tuple[str, Path]],
    path: Path,
) -> None:
    """Top: seven cavity fields; bottom: phase diagram (embedded)."""
    import h5py
    from matplotlib import gridspec

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 7, figure=fig, height_ratios=[1.0, 0.85], hspace=0.35)
    titles = [f"γ={r['gamma']:.1f}" for r in gamma_rows_phase[: len(run_dirs_ordered)]]
    for j, ((_, rdir), ttl) in enumerate(zip(run_dirs_ordered, titles, strict=False)):
        ax = fig.add_subplot(gs[0, j])
        summ_path = rdir / "summary.json"
        L, R = 200.0, 80.0
        if summ_path.is_file():
            with summ_path.open() as f:
                summ = json.loads(f.read())
            prm = summ.get("parameters") or {}
            L = float(prm.get("L", L))
            R = float(prm.get("R", R))
        h5p = rdir / "snapshots.h5"
        if h5p.is_file():
            with h5py.File(h5p, "r") as h5:
                keys = sorted(h5.keys(), key=lambda x: int(x.split("_")[1]))
                pm = np.asarray(h5[keys[-1]]["phi_m"])
                pc = np.asarray(h5[keys[-1]]["phi_c"])
            z = _antiphase_masked_crop(pm, pc, L=L, R=R)
            cmap = _antiphase_cmap()
            ax.imshow(
                z,
                origin="lower",
                cmap=cmap,
                vmin=-1.0,
                vmax=1.0,
                interpolation="bilinear",
            )
        ax.axis("off")
        ax.set_title(ttl, fontsize=10)

    gammas = [float(r["gamma"]) for r in gamma_rows_phase]
    cv_pcts = [float(r["CV_q_pct"]) for r in gamma_rows_phase]
    nb = [int(r["N_bands"]) for r in gamma_rows_phase]
    labyrinth = [bool(r.get("labyrinth")) for r in gamma_rows_phase]
    axa = fig.add_subplot(gs[1, :3])
    axb = fig.add_subplot(gs[1, 4:])
    axa.plot(gammas, cv_pcts, "o-", color="tab:blue")
    axa.axhline(50.0, color="gray", ls="--", lw=1)
    axa.set_xlabel(r"$\gamma$")
    axa.set_ylabel(r"CV$(q)$ %")
    axa.set_title("A")
    axb.plot(gammas, nb, "s-", color="tab:green")
    for g, nn, lab in zip(gammas, nb, labyrinth, strict=False):
        if lab:
            axb.scatter([g], [nn], c="red", s=70, zorder=5, marker="x")
    axb.set_xlabel(r"$\gamma$")
    axb.set_ylabel(r"$N_{\mathrm{bands}}$")
    axb.set_title("B")
    plt.suptitle(r"$\gamma$ scan — fields and phase diagram", fontsize=13)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paper_main_sweep_row(
    run_dirs: list[tuple[str, Path]],
    path: Path,
) -> None:
    """Six panels, 300 DPI, ~800 px per panel width for publication."""
    import h5py

    ncols = len(run_dirs)
    if ncols == 0:
        return
    w_in = (800.0 / 300.0) * ncols
    h_in = 800.0 / 300.0
    fig, axes = plt.subplots(1, ncols, figsize=(w_in, h_in))
    axes_list = np.atleast_1d(np.asarray(axes)).ravel().tolist()
    for ax, (cid, rdir) in zip(axes_list, run_dirs, strict=False):  # noqa: B905
        summ_path = rdir / "summary.json"
        L, R = 200.0, 80.0
        if summ_path.is_file():
            with summ_path.open() as f:
                summ = json.loads(f.read())
            prm = summ.get("parameters") or {}
            L = float(prm.get("L", L))
            R = float(prm.get("R", R))
        h5p = rdir / "snapshots.h5"
        if not h5p.is_file():
            ax.axis("off")
            continue
        with h5py.File(h5p, "r") as h5:
            keys = sorted(h5.keys(), key=lambda x: int(x.split("_")[1]))
            pm = np.asarray(h5[keys[-1]]["phi_m"])
            pc = np.asarray(h5[keys[-1]]["phi_c"])
        z = _antiphase_masked_crop(pm, pc, L=L, R=R)
        cmap = _antiphase_cmap()
        ax.imshow(
            z,
            origin="lower",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        ax.axis("off")
        ax.set_title(cid, fontsize=10, pad=4)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08, wspace=0.05)
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_paper_canonical_antiphase_slice(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    R: float,
    path: Path,
    rho_title: float,
) -> None:
    xm, vm = horizontal_centerline(phi_m, L, R)
    _, vc = horizontal_centerline(phi_c, L, R)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(xm, vm, color="tab:blue", lw=2.0, label=r"$\phi_m$")
    ax.plot(xm, vc, color="tab:orange", lw=2.0, label=r"$\phi_c$")
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(
        rf"Moganite–chalcedony anti-phased banding ($\rho={rho_title:.2f}$)",
        fontsize=12,
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close(fig)


def write_animation(
    frames: list[np.ndarray],
    path: Path,
    *,
    fps: float = 30.0,
) -> None:
    import imageio.v2 as imageio

    out = [(np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8) for f in frames]
    imageio.mimsave(path, out, fps=fps, macro_block_size=1)
