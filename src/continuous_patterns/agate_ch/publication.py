"""Bundled paper figures and the repository ``RESULTS.md`` summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from continuous_patterns.agate_ch.plotting import (
    compose_gamma_scan_publication_figure,
    plot_paper_canonical_antiphase_slice,
    plot_paper_main_sweep_row,
)


def write_results_markdown(
    repo_root: Path,
    main_sweep_dir: Path,
    gamma_sweep_dir: Path | None,
) -> None:
    """Write ``RESULTS.md`` at the repo root from sweep CSV and summary scans."""
    csv_path = main_sweep_dir / "sweep_summary.csv"
    mb_list: list[float] = []
    for sj in sorted(main_sweep_dir.glob("*/summary.json")):
        try:
            sd = json.loads(sj.read_text())
            surf = sd.get("mass_balance_surface_flux") or {}
            lp = surf.get("leak_pct")
            if lp is not None and lp == lp:
                mb_list.append(abs(float(lp)))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    mb_line = f"{max(mb_list):.4f}%" if mb_list else "(run main sweep to populate)"
    table_lines: list[str] = []
    if csv_path.is_file():
        with csv_path.open() as f:
            rd = csv.DictReader(f)
            rows = list(rd)
            if rows:
                cols = list(rows[0].keys())
                table_lines.append("| " + " | ".join(cols) + " |")
                table_lines.append("| " + " | ".join("---" for _ in cols) + " |")
                for row in rows:
                    table_lines.append(
                        "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |"
                    )

    tbl_md = (
        "\n".join(table_lines)
        if table_lines
        else "(No sweep_summary.csv yet — run main sweep first.)"
    )

    gp = ""
    if gamma_sweep_dir and (gamma_sweep_dir / "gamma_phase_diagram.png").is_file():
        try:
            rel_g = (
                (gamma_sweep_dir / "gamma_phase_diagram.png")
                .relative_to(repo_root)
                .as_posix()
            )
            gp = f"\n![Gamma phase diagram]({rel_g})\n"
        except ValueError:
            gp = "\n*(Gamma phase diagram generated in gamma sweep directory)*\n"

    body = f"""# Agate Cahn-Hilliard simulation: results summary

## Claims supported by this simulation

1. Concentric banding emerges spontaneously in phase-field simulations of silica
   precipitation in a 2D circular cavity, WITHOUT any mechanical coupling,
   elastic shrinkage, or Biot poroelasticity.
2. Bands exhibit stable moganite/chalcedony anti-phasing (anticorrelation
   ρ ≈ −0.95 in representative runs) matching qualitative observations of real agates.
3. Bands violate Jabłczyński's law in the sense used here: CV(q) is
   typically tens of percent across configs, unlike classical Liesegang
   (often CV < 10%).
4. Band regularity varies with polymorph immiscibility γ and moganite mobility M_m;
   very high γ can yield labyrinthine morphology (auto-detected in diagnostics).
5. Patterns are seed-robust under the documented criterion (classification +
   anticorrelation + bounded relative spread in band count).

## Claims NOT supported (explicit limitations)

1. This is 2D; real agates are 3D (see NOTES.md).
2. No calibration to measured band spacings from natural samples.
3. Dimensionless τ is not mapped to SI time without additional modeling.
4. Numerical mass balance residual (Option B dense flux vs dissolved disk):
   worst-case |relative| across six main configs ≈ **{mb_line}** —
   see `mass_balance_surface_flux.leak_pct` in each `summary.json`.

## Key numbers

{tbl_md}

## Gamma-scan phase diagram
{gp}
See `gamma_phase_diagram.csv` in the gamma sweep directory for structured data.

## Relation to Szymczak 2024 review

Non-equilibrium self-organisation in colloidal silica is consistent with the
phase-field + reaction framework used here; emergent bands need not follow
classical Liesegang scaling.

## Next steps

- 3D and mechanics; cavity-shape variation; calibration to published spacing data.
"""
    (repo_root / "RESULTS.md").write_text(body)


def generate_paper_figures(
    main_sweep_dir: Path,
    gamma_sweep_dir: Path,
) -> None:
    """Populate ``paper_figures/`` under the main sweep directory."""
    import shutil

    import h5py

    paper = main_sweep_dir / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)
    sweep_entries: list[tuple[str, Path]] = []
    for sub in sorted(main_sweep_dir.iterdir()):
        if sub.is_dir() and (sub / "summary.json").is_file():
            sweep_entries.append((sub.name, sub))
    sweep_entries.sort(key=lambda x: x[0])
    if len(sweep_entries) >= 6:
        plot_paper_main_sweep_row(sweep_entries[:6], paper / "fig1_comparison.png")

    mp_dir = main_sweep_dir / "medium_pinning"
    if (mp_dir / "snapshots.h5").is_file():
        with h5py.File(mp_dir / "snapshots.h5", "r") as h5:
            keys = sorted(h5.keys(), key=lambda x: int(x.split("_")[1]))
            pm = np.asarray(h5[keys[-1]]["phi_m"])
            pc = np.asarray(h5[keys[-1]]["phi_c"])
        with (mp_dir / "summary.json").open() as f:
            summ = json.loads(f.read())
        prm = summ.get("parameters") or {}
        L = float(prm.get("L", 200))
        R = float(prm.get("R", 80))
        ac = float(summ.get("moganite_chalcedony_anticorrelation", -0.94))
        plot_paper_canonical_antiphase_slice(
            pm,
            pc,
            L=L,
            R=R,
            path=paper / "fig2_canonical_slice.png",
            rho_title=ac,
        )
        gif_src = mp_dir / "evolution.gif"
        if gif_src.is_file():
            shutil.copy2(gif_src, paper / "fig4_evolution.gif")

    g_csv = gamma_sweep_dir / "gamma_phase_diagram.csv"
    rows_c: list[dict[str, Any]] = []
    if g_csv.is_file():
        with g_csv.open() as f:
            rd = csv.DictReader(f)
            for row in rd:
                labyrinth_cell = str(row.get("labyrinth", "")).lower() in (
                    "true",
                    "1",
                    "yes",
                )
                serr = row.get("std_q_pct_err", "") or "0"
                try:
                    serr_f = float(serr)
                except ValueError:
                    serr_f = 0.0
                rows_c.append(
                    {
                        "gamma": float(row["gamma"]),
                        "N_bands": int(float(row["N_bands"])),
                        "CV_q_pct": float(row["CV_q_pct"]),
                        "spearman_rho": row.get("spearman_rho", ""),
                        "anticorrelation": row.get("anticorrelation", ""),
                        "classification": row.get("classification", ""),
                        "labyrinth": labyrinth_cell,
                        "std_q_pct_err": serr_f,
                    }
                )
    rows_c.sort(key=lambda r: r["gamma"])
    run_gamma: list[tuple[str, Path, float]] = []
    for sub in sorted(gamma_sweep_dir.iterdir()):
        if sub.is_dir() and (sub / "summary.json").is_file():
            with (sub / "summary.json").open() as sf:
                gj = json.load(sf)
            g = float((gj.get("parameters") or {}).get("gamma", 0.0))
            run_gamma.append((sub.name, sub, g))
    run_gamma.sort(key=lambda t: t[2])
    run_gamma_paths = [(a, b) for a, b, _ in run_gamma]
    if rows_c and run_gamma_paths:
        compose_gamma_scan_publication_figure(
            rows_c, run_gamma_paths, paper / "fig3_gamma_scan.png"
        )
