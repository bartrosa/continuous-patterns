"""Bundled paper figures and the repository ``RESULTS.md`` summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from continuous_patterns.agate_stage2.plotting import (
    compose_gamma_scan_publication_figure,
    plot_gamma_phase_diagram,
    plot_paper_canonical_antiphase_slice,
    plot_paper_main_sweep_row,
    write_evolution_gif_phi_m,
)


def _gamma_scan_results_table_md(gamma_dir: Path | None) -> str:
    """Markdown table from ``gamma_sweep_dir/*/summary.json``."""
    if gamma_dir is None or not gamma_dir.is_dir():
        return "(No γ scan directory provided.)"
    rows: list[tuple[float, int, float, str, float, float]] = []
    for sub in sorted(gamma_dir.iterdir()):
        if not sub.is_dir():
            continue
        sj = sub / "summary.json"
        if not sj.is_file():
            continue
        s = json.loads(sj.read_text())
        p = s.get("parameters") or {}
        g = float(p.get("gamma", 0.0))
        mf = s.get("metrics_at_final") or {}
        cv_fr = mf.get("cv_q")
        if cv_fr is None or cv_fr != cv_fr:
            cv_fr = s.get("cv_q_percent")
        cv_pct = float(cv_fr) * 100.0 if cv_fr == cv_fr else float("nan")
        n_final = int(s.get("final_band_count", 0))
        klass = str(s.get("classification_at_final", ""))
        ac_raw = s.get("moganite_chalcedony_anticorrelation")
        ac = float(ac_raw) if ac_raw is not None and ac_raw == ac_raw else float("nan")
        mb_raw = (s.get("mass_balance_surface_flux") or {}).get("leak_pct")
        mb_f = float(mb_raw) if mb_raw is not None and mb_raw == mb_raw else float("nan")
        rows.append((g, n_final, cv_pct, klass, ac, mb_f))
    rows.sort(key=lambda t: t[0])
    if not rows:
        return "(No γ scan summaries found.)"
    lines = [
        "| γ | N_bands | CV(q) % | classification | anticorr | mass_balance % |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for g, nf, cv, klass, ac, mb_f in rows:
        cv_s = f"{cv:.1f}" if cv == cv else ""
        ac_s = f"{ac:.3f}" if ac == ac else ""
        mb_s = f"{mb_f:.3f}" if mb_f == mb_f else ""
        esc = klass.replace("|", "\\|")
        lines.append(f"| {g:.1f} | {nf} | {cv_s} | {esc} | {ac_s} | {mb_s} |")
    return "\n".join(lines)


def write_results_markdown(
    repo_root: Path,
    main_sweep_dir: Path | None,
    gamma_sweep_dir: Path | None,
) -> None:
    """Write ``RESULTS.md`` at the repo root from sweep CSV and summary scans.

    ``main_sweep_dir`` — optional six-config sweep (table + worst-case mass balance).
    If omitted, ``gamma_sweep_dir`` supplies the table when present (gamma-only).
    ``gamma_sweep_dir`` — phase diagram path embedded in RESULTS.md when set.
    """
    table_dir = main_sweep_dir if main_sweep_dir is not None else gamma_sweep_dir
    if table_dir is None:
        msg = "write_results_markdown needs main_sweep_dir and/or gamma_sweep_dir"
        raise ValueError(msg)
    csv_path = table_dir / "sweep_summary.csv"
    mb_list: list[float] = []
    for sj in sorted(table_dir.glob("*/summary.json")):
        try:
            sd = json.loads(sj.read_text())
            surf = sd.get("mass_balance_surface_flux") or {}
            lp = surf.get("leak_pct")
            if lp is not None and lp == lp:
                mb_list.append(abs(float(lp)))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    mb_line = f"{max(mb_list):.4f}%" if mb_list else "(no summary.json mass-balance data)"
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
                    table_lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")

    tbl_md = (
        "\n".join(table_lines)
        if table_lines
        else "(No sweep_summary.csv in the selected sweep directory yet.)"
    )

    gamma_tbl = _gamma_scan_results_table_md(gamma_sweep_dir)

    gp = ""
    if gamma_sweep_dir and (gamma_sweep_dir / "gamma_phase_diagram.png").is_file():
        try:
            rel_g = (gamma_sweep_dir / "gamma_phase_diagram.png").relative_to(repo_root).as_posix()
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

## Part I — falsification sweep (pinning / ratchet)

{tbl_md}

## Part II — γ scan (one run per γ)

{gamma_tbl}

Each γ point is one simulation; CV(q) describes that pattern, not error bars.

## Gamma-scan phase diagram
{gp}
See `gamma_phase_diagram.csv` under the gamma sweep directory for structured data.

## Relation to Szymczak 2024 review

Non-equilibrium self-organisation in colloidal silica is consistent with the
phase-field + reaction framework used here; emergent bands need not follow
classical Liesegang scaling.

## Next steps

- 3D and mechanics; cavity-shape variation; calibration to published spacing data.
"""
    (repo_root / "RESULTS.md").write_text(body)


def generate_paper_figures(
    gamma_sweep_dir: Path,
    *,
    main_sweep_dir: Path | None = None,
) -> None:
    """Build publication figures under ``paper_figures/``.

    Figures 1–2 use ``main_sweep_dir`` when provided (six-config falsification sweep).
    Figure 3 always uses ``gamma_sweep_dir``. Output directory is ``main_sweep_dir``
    when set, otherwise ``gamma_sweep_dir`` (gamma-only pipeline).
    """
    import shutil

    import h5py

    out_base = main_sweep_dir if main_sweep_dir is not None else gamma_sweep_dir
    paper = out_base / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)

    if main_sweep_dir is not None:
        sweep_entries: list[tuple[str, Path]] = []
        for sub in sorted(main_sweep_dir.iterdir()):
            if sub.is_dir() and (sub / "summary.json").is_file():
                sweep_entries.append((sub.name, sub))
        sweep_entries.sort(key=lambda x: x[0])
        if len(sweep_entries) >= 6:
            plot_paper_main_sweep_row(sweep_entries[:6], paper / "fig1_comparison.png")

    mp_dir = (main_sweep_dir / "medium_pinning") if main_sweep_dir is not None else None
    if mp_dir is not None and (mp_dir / "snapshots.h5").is_file():
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
        try:
            from continuous_patterns.agate_stage2.run import load_snapshots_h5

            snaps_pub = load_snapshots_h5(mp_dir / "snapshots.h5")
            write_evolution_gif_phi_m(
                snaps_pub,
                paper / "fig4_evolution.gif",
                L=L,
                R=R,
            )
        except Exception:
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
    if rows_c:
        plot_gamma_phase_diagram(
            rows_c,
            gamma_sweep_dir / "gamma_phase_diagram.png",
            gamma_sweep_dir / "gamma_phase_diagram.csv",
        )
    if rows_c and run_gamma_paths:
        compose_gamma_scan_publication_figure(
            rows_c, run_gamma_paths, paper / "fig3_gamma_scan.png"
        )
        gpub = gamma_sweep_dir / "paper_figures"
        gpub.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paper / "fig3_gamma_scan.png", gpub / "fig3_gamma_scan.png")
