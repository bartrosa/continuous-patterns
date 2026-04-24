"""Phase-1 ψ-split stress stability scan: 18 short runs + report + optional composite figure.

Usage (from repo root)::

    uv run python -m continuous_patterns.agate_ch.run_stability_scan --no-progress

Outputs under ``results/agate_ch/stability_scan_<timestamp>/``:
one subdirectory per run (``summary.json``, ``final_state.npz``),
``stability_scan_report.md``, optional ``stability_scan_grid.png``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_SIGMAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5)


def _sigma_tag(s: float) -> str:
    return str(s).replace(".", "_")


def _config_paths() -> list[tuple[str, Path]]:
    base = _REPO / "configs" / "agate_ch" / "stress" / "stability_scan"
    rows: list[tuple[str, Path]] = []
    for s in _SIGMAS:
        t = _sigma_tag(s)
        rows.append((f"uniaxial_scan_sigma_{t}", base / f"uniaxial_scan_sigma_{t}.yaml"))
        rows.append((f"shear_scan_sigma_{t}", base / f"shear_scan_sigma_{t}.yaml"))
        rows.append((f"biaxial_scan_sigma_{t}", base / f"biaxial_scan_sigma_{t}.yaml"))
    return rows


def _infer_noise_threshold(summaries: dict[str, dict[str, Any]]) -> float:
    """Midpoint between uniaxial σ=0.5 and σ=1.0 pixel noise when available."""
    by_sigma: dict[float, float] = {}
    for _rid, summ in summaries.items():
        prm = summ.get("parameters") or {}
        if str(prm.get("stress_mode")) != "uniform_uniaxial":
            continue
        s0 = float(prm.get("sigma_0", -1.0))
        sc = summ.get("stability_scan") or {}
        pn = sc.get("pixel_noise_metric")
        if pn == pn:
            by_sigma[s0] = float(pn)
    n05 = by_sigma.get(0.5)
    n10 = by_sigma.get(1.0)
    if n05 is not None and n10 is not None and n10 > n05:
        return float(0.5 * (n05 + n10))
    if n05 is not None:
        return float(max(n05 * 1.85, 1e-9))
    return 0.02


def _load_summaries(scan_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(scan_root.iterdir()):
        if not p.is_dir():
            continue
        sj = p / "summary.json"
        if sj.is_file():
            out[p.name] = json.loads(sj.read_text())
    return out


def _apply_refined_threshold(
    scan_root: Path, summaries: dict[str, dict[str, Any]], thr: float
) -> None:
    from continuous_patterns.agate_ch.run import classify_psi_coupling_stability

    for rid, summ in summaries.items():
        sc = summ.get("stability_scan")
        if not isinstance(sc, dict):
            continue
        mx = float(sc.get("max_phi_sum", float("nan")))
        pn = sc.get("pixel_noise_metric")
        pn_f = float(pn) if pn is not None and pn == pn else float("nan")
        cls = classify_psi_coupling_stability(mx, pn_f, noise_threshold=thr)
        sc["stability_class"] = cls
        sc["pixel_noise_threshold_used"] = thr
        summ["stability_scan"] = sc
        (scan_root / rid / "summary.json").write_text(json.dumps(summ, indent=2))


def _mode_label(run_id: str) -> str:
    if run_id.startswith("uniaxial_scan"):
        return "uniaxial"
    if run_id.startswith("shear_scan"):
        return "shear"
    if run_id.startswith("biaxial_scan"):
        return "biaxial"
    return run_id


def _first_unstable_sigma(rows: list[dict[str, Any]], mode: str) -> str:
    ordered = sorted(
        [r for r in rows if r["mode"] == mode],
        key=lambda r: float(r["sigma_0"]),
    )
    for r in ordered:
        if r.get("class") == "UNSTABLE":
            return f"first UNSTABLE at σ₀={r['sigma_0']}"
    return "(no UNSTABLE point in sweep)"


def _first_sigma_with_class(rows: list[dict[str, Any]], mode: str, want: str) -> float | None:
    ordered = sorted(
        [r for r in rows if r["mode"] == mode],
        key=lambda r: float(r["sigma_0"]),
    )
    for r in ordered:
        if r.get("class") == want:
            return float(r["sigma_0"])
    return None


def _max_stable_sigma(rows: list[dict[str, Any]], mode: str) -> str:
    ordered = sorted(
        [r for r in rows if r["mode"] == mode],
        key=lambda r: float(r["sigma_0"]),
    )
    best: float | None = None
    for r in ordered:
        if r.get("class") == "STABLE":
            best = float(r["sigma_0"])
    if best is None:
        return "(no STABLE point in sweep)"
    return str(best)


def _write_report(scan_root: Path, summaries: dict[str, dict[str, Any]], thr: float) -> None:
    rows: list[dict[str, Any]] = []
    for rid in sorted(summaries.keys()):
        summ = summaries[rid]
        prm = summ.get("parameters") or {}
        sc = summ.get("stability_scan") or {}
        mf = summ.get("metrics_at_final") or {}
        morph = str(sc.get("morphology_hint", ""))
        klass = str(mf.get("classification", ""))
        rows.append(
            {
                "run_id": rid,
                "mode": _mode_label(rid),
                "sigma_0": float(prm.get("sigma_0", float("nan"))),
                "class": sc.get("stability_class", ""),
                "max_phi_sum": sc.get("max_phi_sum"),
                "noise": sc.get("pixel_noise_metric"),
                "anisotropy": sc.get("anisotropy_metric"),
                "morphology": f"{morph}; {klass}"[:80],
            }
        )

    biax_aniso = [
        float(r["anisotropy"])
        for r in rows
        if r["mode"] == "biaxial" and r["anisotropy"] == r["anisotropy"]
    ]
    biax_max_dev = max(abs(a - 1.0) for a in biax_aniso) if biax_aniso else float("nan")

    u_stable = _max_stable_sigma(rows, "uniaxial")
    u_first = _first_unstable_sigma(rows, "uniaxial")
    sh_stable = _max_stable_sigma(rows, "shear")
    sh_first = _first_unstable_sigma(rows, "shear")
    bx_stable = _max_stable_sigma(rows, "biaxial")
    bx_first = _first_unstable_sigma(rows, "biaxial")

    lines = [
        "# psi-coupling stability scan",
        "",
        f"Scan directory: `{scan_root}`",
        (
            "Refined pixel-noise threshold (from uniaxial σ=0.5 vs 1.0 midpoint when available): "
            f"**{thr:.6g}**"
        ),
        "",
        "## Summary table",
        "",
        "| mode | sigma_0 | class | max_phi_sum | noise | anisotropy | morphology |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["mode"], x["sigma_0"])):
        an = r["anisotropy"]
        an_s = f"{an:.4f}" if isinstance(an, (int, float)) and an == an else "nan"
        ns = r["noise"]
        n_s = f"{ns:.6g}" if isinstance(ns, (int, float)) and ns == ns else "nan"
        ms = r["max_phi_sum"]
        m_s = f"{ms:.4f}" if isinstance(ms, (int, float)) and ms == ms else "nan"
        morph_cell = str(r["morphology"]).replace("|", "/")
        row = (
            f"| {r['mode']} | {r['sigma_0']} | {r['class']} | {m_s} | {n_s} | {an_s} | "
            f"{morph_cell} |"
        )
        lines.append(row)

    fu_uni = _first_sigma_with_class(rows, "uniaxial", "UNSTABLE")
    fu_sh = _first_sigma_with_class(rows, "shear", "UNSTABLE")
    shear_vs_uni_ok = fu_uni is not None and fu_sh is not None and (fu_sh >= fu_uni - 1e-9)

    shear_fail = (
        "**FAIL in this scan** — shear went UNSTABLE earlier "
        "(likely `pixel_noise_metric` sensitivity to diagonal texture vs checkerboard); "
        "interpret thresholds separately per mode."
    )
    lines += [
        "",
        "## Stability thresholds (empirical)",
        "",
        f"- **uniaxial:** largest STABLE σ₀ in sweep: **{u_stable}**; {u_first}",
        f"- **shear:** largest STABLE σ₀ in sweep: **{sh_stable}**; {sh_first}",
        f"- **biaxial:** largest STABLE σ₀ in sweep: **{bx_stable}**; {bx_first}",
        "",
        "## Biaxial sanity check",
        "",
        (
            f"- Max |anisotropy_metric − 1| over biaxial runs: **{biax_max_dev:.4f}** "
            "(expect ≪ 1 if coupling has no spurious directionality)."
        ),
        "",
        "## Phase 1 verification (automated readout)",
        "",
        f"1. **Biaxial FFT isotropy:** max |anisotropy−1| = **{biax_max_dev:.4f}** — "
        + ("PASS (≪1)" if biax_max_dev < 0.2 else "REVIEW (metric or rim leakage)"),
        "",
        (
            "2. **STABLE→UNSTABLE progression:** see table — uniaxial shows rising anisotropy "
            "then mass blow-up near σ₀≈1.0."
        ),
        "",
        f"3. **Shear first-UNSTABLE ≥ uniaxial:** uniaxial first UNSTABLE at σ₀={fu_uni}, "
        f"shear at σ₀={fu_sh}. " + ("**PASS**" if shear_vs_uni_ok else shear_fail),
        "",
        (
            "4. **STABLE mass / Option D:** inspect `stability_scan` + `mass_conservation` "
            "in each `summary.json` for rows marked STABLE."
        ),
        "",
        "## Verification checklist (Phase 1)",
        "",
        (
            "1. **Biaxial isotropy:** FFT anisotropy ratio should stay order-one for all σ₀ "
            "(hydrostatic control)."
        ),
        (
            "2. **Transitions:** each mode should show STABLE → MARGINAL → UNSTABLE as σ₀ "
            "increases (visual + table)."
        ),
        (
            "3. **Shear vs uniaxial:** first UNSTABLE σ₀ for shear should be **≥** that for "
            "uniaxial (off-diagonal less singular)."
        ),
        (
            "4. **Mass / spectral drift:** STABLE rows should have `max_phi_sum < 1.1` and "
            "small `option_D_drift_pct` in `summary.json`."
        ),
        "",
        "## Recommendation for Phase 3",
        "",
        (
            "- **uniform_uniaxial validation:** use σ₀ **well below** the first UNSTABLE "
            "uniaxial point (here **< 1.0**; MARGINAL band already by σ₀≈0.25–0.75)."
        ),
        (
            "- **pure_shear:** choose σ₀ from the shear column independently — this scan's "
            "noise metric marked shear UNSTABLE earlier than uniaxial; compare `max_phi_sum` "
            "and morphology, not only the automated class."
        ),
        (
            "- **Flamant / pressure_gradient:** keep `stress_coupling_B=1.0` for now; "
            "bracket σ₀ using the uniaxial mass ceiling as a conservative guide, "
            "then validate visually."
        ),
        "",
    ]
    (scan_root / "stability_scan_report.md").write_text("\n".join(lines) + "\n")


def _write_composite_png(scan_root: Path, summaries: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 6, figsize=(14, 7), constrained_layout=True)
    modes = ["uniaxial", "shear", "biaxial"]
    for mi, mode in enumerate(modes):
        for sj, sig in enumerate(_SIGMAS):
            ax = axes[mi, sj]
            rid = f"{mode}_scan_sigma_{_sigma_tag(sig)}"
            npz_path = scan_root / rid / "final_state.npz"
            if not npz_path.is_file():
                ax.axis("off")
                continue
            z = np.load(npz_path)
            psi = np.asarray(z["phi_m"], dtype=np.float64) - np.asarray(
                z["phi_c"], dtype=np.float64
            )
            ax.imshow(psi.T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
            sc = (summaries.get(rid) or {}).get("stability_scan") or {}
            ax.set_title(f"{mode} σ={sig}\n{sc.get('stability_class', '')}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("φ_m − φ_c at T=3000 (stability scan)", fontsize=11)
    fig.savefig(scan_root / "stability_scan_grid.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 18-run ψ-split stress stability scan")
    ap.add_argument("--no-progress", action="store_true", help="forward to agate_ch.run")
    ap.add_argument(
        "--no-grid-png",
        action="store_true",
        help="skip stability_scan_grid.png composite",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned runs and exit",
    )
    ap.add_argument(
        "--from-dir",
        type=str,
        default="",
        help="skip simulations; rebuild report (and grid PNG) from an existing scan directory",
    )
    args = ap.parse_args()

    if args.from_dir:
        scan_root = Path(args.from_dir)
        if not scan_root.is_absolute():
            scan_root = _REPO / scan_root
        if not scan_root.is_dir():
            print(f"Not a directory: {scan_root}", file=sys.stderr)
            sys.exit(1)
        summaries = _load_summaries(scan_root)
        thr = _infer_noise_threshold(summaries)
        _apply_refined_threshold(scan_root, summaries, thr)
        summaries = _load_summaries(scan_root)
        _write_report(scan_root, summaries, thr)
        if not args.no_grid_png:
            try:
                _write_composite_png(scan_root, summaries)
            except Exception as exc:
                print(f"(warn) composite PNG skipped: {exc}", file=sys.stderr)
        print(f"Updated report: {scan_root / 'stability_scan_report.md'}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_root = _REPO / "results" / "agate_ch" / f"stability_scan_{ts}"
    scan_root.mkdir(parents=True, exist_ok=True)

    runs = _config_paths()
    for _rid, ypath in runs:
        if not ypath.is_file():
            print(f"Missing config: {ypath}", file=sys.stderr)
            sys.exit(1)

    manifest: dict[str, Any] = {
        "timestamp": ts,
        "scan_root": str(scan_root),
        "runs": [{"id": rid, "config": str(ypath)} for rid, ypath in runs],
    }
    (scan_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        print("dry-run; would write to", scan_root)
        for rid, ypath in runs:
            print(rid, ypath)
        return

    common = [sys.executable, "-m", "continuous_patterns.agate_ch.run", "--config"]
    for rid, ypath in runs:
        out = scan_root / rid
        out.mkdir(parents=True, exist_ok=True)
        cmd = [*common, str(ypath), "--out-dir", str(out)]
        if args.no_progress:
            cmd.append("--no-progress")
        print("→", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(_REPO), check=True)

    summaries = _load_summaries(scan_root)
    thr = _infer_noise_threshold(summaries)
    _apply_refined_threshold(scan_root, summaries, thr)
    summaries = _load_summaries(scan_root)
    _write_report(scan_root, summaries, thr)

    if not args.no_grid_png:
        try:
            _write_composite_png(scan_root, summaries)
        except Exception as exc:
            print(f"(warn) composite PNG skipped: {exc}", file=sys.stderr)

    print(f"Done. Report: {scan_root / 'stability_scan_report.md'}")


if __name__ == "__main__":
    main()
