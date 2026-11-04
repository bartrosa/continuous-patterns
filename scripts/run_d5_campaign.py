#!/usr/bin/env python3
"""EXP-D5: D5H reaction-limited sweep + master figure banners + integrity gate."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]

_BANNED_FORWARD = re.compile(r"\b(SK|Szymczak|Kossacki|D2B)\b", re.IGNORECASE)
_D5H_ROOT = _ROOT / "results/slab_twophase/d5h_heaney_reaction_limited"
_D5_CFG = _ROOT / "experiments/comparisons/d5_master_comparison.yaml"


def _forward_facing_paths(master_dir: Path, report_path: Path) -> list[Path]:
    return [
        _D5_CFG,
        master_dir / "comparison_config.yaml",
        master_dir / "table_comparison.md",
        master_dir / "metrics.json",
        report_path,
    ]


def _grep_forward_facing(master_dir: Path, report_path: Path) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for p in _forward_facing_paths(master_dir, report_path):
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if _BANNED_FORWARD.search(ln)]
        if lines:
            hits[str(p.relative_to(_ROOT))] = lines
    return hits


def _run(cmd: list[str], *, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    inner = " ".join(shlex.quote(part) for part in ["uv", "run", *cmd])
    script = f"set -o pipefail; {inner} 2>&1 | tee {shlex.quote(str(log))}"
    print(f"[run_d5_campaign] {script}")
    subprocess.run(script, shell=True, executable="/bin/bash", check=True, cwd=_ROOT)


def _newest_timestamp_dir(root: Path) -> Path:
    subs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subs:
        raise RuntimeError(f"No timestamp directories under {root}")
    return subs[-1]


def _assert_integrity(
    *, master_dir: Path, metrics: dict[str, Any], verdict: dict[str, Any]
) -> None:
    exps = metrics.get("experiments", {})
    if not isinstance(exps, dict):
        raise RuntimeError("metrics.json missing experiments block")

    e9 = exps.get("E9_Heaney_movingfront")
    if not e9:
        raise RuntimeError("E9_Heaney_movingfront missing from metrics")
    ingest = str(e9.get("ingest_source") or e9.get("run_dir") or "")
    if "d5h_heaney_reaction_limited" not in ingest:
        raise RuntimeError(f"E9 must ingest from d5h_heaney_reaction_limited, got {ingest!r}")

    master_run = str(verdict.get("master_comparison_run", ""))
    for label, row in exps.items():
        status = str(row.get("row_status", ""))
        if status == "WARN":
            raise RuntimeError(f"{label}: WARN status forbidden (got WARN)")
        if status != "OK":
            raise RuntimeError(f"{label}: status={status!r} (expected OK)")

    fig = master_dir / "figure_master_comparison.png"
    if not fig.is_file():
        raise RuntimeError(f"missing {fig}")

    table = master_dir / "table_comparison.md"
    if table.is_file() and "clipped_mass_frac" in table.read_text(encoding="utf-8"):
        raise RuntimeError("table_comparison.md still contains clipped_mass_frac")

    cfg = yaml.safe_load((master_dir / "comparison_config.yaml").read_text(encoding="utf-8"))
    if not cfg.get("comparison", {}).get("figure_banners"):
        raise RuntimeError("comparison_config missing figure_banners: true")

    print(f"[integrity] PASS — E9 from d5h ({master_run}), banners enabled, all rows OK")


def _write_report_d5(
    path: Path,
    *,
    master_dir: Path,
    d5h_dir: Path,
    verdict: dict[str, Any],
    metrics: dict[str, Any],
    grep_hits: dict[str, list[str]],
) -> None:
    lines = [
        "# REPORT_D5 — reaction-limited Heaney moving front + figure banners",
        "",
        "## PART A — reaction-limited sweep",
        f"- Sweep dir: `{d5h_dir}` <!-- d5h.sweep_dir -->",
        f"- Verdict: `{verdict.get('master_comparison_run')}` oscillates={verdict.get('any_oscillates')} "  # noqa: E501
        f"<!-- d5_part_a_verdict.json -->",
        f"- Baseline physics: {verdict.get('baseline_physics', {})} "
        f"<!-- d5_part_a_verdict.json baseline_physics -->",
        f"- Prediction score: {verdict.get('prediction_score', {})}",
        f"- Root cause: {verdict.get('root_cause_verdict', '')}",
        "",
        "### Per-run (Da relationship)",
    ]
    for r in verdict.get("per_run", []):
        lines.append(
            f"- **{r['run_label']}**: cycles={r.get('n_cycles')} oscillates={r.get('oscillates')} "
            f"c0={r.get('c0')} D={r.get('D_abs')} k_L={r.get('k_L_abs')} "
            f"Da_start={r.get('da_start_osc')} Da_stop={r.get('da_stop_osc')} "
            f"frac_Da>1={r.get('frac_time_da_gt_1')} "
            f"sustained={r.get('sustained')} dying={r.get('dying')} "
            f"<!-- d5h.{r['run_label']}.verdict -->"
        )
        if r.get("band_metrics"):
            bm = r["band_metrics"]
            lines.append(
                f"  - bands: lambda={bm.get('lambda_band')} tau={bm.get('tau_cycle')} "
                f"d={bm.get('d_spacing')} v_open={bm.get('v_at_open')} "
                f"<!-- d5h.{r['run_label']}.bands -->"
            )
    lines += [
        "",
        "## PART B — master figure banners",
        f"- Master dir: `{master_dir}` <!-- master.invocation_id -->",
        "- Tier-1 Heaney: kinetic hysteresis clock | sets PERIOD",
        "- Tier-1 CH: phase-field demixing | sets WAVELENGTH",
        "- Tier-2 variant banners on E1, E2, E3, E9, E10, E14, E7, E16 blocks",
        "- Figure subtitle + 3-entry legend retained",
        "",
        "## Terminology gate (forward-facing deliverables)",
    ]
    if grep_hits:
        for fp, ln in grep_hits.items():
            lines.append(f"- FAIL `{fp}`: {ln[:3]}")
    else:
        lines.append("- **PASS** — no legacy tokens in config, table, metrics, or this report.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finalize_gates(*, master_dir: Path, d5h_dir: Path) -> None:
    verdict = json.loads((d5h_dir / "d5_part_a_verdict.json").read_text(encoding="utf-8"))
    metrics = json.loads((master_dir / "metrics.json").read_text(encoding="utf-8"))
    report_path = master_dir / "REPORT_D5.md"

    pre_hits = _grep_forward_facing(master_dir, report_path)
    pre_hits.pop(str(report_path.relative_to(_ROOT)), None)

    _write_report_d5(
        report_path,
        master_dir=master_dir,
        d5h_dir=d5h_dir,
        verdict=verdict,
        metrics=metrics,
        grep_hits=pre_hits,
    )

    grep_hits = _grep_forward_facing(master_dir, report_path)
    if grep_hits:
        raise RuntimeError(f"grep gate FAIL: {grep_hits}")

    _assert_integrity(master_dir=master_dir, metrics=metrics, verdict=verdict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-D5 campaign")
    parser.add_argument(
        "--master-dir",
        type=Path,
        default=_ROOT / "results/master_comparison/20260609T132016Z",
    )
    parser.add_argument(
        "--skip-d5h", action="store_true", help="Skip D5H sweep (verdict must exist)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rewrite REPORT_D5 + integrity gates only (no finalize)",
    )
    args = parser.parse_args(argv)

    master_dir = args.master_dir if args.master_dir.is_absolute() else _ROOT / args.master_dir

    if not args.skip_d5h and not args.report_only:
        _run(["python", "scripts/run_d5h_sweep.py"], log=Path("logs/d5_d5h_sweep.log"))
        d5h_dir = _newest_timestamp_dir(_D5H_ROOT)
        _run(["python", "scripts/analyze_d5.py", str(d5h_dir)], log=Path("logs/d5_d5h_analyze.log"))
    else:
        d5h_dir = _newest_timestamp_dir(_D5H_ROOT)

    if not args.report_only:
        import shutil

        shutil.copy2(_D5_CFG, master_dir / "comparison_config.yaml")
        _run(
            [
                "python",
                "scripts/run_banding_comparison.py",
                "--finalize-outputs",
                str(master_dir),
            ],
            log=Path("logs/d5_finalize_master.log"),
        )
        metrics_probe = json.loads((master_dir / "metrics.json").read_text(encoding="utf-8"))
        e9_probe = metrics_probe.get("experiments", {}).get("E9_Heaney_movingfront", {})
        ingest_probe = str(e9_probe.get("ingest_source") or e9_probe.get("run_dir") or "")
        if "d5h_heaney_reaction_limited" not in ingest_probe:
            raise RuntimeError(f"finalize did not update E9 to d5h ingest (got {ingest_probe!r})")

    _finalize_gates(master_dir=master_dir, d5h_dir=d5h_dir)
    print(f"D5 complete -> {master_dir}")
    print(f"  D5H: {d5h_dir}")
    print(f"  figure: {master_dir / 'figure_master_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
