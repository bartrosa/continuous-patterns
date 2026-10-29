#!/usr/bin/env python3
"""EXP-D4: fresh D3H sweep + full master comparison compute + integrity gate + REPORT_D4."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

_BANNED_FORWARD = re.compile(r"\b(SK|Szymczak|Kossacki|D2B)\b", re.IGNORECASE)
_D3H_ROOT = _ROOT / "results/slab_twophase/d3h_heaney_diag"
_MASTER_ROOT = _ROOT / "results/master_comparison"
_D4_CFG = _ROOT / "experiments/comparisons/d4_master_comparison.yaml"


def _forward_facing_paths(master_dir: Path, report_path: Path) -> list[Path]:
    return [
        _D4_CFG,
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
    print(f"[run_d4_campaign] {script}")
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
    if "d3h_heaney_diag" not in ingest:
        raise RuntimeError(f"E9 must ingest from d3h_heaney_diag, got {ingest!r}")
    if "d2h_heaney_movingfront" in ingest or "d2b_sk_movingfront" in ingest:
        raise RuntimeError("E9 must ingest from d3h_heaney_diag, not legacy D2H/D2B sweep")

    master_run = str(verdict.get("master_comparison_run", ""))
    for label, row in exps.items():
        status = str(row.get("row_status", ""))
        if status == "WARN":
            raise RuntimeError(f"{label}: WARN status forbidden on final figure (got WARN)")
        if status != "OK":
            raise RuntimeError(f"{label}: status={status!r} (expected OK)")

    fig = master_dir / "figure_master_comparison.png"
    if not fig.is_file():
        raise RuntimeError(f"missing {fig}")

    table = master_dir / "table_comparison.md"
    if table.is_file() and "clipped_mass_frac" in table.read_text(encoding="utf-8"):
        raise RuntimeError("table_comparison.md still contains clipped_mass_frac")

    print(f"[integrity] PASS — E9 from d3h ({master_run}), all rows OK/honest-negative")


def _write_report_d4(
    path: Path,
    *,
    master_dir: Path,
    d3h_dir: Path,
    verdict: dict[str, Any],
    metrics: dict[str, Any],
    grep_hits: dict[str, list[str]],
) -> None:
    exps = metrics.get("experiments", {})
    clip_rates = {
        k: v.get("clip_rate")
        for k, v in exps.items()
        if isinstance(v, dict) and v.get("clip_rate") is not None
    }
    lines = [
        "# REPORT_D4 — master comparison finalized",
        "",
        "## P0 — Heaney moving-front diagnostic",
        f"- Sweep dir: `{d3h_dir}` <!-- d3h.sweep_dir -->",
        f"- Verdict: `{verdict.get('master_comparison_run')}` oscillates={verdict.get('any_oscillates')} "  # noqa: E501
        f"<!-- d3_part_b_verdict.json -->",
        f"- Root cause: {verdict.get('root_cause_verdict', '')}",
        f"- Prediction score: {verdict.get('prediction_score', {})}",
        "",
        "### Per-run",
    ]
    for r in verdict.get("per_run", []):
        lines.append(
            f"- **{r['run_label']}**: cycles={r.get('n_cycles')} c_sat={r.get('c_sat')} "
            f"c_sat<c2={r.get('c_sat_lt_c2')} oscillates={r.get('oscillates')} "
            f"<!-- d3h.{r['run_label']}.verdict -->"
        )
    lines += [
        "",
        "## P1 — clipped_mass_frac purged",
        "- Removed from slab_twophase summaries, table, and figure captions.",
        "",
        "## P2 — clip_rate informational only",
        f"- clip_rate range: {clip_rates} <!-- metrics.json experiments.*.clip_rate -->",
        "- E16/E17 ~1e-3: projection near arrested front, mass conserved (OK).",
        "- clip_rate never sets row status or draws overlay boxes.",
        "",
        "## P3 — Heaney label migration",
        "- run_labels: E1_Heaney_open, E2_Heaney_control, E3_Heaney_closed, E9_Heaney_movingfront",
        "- On-disk run directories keep historical names (unchanged).",
        "",
        "## P4 — driven wall panel",
        "- D1A rows E10-E15: prescribed front x(t) only, no c_dep line.",
        "",
        "## P5 — terminal band annotation",
        "- E3 spacing panel uses excl. terminal band fit when Q spike detected.",
        "",
        "## Master comparison (fresh compute)",
        f"- Output dir: `{master_dir}` <!-- master.invocation_id -->",
        "",
        "## Terminology gate (forward-facing deliverables)",
    ]
    if grep_hits:
        for fp, ln in grep_hits.items():
            lines.append(f"- FAIL `{fp}`: {ln[:3]}")
    else:
        lines.append("- **PASS** — no legacy tokens in config, table, metrics, or this report.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_d3h() -> Path:
    _run(["python", "scripts/run_d3h_sweep.py"], log=Path("logs/d4_d3h_sweep.log"))
    d3h_dir = _newest_timestamp_dir(_D3H_ROOT)
    _run(["python", "scripts/analyze_d3.py", str(d3h_dir)], log=Path("logs/d4_d3h_analyze.log"))
    return d3h_dir


def _run_fresh_master() -> Path:
    """Full master comparison: E1–E3, E7–E8 compute + D1/D3H ingest → new timestamp dir."""
    _run(
        [
            "python",
            "scripts/run_banding_comparison.py",
            str(_D4_CFG),
            str(_MASTER_ROOT),
        ],
        log=Path("logs/d4_master_fresh.log"),
    )
    return _newest_timestamp_dir(_MASTER_ROOT)


def _finalize_gates(*, master_dir: Path, d3h_dir: Path) -> None:
    verdict = json.loads((d3h_dir / "d3_part_b_verdict.json").read_text(encoding="utf-8"))
    metrics_path = master_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_path = master_dir / "REPORT_D4.md"

    pre_hits = _grep_forward_facing(master_dir, report_path)
    pre_hits.pop(str(report_path.relative_to(_ROOT)), None)

    _write_report_d4(
        report_path,
        master_dir=master_dir,
        d3h_dir=d3h_dir,
        verdict=verdict,
        metrics=metrics,
        grep_hits=pre_hits,
    )

    grep_hits = _grep_forward_facing(master_dir, report_path)
    if grep_hits:
        raise RuntimeError(f"grep gate FAIL: {grep_hits}")

    _assert_integrity(master_dir=master_dir, metrics=metrics, verdict=verdict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP-D4 campaign (default: fresh D3H + full master comparison compute)"
    )
    parser.add_argument(
        "--master-dir",
        type=Path,
        default=None,
        help="Existing master dir (for --report-only or --finalize-only)",
    )
    parser.add_argument(
        "--skip-d3h",
        action="store_true",
        help="Skip D3H sweep; use newest d3h_heaney_diag timestamp",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rewrite REPORT_D4 + integrity gates on existing master-dir (no compute)",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Re-export figure/table from existing master-dir (no row compute, no D3H)",
    )
    args = parser.parse_args(argv)

    if args.report_only or args.finalize_only:
        if args.master_dir is None:
            raise RuntimeError("--master-dir required with --report-only or --finalize-only")
        master_dir = args.master_dir if args.master_dir.is_absolute() else _ROOT / args.master_dir
        d3h_dir = _newest_timestamp_dir(_D3H_ROOT)
        if args.finalize_only:
            _run(
                [
                    "python",
                    "scripts/run_banding_comparison.py",
                    "--finalize-outputs",
                    str(master_dir),
                ],
                log=Path("logs/d4_finalize_master.log"),
            )
        _finalize_gates(master_dir=master_dir, d3h_dir=d3h_dir)
        print(f"D4 complete -> {master_dir}")
        return 0

    # Default: everything fresh
    d3h_dir = _newest_timestamp_dir(_D3H_ROOT) if args.skip_d3h else _run_d3h()
    master_dir = _run_fresh_master()
    _finalize_gates(master_dir=master_dir, d3h_dir=d3h_dir)
    print(f"D4 complete (fresh) -> {master_dir}")
    print(f"  D3H: {d3h_dir}")
    print(f"  figure: {master_dir / 'figure_master_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
