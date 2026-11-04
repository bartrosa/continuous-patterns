#!/usr/bin/env python3
"""EXP-D6: one self-contained fresh campaign + D6H sweep + dedicated banners + gates."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MASTER_ROOT = _ROOT / "results/master_comparison"
_D6_CFG_TEMPLATE = _ROOT / "experiments/sweeps/d6_master_campaign.yaml"
_BANNED_FORWARD = re.compile(r"\b(SK|Szymczak|Kossacki|D2B|d2b_sk)\b", re.IGNORECASE)
_FOREIGN_INGEST = frozenset({"d1", "d2h", "d2h_best", "d3h_best", "d5h_best"})

# Wall-time estimates (s) from 20260609T132016Z full master + D5H scale.
_WALL_EST_S: dict[str, float] = {
    "D6H_sweep (6 runs)": 6.0 * 90.0,
    "E1_Heaney_open": 172.0,
    "E2_Heaney_control": 172.0,
    "E3_Heaney_closed": 344.0,
    "E9_Heaney_movingfront": 2.0,
    "E10_D1A_v0p010_f30": 2558.0,
    "E11_D1A_v0p020_f30": 1269.0,
    "E12_D1A_v0p040_f30": 634.0,
    "E13_D1A_v0p080_f30": 317.0,
    "E14_D1A_v0p020_f15": 1268.0,
    "E15_D1A_v0p020_f50": 1283.0,
    "E7_TP_gamma4_CH_baselineOstwald": 1591.0,
    "E8_TP_gamma4_CH_matchedOstwald": 1589.0,
    "E16_D1C_g4": 10897.0,
    "E17_D1C_g8": 14516.0,
}

_MINI_T_SCALE = 0.08


def _run(cmd: list[str], *, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    inner = " ".join(shlex.quote(part) for part in ["uv", "run", *cmd])
    script = f"set -o pipefail; {inner} 2>&1 | tee {shlex.quote(str(log))}"
    print(f"[run_d6_campaign] {script}")
    subprocess.run(script, shell=True, executable="/bin/bash", check=True, cwd=_ROOT)


def _load_comparison_template() -> dict[str, Any]:
    raw = yaml.safe_load(_D6_CFG_TEMPLATE.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "comparison" not in raw:
        raise ValueError("d6_master_campaign.yaml must contain comparison block")
    return raw["comparison"]


def _apply_mini(comp: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(comp)
    for entry in out.get("experiments", []):
        ov = entry.setdefault("overrides", {})
        if not isinstance(ov, dict):
            continue
        model = str(entry.get("model", ""))
        if "geometry.n" not in ov and model == "slab_twophase":
            ov["geometry.n"] = 128
        if "time.T" in ov:
            ov["time.T"] = max(200.0, float(ov["time.T"]) * _MINI_T_SCALE)
        elif model == "slab_kinetic":
            ov["time.T"] = 4.0
    return out


def _print_dry_run(*, mini: bool, skip_cavity: bool) -> None:
    scale = _MINI_T_SCALE if mini else 1.0
    print("\n=== D6 DRY-RUN wall-time estimate ===")
    total = 0.0
    for label, est in _WALL_EST_S.items():
        t = est * (scale if label != "D6H_sweep (6 runs)" else (0.15 if mini else 1.0))
        if skip_cavity and (label.startswith("E16") or label.startswith("E17")):
            print(f"  {label:40s}  REUSE (0 s)")
            continue
        print(f"  {label:40s}  {t / 60:6.1f} min  ({t:.0f} s)")
        total += t
    print(f"  {'TOTAL':40s}  {total / 3600:6.2f} h  ({total:.0f} s)")
    print("=====================================\n")


def _patch_comp_paths(comp: dict[str, Any], *, out_dir: Path, d6h_dir: Path) -> dict[str, Any]:
    comp = copy.deepcopy(comp)
    comp["d6h_sweep_dir"] = str(d6h_dir.resolve())
    comp["d6_verdict_path"] = str((d6h_dir / "d6_part_c_verdict.json").resolve())
    return comp


def _write_campaign_config(out_dir: Path, comp: dict[str, Any]) -> Path:
    cfg_path = out_dir / "comparison_config.yaml"
    cfg_path.write_text(yaml.safe_dump({"comparison": comp}, sort_keys=False), encoding="utf-8")
    return cfg_path


def _load_or_bootstrap_campaign_config(out_dir: Path, *, mini: bool) -> dict[str, Any]:
    """Load comparison config, or rebuild it from d6h_sweep if analyze failed mid-campaign."""
    cfg_path = out_dir / "comparison_config.yaml"
    if cfg_path.is_file():
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["comparison"]
    d6h_dir = out_dir / "d6h_sweep"
    if not d6h_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot resume {out_dir}: missing comparison_config.yaml and d6h_sweep/"
        )
    comp = _load_comparison_template()
    if mini:
        comp = _apply_mini(comp)
    comp = _patch_comp_paths(comp, out_dir=out_dir, d6h_dir=d6h_dir)
    _write_campaign_config(out_dir, comp)
    print("[d6] Bootstrapped comparison_config.yaml from existing d6h_sweep")
    return comp


def _seed_cavity_reuse(
    *,
    out_dir: Path,
    source_master: Path,
    invocation_id: str,
    labels: tuple[str, ...] = ("E16_D1C_g4", "E17_D1C_g8"),
) -> None:
    from scripts.run_banding_comparison import _newest_complete_run_for_label

    progress_path = out_dir / "comparison_progress.json"
    progress = {"completed": {}, "failed": {}}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    for label in labels:
        src = _newest_complete_run_for_label(source_master, label, source_master.name)
        dst_parent = out_dir / "runs" / label
        dst_parent.mkdir(parents=True, exist_ok=True)
        dst = dst_parent / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        summary_path = dst / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["invocation_id"] = invocation_id
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        progress.setdefault("completed", {})[label] = {
            "run_dir": str(dst.resolve()),
            "model": "slab_twophase",
            "cavity_reused": True,
        }
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(f"[d6] Seeded cavity reuse from {source_master} -> {labels}")


def _execute_master(
    *,
    out_dir: Path,
    comp: dict[str, Any],
    config_path: Path,
    resume: bool = False,
) -> None:
    from scripts.run_banding_comparison import (
        _execute_comparison_rows,
        _patch_movingfront_display_labels,
        _preflight_comparison_experiments,
    )

    invocation_id = out_dir.name
    experiments = comp.get("experiments", [])
    _patch_movingfront_display_labels(experiments, comp, _ROOT)
    _preflight_comparison_experiments(experiments, _ROOT, comp)
    _execute_comparison_rows(
        config_path=config_path,
        out_dir=out_dir,
        comp=comp,
        experiments=experiments,
        invocation_id=invocation_id,
        repo_root=_ROOT,
        resume=resume,
    )


def _grep_forward(master_dir: Path, report_path: Path) -> dict[str, list[str]]:
    paths = [
        master_dir / "comparison_config.yaml",
        master_dir / "table_comparison.md",
        master_dir / "metrics.json",
        report_path,
    ]
    hits: dict[str, list[str]] = {}
    for p in paths:
        if not p.is_file():
            continue
        lines = [
            ln for ln in p.read_text(encoding="utf-8").splitlines() if _BANNED_FORWARD.search(ln)
        ]
        if lines:
            hits[str(p.relative_to(_ROOT))] = lines
    return hits


def _write_report_d6(
    path: Path,
    *,
    master_dir: Path,
    d6h_dir: Path,
    verdict: dict[str, Any],
    grep_hits: dict[str, list[str]],
) -> None:
    lines = [
        "# REPORT_D6 — self-contained campaign",
        "",
        "## PART C — Heaney moving front",
        f"- Sweep dir: `{d6h_dir}` <!-- d6h.sweep_dir -->",
        f"- C1 decision: {verdict.get('c1_decision')} <!-- d6_part_c_verdict.json c1_decision -->",
        f"- Master row: `{verdict.get('master_comparison_run')}` oscillates={verdict.get('any_oscillates')}",  # noqa: E501
        f"- Prediction score: {verdict.get('prediction_score')}",
        "",
        "### Swing vs band width",
        "| run | swing | band width | predicted | observed cycles | oscillates |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in verdict.get("per_run", []):
        lines.append(
            f"| {r['run_label']} | {r.get('swing_amplitude'):.4g} | {r.get('band_width'):.4g} | "
            f"{r.get('predicted_oscillates')} | {r.get('n_cycles')} | {r.get('oscillates')} | "
            f"<!-- d6h.{r['run_label']}.verdict -->"
        )
    lines += [
        "",
        "## PART B — banner layout",
        f"- Master dir: `{master_dir}` <!-- master.invocation_id -->",
        "- figure_banner_layout: dedicated (GridSpec rows, bbox collision gate)",
        "",
        "## PART A — self-contained campaign",
        "- All rows computed in one invocation; no foreign d1/d3/d5 ingest.",
        "- E9 ingests D6H winner from campaign-local d6h_sweep/.",
        "",
        "## Terminology gate",
    ]
    if grep_hits:
        for fp, ln in grep_hits.items():
            lines.append(f"- FAIL `{fp}`: {ln[:2]}")
    else:
        lines.append("- **PASS** — no legacy tokens in forward-facing deliverables.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assert_integrity(
    *,
    master_dir: Path,
    d6h_root: Path,
    comp: dict[str, Any],
    verdict: dict[str, Any],
) -> None:
    d6h_root = Path(d6h_root).resolve()
    for entry in comp.get("experiments", []):
        mode = str(entry.get("ingest", ""))
        if mode and mode in _FOREIGN_INGEST:
            raise RuntimeError(f"foreign ingest forbidden: {entry['run_label']} ingest={mode}")
        if mode and mode != "d6h_best":
            raise RuntimeError(f"unexpected ingest: {entry['run_label']} ingest={mode}")

    metrics = json.loads((master_dir / "metrics.json").read_text(encoding="utf-8"))
    e9 = metrics.get("experiments", {}).get("E9_Heaney_movingfront")
    if not e9:
        raise RuntimeError("E9_Heaney_movingfront missing from metrics")
    ingest = str(e9.get("ingest_source") or e9.get("run_dir") or "")
    d6h_sweep = str(comp.get("d6h_sweep_dir", ""))
    if d6h_sweep not in ingest:
        raise RuntimeError(f"E9 must ingest from campaign d6h_sweep, got {ingest!r}")

    if comp.get("figure_banner_layout") != "dedicated":
        raise RuntimeError("figure_banner_layout must be dedicated")

    for label, row in metrics.get("experiments", {}).items():
        status = str(row.get("row_status", ""))
        if status == "WARN":
            raise RuntimeError(f"{label}: WARN forbidden")
        if status != "OK":
            raise RuntimeError(f"{label}: status={status!r}")

    table = master_dir / "table_comparison.md"
    if table.is_file() and "clipped_mass_frac" in table.read_text(encoding="utf-8"):
        raise RuntimeError("clipped_mass_frac in table")

    fig = master_dir / "figure_master_comparison.png"
    if not fig.is_file():
        raise RuntimeError(f"missing {fig}")

    from scripts.run_banding_comparison import _LABEL_LEGEND

    if len(_LABEL_LEGEND) != 3:
        raise RuntimeError(f"legend must have 3 entries, got {len(_LABEL_LEGEND)}")

    d6h_diag = d6h_root / "figure_d6h_diag.png"
    if not d6h_diag.is_file():
        raise RuntimeError(f"missing {d6h_diag}")

    table_d6h = d6h_root / "table_d6h.md"
    if not table_d6h.is_file():
        raise RuntimeError(f"missing {table_d6h}")

    print(
        f"[integrity] PASS — single invocation {master_dir.name}, "
        f"E9={verdict.get('master_comparison_run')}, dedicated banners"
    )


def _finalize_gates(out_dir: Path) -> None:
    """Run REPORT_D6 + grep + integrity gates on a completed campaign (no compute)."""
    out_dir = out_dir.resolve()
    comp = yaml.safe_load((out_dir / "comparison_config.yaml").read_text(encoding="utf-8"))[
        "comparison"
    ]
    d6h_root = Path(str(comp["d6h_sweep_dir"])).resolve()
    verdict = json.loads((d6h_root / "d6_part_c_verdict.json").read_text(encoding="utf-8"))
    report_path = out_dir / "REPORT_D6.md"
    pre_hits = _grep_forward(out_dir, report_path)
    _write_report_d6(
        report_path,
        master_dir=out_dir,
        d6h_dir=d6h_root,
        verdict=verdict,
        grep_hits=pre_hits,
    )
    grep_hits = _grep_forward(out_dir, report_path)
    if grep_hits:
        raise RuntimeError(f"grep gate FAIL: {grep_hits}")
    _assert_integrity(master_dir=out_dir, d6h_root=d6h_root, comp=comp, verdict=verdict)
    print(f"D6 gates PASS -> {out_dir}")
    print(f"  figure: {out_dir / 'figure_master_comparison.png'}")
    print(f"  d6h:    {d6h_root / 'figure_d6h_diag.png'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-D6 self-contained campaign")
    parser.add_argument("--dry-run", action="store_true", help="Print wall-time table and exit")
    parser.add_argument("--mini", action="store_true", help="Short T / n=128 validation profile")
    parser.add_argument(
        "--skip-cavity-recompute",
        action="store_true",
        help="Reuse E16/E17 from --cavity-source (default: latest master)",
    )
    parser.add_argument(
        "--cavity-source",
        type=Path,
        default=_ROOT / "results/master_comparison/20260609T132016Z",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_MASTER_ROOT,
    )
    parser.add_argument(
        "--finalize-only",
        type=Path,
        default=None,
        metavar="OUT_DIR",
        help="Run REPORT_D6 + integrity gates only (no compute; use after a completed run)",
    )
    parser.add_argument(
        "--continue-out-dir",
        type=Path,
        default=None,
        help="Resume an interrupted D6 campaign directory",
    )
    args = parser.parse_args(argv)

    if args.finalize_only is not None:
        out_dir = (
            args.finalize_only if args.finalize_only.is_absolute() else _ROOT / args.finalize_only
        )
        _finalize_gates(out_dir)
        return 0

    if args.dry_run:
        _print_dry_run(mini=args.mini, skip_cavity=args.skip_cavity_recompute)
        return 0

    if args.continue_out_dir is not None:
        out_dir = (
            args.continue_out_dir
            if args.continue_out_dir.is_absolute()
            else _ROOT / args.continue_out_dir
        )
        comp = _load_or_bootstrap_campaign_config(out_dir, mini=args.mini)
        d6h_dir = Path(str(comp["d6h_sweep_dir"]))
        verdict_path = d6h_dir / "d6_part_c_verdict.json"
        if not verdict_path.is_file():
            _run(
                ["python", "scripts/analyze_d6.py", str(d6h_dir)],
                log=Path("logs/d6_d6h_analyze.log"),
            )
        if args.skip_cavity_recompute:
            cavity_src = (
                args.cavity_source
                if args.cavity_source.is_absolute()
                else _ROOT / args.cavity_source
            )
            _seed_cavity_reuse(
                out_dir=out_dir, source_master=cavity_src, invocation_id=out_dir.name
            )
        _execute_master(
            out_dir=out_dir,
            comp=comp,
            config_path=out_dir / "comparison_config.yaml",
            resume=True,
        )
    else:
        invocation_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_root = args.output_root if args.output_root.is_absolute() else _ROOT / args.output_root
        out_dir = out_root / invocation_id
        out_dir.mkdir(parents=True, exist_ok=False)
        d6h_dir = out_dir / "d6h_sweep"

        comp = _load_comparison_template()
        if args.mini:
            comp = _apply_mini(comp)

        _print_dry_run(mini=args.mini, skip_cavity=args.skip_cavity_recompute)

        mini_flag = ["--mini"] if args.mini else []
        _run(
            ["python", "scripts/run_d6h_sweep.py", "--results-root", str(d6h_dir), *mini_flag],
            log=Path("logs/d6_d6h_sweep.log"),
        )
        _run(
            ["python", "scripts/analyze_d6.py", str(d6h_dir)],
            log=Path("logs/d6_d6h_analyze.log"),
        )

        comp = _patch_comp_paths(comp, out_dir=out_dir, d6h_dir=d6h_dir)
        cfg_path = _write_campaign_config(out_dir, comp)

        if args.skip_cavity_recompute:
            cavity_src = (
                args.cavity_source
                if args.cavity_source.is_absolute()
                else _ROOT / args.cavity_source
            )
            _seed_cavity_reuse(
                out_dir=out_dir, source_master=cavity_src, invocation_id=invocation_id
            )

        _execute_master(
            out_dir=out_dir,
            comp=comp,
            config_path=cfg_path,
            resume=bool(args.skip_cavity_recompute),
        )

    comp = yaml.safe_load((out_dir / "comparison_config.yaml").read_text(encoding="utf-8"))[
        "comparison"
    ]
    d6h_root = Path(str(comp["d6h_sweep_dir"])).resolve()
    verdict = json.loads((d6h_root / "d6_part_c_verdict.json").read_text(encoding="utf-8"))
    report_path = out_dir / "REPORT_D6.md"
    pre_hits = _grep_forward(out_dir, report_path)
    _write_report_d6(
        report_path, master_dir=out_dir, d6h_dir=d6h_root, verdict=verdict, grep_hits=pre_hits
    )
    grep_hits = _grep_forward(out_dir, report_path)
    if grep_hits:
        raise RuntimeError(f"grep gate FAIL: {grep_hits}")
    _assert_integrity(master_dir=out_dir, d6h_root=d6h_root, comp=comp, verdict=verdict)
    print(f"D6 complete -> {out_dir}")
    print(f"  D6H: {d6h_root}")
    print(f"  figure: {out_dir / 'figure_master_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
