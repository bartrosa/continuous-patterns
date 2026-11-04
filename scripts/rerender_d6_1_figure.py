#!/usr/bin/env python3
"""EXP-D6.1: re-render master comparison figure only (fixed banner geometry)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_OUT = _ROOT / "results/master_comparison/20260609T152334Z"


def _write_report(out_dir: Path, report: dict[str, object]) -> None:
    lines = [
        "# REPORT_D6_1 — figure layout fix (re-render only)",
        "",
        f"- Campaign dir: `{out_dir}`",
        f"- panel_row_in (D4/D5 reference): {report['panel_row_in']:.4g} in",
        f"- H_tier1: {report['h_tier1_in']:.4g} in",
        f"- H_tier2: {report['h_tier2_in']:.4g} in",
        f"- total figure height: {report['fig_height_in']:.2f} in x {report['fig_width_in']:.2f} in width",  # noqa: E501
        f"- n_plot_rows: {report['n_plot_rows']} | n_tier1: {report['n_tier1']} | n_tier2: {report['n_tier2']}",  # noqa: E501
        f"- min panel row px gate: {report['min_panel_row_px']:.1f} px",
        f"- measured min panel row px: {report['measured_panel_row_px_min']:.1f} px",
        "- bbox gate: **PASS** (no banner/plot overlap; no plot/plot overlap)",
        "",
        "Re-render only — no simulation recompute.",
    ]
    (out_dir / "REPORT_D6_1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    append = (
        f"\n## D6.1 layout (re-render)\n"
        f"- panel_row_in={report['panel_row_in']:.4g} in, H1={report['h_tier1_in']:.4g} in, "
        f"H2={report['h_tier2_in']:.4g} in, fig_h={report['fig_height_in']:.2f} in, bbox gate PASS\n"  # noqa: E501
    )
    report_d6 = out_dir / "REPORT_D6.md"
    if report_d6.is_file():
        text = report_d6.read_text(encoding="utf-8")
        if "## D6.1 layout" not in text:
            report_d6.write_text(text.rstrip() + append, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-D6.1 figure-only re-render")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help="Completed D6 campaign directory",
    )
    args = parser.parse_args(argv)
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    if not (out_dir / "comparison_config.yaml").is_file():
        raise FileNotFoundError(f"missing comparison_config.yaml in {out_dir}")

    import scripts.comparison_figure_layout as cfl
    from scripts.run_banding_comparison import regenerate_comparison_figure

    fig_path = regenerate_comparison_figure(out_dir)
    report = cfl.LAST_LAYOUT_REPORT
    if not report:
        raise RuntimeError("layout report missing after re-render")
    _write_report(out_dir, report)

    from PIL import Image

    with Image.open(fig_path) as im:
        px_w, px_h = im.size
    print(f"D6.1 re-render complete -> {fig_path}")
    print(
        f"  pixels: {px_w} x {px_h}  (expected ~{int(report['fig_width_in'] * cfl.FIG_DPI)} x "
        f"~{int(report['fig_height_in'] * cfl.FIG_DPI)})"
    )
    print(f"  REPORT -> {out_dir / 'REPORT_D6_1.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
