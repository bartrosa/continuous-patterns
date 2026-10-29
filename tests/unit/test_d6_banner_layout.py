"""Dedicated banner layout must not overlap plot axes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_layout_module():
    script = REPO / "scripts/comparison_figure_layout.py"
    spec = importlib.util.spec_from_file_location("comparison_figure_layout", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dedicated_banners_disjoint(tmp_path: Path) -> None:
    layout = _load_layout_module()
    write_comparison_figure_dedicated = layout.write_comparison_figure_dedicated
    n = 2
    records = [type("R", (), {"model": "slab_kinetic", "run_label": f"E{i}"})() for i in range(n)]
    metrics_list = [{"n_bands": 1, "run_dir": None, "row_status": "OK"} for _ in range(n)]
    configs = [{"physics": {"c1": 0.4, "c2": 0.2}} for _ in range(n)]
    experiments = [
        {
            "run_label": "E1_Heaney_open",
            "mechanism_class": "Heaney kinetic clock",
            "variant_banner": "Open slab | test",
            "display_label": "E1",
        },
        {
            "run_label": "E10_test",
            "mechanism_class": "CH driven eutectic",
            "variant_banner": "Driven | test",
            "display_label": "E10",
        },
    ]

    def _noop_ax(ax, *args, **kwargs):
        ax.plot([0, 1], [0, 1])
        return [ax]

    def _noop_spacing(ax, *args, **kwargs):
        ax.plot([0, 1], [0, 1])
        return [ax]

    out = tmp_path / "test_d6_banner.png"
    write_comparison_figure_dedicated(
        out,
        records=records,
        metrics_list=metrics_list,
        configs=configs,
        front_origins=[{"origin": "LEFT"} for _ in range(n)],
        regressions=[{"status": "SKIP"} for _ in range(n)],
        table_rows=[{"row_status": "OK", "invalid_reasons": []} for _ in range(n)],
        experiments=experiments,
        invocation_id="TEST",
        footer_note="test",
        figure_subtitle="Test subtitle",
        label_legend=[
            ("SiO2 phase A", "#d62728"),
            ("SiO2 phase B", "#1f77b4"),
            ("mixture", "#7f7f7f"),
        ],
        panel_titles=("Wall", "Kymo", "Bands", "Spacing"),
        plot_wall_dynamics=_noop_ax,
        plot_kymograph=_noop_ax,
        plot_band_strip=_noop_ax,
        plot_spacing_laws=_noop_spacing,
        load_run_snapshots=lambda _p: None,
        load_run_history=lambda _p: None,
        supply_regime=lambda _c: "open",
        fmt_law=lambda v, _m: str(v),
        status_draws_overlay=lambda _s: False,
    )
    assert out.is_file()
