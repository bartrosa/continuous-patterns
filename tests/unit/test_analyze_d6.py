"""analyze_d6 table + figure helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _load_analyze_d6():
    script = REPO / "scripts/analyze_d6.py"
    spec = importlib.util.spec_from_file_location("analyze_d6", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_write_table_and_plot(tmp_path: Path) -> None:
    mod = _load_analyze_d6()
    per_run = [
        {
            "run_label": "D6H_narrow_k1",
            "c1": 0.24,
            "c2": 0.21,
            "band_width": 0.03,
            "k_L_abs": 0.5,
            "swing_amplitude": 0.04,
            "predicted_oscillates": True,
            "n_cycles": 0,
            "oscillates": False,
            "Da_trace": [0.5, 1.2, 0.8],
        }
    ]
    plot_runs = [
        {
            **per_run[0],
            "snap_t": np.linspace(0, 10, 20),
            "snap_c_dep": np.linspace(0.22, 0.25, 20),
            "snap_S": ["L"] * 20,
            "snap_Da": np.linspace(0.4, 1.1, 20),
        }
    ]
    table_path = tmp_path / "table_d6h.md"
    fig_path = tmp_path / "figure_d6h_diag.png"
    mod._write_table_d6h(table_path, per_run, master_run="D6H_narrow_k1")
    mod._plot_d6h_diag(fig_path, plot_runs)
    assert table_path.is_file()
    assert "D6H_narrow_k1" in table_path.read_text(encoding="utf-8")
    assert fig_path.is_file()


def test_d6_cavity_rows_pass_preflight() -> None:
    import importlib.util

    import yaml

    script = REPO / "scripts/run_banding_comparison.py"
    spec = importlib.util.spec_from_file_location("run_banding_comparison", script)
    assert spec is not None and spec.loader is not None
    bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc)

    comp = yaml.safe_load(
        (REPO / "experiments/sweeps/d6_master_campaign.yaml").read_text(encoding="utf-8")
    )["comparison"]
    cavity = [e for e in comp["experiments"] if e.get("cavity_row")]
    assert {e["run_label"] for e in cavity} == {"E16_D1C_g4", "E17_D1C_g8"}
    bc._preflight_comparison_experiments(cavity, REPO, comp)
