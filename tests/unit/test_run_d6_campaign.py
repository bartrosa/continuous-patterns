"""D6 integrity gate on completed campaign artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DONE = REPO / "results/master_comparison/20260609T152334Z"


def _load_d6():
    script = REPO / "scripts/run_d6_campaign.py"
    spec = importlib.util.spec_from_file_location("run_d6_campaign", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_assert_integrity_accepts_path_d6h_root(tmp_path: Path) -> None:
    d6 = _load_d6()
    d6h = tmp_path / "d6h_sweep"
    d6h.mkdir()
    (d6h / "figure_d6h_diag.png").write_bytes(b"png")
    (d6h / "table_d6h.md").write_text("# ok\n", encoding="utf-8")
    master = tmp_path / "master"
    master.mkdir()
    (master / "figure_master_comparison.png").write_bytes(b"png")
    (master / "table_comparison.md").write_text("| ok |\n", encoding="utf-8")
    metrics = {
        "experiments": {
            "E9_Heaney_movingfront": {
                "row_status": "OK",
                "ingest_source": str(d6h / "D6H_med_k1" / "run"),
            }
        }
    }
    (master / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    comp = {
        "figure_banner_layout": "dedicated",
        "d6h_sweep_dir": str(d6h.resolve()),
        "experiments": [{"run_label": "E9_Heaney_movingfront", "ingest": "d6h_best"}],
    }
    d6._assert_integrity(
        master_dir=master,
        d6h_root=d6h,
        comp=comp,
        verdict={"master_comparison_run": "D6H_med_k1"},
    )


def test_finalize_only_on_completed_campaign() -> None:
    import pytest

    if not (DONE / "figure_master_comparison.png").is_file():
        pytest.skip("20260609T152334Z campaign not on disk")
    d6 = _load_d6()
    d6._finalize_gates(DONE)
