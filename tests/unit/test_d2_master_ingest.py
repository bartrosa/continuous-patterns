"""D2 master comparison ingest smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
D2H_RUN = (
    REPO
    / "results/slab_twophase/d2h_heaney_movingfront/20260609T105809Z/D2H_poster/20260609T105810Z"
)


@pytest.mark.skipif(not D2H_RUN.is_dir(), reason="D2H sweep not on disk")
def test_d2h_ingest_row_handles_null_lambda_ch() -> None:
    import importlib.util

    script = REPO / "scripts/run_banding_comparison.py"
    spec = importlib.util.spec_from_file_location("run_banding_comparison", script)
    assert spec is not None and spec.loader is not None
    bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc)

    from continuous_patterns.core.io import load_run_config
    from continuous_patterns.experiments.checkpoint import load_sim_result_from_run_dir

    comp = yaml.safe_load(
        (REPO / "experiments/comparisons/d4_master_comparison.yaml").read_text(encoding="utf-8")
    )["comparison"]
    entry = dict(next(e for e in comp["experiments"] if e["run_label"] == "E9_Heaney_movingfront"))
    entry["ingest"] = "d2h"
    entry["ingest_run"] = "D2H_poster"
    comp["d2h_sweep_dir"] = "results/slab_twophase/d2h_heaney_movingfront"
    run_dir = bc._resolve_ingest_run_dir(entry, comp, REPO)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary.get("lambda_ch") is None

    cfg = load_run_config(run_dir / "config.yaml")
    result = load_sim_result_from_run_dir(run_dir, "slab_twophase")
    _, _, _, row, _, _, _ = bc._build_comparison_row(entry, result, cfg, {})
    assert row["row_status"] in ("OK", "WARN", "RUN_INVALID", "NUMERICALLY_INVALID")
