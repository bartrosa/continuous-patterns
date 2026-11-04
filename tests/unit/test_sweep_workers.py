"""Tests for parallel sweep execution in ``run_sweep_axes``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]


def test_workers_manifest_complete(tmp_path: Path) -> None:
    sweep_yaml = tmp_path / "mini_workers.yaml"
    sweep_yaml.write_text(
        yaml.safe_dump(
            {
                "sweep": {
                    "name": "mini_workers_test",
                    "base_config": "experiments/canonical/slab_kinetic/baseline.yaml",
                    "axes": [
                        {
                            "name": "c1",
                            "path": "physics.c1",
                            "values": [0.40, 0.42, 0.45, 0.49],
                        }
                    ],
                },
                "overrides": {
                    "time.T": 0.02,
                    "geometry.n": 64,
                    "physics.c2": 0.10,
                    "solver.backend": "numpy",
                },
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "sweep_out"
    cmd = [
        sys.executable,
        str(_REPO / "scripts" / "run_sweep_axes.py"),
        str(sweep_yaml),
        str(out_dir),
        "--workers",
        "4",
    ]
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    proc = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, check=False, env=env)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    manifests = list(out_dir.rglob("manifest_axes.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    success = [e for e in manifest["entries"] if e.get("status") == "success"]
    assert len(success) == 4
    for e in success:
        rel = e.get("relative_path")
        assert rel
        summary = manifests[0].parent / rel / "summary.json"
        assert summary.is_file(), f"missing summary for {rel}"
