"""Tests for ``plot_jablczynski`` (three-panel Jabłczyński figure)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from continuous_patterns.core.plotting import plot_jablczynski


def test_plot_jablczynski_normal_case(tmp_path: Path) -> None:
    """Plot with 10 synthetic bands — all 3 panels render."""
    positions = np.linspace(10, 80, 10).tolist()
    spacings = np.diff(positions).tolist()
    q_ratios = [spacings[i + 1] / spacings[i] for i in range(len(spacings) - 1)]
    Q_positions = [positions[i + 1] / positions[i] for i in range(len(positions) - 1)]

    jab = {
        "n_bands": 10,
        "peak_positions": positions,
        "spacings": spacings,
        "q_ratios": q_ratios,
        "q_cv": float(np.std(q_ratios) / np.mean(q_ratios)),
        "Q_positions": Q_positions,
        "Q_cv": float(np.std(Q_positions) / np.mean(Q_positions)),
        "used_field": "phi_c",
    }

    out = tmp_path / "jab.png"
    plot_jablczynski(jab, out, title="test")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_jablczynski_insufficient_bands(tmp_path: Path) -> None:
    """``n_bands < 3`` → annotation plot, no crash."""
    jab = {
        "n_bands": 2,
        "peak_positions": [10.0, 20.0],
        "spacings": [10.0],
        "q_ratios": [],
        "q_cv": 0.0,
        "Q_positions": [2.0],
        "Q_cv": 0.0,
    }

    out = tmp_path / "jab_few.png"
    plot_jablczynski(jab, out, title="few bands test")
    assert out.exists()


def test_plot_jablczynski_zero_bands(tmp_path: Path) -> None:
    """``n_bands = 0`` → should not crash."""
    jab: dict = {"n_bands": 0, "peak_positions": [], "spacings": []}
    out = tmp_path / "jab_zero.png"
    plot_jablczynski(jab, out)
    assert out.exists()
