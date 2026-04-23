#!/usr/bin/env python3
"""Read-only diagnostics for Experiment 2 Stage I → Stage II (Run A vs Run B).

Compares Run A ``final_state.npz`` to Run B's first HDF5 snapshot, analyzes
inside/outside-disk statistics, renders a four-row diagnostic figure, embeds
Solver/runner code excerpts, and writes a preliminary hypothesis (IC mismatch vs
cavity-mask dynamics). Does **not** modify the solver or re-run simulations.

Outputs (under ``results/agate_ch/``):

    stage_seq_diagnosis.json
    stage_seq_diagnosis.png
    stage_seq_diagnosis_report.md

Example:
    python -m continuous_patterns.agate_ch.diagnose_stage_seq \\
        --run-a results/agate_ch/stage_seq_run_a_YYYYMMDD \\
        --run-b results/agate_ch/stage_seq_run_b_YYYYMMDD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import h5py
except ImportError as exc:
    raise SystemExit("h5py required (agate optional extra)") from exc


def _repo_root() -> Path:
    """Return the repository root directory.

    Returns:
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[3]


def _results_agate_ch() -> Path:
    """Directory for agate_ch diagnosis artifacts.

    Returns:
        ``<repo>/results/agate_ch``.
    """
    return _repo_root() / "results" / "agate_ch"


def _load_h5_fields(h5_path: Path, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load ``c``, ``phi_m``, ``phi_c`` from the ``index``-th snapshot group in HDF5.

    Args:
        h5_path: Path to ``snapshots.h5``.
        index: Zero-based index into time-sorted ``t_*`` groups.

    Returns:
        Tuple ``(c, phi_m, phi_c, group_name)``.

    Raises:
        ValueError: If the HDF5 file contains no snapshot groups.
    """
    with h5py.File(h5_path, "r") as h5:
        keys = sorted(h5.keys(), key=lambda x: int(str(x).split("_")[1]))
        if not keys:
            raise ValueError(f"empty HDF5: {h5_path}")
        k = keys[index]
        g = h5[k]
        c = np.asarray(g["c"], dtype=np.float64)
        pm = np.asarray(g["phi_m"], dtype=np.float64)
        pc = np.asarray(g["phi_c"], dtype=np.float64)
        return c, pm, pc, str(k)


def _stats_global(a: np.ndarray) -> dict[str, float]:
    """Compute min, max, mean, and std over the full array.

    Args:
        a: Field values on the grid.

    Returns:
        Dictionary with keys ``min``, ``max``, ``mean``, ``std``.
    """
    a = np.asarray(a, dtype=np.float64)
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def _stats_mask(a: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute statistics of ``a`` restricted to ``mask`` (boolean).

    Args:
        a: Field values.
        mask: Boolean mask; same shape as ``a``.

    Returns:
        Stats dict; if no True entries, returns NaNs for all entries.
    """
    v = np.asarray(a, dtype=np.float64)[mask]
    if v.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
    }


def _disk_masks(
    n: int, L: float, *, outside_r: float = 85.0, inside_r: float = 75.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build radial masks for ``outside`` (annulus beyond cavity) and ``inside`` analyses.

    Args:
        n: Grid resolution per axis.
        L: Physical domain side length (square domain ``[0,L]^2``).
        outside_r: Cells with radius ``> outside_r`` are "outside" the cavity band.
        inside_r: Cells with radius ``< inside_r`` are "inside" the cavity core.

    Returns:
        Tuple ``(outside_mask, inside_mask)`` boolean arrays shaped ``(n, n)``.
    """
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    rv = np.sqrt((X - L / 2.0) ** 2 + (Y - L / 2.0) ** 2)
    outside_mask = rv > outside_r
    inside_mask = rv < inside_r
    return outside_mask, inside_mask


def _inside_metrics(pm: np.ndarray, pc: np.ndarray, inside: np.ndarray) -> dict[str, float]:
    """Summarize phase contrast and mixed-phase fraction inside a mask.

    Args:
        pm: Moganite phase field.
        pc: Chalcedony phase field.
        inside: Boolean mask for the interior region.

    Returns:
        Keys: ``std_phi_m``, ``mean_abs_phi_m_minus_phi_c``, ``fraction_mixed_phi_m``.
    """
    pm_i = pm[inside]
    pc_i = pc[inside]
    contrast = np.mean(np.abs(pm_i - pc_i))
    mixed_frac = float(np.mean((pm_i > 0.3) & (pm_i < 0.7)))
    return {
        "std_phi_m": float(np.std(pm_i)),
        "mean_abs_phi_m_minus_phi_c": float(contrast),
        "fraction_mixed_phi_m": mixed_frac,
    }


def _read_source_slice(rel: str, start_line: int, end_line: int) -> str:
    """Read a line-numbered excerpt from a file under ``agate_ch/``.

    Args:
        rel: Path relative to ``src/continuous_patterns/agate_ch/``.
        start_line: First 1-based line to include.
        end_line: Last 1-based line to include (inclusive).

    Returns:
        Text block with ``lineno | source`` formatting for Markdown reports.

    Raises:
        OSError: If the source file cannot be read.
    """
    path = _repo_root() / "src" / "continuous_patterns" / "agate_ch" / rel
    lines = path.read_text().splitlines()
    out: list[str] = []
    for i in range(start_line - 1, min(end_line, len(lines))):
        out.append(f"{i + 1:4d} | {lines[i]}")
    return "\n".join(out)


def run_diagnosis(
    run_a_dir: Path,
    run_b_dir: Path,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Execute all diagnosis checks and write JSON, PNG, and Markdown reports.

    Args:
        run_a_dir: Run A output directory (must contain ``final_state.npz``).
        run_b_dir: Run B output directory (must contain ``snapshots.h5`` and
            ``final_state.npz``).
        rtol: Relative tolerance for ``numpy.allclose`` (Run A vs Run B init).
        atol: Absolute tolerance for ``numpy.allclose``.

    Returns:
        Nested dict with ``checks`` (check1–6), ``preliminary_diagnosis``, and
        ``output_files`` paths.

    Raises:
        FileNotFoundError: If required input files are missing.
    """
    report: dict[str, Any] = {
        "run_a_dir": str(run_a_dir),
        "run_b_dir": str(run_b_dir),
        "checks": {},
    }

    # ----- Check 1 -----
    summ_b_path = run_b_dir / "summary.json"
    if not summ_b_path.is_file():
        raise FileNotFoundError(summ_b_path)
    summ_b = json.loads(summ_b_path.read_text())
    p = summ_b.get("parameters") or {}
    expected = {
        "gamma": 6.0,
        "enable_reaction": False,
        "enable_dirichlet": False,
        "initial_condition_type": "from_snapshot",
        "T_total": 20000.0,
    }
    ic_type = p.get("initial_condition")
    snap_path = p.get("snapshot_path", "")
    check1 = {
        "gamma": p.get("gamma"),
        "enable_reaction": p.get("enable_reaction"),
        "enable_dirichlet": p.get("enable_dirichlet"),
        "initial_condition": ic_type,
        "snapshot_path": snap_path,
        "T": p.get("T"),
        "seed": p.get("seed"),
        "warnings": [],
    }
    if float(p.get("gamma", -1)) != expected["gamma"]:
        check1["warnings"].append(f"gamma expected {expected['gamma']}, got {p.get('gamma')}")
    if p.get("enable_reaction") is not expected["enable_reaction"]:
        check1["warnings"].append(
            f"enable_reaction expected {expected['enable_reaction']}, "
            f"got {p.get('enable_reaction')}"
        )
    if p.get("enable_dirichlet") is not expected["enable_dirichlet"]:
        check1["warnings"].append(
            f"enable_dirichlet expected {expected['enable_dirichlet']}, "
            f"got {p.get('enable_dirichlet')}"
        )
    if str(ic_type) != expected["initial_condition_type"]:
        check1["warnings"].append(
            f"initial_condition expected {expected['initial_condition_type']}, got {ic_type}"
        )
    if float(p.get("T", -1)) != expected["T_total"]:
        check1["warnings"].append(f"T expected {expected['T_total']}, got {p.get('T')}")
    report["checks"]["check1_config_run_b"] = check1

    # ----- Load arrays -----
    npz_a = run_a_dir / "final_state.npz"
    npz_b_fin = run_b_dir / "final_state.npz"
    h5_b = run_b_dir / "snapshots.h5"
    if not npz_a.is_file():
        raise FileNotFoundError(npz_a)
    if not h5_b.is_file():
        raise FileNotFoundError(h5_b)

    da = np.load(npz_a, allow_pickle=False)
    A_pm = np.asarray(da["phi_m"], dtype=np.float64)
    A_pc = np.asarray(da["phi_c"], dtype=np.float64)
    A_c = np.asarray(da["c"], dtype=np.float64)

    B0_c, B0_pm, B0_pc, b0_key = _load_h5_fields(h5_b, 0)

    if not npz_b_fin.is_file():
        raise FileNotFoundError(npz_b_fin)
    db = np.load(npz_b_fin, allow_pickle=False)
    Bf_pm = np.asarray(db["phi_m"], dtype=np.float64)
    Bf_pc = np.asarray(db["phi_c"], dtype=np.float64)
    Bf_c = np.asarray(db["c"], dtype=np.float64)

    n = A_pm.shape[0]
    L = float(p.get("L", 200.0))
    outside_mask, inside_mask = _disk_masks(n, L)

    # ----- Check 2 -----
    check2: dict[str, Any] = {
        "run_b_first_h5_group": b0_key,
        "shapes": {
            "A_final": list(A_pm.shape),
            "B_init": list(B0_pm.shape),
        },
        "global_stats": {
            "A_final_phi_m": _stats_global(A_pm),
            "B_init_phi_m": _stats_global(B0_pm),
        },
        "fields": {},
    }
    all_ok = True
    for name, a_arr, b_arr in (
        ("phi_m", A_pm, B0_pm),
        ("phi_c", A_pc, B0_pc),
        ("c", A_c, B0_c),
    ):
        close = bool(np.allclose(a_arr, b_arr, rtol=rtol, atol=atol))
        mad = float(np.max(np.abs(a_arr - b_arr)))
        check2["fields"][name] = {
            "allclose": close,
            "max_abs_diff": mad,
            "stats_A": _stats_global(a_arr),
            "stats_B_init": _stats_global(b_arr),
        }
        all_ok = all_ok and close
    check2["all_fields_allclose"] = all_ok
    check2.setdefault("warnings", [])
    if not all_ok:
        check2["warnings"].append(
            "Run B first HDF5 snapshot differs from Run A final_state.npz "
            "(IC path or export mismatch)."
        )
    report["checks"]["check2_a_final_vs_b_init"] = check2

    # ----- Check 3 -----
    check3: dict[str, Any] = {}
    for label, pm, pc, c in (
        ("run_A_final", A_pm, A_pc, A_c),
        ("run_B_init", B0_pm, B0_pc, B0_c),
        ("run_B_final", Bf_pm, Bf_pc, Bf_c),
    ):
        check3[label] = {
            "phi_m_outside": _stats_mask(pm, outside_mask),
            "phi_c_outside": _stats_mask(pc, outside_mask),
            "c_outside": _stats_mask(c, outside_mask),
        }
    b_init_std_pm = check3["run_B_init"]["phi_m_outside"]["std"]
    check3["warnings"] = []
    if b_init_std_pm is not None and b_init_std_pm > 0.01:
        check3["warnings"].append(
            "Run B init outside-disk std(phi_m) > 0.01 — "
            "snapshot IC may not preserve zeros outside cavity."
        )
    report["checks"]["check3_outside_disk"] = check3

    # ----- Check 4 -----
    check4: dict[str, Any] = {}
    for label, pm, pc in (
        ("run_A_final", A_pm, A_pc),
        ("run_B_init", B0_pm, B0_pc),
        ("run_B_final", Bf_pm, Bf_pc),
    ):
        check4[label] = _inside_metrics(pm, pc, inside_mask)
    report["checks"]["check4_inside_disk"] = check4

    # ----- Check 5 figure -----
    out_dir = _results_agate_ch()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "stage_seq_diagnosis.png"

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    col_titles = (r"$\phi_m$", r"$\phi_c$", r"$\phi_m-\phi_c$", r"$c$")

    def row_imshow(
        row: int, pm: np.ndarray, pc: np.ndarray, c: np.ndarray, title_prefix: str
    ) -> None:
        diff = pm - pc
        ims = [
            (pm, "viridis", 0.0, 1.0),
            (pc, "viridis", 0.0, 1.0),
            (diff, "RdBu_r", -1.0, 1.0),
            (c, "viridis", None, None),
        ]
        for col, ((arr, cmap, vmin, vmax)) in enumerate(ims):
            ax = axes[row, col]
            if vmin is None:
                im = ax.imshow(arr, origin="lower", cmap=cmap)
            else:
                im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{title_prefix} — {col_titles[col]}")
            ax.set_xticks([])
            ax.set_yticks([])

    row_imshow(0, A_pm, A_pc, A_c, "A final")
    row_imshow(1, B0_pm, B0_pc, B0_c, "B init")
    row_imshow(2, Bf_pm, Bf_pc, Bf_c, "B final")
    for col, (name, a_arr, b_arr) in enumerate(
        (
            ("phi_m", A_pm, B0_pm),
            ("phi_c", A_pc, B0_pc),
            ("phi_m-phi_c", A_pm - A_pc, B0_pm - B0_pc),
            ("c", A_c, B0_c),
        )
    ):
        d = b_arr - a_arr
        ax = axes[3, col]
        vmax = float(np.max(np.abs(d))) + 1e-12
        im = ax.imshow(d, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"diff B_init - A_final {name}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Stage sequence diagnosis (rows: A final, B init, B final, diff)", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    report["checks"]["check5_figure"] = {"path": str(png_path)}

    max_diff_row4 = max(
        float(np.max(np.abs(B0_pm - A_pm))),
        float(np.max(np.abs(B0_pc - A_pc))),
        float(np.max(np.abs(B0_c - A_c))),
        float(np.max(np.abs((B0_pm - B0_pc) - (A_pm - A_pc)))),
    )
    report["checks"]["check5_max_abs_diff_Binit_minus_Afinal"] = max_diff_row4

    # ----- Check 6 code excerpts (paths relative to agate_ch package) -----
    check6 = {
        "initial_state_from_snapshot_solver_py_lines_289_331": _read_source_slice(
            "solver.py", 289, 331
        ),
        "build_initial_state_solver_py_lines_333_372": _read_source_slice("solver.py", 333, 372),
        "imex_step_cavity_mask_solver_py_lines_199_208": _read_source_slice("solver.py", 199, 208),
        "run_simulation_ic_run_py_lines_782_811": _read_source_slice("run.py", 782, 811),
    }
    report["checks"]["check6_code_excerpts"] = check6

    # ----- Preliminary diagnosis -----
    reasons: list[str] = []
    hypothesis = "undetermined"

    init_out_std = check3["run_B_init"]["phi_m_outside"]["std"]
    fin_out_std = check3["run_B_final"]["phi_m_outside"]["std"]
    fin_out_mean_pm = check3["run_B_final"]["phi_m_outside"]["mean"]

    if not check2["all_fields_allclose"]:
        hypothesis = "B_or_C"
        reasons.append("Check2: B_init differs from A_final — IC pipeline or HDF5 step-0 mismatch.")
    elif init_out_std <= 0.01 and fin_out_std > 0.05:
        hypothesis = "A"
        reasons.append(
            "Check3: B_init outside disk ~flat (snapshot OK), but B_final outside shows "
            "variation — consistent with χ cavity projection tied to Dirichlet flags in older code "
            "(fixed: apply_cavity_mask independent of disable_dirichlet), so γ-CH evolved φ on the "
            "periodic grid outside χ."
        )
    elif init_out_std > 0.01:
        hypothesis = "B_or_C"
        reasons.append(
            "Check3: non-trivial outside-disk stats already at B init — implies "
            "snapshot/HDF5 export differs from A final or order fields wrong."
        )
    elif abs(fin_out_mean_pm - 0.5) < 0.15 and fin_out_std > 0.1:
        hypothesis = "A"
        reasons.append(
            "B final outside mean(phi_m) near 0.5 with large std — typical spinodal mixture on "
            "unconstrained domain."
        )

    report["preliminary_diagnosis"] = {
        "hypothesis_label": hypothesis,
        "hypothesis_legend": {
            "A": (
                "χ cavity projection was skipped when Dirichlet disabled (legacy) — "
                "dynamics outside χ."
            ),
            "B_or_C": "IC mismatch: noise/double init or partial load (Checks 2–3).",
        },
        "reasons": reasons,
    }

    json_path = out_dir / "stage_seq_diagnosis.json"

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    json_path.write_text(json.dumps(_json_safe(report), indent=2))

    # ----- Markdown report -----
    md_lines = [
        "# Stage sequence diagnosis report",
        "",
        f"- Run A: `{run_a_dir}`",
        f"- Run B: `{run_b_dir}`",
        "",
        "## Check 1 — Run B configuration",
        "",
        "```json",
        json.dumps(check1, indent=2),
        "```",
        "",
        "## Check 2 — A final vs B init (HDF5 first snapshot)",
        "",
        "```json",
        json.dumps({k: v for k, v in check2.items() if k != "warnings"}, indent=2),
        "```",
        "",
        "## Check 3 — Outside disk (r > 85)",
        "",
        "```json",
        json.dumps(check3, indent=2),
        "```",
        "",
        "## Check 4 — Inside disk (r < 75)",
        "",
        "```json",
        json.dumps(check4, indent=2),
        "```",
        "",
        "## Check 5 — Figure",
        "",
        "![diagnosis](stage_seq_diagnosis.png)",
        "",
        f"Max abs diff (B_init − A_final) across scalar panels: **{max_diff_row4:.6e}**",
        "",
        "## Check 6 — Code excerpts (read-only)",
        "",
        "### `initial_state_from_snapshot` (solver.py)",
        "",
        "```python",
        check6["initial_state_from_snapshot_solver_py_lines_289_331"],
        "```",
        "",
        "### `build_initial_state` (solver.py)",
        "",
        "```python",
        check6["build_initial_state_solver_py_lines_333_372"],
        "```",
        "",
        "### Cavity mask in `imex_step` (solver.py)",
        "",
        "```python",
        check6["imex_step_cavity_mask_solver_py_lines_199_208"],
        "```",
        "",
        "### `run_simulation` IC + snapshots (run.py)",
        "",
        "```python",
        check6["run_simulation_ic_run_py_lines_782_811"],
        "```",
        "",
        "## Preliminary diagnosis",
        "",
        f"**Hypothesis:** `{hypothesis}`",
        "",
        *(f"- {r}" for r in reasons),
        "",
        "---",
        "Artifacts: `stage_seq_diagnosis.json`, `stage_seq_diagnosis.png`, this file.",
        "",
    ]
    md_path = out_dir / "stage_seq_diagnosis_report.md"
    md_path.write_text("\n".join(md_lines))

    report["output_files"] = {
        "json": str(json_path),
        "png": str(png_path),
        "md": str(md_path),
    }

    return report


def main() -> None:
    """CLI entry: parse paths, run :func:`run_diagnosis`, print summaries to stdout."""
    root = _repo_root()
    ap = argparse.ArgumentParser(
        description="Diagnose stage_seq Run A / Run B consistency (read-only)."
    )
    ap.add_argument(
        "--run-a",
        type=str,
        default=str(root / "results" / "agate_ch" / "stage_seq_run_a_20260423_115846"),
        help="Run A output directory",
    )
    ap.add_argument(
        "--run-b",
        type=str,
        default=str(root / "results" / "agate_ch" / "stage_seq_run_b_20260423_115846"),
        help="Run B output directory",
    )
    args = ap.parse_args()
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    if not run_a.is_absolute():
        run_a = root / run_a
    if not run_b.is_absolute():
        run_b = root / run_b

    rep = run_diagnosis(run_a, run_b)
    print(json.dumps(rep["checks"]["check1_config_run_b"], indent=2))
    print("\n--- Check 2 ---")
    print(json.dumps(rep["checks"]["check2_a_final_vs_b_init"], indent=2))
    print("\n--- Check 3 (outside) ---")
    print(json.dumps(rep["checks"]["check3_outside_disk"], indent=2))
    print("\n--- Preliminary diagnosis ---")
    print(json.dumps(rep["preliminary_diagnosis"], indent=2))
    print("\nWritten:", rep["output_files"])


if __name__ == "__main__":
    main()
