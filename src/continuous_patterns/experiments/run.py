"""Canonical CLI: nested config → simulate → ``results/`` tree.

``python -m continuous_patterns.experiments.run --config …`` (``docs/ARCHITECTURE.md`` §6.1).

Environment: ``CP_OVERRIDE_T`` — if set to a float string, overrides ``time.T`` after
``load_run_config`` (used by ``scripts/reproduce_canonical.py`` mini mode).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from continuous_patterns.core.io import (
    ResultPaths,
    allocate_run_dir,
    load_run_config,
    save_final_state_npz,
    save_run_config,
    save_snapshots_h5,
    save_summary,
    write_figures_final,
)
from continuous_patterns.core.plotting import (
    parse_run_stamp_utc,
    plot_jablczynski,
    write_evolution_gif,
)
from continuous_patterns.core.types import SimResult
from continuous_patterns.models import agate_ch, agate_stage2

MODEL_DISPATCH: dict[str, Any] = {
    "agate_ch": agate_ch.simulate,
    "agate_stage2": agate_stage2.simulate,
}

logger = logging.getLogger(__name__)

_LOG_FORMAT = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _has_stdout_console(pkg: logging.Logger) -> bool:
    return any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in pkg.handlers
    )


def _setup_run_logger(paths: ResultPaths | None, log_level: str) -> logging.Handler | None:
    """Attach console (stdout) and optional file handler to ``continuous_patterns`` logger.

    Returns the file handler when ``paths`` is set (caller removes it in ``finally``).
    """
    pkg = logging.getLogger("continuous_patterns")
    pkg.setLevel(logging.DEBUG)
    pkg.propagate = False

    if not _has_stdout_console(pkg):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console.setFormatter(_LOG_FORMAT)
        pkg.addHandler(console)

    file_handler: logging.Handler | None = None
    if paths is not None:
        log_path = paths.log_file
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_LOG_FORMAT)
        pkg.addHandler(fh)
        file_handler = fh

    return file_handler


def run_one(
    cfg: dict[str, Any],
    *,
    results_root: Path | None = None,
    chunk_size: int = 2000,
    write_artifacts: bool = True,
    show_progress: bool = True,
    log_level: str = "INFO",
) -> SimResult:
    """Run one simulation; optionally write ``results/`` artifacts.

    Dispatches on ``cfg["experiment"]["model"]``. When ``write_artifacts`` is
    true, ``results_root`` must be set (defaults in the CLI to ``./results``).
    """
    model_name = cfg["experiment"]["model"]
    if model_name not in MODEL_DISPATCH:
        raise ValueError(f"Unknown model: {model_name!r}. Available: {sorted(MODEL_DISPATCH)}")
    simulate_fn = MODEL_DISPATCH[model_name]

    cfg.setdefault("output", {})
    cfg["output"]["record_spectral_mass_diagnostic"] = True

    paths: ResultPaths | None = None
    if write_artifacts:
        if results_root is None:
            raise ValueError("results_root is required when write_artifacts=True")
        paths = allocate_run_dir(
            experiment_name=str(cfg["experiment"]["name"]),
            results_root=Path(results_root),
        )

    file_handler = _setup_run_logger(paths, log_level)
    try:
        logger.info("Dispatching to model %s", model_name)
        t_sim0 = time.perf_counter()
        result = simulate_fn(cfg, chunk_size=chunk_size, show_progress=show_progress)
        wall_s = time.perf_counter() - t_sim0

        if write_artifacts:
            assert paths is not None
            save_run_config(paths.config_yaml, cfg)
            save_summary(paths.summary_json, result.diagnostics)
            save_final_state_npz(
                paths.final_state_npz,
                phi_m=np.asarray(result.state_final.phi_m),
                phi_c=np.asarray(result.state_final.phi_c),
                c=np.asarray(result.state_final.c),
                chi=None,
            )
            gcfg = cfg["geometry"]
            include_panel = bool(cfg.get("output", {}).get("include_params_panel", True))
            ts_human = parse_run_stamp_utc(paths.root.name)
            exp_name = str(cfg["experiment"]["name"])
            fig_title = f"{exp_name} — {ts_human}" if ts_human else exp_name
            params_for_panel: dict[str, Any] | None = None
            if include_panel:
                params_for_panel = dict(cfg)
                params_for_panel["_diagnostics"] = {**result.diagnostics, "wall_time_s": wall_s}
            write_figures_final(
                paths.root,
                phi_m=np.asarray(result.state_final.phi_m),
                phi_c=np.asarray(result.state_final.phi_c),
                c=np.asarray(result.state_final.c),
                L=float(gcfg["L"]),
                R=float(gcfg.get("R", 0.0)),
                title=fig_title,
                params=params_for_panel,
                include_params_panel=include_panel,
            )
            out = cfg.get("output", {})
            if bool(out.get("save_jablczynski_plot", False)):
                jab = result.diagnostics.get("jab_canonical")
                if isinstance(jab, dict) and jab:
                    meta = result.meta if isinstance(result.meta, dict) else {}
                    plot_jablczynski(
                        jab,
                        paths.jablczynski_plot,
                        title=f"{exp_name} — Jabłczyński analysis",
                        radial_centers=meta.get("radial_centers"),
                        radial_profile=meta.get("radial_profile"),
                    )
                    logger.info("Wrote Jabłczyński plot to %s", paths.jablczynski_plot)
            if bool(out.get("save_snapshots_h5", False)):
                h5_snaps = result.meta.get("h5_snapshots")
                if isinstance(h5_snaps, list) and h5_snaps:
                    save_snapshots_h5(
                        paths.snapshots_h5,
                        h5_snaps,
                        dt=float(cfg["time"]["dt"]),
                        cfg_summary={
                            "experiment": cfg.get("experiment", {}),
                            "geometry": cfg.get("geometry", {}),
                            "physics": cfg.get("physics", {}),
                            "time": cfg.get("time", {}),
                        },
                    )
                    logger.info(
                        "Wrote %d snapshots to %s",
                        len(h5_snaps),
                        paths.snapshots_h5,
                    )
                    h5_snaps.clear()
            if bool(out.get("record_evolution_gif", False)):
                gif_snaps = result.meta.get("gif_snapshots")
                if isinstance(gif_snaps, list) and gif_snaps:
                    gif_path = paths.root / "evolution_phi_m.gif"
                    fps = int(out.get("gif_fps", 10))
                    write_evolution_gif(
                        gif_snaps,
                        gif_path,
                        L=float(gcfg["L"]),
                        R=float(gcfg.get("R", 0.0)),
                        fps=fps,
                        field_name="phi_m",
                    )
                    logger.info("Wrote evolution GIF to %s", gif_path)
            logger.info("Artifacts written to %s", paths.root)
            return replace(result, paths=paths)

        return result
    finally:
        if file_handler is not None:
            pkg = logging.getLogger("continuous_patterns")
            pkg.removeHandler(file_handler)
            file_handler.close()


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--config``, ``--out-dir``, ``--chunk-size``, ``--no-write``, logging flags."""
    parser = argparse.ArgumentParser(
        prog="continuous_patterns.experiments.run",
        description="Run one simulation from a nested YAML config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to nested YAML config.",
    )
    parser.add_argument(
        "--user-settings",
        type=Path,
        default=None,
        help=(
            "Optional solver-overrides YAML "
            "(default: experiments/solver_settings.yaml when present)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Results root directory (default: ./results).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="IMEX chunk size.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing artifacts (smoke testing).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console log level (run.log is always DEBUG when writing artifacts).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_run_config(
            Path(args.config),
            user_settings_path=args.user_settings,
        )
    except ValueError as e:
        print(f"Config load failed: {e}", file=sys.stderr)
        return 2

    ot = os.environ.get("CP_OVERRIDE_T")
    if ot is not None and str(ot).strip() != "":
        cfg.setdefault("time", {})
        cfg["time"]["T"] = float(ot)

    try:
        result = run_one(
            cfg,
            results_root=None if args.no_write else args.out_dir,
            chunk_size=int(args.chunk_size),
            write_artifacts=not args.no_write,
            show_progress=not args.no_progress,
            log_level=str(args.log_level),
        )
    except Exception as e:
        print(f"Simulation failed: {e}", file=sys.stderr)
        return 1

    if result.paths is not None:
        print(f"Run complete: {result.paths.root}")
    else:
        print("Run complete (no artifacts written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
