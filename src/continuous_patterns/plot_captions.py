"""Reusable figure footers listing experiment parameters (traceability on PNG exports).

GIF and video writers are intentionally excluded — only static raster figures
(typically Matplotlib ``savefig``) get the parameter strip.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CAPTION_OVERRIDE = "__caption_override__"


def format_run_parameters_caption(cfg: Mapping[str, Any] | None) -> str:
    """Render a multi-line monospace block of run parameters for figure footers.

    Args:
        cfg: Flat simulation parameters (e.g. merged YAML), or an aggregate dict
            for sweep summaries. If ``cfg`` contains ``__caption_override__``,
            that string is used after text wrapping (other keys ignored).

    Returns:
        Wrapped text suitable for ``Figure.text`` (may be empty if ``cfg`` is
        ``None`` or empty).
    """
    if not cfg:
        return ""
    if _CAPTION_OVERRIDE in cfg:
        raw = str(cfg[_CAPTION_OVERRIDE])
    else:
        public = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
        try:
            raw = json.dumps(public, sort_keys=True, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            raw = repr(dict(public))
    lines: list[str] = []
    for segment in raw.split("\n"):
        lines.extend(
            textwrap.wrap(
                segment,
                width=96,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    return "\n".join(lines)


def figure_add_parameter_footer(
    fig: Any,
    cfg: Mapping[str, Any] | None,
    *,
    fontsize: float = 5.5,
) -> list[Any]:
    """Reserve bottom space and draw parameter text on a Matplotlib figure.

    Args:
        fig: A :class:`matplotlib.figure.Figure`.
        cfg: Parameters to render; skipped if ``None`` or empty caption.
        fontsize: Footnote font size (monospace; default tuned for readability on typical DPI).

    Returns:
        List of artists to pass as ``bbox_extra_artists`` when using
        ``bbox_inches=\"tight\"``, or an empty list.
    """
    if not cfg:
        return []
    text = format_run_parameters_caption(cfg)
    if not text.strip():
        return []
    nlines = max(1, text.count("\n") + 1)
    bottom_frac = min(0.52, max(0.13, 0.085 + nlines * 0.012))
    fig.subplots_adjust(bottom=bottom_frac)
    t = fig.text(
        0.5,
        0.002,
        text,
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        family="monospace",
        color="#1a1a1a",
    )
    return [t]


def figure_save_png_with_params(
    fig: Any,
    path: Path | str,
    cfg: Mapping[str, Any] | None,
    *,
    dpi: float | None = None,
    **savefig_kw: Any,
) -> None:
    """Save a PNG with an optional experiment-parameter footer.

    Args:
        fig: Matplotlib figure handle.
        path: Output path (``.png``).
        cfg: Parameters footer; omit for a plain save (same as vanilla
            ``savefig`` aside from defaults).
        dpi: Dots per inch; forwarded to ``savefig``.
        **savefig_kw: Extra arguments to ``Figure.savefig``. When a footer is
            present, ``bbox_inches='tight'``, ``bbox_extra_artists``, and
            ``pad_inches`` are set unless already provided.

    Note:
        Call **after** ``tight_layout()`` so the footer margins apply last.
    """
    path = Path(path)
    extra = figure_add_parameter_footer(fig, cfg)
    kw = dict(savefig_kw)
    if dpi is not None:
        kw["dpi"] = dpi
    if extra:
        kw.setdefault("bbox_inches", "tight")
        kw.setdefault("bbox_extra_artists", extra)
        kw.setdefault("pad_inches", 0.28)
    fig.savefig(path, **kw)


def pyplot_save_png_with_params(
    path: Path | str,
    cfg: Mapping[str, Any] | None,
    **savefig_kw: Any,
) -> None:
    """Like :func:`figure_save_png_with_params` but uses the current pyplot figure."""
    import matplotlib.pyplot as plt

    figure_save_png_with_params(plt.gcf(), path, cfg, **savefig_kw)
