"""src/viz - Visualization modules for the Spectral Separability Explorer.

This package provides 7 plotting modules used by Tab 5 (Results) and the
export pipeline. Each module exposes pure functions that consume the same
input shape::

    df:        pd.DataFrame
    class_col: str
    bands:     list[str]
    *, palette: dict | None = None
    **kwargs

and return a ``plotly.graph_objects.Figure`` ready to be displayed in
Gradio (via ``gr.Plot``) or exported via the ``fig_to_png`` / ``fig_to_svg``
helpers below.

Public API
==========

Re-exports (from ``src.core`` — single source of truth for JM thresholds and
the 4-bucket palette, per HANDOFF Decision #17):

    JM_BUCKETS, JM_BUCKET_COLORS,
    JM_THRESHOLD_POOR, JM_THRESHOLD_MODERATE, JM_THRESHOLD_GOOD, JM_MAX

Style constants:

    DEFAULT_PLOT_TEMPLATE, DEFAULT_PLOT_LAYOUT,
    DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE,
    DEFAULT_TITLE_FONT_SIZE, DEFAULT_AXIS_TITLE_FONT_SIZE,
    DEFAULT_EXPORT_WIDTH, DEFAULT_EXPORT_HEIGHT, DEFAULT_EXPORT_SCALE

Helpers:

    generate_class_palette(class_ids, base="tab10") -> dict
    order_bands_by_wavelength(bands, wavelengths=None) -> list
    apply_modern_layout(fig, **overrides) -> go.Figure
    fig_to_png(fig, path, *, width, height, scale) -> Path
    fig_to_svg(fig, path, *, width, height) -> Path
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.colors as _mcolors
import plotly.graph_objects as go
from matplotlib import colormaps as _colormaps

# ---------------------------------------------------------------- Re-exports
# Source of truth for JM thresholds + 4-bucket palette (HANDOFF Decision #17).
# These constants flow through every JM-derived plot (heatmap colour scale,
# bucket legend, ranked-pairs row colouring, comparative subset summaries).
from src.core import (
    JM_BUCKETS,
    JM_BUCKET_COLORS,
    JM_MAX,
    JM_THRESHOLD_GOOD,
    JM_THRESHOLD_MODERATE,
    JM_THRESHOLD_POOR,
)


# ---------------------------------------------------------------- Constants
#: Plotly base template — clean white background, subtle gridlines.
#: Matches the Spectral 3D Explorer visual aesthetic.
DEFAULT_PLOT_TEMPLATE: str = "plotly_white"

#: Modern font stack (system fonts → Inter when installed, with Apple
#: BlinkMac fallback for macOS rendering and a generic sans-serif fallback).
DEFAULT_FONT_FAMILY: str = (
    "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
)

#: Default font sizes (px).
DEFAULT_FONT_SIZE: int = 12
DEFAULT_TITLE_FONT_SIZE: int = 16
DEFAULT_AXIS_TITLE_FONT_SIZE: int = 13

#: Default layout dict applied by :func:`apply_modern_layout`. Kept as a plain
#: dict so callers can inspect / override individual keys without re-defining
#: the entire layout.
DEFAULT_PLOT_LAYOUT: dict[str, Any] = dict(
    template=DEFAULT_PLOT_TEMPLATE,
    font=dict(
        family=DEFAULT_FONT_FAMILY,
        size=DEFAULT_FONT_SIZE,
        color="#1f2937",  # Tailwind slate-800
    ),
    title=dict(
        font=dict(
            family=DEFAULT_FONT_FAMILY,
            size=DEFAULT_TITLE_FONT_SIZE,
            color="#111827",  # Tailwind slate-900
        ),
        x=0.5,
        xanchor="center",
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=70, r=30, t=70, b=60),
    hoverlabel=dict(
        font=dict(family=DEFAULT_FONT_FAMILY, size=12),
        bgcolor="white",
        bordercolor="#d1d5db",  # Tailwind gray-300
    ),
    legend=dict(
        font=dict(family=DEFAULT_FONT_FAMILY, size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e5e7eb",  # Tailwind gray-200
        borderwidth=1,
    ),
)

#: Default static-export sizes (used by :func:`fig_to_png` / :func:`fig_to_svg`).
DEFAULT_EXPORT_WIDTH: int = 1200
DEFAULT_EXPORT_HEIGHT: int = 800
DEFAULT_EXPORT_SCALE: int = 2  # 2× for retina/print clarity (PNG only)


# ---------------------------------------------------------------- Helpers
def generate_class_palette(
    class_ids: Sequence[Any],
    base: str = "tab10",
) -> dict[Any, str]:
    """Build a stable ``{class_id: "#rrggbb"}`` mapping from a Matplotlib cmap.

    Same input → same output (deterministic). Wraps around modulo ``cmap.N``
    when ``len(class_ids)`` exceeds the cmap size.

    Parameters
    ----------
    class_ids
        Sequence of class identifiers (any hashable). The i-th id maps to the
        i-th colour of the cmap.
    base
        Matplotlib qualitative cmap name. Use ``"tab10"`` for ≤10 classes
        (recommended for the 7 thesis classes) or ``"tab20"`` for up to 20.

    Returns
    -------
    dict
        Mapping ``{class_id: "#rrggbb"}``.

    Examples
    --------
    >>> palette = generate_class_palette([0, 1, 2])
    >>> set(palette) == {0, 1, 2}
    True
    >>> all(v.startswith("#") and len(v) == 7 for v in palette.values())
    True
    """
    cmap = _colormaps[base]
    n = max(int(getattr(cmap, "N", 10)), 1)
    return {
        cid: _mcolors.to_hex(cmap(i % n))
        for i, cid in enumerate(class_ids)
    }


def order_bands_by_wavelength(
    bands: Sequence[str],
    wavelengths: Mapping[str, float] | None = None,
) -> list[str]:
    """Reorder bands by ascending centre wavelength when a mapping is supplied.

    Bands missing from ``wavelengths`` are placed at the end while preserving
    their relative input order. When ``wavelengths`` is ``None`` or empty, a
    plain copy of ``bands`` is returned (preserves input order, e.g. for
    sensors without published spectral data).

    Parameters
    ----------
    bands
        Band names to reorder.
    wavelengths
        Optional mapping ``{band_name: centre_nm}``. Typically obtained from
        :func:`src.core.get_band_wavelengths` for known presets.

    Returns
    -------
    list
        Reordered band names.

    Examples
    --------
    >>> wl = {"Blue": 475, "Green": 560, "Red": 668}
    >>> order_bands_by_wavelength(["Red", "Blue", "Green"], wl)
    ['Blue', 'Green', 'Red']
    >>> order_bands_by_wavelength(["NIR", "Red"], {"Red": 668})
    ['Red', 'NIR']
    >>> order_bands_by_wavelength(["B1", "B2"], None)
    ['B1', 'B2']
    """
    if not wavelengths:
        return list(bands)
    bands_list = list(bands)
    return sorted(
        bands_list,
        key=lambda b: (
            0 if b in wavelengths else 1,    # known bands first
            wavelengths.get(b, 0.0),         # then by ascending wavelength
            bands_list.index(b),             # tiebreak: input order
        ),
    )


def apply_modern_layout(fig: go.Figure, **overrides: Any) -> go.Figure:
    """Apply :data:`DEFAULT_PLOT_LAYOUT` to a Plotly figure (in-place + returned).

    Parameters
    ----------
    fig
        Plotly figure to mutate.
    **overrides
        Layout properties to override on top of the defaults
        (e.g. ``title="JM Heatmap — 7D"``, ``height=600``,
        ``xaxis_title="Wavelength (nm)"``).

    Returns
    -------
    go.Figure
        The same figure (returned for chaining).

    Notes
    -----
    Mutates ``fig`` in-place via ``fig.update_layout`` — call once per
    figure as the final step before returning from a plot factory.
    """
    fig.update_layout(**DEFAULT_PLOT_LAYOUT)
    if overrides:
        fig.update_layout(**overrides)
    return fig


def fig_to_png(
    fig: go.Figure,
    path: str | Path,
    *,
    width: int = DEFAULT_EXPORT_WIDTH,
    height: int = DEFAULT_EXPORT_HEIGHT,
    scale: int = DEFAULT_EXPORT_SCALE,
) -> Path:
    """Export a Plotly figure to PNG via ``kaleido``.

    Creates parent directories as needed.

    Parameters
    ----------
    fig
        Plotly figure to export.
    path
        Output filepath (str or Path). Suffix is not enforced — caller
        decides; ``.png`` recommended.
    width, height
        Output pixel dimensions before scaling.
    scale
        Resolution multiplier (e.g. ``2`` → 2400×1600 for default 1200×800).
        Recommended for print or high-DPI displays.

    Returns
    -------
    Path
        The resolved output path.

    Raises
    ------
    ValueError
        Propagated from kaleido if the figure cannot be rendered.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), format="png", width=width, height=height, scale=scale)
    return out


def fig_to_svg(
    fig: go.Figure,
    path: str | Path,
    *,
    width: int = DEFAULT_EXPORT_WIDTH,
    height: int = DEFAULT_EXPORT_HEIGHT,
) -> Path:
    """Export a Plotly figure to SVG via ``kaleido``.

    Creates parent directories as needed. SVG export is resolution-independent,
    so no ``scale`` parameter is provided (figures scale losslessly).

    Parameters
    ----------
    fig
        Plotly figure to export.
    path
        Output filepath (str or Path). ``.svg`` suffix recommended.
    width, height
        Output dimensions in SVG user units.

    Returns
    -------
    Path
        The resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), format="svg", width=width, height=height)
    return out


# ---------------------------------------------------------------- Public API
__all__ = [
    # Re-exported from src.core (source of truth)
    "JM_BUCKETS",
    "JM_BUCKET_COLORS",
    "JM_MAX",
    "JM_THRESHOLD_GOOD",
    "JM_THRESHOLD_MODERATE",
    "JM_THRESHOLD_POOR",
    # Style constants
    "DEFAULT_PLOT_TEMPLATE",
    "DEFAULT_PLOT_LAYOUT",
    "DEFAULT_FONT_FAMILY",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_TITLE_FONT_SIZE",
    "DEFAULT_AXIS_TITLE_FONT_SIZE",
    "DEFAULT_EXPORT_WIDTH",
    "DEFAULT_EXPORT_HEIGHT",
    "DEFAULT_EXPORT_SCALE",
    # Helpers
    "generate_class_palette",
    "order_bands_by_wavelength",
    "apply_modern_layout",
    "fig_to_png",
    "fig_to_svg",
]
