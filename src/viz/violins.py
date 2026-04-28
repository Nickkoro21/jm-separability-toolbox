"""src/viz/violins.py — Per-band violin plots with classes side-by-side.

Same band-faceted layout as :mod:`src.viz.boxplots`, but each class is
rendered as a violin (kernel-density estimate) rather than a box. The
violin shape reveals **distribution features that boxplots hide**:
bimodality, skewness, long tails, density peaks. An optional inner
box gives the median + IQR landmarks alongside the KDE shape — the
standard "violin + inner box" combo favoured in scientific reporting.

Compared to ``boxplots.py``:
    * Violin = full distribution shape (KDE)
    * Boxplot = five-number summary
    * Inner box (default ON) brings them together: shape + summary

Public API
----------
make_violins(df, class_col, bands, *, wavelengths=None,
             class_labels=None, palette=None,
             show_inner_box=True, show_points=False,
             n_cols=None, height_per_row=320) -> go.Figure
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz import (
    apply_modern_layout,
    generate_class_palette,
    order_bands_by_wavelength,
)

# ─── Type aliases ────────────────────────────────────────────────────────
# Plotly's `points` parameter accepts a string mode or False for "off".
ShowPointsOption = Union[Literal["outliers", "all", "suspectedoutliers"], bool]


# ─── Internal helpers ────────────────────────────────────────────────────
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#rrggbb`` (or ``#rgb``) string to ``rgba(r,g,b,a)``."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _empty_figure(message: str) -> go.Figure:
    """Return a clean, centred placeholder figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#6b7280"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_modern_layout(fig, height=400)


def _resolve_class_label(cid: Any, mapping: Mapping[Any, str] | None) -> str:
    """Resolve a class id to a display string. Falls back to ``str(cid)``."""
    if mapping is None:
        return str(cid)
    return str(mapping.get(cid, cid))


def _validate_inputs(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
) -> str | None:
    """Return an error message if inputs are invalid, else None."""
    if df is None or len(df) == 0:
        return "No data available."
    if class_col not in df.columns:
        return f"Class column '{class_col}' not found."
    bands_list = list(bands)
    if not bands_list:
        return "At least one band is required."
    missing = [b for b in bands_list if b not in df.columns]
    if missing:
        return f"Missing band columns: {', '.join(missing)}"
    if df[class_col].nunique() < 1:
        return "At least 1 class required."
    return None


def _choose_grid(n_bands: int, user_n_cols: int | None) -> tuple[int, int]:
    """Compute ``(n_rows, n_cols)`` for the band-faceted subplot grid."""
    if user_n_cols is not None and user_n_cols > 0:
        n_cols = user_n_cols
    else:
        n_cols = max(1, math.ceil(math.sqrt(n_bands)))
    n_rows = max(1, math.ceil(n_bands / n_cols))
    return n_rows, n_cols


def _band_subplot_title(
    band: str,
    wavelengths: Mapping[str, float] | None,
) -> str:
    """Format a subplot title.

    Returns ``"Red (668 nm)"`` when the band has a known wavelength,
    otherwise just ``"Red"``.
    """
    if wavelengths and band in wavelengths:
        return f"{band} ({wavelengths[band]:.0f} nm)"
    return band


# ─── Public API ──────────────────────────────────────────────────────────
def make_violins(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
    *,
    wavelengths: Mapping[str, float] | None = None,
    class_labels: Mapping[Any, str] | None = None,
    palette: Mapping[Any, str] | None = None,
    show_inner_box: bool = True,
    show_points: ShowPointsOption = False,
    n_cols: int | None = None,
    height_per_row: int = 320,
) -> go.Figure:
    """Per-band violin plots with classes side-by-side.

    Each subplot represents one band; within each subplot every class
    appears as a violin (KDE shape) with an optional inner mini-boxplot
    showing median + IQR. Y-axes are independent per subplot since
    different bands carry different units.

    The KDE is computed by Plotly with ``spanmode="hard"`` so the
    density is clipped to the actual data range, avoiding visually
    misleading extrapolation beyond observed values.

    Parameters
    ----------
    df, class_col, bands
        Same input contract as the other ``src.viz`` modules.
    wavelengths
        Optional ``{band_name: centre_nm}`` mapping. When provided,
        bands are ordered ascending by wavelength and subplot titles
        include the centre wavelength.
    class_labels
        Optional ``{class_id: display_name}`` mapping for legend +
        x-tick labels.
    palette
        Optional ``{class_id: "#rrggbb"}``. Missing classes are filled
        in from a tab10/tab20 default palette.
    show_inner_box
        When True (default), the median + IQR are overlaid as a small
        boxplot inside each violin — the standard "violin + inner box"
        combo for scientific reporting.
    show_points
        Sample-point overlay. ``False`` (default) shows no points,
        ``"outliers"`` shows Tukey outliers only, ``"all"`` shows every
        sample. Use sparingly — with 100+ samples per class, points
        become noise.
    n_cols
        Number of grid columns. Defaults to ``ceil(sqrt(n_bands))``.
    height_per_row
        Pixel height per row of subplots. Total figure height is
        ``n_rows * height_per_row + 120``.

    Returns
    -------
    go.Figure
        Subplot-grid figure with one panel per band. A unified
        horizontal legend at the top has one entry per class; toggling
        a class hides its violins across all subplots via
        ``legendgroup``.
    """
    bands = list(bands)
    err = _validate_inputs(df, class_col, bands)
    if err is not None:
        return _empty_figure(err)

    # ── Order bands (by wavelength when all known) ──────────────────
    all_have_wl = bool(wavelengths) and all(b in wavelengths for b in bands)
    if all_have_wl:
        ordered_bands = order_bands_by_wavelength(bands, wavelengths)
    else:
        ordered_bands = list(bands)

    # ── Resolve classes + palette ───────────────────────────────────
    unique_classes = list(df[class_col].drop_duplicates())
    if not unique_classes:
        return _empty_figure("No classes found in data.")

    base = "tab20" if len(unique_classes) > 10 else "tab10"
    default_palette = generate_class_palette(unique_classes, base=base)
    if palette:
        default_palette.update(dict(palette))
    resolved_palette: dict[Any, str] = default_palette
    class_label_strs = [
        _resolve_class_label(cid, class_labels) for cid in unique_classes
    ]

    # ── Grid layout ─────────────────────────────────────────────────
    n_rows, n_cols_resolved = _choose_grid(len(ordered_bands), n_cols)
    subplot_titles = [_band_subplot_title(b, wavelengths) for b in ordered_bands]
    subplot_titles += [""] * (n_rows * n_cols_resolved - len(ordered_bands))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols_resolved,
        subplot_titles=subplot_titles,
        shared_yaxes=False,         # bands have different units
        shared_xaxes=False,
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
    )

    # ── Add violins per (band, class) ───────────────────────────────
    n_traces_added = 0
    for band_idx, band in enumerate(ordered_bands):
        row = band_idx // n_cols_resolved + 1
        col = band_idx % n_cols_resolved + 1
        is_first_subplot = (band_idx == 0)

        for cid_idx, cid in enumerate(unique_classes):
            label = class_label_strs[cid_idx]
            color = resolved_palette.get(cid, "#6b7280")
            fill_rgba = _hex_to_rgba(color, 0.30)

            class_mask = df[class_col] == cid
            values = df.loc[class_mask, band].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if values.size == 0:
                continue   # skip empty class for this band

            fig.add_trace(
                go.Violin(
                    y=values,
                    name=label,
                    legendgroup=label,
                    showlegend=is_first_subplot,
                    # Violin body
                    fillcolor=fill_rgba,
                    line_color=color,
                    line_width=1.5,
                    # Inner mini-boxplot
                    box_visible=show_inner_box,
                    box_fillcolor="rgba(255,255,255,0.85)",
                    box_line_color="#1f2937",
                    box_line_width=1,
                    box_width=0.25,
                    # Mean line dashed inside the violin
                    meanline_visible=True,
                    meanline_color="#1f2937",
                    meanline_width=1.2,
                    # Optional sample points
                    points=show_points,
                    pointpos=0,
                    marker_color=color,
                    marker_size=3,
                    marker_opacity=0.55,
                    # KDE clipped to actual data range (scientific honesty)
                    spanmode="hard",
                    width=0.85,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        f"Band: {band}<br>"
                        "Median = %{median:.3f}<br>"
                        "Q1 = %{q1:.3f}<br>"
                        "Q3 = %{q3:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            n_traces_added += 1

    if n_traces_added == 0:
        return _empty_figure("No finite values to display for the selected bands.")

    # ── Common axis settings ────────────────────────────────────────
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=9),
        tickangle=-30,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=False,
        tickfont=dict(size=9),
    )

    # Y-axis title only on the leftmost column.
    for r in range(1, n_rows + 1):
        fig.update_yaxes(
            title=dict(text="Value", font=dict(size=11)),
            row=r, col=1,
        )

    # Style subplot titles consistently with the modern layout.
    for ann in fig.layout.annotations:
        ann.font = dict(family="Inter, system-ui, sans-serif", size=12, color="#111827")

    # Hide axes on unused (padded) grid cells.
    n_used = len(ordered_bands)
    n_total = n_rows * n_cols_resolved
    for empty_idx in range(n_used, n_total):
        r = empty_idx // n_cols_resolved + 1
        c = empty_idx % n_cols_resolved + 1
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    apply_modern_layout(
        fig,
        title="Violin Plots — Per Band, Classes Side-by-Side",
        height=n_rows * height_per_row + 120,
        violinmode="group",         # classes laid out side-by-side per subplot
        showlegend=True,
        legend=dict(
            title=dict(text="Class", font=dict(size=11)),
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
        margin=dict(l=70, r=30, t=120, b=70),  # extra top for horizontal legend
        hovermode="closest",
    )
    return fig


__all__ = ["make_violins"]
