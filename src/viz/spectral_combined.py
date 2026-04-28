"""src/viz/spectral_combined.py — Spectral signature plot, all classes overlaid.

Renders the canonical "spectral signature" plot used to visually inspect
class separability before any quantitative analysis. One line per class
with per-class colour, optional ±1σ envelope as a shaded band.

The x-axis adapts to the input bands:

*   When **all** requested bands have a known wavelength (via the
    ``wavelengths`` argument), the x-axis is numeric and bands are ordered
    ascending by wavelength. Tick labels show ``"<band>\\n<wl> nm"``.
*   When **any** band lacks a wavelength (e.g. ``nDSM``, ``Thermal`` in a
    7D subset), the x-axis falls back to a categorical scale in input
    order. This keeps the function useful for mixed/non-spectral subsets
    without sacrificing scientific correctness when all bands are spectral.

Public API
----------
make_spectral_combined(df, class_col, bands, *, wavelengths=None,
                       class_labels=None, palette=None,
                       show_std=True, height=460) -> go.Figure
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core import (
    CATEGORY_DISPLAY,
    CATEGORY_LABELS,
    CATEGORY_OTHER,
    group_bands_by_category,
)
from src.viz import (
    apply_modern_layout,
    generate_class_palette,
    order_bands_by_wavelength,
)


# ─── Internal helpers ────────────────────────────────────────────────────
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#rrggbb`` string to a Plotly-compatible rgba(r,g,b,a) string.

    Accepts both 3-digit (``#abc``) and 6-digit (``#aabbcc``) hex shorthands.
    """
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


# ─── Public API ──────────────────────────────────────────────────────────
def make_spectral_combined(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
    *,
    wavelengths: Mapping[str, float] | None = None,
    class_labels: Mapping[Any, str] | None = None,
    palette: Mapping[Any, str] | None = None,
    show_std: bool = True,
    height: int = 460,
) -> go.Figure:
    """Spectral signature plot — one line per class, all overlaid.

    Per-class mean is plotted as a line with markers. When ``show_std`` is
    True (default), the ±1σ band is rendered as a translucent envelope
    underneath each class line, using the class colour at low alpha.

    Parameters
    ----------
    df
        DataFrame containing the spectral samples.
    class_col
        Name of the column holding class identifiers.
    bands
        Sequence of column names to plot.
    wavelengths
        Optional ``{band_name: centre_nm}`` mapping. When *all* requested
        bands have a wavelength entry, the x-axis is numeric and bands
        are ordered ascending by wavelength. Otherwise the x-axis falls
        back to a categorical scale in input order.
    class_labels
        Optional ``{class_id: display_name}`` mapping for legend entries.
    palette
        Optional ``{class_id: "#rrggbb"}`` mapping. Missing classes are
        filled in from a tab10/tab20 default palette.
    show_std
        When True (default), draw a ±1σ shaded envelope per class.
    height
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure with one or three traces per class (line + optional
        envelope upper/lower) plus a unified legend on the right side.

    Notes
    -----
    Envelope traces use ``legendgroup`` so toggling a class in the legend
    hides its mean line **and** its envelope simultaneously, even though
    only the mean line owns the visible legend entry.
    """
    bands = list(bands)
    err = _validate_inputs(df, class_col, bands)
    if err is not None:
        return _empty_figure(err)

    # Group bands by physical quantity (Reflectance / Height /
    # Temperature / Other / Index). Empty categories are dropped.
    grouped_categories = group_bands_by_category(bands, wavelengths or {})
    n_panels = len(grouped_categories)
    if n_panels == 0:
        return _empty_figure("No bands could be grouped.")

    # Compute per-class stats once across ALL requested bands. The panel
    # loop below slices columns from these dataframes per panel.
    grouped_df = df.groupby(class_col, sort=False)
    mean_df = grouped_df[bands].mean()
    std_df = grouped_df[bands].std(ddof=0).fillna(0.0)
    unique_classes = list(mean_df.index)

    # Resolve palette.
    base = "tab20" if len(unique_classes) > 10 else "tab10"
    default_palette = generate_class_palette(unique_classes, base=base)
    if palette:
        default_palette.update(dict(palette))
    resolved_palette: dict[Any, str] = default_palette

    # Build the subplot grid (single Figure with N stacked rows).
    subplot_titles = (
        [
            f"{CATEGORY_DISPLAY[cat]} — {len(members)} band"
            + ("s" if len(members) != 1 else "")
            for cat, members in grouped_categories.items()
        ]
        if n_panels > 1
        else None
    )
    fig = make_subplots(
        rows=n_panels,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12 if n_panels > 1 else 0.0,
        subplot_titles=subplot_titles,
    )

    # Style subplot title annotations (when present).
    if n_panels > 1:
        for ann in list(fig.layout.annotations):
            ann.font = dict(size=12, color="#374151")  # gray-700
            ann.x = 0.0
            ann.xanchor = "left"

    # Plot each panel.
    for row_idx, (category, panel_bands) in enumerate(
        grouped_categories.items(), start=1,
    ):
        # Decision #32 applied *within* this panel only.
        wl = wavelengths or {}
        panel_has_all_wl = bool(wl) and all(b in wl for b in panel_bands)
        if panel_has_all_wl and panel_bands:
            ordered = order_bands_by_wavelength(panel_bands, wl)
            x_values: list[Any] = [float(wl[b]) for b in ordered]
            x_axis_title = "Wavelength (nm)"
            tickvals: list[Any] | None = list(x_values)
            ticktext: list[str] | None = [
                f"{b}<br>{wl[b]:.0f} nm" for b in ordered
            ]
        else:
            ordered = list(panel_bands)
            x_values = list(ordered)
            x_axis_title = "Band"
            tickvals = None
            ticktext = None

        is_single_band = len(ordered) == 1

        for cid in unique_classes:
            means = mean_df.loc[cid, ordered].to_numpy(dtype=float)
            stds = std_df.loc[cid, ordered].to_numpy(dtype=float)
            if not np.isfinite(means).any():
                continue

            label = _resolve_class_label(cid, class_labels)
            color = resolved_palette.get(cid, "#6b7280")
            rgba_fill = _hex_to_rgba(color, 0.18)
            # Legend on the first panel only; legendgroup syncs the toggle.
            showlegend = (row_idx == 1)

            customdata = np.column_stack([
                np.array(ordered, dtype=object),
                stds,
            ])
            hover_tmpl = (
                "<b>%{fullData.name}</b><br>"
                "%{customdata[0]}: mean = %{y:.3f} ± %{customdata[1]:.3f}"
                "<extra></extra>"
            )

            if is_single_band:
                # Markers + vertical ±1σ error bars.
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=means,
                        error_y=dict(
                            type="data", array=stds, visible=show_std,
                            thickness=1.5, width=6, color=color,
                        ),
                        mode="markers",
                        name=label,
                        legendgroup=label,
                        showlegend=showlegend,
                        marker=dict(
                            size=10, color=color,
                            line=dict(color="white", width=1),
                        ),
                        customdata=customdata,
                        hovertemplate=hover_tmpl,
                    ),
                    row=row_idx, col=1,
                )
            else:
                if show_std:
                    upper = means + stds
                    lower = means - stds
                    fig.add_trace(
                        go.Scatter(
                            x=x_values, y=upper, mode="lines",
                            line=dict(width=0),
                            showlegend=False, hoverinfo="skip",
                            legendgroup=label,
                        ),
                        row=row_idx, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_values, y=lower, mode="lines",
                            line=dict(width=0),
                            fill="tonexty", fillcolor=rgba_fill,
                            showlegend=False, hoverinfo="skip",
                            legendgroup=label,
                        ),
                        row=row_idx, col=1,
                    )

                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=means,
                        mode="lines+markers",
                        name=label,
                        legendgroup=label,
                        showlegend=showlegend,
                        line=dict(color=color, width=2),
                        marker=dict(
                            size=7, color=color,
                            line=dict(color="white", width=1),
                        ),
                        customdata=customdata,
                        hovertemplate=hover_tmpl,
                    ),
                    row=row_idx, col=1,
                )

        # Per-panel axes.
        x_kwargs: dict[str, Any] = dict(
            title_text=x_axis_title,
            title_font=dict(size=11),
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="#f3f4f6",
            zeroline=False,
            row=row_idx, col=1,
        )
        if tickvals is not None and ticktext is not None:
            x_kwargs["tickmode"] = "array"
            x_kwargs["tickvals"] = tickvals
            x_kwargs["ticktext"] = ticktext
        fig.update_xaxes(**x_kwargs)

        fig.update_yaxes(
            title_text=CATEGORY_LABELS[category],
            title_font=dict(size=11),
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="#f3f4f6",
            zeroline=False,
            row=row_idx, col=1,
        )

        # In-panel warning for the Other category.
        if category == CATEGORY_OTHER:
            xref = "x domain" if row_idx == 1 else f"x{row_idx} domain"
            yref = "y domain" if row_idx == 1 else f"y{row_idx} domain"
            fig.add_annotation(
                xref=xref, yref=yref,
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                text="⚠ Unrecognized band types — shown without unit context",
                showarrow=False,
                font=dict(size=10, color="#92400e"),  # amber-800
                bgcolor="rgba(255, 251, 235, 0.92)",  # amber-50
                bordercolor="#f59e0b",                # amber-500
                borderwidth=1,
                borderpad=4,
            )

    if not fig.data:
        return _empty_figure("No class has finite values for the selected bands.")

    # Layout.
    title_text = "Spectral Signatures — All Classes"
    title_text += " (mean ± 1σ)" if show_std else " (mean)"

    # Backward compatible: 1 panel honours the caller's ``height`` exactly.
    # Multi-panel scales linearly so each panel keeps a readable y-range.
    final_height = height if n_panels == 1 else max(height, 300 * n_panels + 80)

    apply_modern_layout(
        fig,
        title=title_text,
        height=final_height,
        showlegend=True,
        legend=dict(
            title=dict(text="Class", font=dict(size=11)),
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
        margin=dict(l=70, r=160, t=70, b=70),
        hovermode="closest",
    )
    return fig


__all__ = ["make_spectral_combined"]
