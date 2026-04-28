"""src/viz/spectral_per_class.py — Per-class spectral signatures in a faceted grid.

This module renders one subplot per class arranged in an automatic grid,
making class-by-class spectral patterns easier to compare than the
overlaid ``spectral_combined`` plot when many classes are present.

Key features
------------
*   Auto grid layout (default ``n_cols = ceil(sqrt(n_classes))``).
*   **Shared y-axis range** across all subplots so visual amplitude
    comparison between classes is meaningful.
*   Same x-axis adaptation as :mod:`src.viz.spectral_combined`: numeric
    when *all* bands have wavelengths, categorical otherwise.
*   Per-class colour from the palette helper, with optional ±1σ envelope.

Public API
----------
make_spectral_per_class(df, class_col, bands, *, wavelengths=None,
                        class_labels=None, palette=None,
                        show_std=True, n_cols=None,
                        height_per_row=220) -> go.Figure
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz import (
    apply_modern_layout,
    generate_class_palette,
    order_bands_by_wavelength,
)


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


def _choose_grid(n_classes: int, user_n_cols: int | None) -> tuple[int, int]:
    """Compute ``(n_rows, n_cols)`` for the subplot grid.

    Default heuristic: ``n_cols = ceil(sqrt(n))`` which gives 2×2 for 4
    classes, 3×3 for 7-9 classes, 4×4 for 10-16, etc. — a balanced grid.
    Caller may override via ``user_n_cols``.
    """
    if user_n_cols is not None and user_n_cols > 0:
        n_cols = user_n_cols
    else:
        n_cols = max(1, math.ceil(math.sqrt(n_classes)))
    n_rows = max(1, math.ceil(n_classes / n_cols))
    return n_rows, n_cols


def _compute_shared_y_range(
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    classes: Sequence[Any],
    show_std: bool,
) -> tuple[float, float]:
    """Compute a global ``[y_min, y_max]`` covering all classes (with padding)."""
    y_min = math.inf
    y_max = -math.inf
    for cid in classes:
        means = mean_df.loc[cid].to_numpy(dtype=float)
        stds = std_df.loc[cid].to_numpy(dtype=float)
        if show_std:
            lows = (means - stds)[np.isfinite(means - stds)]
            highs = (means + stds)[np.isfinite(means + stds)]
        else:
            lows = means[np.isfinite(means)]
            highs = means[np.isfinite(means)]
        if lows.size:
            y_min = min(y_min, float(np.min(lows)))
        if highs.size:
            y_max = max(y_max, float(np.max(highs)))
    if not math.isfinite(y_min) or not math.isfinite(y_max):
        return 0.0, 1.0
    span = max(y_max - y_min, 1e-9)
    return y_min - span * 0.05, y_max + span * 0.05


# ─── Public API ──────────────────────────────────────────────────────────
def make_spectral_per_class(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
    *,
    wavelengths: Mapping[str, float] | None = None,
    class_labels: Mapping[Any, str] | None = None,
    palette: Mapping[Any, str] | None = None,
    show_std: bool = True,
    n_cols: int | None = None,
    height_per_row: int = 220,
) -> go.Figure:
    """Per-class spectral signatures in a faceted subplot grid.

    Each subplot shows the mean spectral curve (and optional ±1σ envelope)
    of one class. All subplots share the same y-axis range across the
    figure so visual amplitude comparison between classes is meaningful.

    Parameters
    ----------
    df, class_col, bands
        Same input contract as :func:`src.viz.spectral_combined.make_spectral_combined`.
    wavelengths
        Optional ``{band_name: centre_nm}`` mapping. Same x-axis logic:
        numeric when *all* bands have wavelengths, categorical otherwise.
    class_labels
        Optional ``{class_id: display_name}`` mapping for subplot titles.
    palette
        Optional ``{class_id: "#rrggbb"}`` mapping. Missing classes are
        filled in from a tab10/tab20 default palette.
    show_std
        When True (default), draw a ±1σ shaded envelope per subplot.
    n_cols
        Number of grid columns. When ``None`` (default), auto-computed
        as ``ceil(sqrt(n_classes))``.
    height_per_row
        Pixel height allocated to each row of subplots. Total figure
        height is ``n_rows * height_per_row + 100``.

    Returns
    -------
    go.Figure
        A subplot-grid figure with one panel per class. Empty grid cells
        (when ``n_classes < n_rows * n_cols``) appear blank with no axes.
    """
    bands = list(bands)
    err = _validate_inputs(df, class_col, bands)
    if err is not None:
        return _empty_figure(err)

    # ── Decide x-axis mode + band ordering ──────────────────────────
    use_numeric_x = bool(wavelengths) and all(b in wavelengths for b in bands)
    if use_numeric_x:
        ordered_bands = order_bands_by_wavelength(bands, wavelengths)
        x_values: list[Any] = [float(wavelengths[b]) for b in ordered_bands]
        x_axis_title = "Wavelength (nm)"
        tickvals: list[Any] | None = list(x_values)
        ticktext: list[str] | None = [
            f"{b}<br>{wavelengths[b]:.0f} nm" for b in ordered_bands
        ]
    else:
        ordered_bands = bands
        x_values = list(ordered_bands)
        x_axis_title = "Band"
        tickvals = None
        ticktext = None

    # ── Compute per-class stats ─────────────────────────────────────
    grouped = df.groupby(class_col, sort=False)
    mean_df = grouped[ordered_bands].mean()
    std_df = grouped[ordered_bands].std(ddof=0).fillna(0.0)
    unique_classes = list(mean_df.index)

    # Filter out classes with no finite values across the requested bands.
    valid_classes = [
        cid for cid in unique_classes
        if np.isfinite(mean_df.loc[cid].to_numpy(dtype=float)).any()
    ]
    if not valid_classes:
        return _empty_figure("No class has finite values for the selected bands.")

    # ── Resolve palette ─────────────────────────────────────────────
    base = "tab20" if len(valid_classes) > 10 else "tab10"
    default_palette = generate_class_palette(valid_classes, base=base)
    if palette:
        default_palette.update(dict(palette))
    resolved_palette: dict[Any, str] = default_palette

    # ── Compute shared y-axis range ────────────────────────────────
    y_min, y_max = _compute_shared_y_range(mean_df, std_df, valid_classes, show_std)

    # ── Build subplot grid ──────────────────────────────────────────
    n_rows, n_cols_resolved = _choose_grid(len(valid_classes), n_cols)
    subplot_titles = [_resolve_class_label(cid, class_labels) for cid in valid_classes]
    # Pad with empty titles for unused cells (so make_subplots is happy).
    subplot_titles += [""] * (n_rows * n_cols_resolved - len(valid_classes))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols_resolved,
        subplot_titles=subplot_titles,
        shared_yaxes=False,         # we set range manually for full control
        shared_xaxes=False,
        horizontal_spacing=0.05,
        vertical_spacing=0.12,
    )

    # ── Populate each subplot with envelope + mean line ────────────
    for idx, cid in enumerate(valid_classes):
        row = idx // n_cols_resolved + 1
        col = idx % n_cols_resolved + 1

        means = mean_df.loc[cid].to_numpy(dtype=float)
        stds = std_df.loc[cid].to_numpy(dtype=float)
        label = _resolve_class_label(cid, class_labels)
        color = resolved_palette.get(cid, "#6b7280")
        rgba_fill = _hex_to_rgba(color, 0.22)

        if show_std:
            upper = means + stds
            lower = means - stds
            # Upper bound first (no fill); the lower trace then fills
            # downward to it via fill="tonexty", producing the envelope.
            fig.add_trace(
                go.Scatter(
                    x=x_values, y=upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values, y=lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=rgba_fill,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )

        customdata = np.column_stack([
            np.array(ordered_bands, dtype=object),
            stds,
        ])
        fig.add_trace(
            go.Scatter(
                x=x_values, y=means,
                mode="lines+markers",
                showlegend=False,
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color, line=dict(color="white", width=1)),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "%{customdata[0]}: mean = %{y:.3f} ± %{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

    # ── Apply axis settings to every subplot ────────────────────────
    xaxis_common: dict[str, Any] = dict(
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=False,
        tickfont=dict(size=9),
    )
    if tickvals is not None and ticktext is not None:
        xaxis_common.update(dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ))

    yaxis_common: dict[str, Any] = dict(
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=False,
        tickfont=dict(size=9),
        range=[y_min, y_max],
    )

    fig.update_xaxes(**xaxis_common)
    fig.update_yaxes(**yaxis_common)

    # X-axis title only on the bottom-row subplots (reduces clutter).
    for c in range(1, n_cols_resolved + 1):
        fig.update_xaxes(
            title=dict(text=x_axis_title, font=dict(size=11)),
            row=n_rows, col=c,
        )
    # Y-axis title only on the leftmost column.
    for r in range(1, n_rows + 1):
        fig.update_yaxes(
            title=dict(text="Band value / Reflectance", font=dict(size=11)),
            row=r, col=1,
        )

    # Style subplot titles to match the modern layout typography.
    for ann in fig.layout.annotations:
        ann.font = dict(family="Inter, system-ui, sans-serif", size=12, color="#111827")

    # Hide axes on unused (padded) cells.
    n_used = len(valid_classes)
    n_total = n_rows * n_cols_resolved
    for empty_idx in range(n_used, n_total):
        r = empty_idx // n_cols_resolved + 1
        c = empty_idx % n_cols_resolved + 1
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    title_text = "Spectral Signatures — Per Class"
    title_text += " (mean ± 1σ)" if show_std else " (mean)"

    apply_modern_layout(
        fig,
        title=title_text,
        height=n_rows * height_per_row + 100,
        showlegend=False,
        margin=dict(l=70, r=30, t=80, b=70),
        hovermode="closest",
    )
    return fig


__all__ = ["make_spectral_per_class"]
