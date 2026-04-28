"""src/viz/jm_matrix.py — JM heatmap for a single subset of bands.

This module renders the pairwise Jeffries–Matusita distance matrix for one
subset of bands (e.g. RGB, 5MS, 7D) as a Plotly heatmap, using the discrete
4-bucket colour scheme defined in ``src.core`` (HANDOFF Decision #17).

Public API
----------
make_jm_heatmap(df, class_col, bands, *, subset_name=None,
                class_labels=None, mask_diagonal=True,
                show="full", height=600) -> go.Figure

Architecture
------------
*   Pure function — no I/O, no global state, no Gradio dependencies.
*   Defensive: returns an empty figure with a placeholder message instead
    of raising when the input cannot produce a meaningful plot (Tab 4
    validation should already prevent these cases, but Tab 5 may be hit
    via direct callers e.g. notebooks or tests).
*   Discrete 4-bucket colour scale aligned with JM thresholds at
    [0, 1.0, 1.5, 1.9, 2.0]. No gradient between buckets — the visual
    must match the bucket counts shown elsewhere in Tab 5.
*   Source-of-truth for colours: ``JM_BUCKET_COLORS`` re-exported via
    ``src.viz``. Never hard-coded here.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.core import interpret_jm
from src.core import jm_matrix as _compute_jm_matrix
from src.viz import (
    JM_BUCKETS,
    JM_BUCKET_COLORS,
    JM_MAX,
    JM_THRESHOLD_GOOD,
    JM_THRESHOLD_MODERATE,
    JM_THRESHOLD_POOR,
    apply_modern_layout,
)

# ─── Type aliases ────────────────────────────────────────────────────────
ShowOption = Literal["full", "upper", "lower"]


# ─── Internal helpers ────────────────────────────────────────────────────
def _build_bucket_colorscale() -> list[list]:
    """Discrete 4-bucket Plotly colorscale aligned with JM thresholds.

    Maps the data range [0, JM_MAX=2.0] to the [0, 1] colour-stop space
    used by Plotly heatmaps::

        0.0 / 2.0 = 0.000   Poor begins
        1.0 / 2.0 = 0.500   Poor → Moderate hard transition
        1.5 / 2.0 = 0.750   Moderate → Good hard transition
        1.9 / 2.0 = 0.950   Good → Excellent hard transition
        2.0 / 2.0 = 1.000   Excellent ends

    Hard transitions are encoded as duplicate stops with different colours,
    yielding discrete bucket bands rather than a smooth gradient.
    """
    poor_norm = JM_THRESHOLD_POOR / JM_MAX        # 0.500
    mod_norm = JM_THRESHOLD_MODERATE / JM_MAX     # 0.750
    good_norm = JM_THRESHOLD_GOOD / JM_MAX        # 0.950
    return [
        [0.0,       JM_BUCKET_COLORS["Poor"]],
        [poor_norm, JM_BUCKET_COLORS["Poor"]],
        [poor_norm, JM_BUCKET_COLORS["Moderate"]],
        [mod_norm,  JM_BUCKET_COLORS["Moderate"]],
        [mod_norm,  JM_BUCKET_COLORS["Good"]],
        [good_norm, JM_BUCKET_COLORS["Good"]],
        [good_norm, JM_BUCKET_COLORS["Excellent"]],
        [1.0,       JM_BUCKET_COLORS["Excellent"]],
    ]


def _build_colorbar_config() -> dict[str, Any]:
    """Custom colorbar with bucket-midpoint ticks and bucket-name labels.

    Tick positions are placed at the midpoint of each bucket range so that
    the bucket label sits visually centred over its colour band:

        Poor:      0.500   (midpoint of [0.0, 1.0])
        Moderate:  1.250   (midpoint of [1.0, 1.5])
        Good:      1.700   (midpoint of [1.5, 1.9])
        Excellent: 1.950   (midpoint of [1.9, 2.0])
    """
    midpoints = [
        (0.0 + JM_THRESHOLD_POOR) / 2,
        (JM_THRESHOLD_POOR + JM_THRESHOLD_MODERATE) / 2,
        (JM_THRESHOLD_MODERATE + JM_THRESHOLD_GOOD) / 2,
        (JM_THRESHOLD_GOOD + JM_MAX) / 2,
    ]
    return dict(
        title=dict(text="JM Distance", font=dict(size=12)),
        tickmode="array",
        tickvals=midpoints,
        ticktext=list(JM_BUCKETS),
        ticks="outside",
        thickness=15,
        len=0.85,
        outlinewidth=0,
    )


def _resolve_class_label(cid: Any, mapping: Mapping[Any, str] | None) -> str:
    """Resolve a class id to a display string. Falls back to ``str(cid)``."""
    if mapping is None:
        return str(cid)
    return str(mapping.get(cid, cid))


def _apply_triangle_mask(matrix: np.ndarray, show: ShowOption) -> np.ndarray:
    """Mask the unwanted triangle of a symmetric matrix with NaN.

    show="full"  → return matrix unchanged
    show="upper" → mask strict lower triangle (i > j) with NaN
    show="lower" → mask strict upper triangle (i < j) with NaN
    """
    if show == "full":
        return matrix
    masked = matrix.copy()
    if show == "upper":
        i_idx, j_idx = np.tril_indices_from(masked, k=-1)
    elif show == "lower":
        i_idx, j_idx = np.triu_indices_from(masked, k=1)
    else:
        raise ValueError(
            f"Invalid show value: {show!r} (use 'full', 'upper', or 'lower')"
        )
    masked[i_idx, j_idx] = np.nan
    return masked


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


# ─── Public API ──────────────────────────────────────────────────────────
def make_jm_heatmap(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
    *,
    subset_name: str | None = None,
    class_labels: Mapping[Any, str] | None = None,
    mask_diagonal: bool = True,
    show: ShowOption = "full",
    height: int = 600,
) -> go.Figure:
    """Build a Plotly heatmap of pairwise JM distances for one band subset.

    Parameters
    ----------
    df
        DataFrame containing the spectral samples. Must include ``class_col``
        and every name in ``bands`` as columns.
    class_col
        Name of the column in ``df`` holding class identifiers.
    bands
        Sequence of column names to use as features. Order is preserved
        when computing covariance (irrelevant to JM but matters for
        downstream debugging).
    subset_name
        Optional label appended to the figure title (e.g. ``"7D"``,
        ``"5MS"``, ``"RGB"``). When ``None`` the title is just
        ``"JM Distance Matrix"``.
    class_labels
        Optional mapping ``{class_id: display_name}``. When provided, axis
        ticks show display names instead of raw ids. Ids missing from the
        mapping fall back to ``str(id)``.
    mask_diagonal
        If True (default), the main diagonal is replaced with NaN and shown
        as ``"—"`` in muted grey. The diagonal is structurally zero (a class
        compared with itself) and carries no information; masking it draws
        the eye to off-diagonal pairs.
    show
        Triangle filter. ``"full"`` (default) shows the entire symmetric
        matrix. ``"upper"`` shows only the upper triangle (i ≤ j).
        ``"lower"`` shows only the lower triangle (i ≥ j).
    height
        Figure height in pixels. Width auto-adjusts via Plotly's
        ``scaleanchor="y"`` so cells render as squares regardless of the
        rendering width in Gradio.

    Returns
    -------
    go.Figure
        A Plotly figure with a discrete 4-bucket heatmap, annotated cells,
        custom colorbar, and modern layout applied.

    Notes
    -----
    Defensive guards: when ``df`` is empty or missing required columns, or
    when fewer than 2 distinct classes are present, an empty figure with a
    placeholder message is returned (the function never raises).

    Examples
    --------
    >>> # Inside Tab 5, given a confirmed Tab 4 state:
    >>> fig = make_jm_heatmap(
    ...     df=state["df"],
    ...     class_col=state["detected_schema"]["class_column"],
    ...     bands=state["subsets"]["7D"],
    ...     subset_name="7D",
    ...     class_labels={0: "Tree", 1: "Building", 2: "Road", ...},
    ... )
    """
    # ── Defensive validation ────────────────────────────────────────
    if df is None or len(df) == 0:
        return _empty_figure("No data available.")
    if class_col not in df.columns:
        return _empty_figure(f"Class column '{class_col}' not found.")
    bands = list(bands)
    missing = [b for b in bands if b not in df.columns]
    if missing:
        return _empty_figure(f"Missing band columns: {', '.join(missing)}")
    if len(bands) < 1:
        return _empty_figure("At least one band is required.")
    n_classes = df[class_col].nunique()
    if n_classes < 2:
        return _empty_figure(f"At least 2 classes required (found {n_classes}).")

    # ── Compute JM matrix via core engine ──────────────────────────
    features = df[bands].to_numpy(dtype=float)
    classes = df[class_col].to_numpy()
    matrix, ordered_labels = _compute_jm_matrix(features, classes)

    # ── Apply diagonal + triangle masks ─────────────────────────────
    display_matrix = np.asarray(matrix, dtype=float).copy()
    if mask_diagonal:
        np.fill_diagonal(display_matrix, np.nan)
    display_matrix = _apply_triangle_mask(display_matrix, show)

    # ── Resolve axis labels ─────────────────────────────────────────
    tick_labels = [_resolve_class_label(cid, class_labels) for cid in ordered_labels]

    # ── Build hover customdata (bucket name per cell, "" for NaN) ──
    n = display_matrix.shape[0]
    customdata = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            v = display_matrix[i, j]
            customdata[i, j] = "" if np.isnan(v) else interpret_jm(v)

    # ── Heatmap trace ───────────────────────────────────────────────
    heatmap = go.Heatmap(
        z=display_matrix,
        x=tick_labels,
        y=tick_labels,
        colorscale=_build_bucket_colorscale(),
        zmin=0.0,
        zmax=JM_MAX,
        colorbar=_build_colorbar_config(),
        customdata=customdata,
        hovertemplate=(
            "<b>%{y} × %{x}</b><br>"
            "JM = %{z:.3f}<br>"
            "Bucket: %{customdata}"
            "<extra></extra>"
        ),
        hoverongaps=False,    # don't show hover on NaN cells
        showscale=True,
        xgap=1,               # 1-px gap between cells (modern look)
        ygap=1,
    )

    # ── Cell-text annotations (per-cell font colour by bucket) ─────
    # Plotly's Heatmap supports `text` + `texttemplate`, but uses one
    # `textfont` for all cells. We need per-cell colours (white on dark
    # buckets, dark on lighter ones), so we use explicit annotations.
    annotations: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(n):
            v = display_matrix[i, j]
            if np.isnan(v):
                annotations.append(dict(
                    x=tick_labels[j], y=tick_labels[i],
                    text="—",
                    showarrow=False,
                    font=dict(color="#9ca3af", size=11),  # muted grey
                ))
            else:
                bucket = interpret_jm(v)
                # White on the two darker buckets (Poor red, Excellent dark green),
                # near-black on the lighter middle two (Moderate amber, Good light green).
                text_color = "white" if bucket in ("Poor", "Excellent") else "#1f2937"
                annotations.append(dict(
                    x=tick_labels[j], y=tick_labels[i],
                    text=f"{v:.3f}",
                    showarrow=False,
                    font=dict(color=text_color, size=11),
                ))

    # ── Compose figure ──────────────────────────────────────────────
    fig = go.Figure(data=[heatmap])
    fig.update_layout(annotations=annotations)

    title_text = "JM Distance Matrix"
    if subset_name:
        title_text = f"{title_text} — {subset_name}"

    apply_modern_layout(
        fig,
        title=title_text,
        height=height,
        xaxis=dict(
            tickangle=-30,
            side="bottom",
            scaleanchor="y",     # equal scale → square cells
            constrain="domain",
        ),
        yaxis=dict(
            autorange="reversed",  # row 0 at the top (matrix convention)
            constrain="domain",
        ),
    )
    return fig


__all__ = ["make_jm_heatmap"]
