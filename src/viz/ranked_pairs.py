"""src/viz/ranked_pairs.py — Ranked class-pair list (worst → best) by JM distance.

This module provides two complementary functions:

*   :func:`compute_ranked_pairs` returns a sortable ``pd.DataFrame`` of
    unique class pairs with their JM distance and bucket. Useful for
    table display in Tab 5 (via ``gr.Dataframe``) and for CSV/Excel
    export in Tab 6.

*   :func:`make_ranked_pairs_bar` renders the same data as a horizontal
    bar chart with per-bar colour by bucket and dashed reference lines
    at the JM threshold boundaries. Useful for at-a-glance identification
    of the most confusable class pairs.

Both functions share the same defensive guards and the same input
contract used by the rest of ``src.viz`` (df + class_col + bands).
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.core import interpret_jm
from src.core import jm_matrix as _compute_jm_matrix
from src.viz import (
    JM_BUCKET_COLORS,
    JM_MAX,
    JM_THRESHOLD_GOOD,
    JM_THRESHOLD_MODERATE,
    JM_THRESHOLD_POOR,
    apply_modern_layout,
)

# ─── Type aliases ────────────────────────────────────────────────────────
SortOption = Literal["ascending", "descending"]


# ─── Internal helpers ────────────────────────────────────────────────────
def _resolve_class_label(cid: Any, mapping: Mapping[Any, str] | None) -> str:
    """Resolve a class id to a display string. Falls back to ``str(cid)``."""
    if mapping is None:
        return str(cid)
    return str(mapping.get(cid, cid))


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


def _validate_inputs(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
) -> str | None:
    """Return an error message if inputs are invalid, else None.

    Centralises the same defensive guards used by both public functions
    in this module.
    """
    if df is None or len(df) == 0:
        return "No data available."
    if class_col not in df.columns:
        return f"Class column '{class_col}' not found."
    bands_list = list(bands)
    missing = [b for b in bands_list if b not in df.columns]
    if missing:
        return f"Missing band columns: {', '.join(missing)}"
    if len(bands_list) < 1:
        return "At least one band is required."
    n_classes = df[class_col].nunique()
    if n_classes < 2:
        return f"At least 2 classes required (found {n_classes})."
    return None


# ─── Public API: compute helper ─────────────────────────────────────────
def compute_ranked_pairs(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
    *,
    class_labels: Mapping[Any, str] | None = None,
) -> pd.DataFrame:
    """Compute the ranked list of unique class pairs by JM distance.

    Pairs are extracted from the strict upper triangle of the JM matrix
    (``i < j``), so each unordered pair appears exactly once. The result is
    sorted ascending by JM (worst first) so the most problematic pairs
    surface at the top — the diagnostic ordering most useful for
    interpreting separability.

    Parameters
    ----------
    df
        DataFrame containing the spectral samples.
    class_col
        Name of the column in ``df`` holding class identifiers.
    bands
        Sequence of column names to use as features.
    class_labels
        Optional mapping ``{class_id: display_name}`` used to populate the
        ``class_a`` and ``class_b`` columns with readable strings. When
        ``None``, raw ids are stringified via ``str()``.

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``class_a`` (str): first class label of the pair
        - ``class_b`` (str): second class label of the pair
        - ``jm``      (float): Jeffries–Matusita distance ∈ [0, 2]
        - ``bucket``  (str): one of "Poor", "Moderate", "Good", "Excellent"

        Sorted ascending by ``jm``. Empty DataFrame with the same schema
        is returned when inputs are invalid.

    Notes
    -----
    NaN entries in the JM matrix (rank-deficient covariance, very small
    classes) are silently skipped. The resulting DataFrame may therefore
    contain fewer than ``n*(n-1)/2`` rows in degenerate cases.

    Use this function when you want the data only — for ``gr.Dataframe``
    display in Tab 5, for CSV export in Tab 6, or for further programmatic
    analysis. For a visual chart, see :func:`make_ranked_pairs_bar`.
    """
    bands = list(bands)
    err = _validate_inputs(df, class_col, bands)
    if err is not None:
        return pd.DataFrame(columns=["class_a", "class_b", "jm", "bucket"])

    features = df[bands].to_numpy(dtype=float)
    classes = df[class_col].to_numpy()
    matrix, ordered_labels = _compute_jm_matrix(features, classes)
    matrix = np.asarray(matrix, dtype=float)

    # Extract strict upper triangle (i < j → unique unordered pairs).
    n = matrix.shape[0]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = float(matrix[i, j])
            if np.isnan(v):
                continue   # skip degenerate pairs
            rows.append({
                "class_a": _resolve_class_label(ordered_labels[i], class_labels),
                "class_b": _resolve_class_label(ordered_labels[j], class_labels),
                "jm": v,
                "bucket": interpret_jm(v),
            })

    out = pd.DataFrame(rows, columns=["class_a", "class_b", "jm", "bucket"])
    out = out.sort_values("jm", ascending=True, kind="mergesort").reset_index(drop=True)
    return out


# ─── Public API: bar chart ──────────────────────────────────────────────
def make_ranked_pairs_bar(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
    *,
    subset_name: str | None = None,
    class_labels: Mapping[Any, str] | None = None,
    top_n: int | None = None,
    sort: SortOption = "ascending",
    height: int | None = None,
) -> go.Figure:
    """Horizontal bar chart of unique class pairs by JM distance.

    Each bar represents one unordered pair (i, j) with i ≠ j. Bar colour
    follows the 4-bucket scheme via ``JM_BUCKET_COLORS``, and dashed
    vertical reference lines at JM = 1.0 / 1.5 / 1.9 mark the bucket
    boundaries (Poor → Moderate → Good → Excellent transitions).

    Parameters
    ----------
    df
        DataFrame containing the spectral samples.
    class_col
        Name of the column in ``df`` holding class identifiers.
    bands
        Sequence of column names to use as features.
    subset_name
        Optional label appended to the figure title (e.g. ``"7D"``).
    class_labels
        Optional mapping ``{class_id: display_name}`` for axis labels.
    top_n
        If set, keep only the worst ``top_n`` pairs (after sorting
        ascending). When ``None`` (default), all pairs are shown.
        Useful for sensors with many classes where the full list would
        be unreadable.
    sort
        Display order. ``"ascending"`` (default) puts the *worst* pair at
        the **top** of the y-axis (worst-first reading order).
        ``"descending"`` puts the best at the top instead.
    height
        Figure height in pixels. When ``None`` (default), auto-computed
        from the number of pairs (~28 px per bar, min 400, plus padding
        for title and reference-line annotations).

    Returns
    -------
    go.Figure
        Plotly figure ready for display in Gradio (``gr.Plot``) or static
        export via :func:`src.viz.fig_to_png` / :func:`src.viz.fig_to_svg`.

    Notes
    -----
    Plotly's horizontal bar y-axis lays out the first row of input at the
    bottom of the chart by default. To put the worst pair at the *top*
    when ``sort="ascending"``, the input rows are reversed before plotting
    — this is an internal detail and does not affect the data returned by
    :func:`compute_ranked_pairs`.
    """
    bands = list(bands)
    err = _validate_inputs(df, class_col, bands)
    if err is not None:
        return _empty_figure(err)

    pairs = compute_ranked_pairs(df, class_col, bands, class_labels=class_labels)
    if pairs.empty:
        return _empty_figure("No valid class pairs to display.")

    # ── Sort + top_n filter ─────────────────────────────────────────
    # compute_ranked_pairs returns ascending by jm.
    # Apply top_n FIRST (always on the ascending list → worst N pairs),
    # then optionally flip for descending display order.
    if top_n is not None and top_n > 0:
        pairs = pairs.head(top_n).reset_index(drop=True)
    if sort == "descending":
        pairs = pairs.iloc[::-1].reset_index(drop=True)

    # Plotly's horizontal bar y-axis displays row 0 at the bottom by
    # default. Reverse before plotting so that "first row of pairs" lands
    # at the TOP of the chart (matches the user's reading expectation).
    pairs_plot = pairs.iloc[::-1].reset_index(drop=True)

    # ── Build labels + colours per bar ──────────────────────────────
    pair_labels = [
        f"{a} × {b}"
        for a, b in zip(pairs_plot["class_a"], pairs_plot["class_b"])
    ]
    bar_colors = [JM_BUCKET_COLORS[b] for b in pairs_plot["bucket"]]
    customdata = pairs_plot["bucket"].to_numpy(dtype=object)

    bar = go.Bar(
        x=pairs_plot["jm"],
        y=pair_labels,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(0,0,0,0.05)", width=1),
        ),
        text=[f"{v:.3f}" for v in pairs_plot["jm"]],
        textposition="outside",
        textfont=dict(family="monospace", size=11, color="#1f2937"),
        customdata=customdata,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "JM = %{x:.3f}<br>"
            "Bucket: %{customdata}"
            "<extra></extra>"
        ),
        cliponaxis=False,
    )

    # ── Auto height ─────────────────────────────────────────────────
    if height is None:
        height = max(400, 28 * len(pairs_plot) + 140)

    # ── Compose figure ──────────────────────────────────────────────
    fig = go.Figure(data=[bar])

    # Reference lines at threshold boundaries (Poor→Moderate→Good→Excellent).
    for thr in (JM_THRESHOLD_POOR, JM_THRESHOLD_MODERATE, JM_THRESHOLD_GOOD):
        fig.add_vline(
            x=thr,
            line=dict(color="#9ca3af", width=1, dash="dash"),
            annotation=dict(
                text=f"{thr:.1f}",
                font=dict(size=10, color="#6b7280"),
                bgcolor="rgba(255,255,255,0.7)",
            ),
            annotation_position="top",
        )

    title_text = "Ranked Class Pairs"
    if subset_name:
        title_text = f"{title_text} — {subset_name}"

    apply_modern_layout(
        fig,
        title=title_text,
        height=height,
        bargap=0.25,
        xaxis=dict(
            title=dict(text="JM Distance", font=dict(size=12)),
            range=[0, JM_MAX * 1.05],   # 5 % headroom for end-of-bar labels
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=""),
            automargin=True,            # auto-expand left margin for long labels
        ),
        showlegend=False,
        margin=dict(l=180, r=40, t=80, b=60),  # extra left for pair labels
    )
    return fig


__all__ = ["compute_ranked_pairs", "make_ranked_pairs_bar"]
