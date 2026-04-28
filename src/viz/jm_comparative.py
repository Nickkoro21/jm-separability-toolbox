"""src/viz/jm_comparative.py — Comparative JM analysis across multiple band subsets.

This module quantifies and visualises the gain (or loss) in class
separability when adding or removing bands. It is the core scientific
narrative of the toolbox: e.g., for the canonical thesis subsets

    BGR  → mean JM = 1.288  (Moderate)
    5MS  → mean JM = 1.551  (Good)
    7D   → mean JM = 1.838  (Good, near Excellent)

we can directly visualise that adding RedEdge + NIR + nDSM + Thermal
moves the mean off-diagonal JM upward across all class pairs — the
key finding the toolbox is built to communicate.

Public API
----------
compute_subset_summary(df, class_col, subsets) -> pd.DataFrame
    Per-subset statistics (mean, min, max, std, bucket counts).

make_jm_comparative_bar(df, class_col, subsets, *, show_error_bars=True,
                        height=420) -> go.Figure
    Vertical bar chart: x = subset, y = mean JM, colour = bucket of mean.
    Optional min/max error bars; horizontal threshold reference lines.

make_jm_bucket_distribution(df, class_col, subsets, *, mode="stacked",
                            normalize=False, height=420) -> go.Figure
    Stacked or grouped bar chart of bucket counts per subset.
    Counts can be normalised to percentages.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.core import count_buckets, interpret_jm
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
ModeOption = Literal["stacked", "grouped"]


# ─── Internal helpers ────────────────────────────────────────────────────
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
    subsets: Mapping[str, Sequence[str]],
) -> str | None:
    """Return an error message if inputs are invalid, else None."""
    if df is None or len(df) == 0:
        return "No data available."
    if class_col not in df.columns:
        return f"Class column '{class_col}' not found."
    if not subsets:
        return "No subsets provided for comparison."
    if df[class_col].nunique() < 2:
        return "At least 2 classes required."
    return None


def _compute_subset_stats(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
) -> dict[str, Any] | None:
    """Compute JM statistics for one subset.

    Returns ``None`` when the subset is invalid (missing bands, no pairs,
    rank-deficient covariance leading to all-NaN). Otherwise returns a
    dict with the stats described in :func:`compute_subset_summary`.

    Notes
    -----
    Mean / min / max / std are computed from the strict upper triangle
    (``i < j``), giving exactly ``n*(n-1)/2`` unique pairs (NaN-filtered).
    Bucket counts are obtained from :func:`src.core.count_buckets`,
    which itself iterates over unique off-diagonal pairs (verified
    against the BGR self-test that reports Total=21 for 7 classes).
    """
    bands = list(bands)
    missing = [b for b in bands if b not in df.columns]
    if missing or len(bands) < 1:
        return None

    features = df[bands].to_numpy(dtype=float)
    classes = df[class_col].to_numpy()
    matrix, _ordered = _compute_jm_matrix(features, classes)
    matrix = np.asarray(matrix, dtype=float)

    n = matrix.shape[0]
    if n < 2:
        return None

    iu, ju = np.triu_indices(n, k=1)
    pair_vals = matrix[iu, ju]
    pair_vals = pair_vals[~np.isnan(pair_vals)]
    if pair_vals.size == 0:
        return None

    bucket_counts = count_buckets(matrix)

    return {
        "n_bands": len(bands),
        "n_pairs": int(pair_vals.size),
        "mean_jm": float(np.mean(pair_vals)),
        "min_jm":  float(np.min(pair_vals)),
        "max_jm":  float(np.max(pair_vals)),
        "std_jm":  float(np.std(pair_vals, ddof=0)),
        "count_poor":      int(bucket_counts.get("Poor", 0)),
        "count_moderate":  int(bucket_counts.get("Moderate", 0)),
        "count_good":      int(bucket_counts.get("Good", 0)),
        "count_excellent": int(bucket_counts.get("Excellent", 0)),
    }


# ─── Public API: compute summary ─────────────────────────────────────────
def compute_subset_summary(
    df: pd.DataFrame,
    class_col: str,
    subsets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Compute a per-subset summary of JM separability statistics.

    Iterates over ``subsets``, computes the JM matrix per subset, extracts
    the unique unordered pairs (strict upper triangle), and aggregates
    central-tendency, spread, and bucket-count statistics.

    Parameters
    ----------
    df
        DataFrame containing the spectral samples.
    class_col
        Name of the column holding class identifiers.
    subsets
        Mapping ``{subset_name: [band_name, ...]}``. Typically the
        ``state["subsets"]`` from Tab 4.

    Returns
    -------
    pd.DataFrame
        One row per valid subset, columns:

        - ``subset_name``     (str)
        - ``n_bands``         (int)
        - ``n_pairs``         (int) — count of NaN-filtered upper-triangle pairs
        - ``mean_jm``         (float)
        - ``min_jm``          (float)
        - ``max_jm``          (float)
        - ``std_jm``          (float)
        - ``count_poor``      (int)
        - ``count_moderate``  (int)
        - ``count_good``      (int)
        - ``count_excellent`` (int)
        - ``mean_bucket``     (str): bucket of ``mean_jm``

        Order matches the iteration order of ``subsets``. Subsets that
        cannot be evaluated (missing bands, fewer than 2 classes,
        all-NaN matrix) are silently excluded from the result.
    """
    cols = [
        "subset_name", "n_bands", "n_pairs",
        "mean_jm", "min_jm", "max_jm", "std_jm",
        "count_poor", "count_moderate", "count_good", "count_excellent",
        "mean_bucket",
    ]

    err = _validate_inputs(df, class_col, subsets)
    if err is not None:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []
    for name, bands in subsets.items():
        stats = _compute_subset_stats(df, class_col, bands)
        if stats is None:
            continue
        stats = dict(stats)
        stats["subset_name"] = name
        stats["mean_bucket"] = interpret_jm(stats["mean_jm"])
        rows.append(stats)

    return pd.DataFrame(rows, columns=cols)


# ─── Public API: comparative bar chart ───────────────────────────────────
def make_jm_comparative_bar(
    df: pd.DataFrame,
    class_col: str,
    subsets: Mapping[str, Sequence[str]],
    *,
    show_error_bars: bool = True,
    height: int = 420,
) -> go.Figure:
    """Vertical bar chart comparing mean JM across subsets.

    Each bar represents one subset. Bar height is the mean off-diagonal
    JM distance for that subset. Bar colour follows the 4-bucket scheme
    and reflects the bucket of the *mean* (not of any individual pair).

    Optional min/max error bars show the spread of pair-wise JM values
    within each subset.

    Horizontal dashed reference lines at JM = 1.0 / 1.5 / 1.9 mark the
    bucket boundaries.

    Parameters
    ----------
    df, class_col, subsets
        Same contract as :func:`compute_subset_summary`.
    show_error_bars
        When True (default), draws min/max whiskers for each subset.
    height
        Figure height in pixels.

    Returns
    -------
    go.Figure
    """
    err = _validate_inputs(df, class_col, subsets)
    if err is not None:
        return _empty_figure(err)

    summary = compute_subset_summary(df, class_col, subsets)
    if summary.empty:
        return _empty_figure("No valid subsets to display.")

    bar_colors = [JM_BUCKET_COLORS[b] for b in summary["mean_bucket"]]

    # 5-column customdata: n_bands, n_pairs, min, max, mean_bucket.
    customdata = np.column_stack([
        summary["n_bands"].to_numpy(dtype=int),
        summary["n_pairs"].to_numpy(dtype=int),
        summary["min_jm"].to_numpy(dtype=float),
        summary["max_jm"].to_numpy(dtype=float),
        summary["mean_bucket"].to_numpy(dtype=object),
    ])

    error_y: dict[str, Any] | None = None
    if show_error_bars:
        # Plotly error bars use array (positive offset) and arrayminus
        # (negative offset) relative to the bar's y value.
        means = summary["mean_jm"].to_numpy(dtype=float)
        mins = summary["min_jm"].to_numpy(dtype=float)
        maxs = summary["max_jm"].to_numpy(dtype=float)
        error_y = dict(
            type="data",
            symmetric=False,
            array=maxs - means,
            arrayminus=means - mins,
            color="#6b7280",
            thickness=1.4,
            width=8,
        )

    bar = go.Bar(
        x=summary["subset_name"],
        y=summary["mean_jm"],
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(0,0,0,0.05)", width=1),
        ),
        text=[f"{v:.3f}" for v in summary["mean_jm"]],
        textposition="outside",
        textfont=dict(family="monospace", size=12, color="#1f2937"),
        customdata=customdata,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Mean JM = %{y:.3f}<br>"
            "Bucket: %{customdata[4]}<br>"
            "Bands: %{customdata[0]}, Pairs: %{customdata[1]}<br>"
            "Range: [%{customdata[2]:.3f}, %{customdata[3]:.3f}]"
            "<extra></extra>"
        ),
        error_y=error_y,
        cliponaxis=False,
    )

    fig = go.Figure(data=[bar])

    # Horizontal threshold reference lines.
    for thr in (JM_THRESHOLD_POOR, JM_THRESHOLD_MODERATE, JM_THRESHOLD_GOOD):
        fig.add_hline(
            y=thr,
            line=dict(color="#9ca3af", width=1, dash="dash"),
            annotation=dict(
                text=f"{thr:.1f}",
                font=dict(size=10, color="#6b7280"),
                bgcolor="rgba(255,255,255,0.7)",
            ),
            annotation_position="right",
        )

    apply_modern_layout(
        fig,
        title="Mean JM Distance — Subset Comparison",
        height=height,
        bargap=0.4,
        xaxis=dict(
            title=dict(text=""),
            tickangle=0,
        ),
        yaxis=dict(
            title=dict(text="Mean JM Distance (off-diagonal)", font=dict(size=12)),
            range=[0, JM_MAX * 1.08],   # 8 % headroom for outside-bar labels + error caps
            zeroline=False,
        ),
        showlegend=False,
    )
    return fig


# ─── Public API: bucket distribution chart ───────────────────────────────
def make_jm_bucket_distribution(
    df: pd.DataFrame,
    class_col: str,
    subsets: Mapping[str, Sequence[str]],
    *,
    mode: ModeOption = "stacked",
    normalize: bool = False,
    height: int = 420,
) -> go.Figure:
    """Bar chart of JM bucket counts per subset.

    Each subset is rendered as either a stacked bar (default — total
    matches ``n_pairs``) or a group of side-by-side bars (one per
    bucket). Counts can be optionally normalised to percentages.

    The four traces (Poor, Moderate, Good, Excellent) are emitted in the
    canonical ``JM_BUCKETS`` order so the legend reads low → high.

    Parameters
    ----------
    df, class_col, subsets
        Same contract as :func:`compute_subset_summary`.
    mode
        ``"stacked"`` (default) for one column per subset with 4 stacked
        segments, or ``"grouped"`` for 4 side-by-side bars per subset.
    normalize
        When True, each subset is normalised so the four bucket values
        sum to 100 (percentage). Useful when comparing subsets with
        different ``n_pairs``.
    height
        Figure height in pixels.

    Returns
    -------
    go.Figure
    """
    err = _validate_inputs(df, class_col, subsets)
    if err is not None:
        return _empty_figure(err)

    summary = compute_subset_summary(df, class_col, subsets)
    if summary.empty:
        return _empty_figure("No valid subsets to display.")

    # We mutate a local copy so the input summary stays untouched.
    summary = summary.copy()

    bucket_cols = ["count_poor", "count_moderate", "count_good", "count_excellent"]
    if normalize:
        totals = summary[bucket_cols].sum(axis=1).replace(0, np.nan)
        for c in bucket_cols:
            summary[c] = (summary[c] / totals * 100.0).fillna(0.0)
        y_axis_title = "Share of pairs (%)"
        text_fmt = "{:.0f}%"
        hover_unit = "%"
    else:
        y_axis_title = "Number of class pairs"
        text_fmt = "{:.0f}"
        hover_unit = ""

    bucket_to_col = {
        "Poor":      "count_poor",
        "Moderate":  "count_moderate",
        "Good":      "count_good",
        "Excellent": "count_excellent",
    }

    fig = go.Figure()
    for bucket in JM_BUCKETS:
        col = bucket_to_col[bucket]
        values = summary[col].to_numpy(dtype=float)
        text_arr = [text_fmt.format(v) for v in values]
        fig.add_trace(go.Bar(
            x=summary["subset_name"],
            y=values,
            name=bucket,
            marker=dict(
                color=JM_BUCKET_COLORS[bucket],
                line=dict(color="rgba(0,0,0,0.05)", width=1),
            ),
            text=text_arr,
            textposition="inside" if mode == "stacked" else "outside",
            textfont=dict(family="monospace", size=10, color="#1f2937"),
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{bucket}: " + "%{y:.0f}" + hover_unit +
                "<extra></extra>"
            ),
            cliponaxis=False,
        ))

    title_text = "Bucket Distribution per Subset"
    if normalize:
        title_text += " (normalised)"

    apply_modern_layout(
        fig,
        title=title_text,
        height=height,
        barmode="stack" if mode == "stacked" else "group",
        bargap=0.25,
        xaxis=dict(
            title=dict(text=""),
            tickangle=0,
        ),
        yaxis=dict(
            title=dict(text=y_axis_title, font=dict(size=12)),
            zeroline=False,
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1.0,
            font=dict(family=None, size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
        margin=dict(l=70, r=30, t=90, b=60),  # extra top for horizontal legend
    )
    return fig


__all__ = [
    "compute_subset_summary",
    "make_jm_comparative_bar",
    "make_jm_bucket_distribution",
]
