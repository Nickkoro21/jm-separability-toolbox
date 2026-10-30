"""src/viz/exhaustive_plots.py — Visuals and tables for the exhaustive band search.

Consumes a :class:`src.core.ExhaustiveResult` (Tab 4, exhaustive mode) and
produces:

make_exhaustive_gain_plot(result, *, height=460) -> go.Figure
    "Separability vs number of bands": one marker per evaluated combination
    (sub-sampled when there are many), plus two frontier lines — the best
    mean JM and the best worst-pair (min) JM attainable at each k. The
    vertical gap between the cloud and the frontier shows how much a good
    band choice matters at that size; the frontier's flattening shows where
    adding bands stops paying off.

format_ranking_table(df, *, top=None) -> pd.DataFrame
    Display-ready copy of a ranking slice: human column names, rounded
    floats, categoricals as plain strings.

format_exhaustive_summary_md(result) -> str
    One-paragraph Markdown summary (combinations, runtime, best overall
    under each criterion) for the accordion header area.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.core import ExhaustiveResult
from src.viz import (
    JM_MAX,
    JM_THRESHOLD_GOOD,
    JM_THRESHOLD_MODERATE,
    JM_THRESHOLD_POOR,
    apply_modern_layout,
)

#: Max scatter points drawn per k (random sub-sample beyond that).
_MAX_POINTS_PER_K: int = 400

#: Display column mapping (internal → header) in display order.
_DISPLAY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("rank_mean", "Rank (mean)"),
    ("rank_min", "Rank (min)"),
    ("k", "k"),
    ("bands", "Bands"),
    ("mean_jm", "Mean JM"),
    ("min_jm", "Min JM"),
    ("max_jm", "Max JM"),
    ("worst_pair", "Worst pair"),
    ("count_excellent", "Excellent"),
    ("count_good", "Good"),
    ("count_moderate", "Moderate"),
    ("count_poor", "Poor"),
    ("mean_bucket", "Mean bucket"),
)
DISPLAY_HEADERS: tuple[str, ...] = tuple(h for _, h in _DISPLAY_COLUMNS)
DISPLAY_DATATYPES: tuple[str, ...] = (
    "number", "number", "number", "str",
    "number", "number", "number", "str",
    "number", "number", "number", "number", "str",
)

_C_MEAN = "#2563eb"   # blue-600
_C_MIN = "#9333ea"    # purple-600
_C_CLOUD = "#94a3b8"  # slate-400


def _empty_figure(message: str, height: int = 400) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="#6b7280"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_modern_layout(fig, height=height)


# ─── Tables ──────────────────────────────────────────────────────────────────

def format_ranking_table(
    df: pd.DataFrame, *, top: int | None = None, decimals: int = 3,
) -> pd.DataFrame:
    """Display-ready copy of a ranking DataFrame slice.

    Parameters
    ----------
    df
        Any slice of ``ExhaustiveResult.ranking`` (same columns).
    top
        Keep only the first ``top`` rows (``None`` = all).
    decimals
        Rounding for the JM columns.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(DISPLAY_HEADERS))
    out = df.head(top) if top is not None else df
    out = out[[c for c, _ in _DISPLAY_COLUMNS]].copy()
    for c in ("mean_jm", "min_jm", "max_jm"):
        out[c] = out[c].astype(float).round(decimals)
    for c in ("worst_pair", "mean_bucket", "bands"):
        out[c] = out[c].astype(str)
    for c in ("rank_mean", "rank_min", "k",
              "count_excellent", "count_good", "count_moderate", "count_poor"):
        out[c] = out[c].astype(int)
    out.columns = list(DISPLAY_HEADERS)
    return out.reset_index(drop=True)


def format_exhaustive_summary_md(result: ExhaustiveResult | None) -> str:
    """Markdown paragraph summarising an exhaustive-search result."""
    if result is None or result.ranking.empty:
        return ""
    r = result
    best_mean = r.top(1, by="mean").iloc[0]
    best_min = r.top(1, by="min").iloc[0]
    best3 = r.best_per_k("mean")
    best3 = best3[best3["k"] == r.min_k]
    smallest = ""
    if not best3.empty:
        b = best3.iloc[0]
        smallest = (
            f"- **Best {int(b['k'])}-band subset** (mean JM): "
            f"`{b['bands']}` → mean {b['mean_jm']:.3f}, "
            f"min {b['min_jm']:.3f} (worst pair: {b['worst_pair']})\n"
        )
    return (
        f"**{r.n_combinations:,} combinations** of {r.min_k}…{r.max_k} bands "
        f"evaluated in **{r.elapsed_s:.1f} s** from {len(r.bands)} candidate "
        f"bands (`{'`, `'.join(r.bands)}`), {len(r.labels)} classes, "
        f"{len(r.labels) * (len(r.labels) - 1) // 2} class pairs each.\n\n"
        f"- **Best overall by mean JM**: `{best_mean['bands']}` → "
        f"mean {best_mean['mean_jm']:.3f}, min {best_mean['min_jm']:.3f}\n"
        f"- **Best overall by worst pair (min JM)**: `{best_min['bands']}` → "
        f"min {best_min['min_jm']:.3f}, mean {best_min['mean_jm']:.3f}\n"
        + smallest +
        "\n*Mean JM* rewards subsets that separate most pairs well; "
        "*min JM* rewards subsets with no weak link. The tables below list "
        "the best combination at every subset size under each criterion, "
        "followed by the global top-25."
    )


# ─── Gain plot ───────────────────────────────────────────────────────────────

def _subsample_per_k(df: pd.DataFrame, max_per_k: int, seed: int = 0) -> pd.DataFrame:
    """Random sub-sample of at most ``max_per_k`` rows per k."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("k", sort=True):
        if len(g) > max_per_k:
            g = g.iloc[rng.choice(len(g), size=max_per_k, replace=False)]
        parts.append(g)
    return pd.concat(parts) if parts else df


def make_exhaustive_gain_plot(
    result: ExhaustiveResult | None, *, height: int = 460,
) -> go.Figure:
    """Separability vs number of bands with best-per-k frontier lines."""
    if result is None or result.ranking.empty:
        return _empty_figure("Run an exhaustive search in Step 4 to see this plot.")

    ranking = result.ranking
    cloud = _subsample_per_k(ranking, _MAX_POINTS_PER_K)
    jitter = (np.random.default_rng(1).random(len(cloud)) - 0.5) * 0.35

    fig = go.Figure()

    # 1) Cloud of all (sub-sampled) combinations — mean JM.
    fig.add_trace(go.Scattergl(
        x=cloud["k"].to_numpy() + jitter,
        y=cloud["mean_jm"],
        mode="markers",
        name="All combinations (mean JM)",
        marker=dict(size=5, color=_C_CLOUD, opacity=0.45),
        customdata=np.column_stack([
            cloud["bands"].astype(str), cloud["min_jm"].round(3),
            cloud["worst_pair"].astype(str),
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Mean JM = %{y:.3f} · Min JM = %{customdata[1]}<br>"
            "Worst pair: %{customdata[2]}<extra></extra>"
        ),
    ))

    # 2) Frontier lines.
    for by, col, color, label in (
        ("mean", "mean_jm", _C_MEAN, "Best mean JM at each k"),
        ("min", "min_jm", _C_MIN, "Best worst-pair (min) JM at each k"),
    ):
        best = result.best_per_k(by)
        fig.add_trace(go.Scatter(
            x=best["k"], y=best[col],
            mode="lines+markers+text",
            name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color, line=dict(color="white", width=1.5)),
            text=[f"{v:.3f}" for v in best[col]],
            textposition="top center" if by == "mean" else "bottom center",
            textfont=dict(size=10, color=color, family="monospace"),
            customdata=np.column_stack([
                best["bands"].astype(str), best["mean_jm"].round(3),
                best["min_jm"].round(3), best["worst_pair"].astype(str),
            ]),
            hovertemplate=(
                "<b>k = %{x}</b> — %{customdata[0]}<br>"
                "Mean JM = %{customdata[1]} · Min JM = %{customdata[2]}<br>"
                "Worst pair: %{customdata[3]}<extra></extra>"
            ),
        ))

    # 3) Threshold lines.
    for thr, name in (
        (JM_THRESHOLD_POOR, "Poor | Moderate"),
        (JM_THRESHOLD_MODERATE, "Moderate | Good"),
        (JM_THRESHOLD_GOOD, "Good | Excellent"),
    ):
        fig.add_hline(
            y=thr, line=dict(color="#9ca3af", width=1, dash="dash"),
            annotation=dict(
                text=f"{thr:.1f}", font=dict(size=10, color="#6b7280"),
                bgcolor="rgba(255,255,255,0.7)",
            ),
            annotation_position="right",
        )

    ks = sorted(ranking["k"].unique().tolist())
    apply_modern_layout(
        fig,
        title="Separability vs Number of Bands — exhaustive search",
        height=height,
        xaxis=dict(
            title=dict(text="Number of bands (k)"),
            tickmode="array", tickvals=ks,
            range=[min(ks) - 0.6, max(ks) + 0.6],
        ),
        yaxis=dict(
            title=dict(text="JM distance"),
            range=[0, JM_MAX * 1.06],
            zeroline=False,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
        ),
        margin=dict(l=70, r=40, t=100, b=60),
    )
    return fig


__all__ = [
    "DISPLAY_DATATYPES",
    "DISPLAY_HEADERS",
    "format_exhaustive_summary_md",
    "format_ranking_table",
    "make_exhaustive_gain_plot",
]
