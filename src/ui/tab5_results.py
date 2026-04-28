"""
Tab 5 — Results / Visualizations.

Sequential workflow: STEP 5 of 6.

Locked until Tab 4 (class filter + band subsets) is confirmed. On unlock,
the tab auto-populates by computing all 9 plot/data outputs from the
``src.viz`` package and rendering them in 6 collapsible accordions:

    1. 📊 Subset Summary          ── summary table + comparative bar +
                                     bucket distribution
    2. 🔥 JM Distance Matrices    ── one heatmap per subset (max 8)
    3. 🥇 Ranked Class Pairs      ── DataFrame + bar per subset (max 8)
    4. 📈 Spectral Signatures     ── combined overlay + per-class faceted
    5. 📦 Boxplots                ── per-band, classes side-by-side
    6. 🎻 Violin Plots            ── per-band, classes side-by-side

Per-subset slots
----------------
JM heatmaps and ranked-pair sections are pre-allocated as ``MAX_SUBSETS``
fixed slots. Each slot wraps its widgets in a ``gr.Group``. On populate,
slots beyond the actual subset count have ``visible=False``.

Schema key compatibility
------------------------
The detected schema (from ``src.core.auto_detect_schema``) uses the keys
``class_col``, ``band_cols``, ``non_spectral_cols``, ``xy_cols``. We read
``class_col`` (NOT ``class_column``) here so the dataclass-to-dict round
trip stays consistent across tabs.

Public API
----------
build(state) -> dict
    Render the tab and return refs needed by app.py.

populate_state_updates(state) -> tuple
    Compute every visualisation and emit updates that fill the tab.
    Order matches :func:`populate_refs`.

clear_state_updates() -> tuple
    Reset every widget to its default. Order matches :func:`populate_refs`.

populate_refs(refs) -> list
    Return the ordered list of widgets that match the
    :func:`populate_state_updates` output tuple. app.py uses this when
    wiring the Tab 4 confirm chain handler.

Constants exposed for re-use:
    ``MAX_SUBSETS``, ``DEFAULT_STATUS``.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd

from src.core import (
    CATEGORY_REFLECTANCE,
    get_unrecognised_bands,
    group_bands_by_category,
)
from src.viz import generate_class_palette
from src.viz.boxplots import make_boxplots
from src.viz.jm_comparative import (
    compute_subset_summary,
    make_jm_bucket_distribution,
    make_jm_comparative_bar,
)
from src.viz.jm_matrix import make_jm_heatmap
from src.viz.ranked_pairs import compute_ranked_pairs, make_ranked_pairs_bar
from src.viz.spectral_combined import make_spectral_combined
from src.viz.violins import make_violins


# ─── Public constants ─────────────────────────────────────────────────────
#: Maximum number of subsets a session may compare side-by-side. Each slot
#: pre-allocates a small bundle of widgets (group + title + plot/df), so
#: keeping this finite avoids unbounded layout costs. Tab 4 does not enforce
#: this limit today; if the user defines >MAX_SUBSETS, the extras are
#: silently dropped from the visualisation.
MAX_SUBSETS: int = 8

DEFAULT_STATUS: str = ""


# ─── Colour tokens (aligned with other tabs) ──────────────────────────────
_C_OK    = "#16a34a"
_C_WARN  = "#d97706"
_C_INFO  = "#8896b3"
_C_BLUE  = "#60a5fa"


# ─── Empty-state defaults ─────────────────────────────────────────────────
_EMPTY_RANKED_DF = pd.DataFrame(columns=["class_a", "class_b", "jm", "bucket"])
_EMPTY_SUMMARY_DF = pd.DataFrame(columns=[
    "subset_name", "n_bands", "n_pairs",
    "mean_jm", "min_jm", "max_jm", "std_jm",
    "count_poor", "count_moderate", "count_good", "count_excellent",
    "mean_bucket",
])


# ─── Internal helpers ─────────────────────────────────────────────────────
def _wavelengths_dict(state: dict) -> dict[str, float]:
    """Convert ``state['wavelengths']`` (list of tuples) to ``{name: nm}``.

    Tab 2 stores wavelengths as ``list[(name, center_nm, fwhm_nm)]``.
    Spectral plotting modules want a ``{band_name: centre_nm}`` mapping.
    Entries with non-numeric centre values are silently skipped.
    """
    raw = state.get("wavelengths") or []
    out: dict[str, float] = {}
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        name, center = entry[0], entry[1]
        try:
            out[str(name)] = float(center)
        except (TypeError, ValueError):
            continue
    return out


def _class_labels_mapping(state: dict) -> dict[Any, str] | None:
    """Extract ``{class_id: display_name}`` from the detected schema.

    Returns ``None`` when the schema does not carry a label mapping
    (e.g. when ``detect_class_column`` could not infer one). Downstream
    plots fall back to ``str(class_id)`` in that case.

    The mapping key may be either the raw id or its string form (JSON
    round-trip flattens int keys to str). Both forms are accepted by
    callers via ``mapping.get(cid) or mapping.get(str(cid))``.
    """
    schema = state.get("detected_schema") or {}
    mapping = schema.get("class_label_mapping")
    if not isinstance(mapping, dict):
        return None
    return mapping


def _filter_df_by_classes(state: dict) -> pd.DataFrame | None:
    """Return ``state['df']`` filtered to the user-selected class ids."""
    df = state.get("df")
    if df is None:
        return None
    schema = state.get("detected_schema") or {}
    class_col = schema.get("class_col")
    selected = state.get("selected_class_ids") or []
    if not class_col or class_col not in df.columns or not selected:
        return df
    return df[df[class_col].isin(selected)].copy()


def _ready_for_compute(state: dict) -> bool:
    """Return True when the upstream state has everything Tab 5 needs."""
    if not state.get("tab4_done"):
        return False
    df = state.get("df")
    if df is None or len(df) == 0:
        return False
    schema = state.get("detected_schema") or {}
    if not schema.get("class_col"):
        return False
    if not state.get("subsets"):
        return False
    if not state.get("selected_class_ids"):
        return False
    return True


def _union_bands(subsets: dict[str, list[str]], df_columns) -> list[str]:
    """Return the union of all band names across ``subsets`` (encounter order).

    Bands not present in ``df_columns`` are skipped so spectral plots do
    not raise on a malformed subset.
    """
    seen: set = set()
    out: list[str] = []
    for bands in subsets.values():
        for b in bands:
            if b not in seen and b in df_columns:
                out.append(b)
                seen.add(b)
    return out


# ─── Build ────────────────────────────────────────────────────────────────
def build(state: gr.State) -> dict:
    """Render Tab 5 widgets, wire internal events, return refs.

    Returns
    -------
    dict
        Refs needed by ``app.py`` for chain wiring & cascade-reset.
    """
    # ── Lock screen ──────────────────────────────────────────────────────
    with gr.Group(visible=True) as lock_msg:
        gr.Markdown(
            f"""
            <div style="text-align:center; padding:32px 24px;
                        background:rgba(245,158,11,0.08);
                        border:1px solid rgba(245,158,11,0.3);
                        border-radius:12px; color:{_C_WARN};">
              <h3 style="margin:0 0 8px 0;">🔒 Locked</h3>
              <p style="margin:0;">Please complete <b>Step 4 — Configure</b>
              first.<br/>Results auto-populate when the configuration is
              confirmed.</p>
            </div>
            """,
        )

    # Per-subset slot widget lists (filled in below).
    heatmap_section_groups: list[Any] = []
    heatmap_titles: list[Any] = []
    heatmap_plots: list[Any] = []
    ranked_section_groups: list[Any] = []
    ranked_titles: list[Any] = []
    ranked_dfs: list[Any] = []
    ranked_bars: list[Any] = []

    # ── Tab content ──────────────────────────────────────────────────────
    with gr.Group(visible=False) as content:
        gr.Markdown(
            """
            ### <span class="step-badge" style="background:rgba(96,165,250,0.15); color:#60a5fa;">STEP 5</span> Visualize separability

            Spectral signatures, distribution plots, JM distance matrices
            (single and comparative), and ranked separability tables —
            all generated automatically from your configuration.
            """,
        )

        # ── 1) Subset Summary ────────────────────────────────────────────
        with gr.Accordion("📊 Subset Summary", open=True):
            gr.Markdown(
                f"<span style='color:{_C_INFO};'>Per-subset statistics — "
                f"mean off-diagonal JM, range, bucket counts.</span>",
            )
            summary_table = gr.Dataframe(
                headers=[
                    "Subset", "Bands", "Pairs",
                    "Mean", "Min", "Max", "Std",
                    "Poor", "Moderate", "Good", "Excellent",
                    "Mean Bucket",
                ],
                datatype=[
                    "str", "number", "number",
                    "number", "number", "number", "number",
                    "number", "number", "number", "number",
                    "str",
                ],
                row_count=(0, "dynamic"),
                col_count=(12, "fixed"),
                interactive=False,
                wrap=True,
                value=_EMPTY_SUMMARY_DF,
            )
            with gr.Row():
                comparative_bar_plot = gr.Plot(
                    label="Mean JM Distance — Subset Comparison",
                )
                bucket_distribution_plot = gr.Plot(
                    label="Bucket Distribution per Subset",
                )

        # ── 2) JM Heatmaps ───────────────────────────────────────────────
        with gr.Accordion("🔥 JM Distance Matrices", open=True):
            gr.Markdown(
                f"<span style='color:{_C_INFO};'>One heatmap per subset, "
                f"discrete 4-bucket colour scheme.</span>",
            )
            for i in range(MAX_SUBSETS):
                with gr.Group(visible=False) as g:
                    title_md = gr.Markdown(value=f"### Subset {i + 1}")
                    plot = gr.Plot(label=None)
                heatmap_section_groups.append(g)
                heatmap_titles.append(title_md)
                heatmap_plots.append(plot)

        # ── 3) Ranked Class Pairs ────────────────────────────────────────
        with gr.Accordion("🥇 Ranked Class Pairs", open=False):
            gr.Markdown(
                f"<span style='color:{_C_INFO};'>Worst pairs first. The "
                f"DataFrame is sortable — click any column header.</span>",
            )
            for i in range(MAX_SUBSETS):
                with gr.Group(visible=False) as g:
                    title_md = gr.Markdown(value=f"### Subset {i + 1}")
                    df_widget = gr.Dataframe(
                        headers=["Class A", "Class B", "JM", "Bucket"],
                        datatype=["str", "str", "number", "str"],
                        row_count=(0, "dynamic"),
                        col_count=(4, "fixed"),
                        interactive=False,
                        wrap=True,
                        value=_EMPTY_RANKED_DF,
                    )
                    bar = gr.Plot(label=None)
                ranked_section_groups.append(g)
                ranked_titles.append(title_md)
                ranked_dfs.append(df_widget)
                ranked_bars.append(bar)

        # ── 4) Spectral Signatures ───────────────────────────────────────
        # Banner shown above the Spectral Signatures accordion when any
        # selected band falls into the "Other" category (unrecognised
        # physical quantity). Hidden by default.
        unknown_bands_banner = gr.Markdown(
            value="",
            visible=False,
            elem_id="tab5_unknown_bands_banner",
        )

        with gr.Accordion("📈 Spectral Signatures", open=False):
            gr.Markdown(
                f"<span style='color:{_C_INFO};'>Mean reflectance per "
                f"class. Bands ordered by wavelength when available. "
                f"Height (nDSM) and Temperature (Thermal) are shown in "
                f"the Boxplot / Violin sections below for clarity.</span>",
            )
            spectral_combined_plot = gr.Plot(
                label="All classes overlaid",
            )

        # ── 5) Boxplots ──────────────────────────────────────────────────
        with gr.Accordion("📦 Boxplots — Per Band", open=False):
            gr.Markdown(
                f"<span style='color:{_C_INFO};'>Per-band 5-number summary, "
                f"classes side-by-side. Outliers shown as points.</span>",
            )
            boxplots_plot = gr.Plot(label=None)

        # ── 6) Violins ───────────────────────────────────────────────────
        with gr.Accordion("🎻 Violin Plots — Per Band", open=False):
            gr.Markdown(
                f"<span style='color:{_C_INFO};'>Per-band KDE shape with "
                f"inner mini-boxplot (median + IQR).</span>",
            )
            violins_plot = gr.Plot(label=None)

        # ── Confirm ──────────────────────────────────────────────────────
        with gr.Row():
            confirm_btn = gr.Button(
                "✓ Confirm results",
                variant="primary",
                interactive=False,
                size="lg",
            )
        status = gr.Markdown(value=DEFAULT_STATUS)

    # ── Internal event wiring ────────────────────────────────────────────
    confirm_btn.click(
        fn=_on_confirm,
        inputs=[state],
        outputs=[state, status],
    )

    return {
        "lock_msg":                 lock_msg,
        "content":                  content,
        "confirm_btn":              confirm_btn,
        "status":                   status,
        # Section refs (used by populate_refs in deterministic order)
        "summary_table":            summary_table,
        "comparative_bar_plot":     comparative_bar_plot,
        "bucket_distribution_plot": bucket_distribution_plot,
        "heatmap_section_groups":   heatmap_section_groups,
        "heatmap_titles":           heatmap_titles,
        "heatmap_plots":            heatmap_plots,
        "ranked_section_groups":    ranked_section_groups,
        "ranked_titles":            ranked_titles,
        "ranked_dfs":               ranked_dfs,
        "ranked_bars":              ranked_bars,
        "spectral_combined_plot":   spectral_combined_plot,
        "boxplots_plot":            boxplots_plot,
        "violins_plot":             violins_plot,
        "unknown_bands_banner":     unknown_bands_banner,
    }


# ─── Internal: confirm handler ────────────────────────────────────────────
def _on_confirm(state: dict):
    """Mark Tab 5 done and emit a success status."""
    new_state = dict(state)
    new_state["tab5_done"] = True
    return new_state, gr.update(value=(
        f"<span style='color:{_C_OK};'>"
        f"✅ Results confirmed. Proceed to <b>Step 6</b> for export.</span>"
    ))


# ─── Public helper: ordered ref list ──────────────────────────────────────
def populate_refs(refs: dict) -> list:
    """Return widgets in the order matching :func:`populate_state_updates`.

    ``app.py`` uses this when wiring the Tab 4 confirm chain handler so
    the ``outputs=[…]`` list and the populate tuple stay in lock-step. If
    the populate ordering changes, this function updates in one place
    and the chain handler keeps working.
    """
    out: list = []
    out.append(refs["summary_table"])
    out.append(refs["comparative_bar_plot"])
    out.append(refs["bucket_distribution_plot"])
    for i in range(MAX_SUBSETS):
        out.append(refs["heatmap_section_groups"][i])
        out.append(refs["heatmap_titles"][i])
        out.append(refs["heatmap_plots"][i])
    for i in range(MAX_SUBSETS):
        out.append(refs["ranked_section_groups"][i])
        out.append(refs["ranked_titles"][i])
        out.append(refs["ranked_dfs"][i])
        out.append(refs["ranked_bars"][i])
    out.append(refs["unknown_bands_banner"])
    out.append(refs["spectral_combined_plot"])
    out.append(refs["boxplots_plot"])
    out.append(refs["violins_plot"])
    out.append(refs["confirm_btn"])
    out.append(refs["status"])
    return out


# ─── Public helper: clear updates (Pattern H) ─────────────────────────────
def clear_state_updates() -> tuple:
    """Reset every Tab 5 widget to its default empty state.

    Order matches :func:`populate_refs`. Called by upstream chain handlers
    on cascade-reset.
    """
    updates: list = []
    # 1) Subset Summary
    updates.append(gr.update(value=_EMPTY_SUMMARY_DF))
    updates.append(gr.update(value=None))
    updates.append(gr.update(value=None))
    # 2) Heatmap slots × MAX_SUBSETS (3 widgets each)
    for i in range(MAX_SUBSETS):
        updates.append(gr.update(visible=False))
        updates.append(gr.update(value=f"### Subset {i + 1}"))
        updates.append(gr.update(value=None))
    # 3) Ranked slots × MAX_SUBSETS (4 widgets each)
    for i in range(MAX_SUBSETS):
        updates.append(gr.update(visible=False))
        updates.append(gr.update(value=f"### Subset {i + 1}"))
        updates.append(gr.update(value=_EMPTY_RANKED_DF))
        updates.append(gr.update(value=None))
    # 4-6) Spectral / Boxplots / Violins
    updates.append(gr.update(visible=False, value=""))   # unknown_bands_banner
    updates.append(gr.update(value=None))   # spectral_combined_plot
    updates.append(gr.update(value=None))   # boxplots_plot
    updates.append(gr.update(value=None))   # violins_plot
    # Confirm + status
    updates.append(gr.update(interactive=False))
    updates.append(gr.update(value=DEFAULT_STATUS))
    return tuple(updates)


# ─── Public helper: populate updates (Pattern I) ──────────────────────────
def populate_state_updates(state: dict) -> tuple:
    """Compute every visualisation and emit updates that fill the tab.

    Order matches :func:`populate_refs`. Returns
    :func:`clear_state_updates` when the upstream state is incomplete
    (defensive — Tab 4 validation should already prevent this).
    """
    if not _ready_for_compute(state):
        return clear_state_updates()

    df_full = state["df"]
    df_filtered = _filter_df_by_classes(state)
    schema = state.get("detected_schema") or {}
    class_col = schema["class_col"]
    subsets: dict[str, list[str]] = state["subsets"]
    selected_class_ids = state["selected_class_ids"] or []
    wavelengths = _wavelengths_dict(state)
    class_labels = _class_labels_mapping(state)

    # Union of bands across all subsets — used for the spectral / box /
    # violin plots (which take a single band list rather than per-subset).
    union_bands = _union_bands(subsets, df_full.columns)

    # Shared palette across every plot ⇒ visual consistency. The same
    # class always gets the same colour everywhere in Tab 5.
    palette = generate_class_palette(
        selected_class_ids,
        base="tab20" if len(selected_class_ids) > 10 else "tab10",
    )

    updates: list = []

    # ── 1) Subset Summary ────────────────────────────────────────────────
    summary_df = compute_subset_summary(df_filtered, class_col, subsets)
    updates.append(gr.update(value=summary_df))
    updates.append(gr.update(value=make_jm_comparative_bar(
        df_filtered, class_col, subsets,
    )))
    updates.append(gr.update(value=make_jm_bucket_distribution(
        df_filtered, class_col, subsets,
    )))

    # ── 2) Heatmap slots ─────────────────────────────────────────────────
    subset_items = list(subsets.items())[:MAX_SUBSETS]
    for i in range(MAX_SUBSETS):
        if i < len(subset_items):
            sname, bands = subset_items[i]
            updates.append(gr.update(visible=True))
            updates.append(gr.update(value=f"### {sname}"))
            updates.append(gr.update(value=make_jm_heatmap(
                df_filtered, class_col, bands,
                subset_name=sname,
                class_labels=class_labels,
            )))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=f"### Subset {i + 1}"))
            updates.append(gr.update(value=None))

    # ── 3) Ranked Pairs slots ────────────────────────────────────────────
    for i in range(MAX_SUBSETS):
        if i < len(subset_items):
            sname, bands = subset_items[i]
            updates.append(gr.update(visible=True))
            updates.append(gr.update(value=f"### {sname}"))
            ranked_df = compute_ranked_pairs(
                df_filtered, class_col, bands,
                class_labels=class_labels,
            )
            updates.append(gr.update(value=ranked_df))
            updates.append(gr.update(value=make_ranked_pairs_bar(
                df_filtered, class_col, bands,
                subset_name=sname,
                class_labels=class_labels,
            )))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=f"### Subset {i + 1}"))
            updates.append(gr.update(value=_EMPTY_RANKED_DF))
            updates.append(gr.update(value=None))

    # ── 4) Spectral Signatures ───────────────────────────────────────────
    # Unknown-bands banner (shown only when any selected band is in the
    # "Other" category). The in-figure annotation in the spectral plot is
    # complementary to this markdown banner.
    unknown_bands = get_unrecognised_bands(union_bands, wavelengths)
    if unknown_bands:
        names_md = ", ".join(f"`{b}`" for b in unknown_bands)
        banner_md = (
            "> ⚠ **Unrecognized bands detected**: " + names_md + ".  \n"
            "> They appear in the **Other** panel of the spectral signature "
            "plot without a unit-aware Y axis. If you know their physical "
            "meaning, rename them (e.g. include `nDSM`, `Thermal`, `NDVI`) "
            "for proper grouping."
        )
        updates.append(gr.update(value=banner_md, visible=True))
    else:
        updates.append(gr.update(value="", visible=False))

    # Filter to reflectance bands only — Height (nDSM) and Temperature
    # (Thermal) are intentionally excluded here. They have incompatible
    # units (m / °C vs unitless reflectance) and are already covered
    # in the Boxplot / Violin sections below.
    reflectance_bands = list(
        group_bands_by_category(union_bands, wavelengths).get(
            CATEGORY_REFLECTANCE, [],
        )
    )
    if reflectance_bands:
        updates.append(gr.update(value=make_spectral_combined(
            df_filtered, class_col, reflectance_bands,
            wavelengths=wavelengths,
            class_labels=class_labels,
            palette=palette,
            show_std=False,
        )))
    else:
        updates.append(gr.update(value=None))

    # ── 5) Boxplots ──────────────────────────────────────────────────────
    if union_bands:
        updates.append(gr.update(value=make_boxplots(
            df_filtered, class_col, union_bands,
            wavelengths=wavelengths,
            class_labels=class_labels,
            palette=palette,
        )))
    else:
        updates.append(gr.update(value=None))

    # ── 6) Violins ───────────────────────────────────────────────────────
    if union_bands:
        updates.append(gr.update(value=make_violins(
            df_filtered, class_col, union_bands,
            wavelengths=wavelengths,
            class_labels=class_labels,
            palette=palette,
        )))
    else:
        updates.append(gr.update(value=None))

    # ── Confirm + status ─────────────────────────────────────────────────
    updates.append(gr.update(interactive=True))
    updates.append(gr.update(value=(
        f"<span style='color:{_C_OK};'>"
        f"✅ Computed <b>{len(subset_items)}</b> subset(s) · "
        f"<b>{len(selected_class_ids)}</b> class(es) · "
        f"<b>{len(union_bands)}</b> unique band(s). "
        f"Click <b>Confirm</b> to proceed to Step 6.</span>"
    )))

    return tuple(updates)


__all__ = [
    "build",
    "clear_state_updates",
    "populate_state_updates",
    "populate_refs",
    "MAX_SUBSETS",
    "DEFAULT_STATUS",
]
