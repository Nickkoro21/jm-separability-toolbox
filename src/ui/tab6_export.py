"""
Tab 6 — Export results.

Sequential workflow: STEP 6 of 6 (terminal — no Tab 7).

Produces a downloadable ZIP bundle containing:
  * results/subset_summary.csv       — user's actual results
  * results/ranked_pairs_<subset>.csv — user's actual results, one per subset
  * example_guide.html               — interpretation guide using example dataset
  * README.txt                       — instructions

The HTML guide is intentionally explicit that the figures inside it
are an EXAMPLE — the user's own results live in the CSV files
alongside it in the bundle.

Public API
----------
build(state) -> dict
    Render the tab and return refs needed by app.py.

clear_state_updates() -> tuple
    Reset every widget to its default. Order matches the chain
    handler outputs= list in app.py.
"""

from __future__ import annotations

import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.io as pio

from src.core import (
    CATEGORY_REFLECTANCE,
    auto_detect_schema,
    group_bands_by_category,
)
from src.viz import generate_class_palette
from src.viz.jm_comparative import (
    compute_subset_summary,
    make_jm_bucket_distribution,
    make_jm_comparative_bar,
)
from src.viz.jm_matrix import make_jm_heatmap
from src.viz.ranked_pairs import compute_ranked_pairs, make_ranked_pairs_bar
from src.viz.spectral_combined import make_spectral_combined


# ── Public constants ─────────────────────────────────────────────────────
DEFAULT_STATUS: str = ""

#: Path to the bundled reference dataset used by the HTML guide.
EXAMPLE_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "examples" / "spectral_samples.csv"
)

#: Wavelengths for the example dataset (MicaSense Altum-PT).
EXAMPLE_WAVELENGTHS: dict[str, float] = {
    "Blue":    475.0,
    "Green":   560.0,
    "Red":     668.0,
    "RedEdge": 717.0,
    "NIR":     842.0,
}

# ── Colour tokens ────────────────────────────────────────────────────────
_C_OK    = "#16a34a"
_C_WARN  = "#d97706"
_C_INFO  = "#8896b3"
_C_ERR   = "#dc2626"



# ── Build ────────────────────────────────────────────────────────────────
def build(state: gr.State) -> dict:
    """Render Tab 6 widgets, wire internal events, return refs."""
    # ── Lock screen ──────────────────────────────────────────────────
    with gr.Group(visible=True) as lock_msg:
        gr.Markdown(
            f"""
            <div style="text-align:center; padding:32px 24px;
                        background:rgba(245,158,11,0.08);
                        border:1px solid rgba(245,158,11,0.3);
                        border-radius:12px; color:{_C_WARN};">
              <h3 style="margin:0 0 8px 0;">🔒 Locked</h3>
              <p style="margin:0;">Please complete <b>Step 5 — Results</b>
              first.</p>
            </div>
            """,
        )

    # ── Tab content ──────────────────────────────────────────────────
    with gr.Group(visible=False) as content:
        gr.Markdown(
            """
            ### <span class="step-badge" style="background:rgba(96,165,250,0.15); color:#60a5fa;">STEP 6</span> Export results

            Generate a ZIP bundle containing **your computed results** as
            CSV files plus a **reference guide** (HTML) showing how to
            interpret JM separability values, using the bundled example
            dataset (`spectral_samples.csv`) as a worked example.

            > 💡 **Tip:** If you only need individual plots, you can
            > download them directly from Step 5 using the camera icon
            > on each Plotly chart.
            """,
        )

        with gr.Row():
            generate_btn = gr.Button(
                "📦 Generate export",
                variant="primary",
                size="lg",
            )

        status = gr.Markdown(value=DEFAULT_STATUS)
        download_file = gr.File(
            label="📥 Download ZIP",
            visible=False,
            interactive=False,
        )

    # ── Internal event wiring ────────────────────────────────────────
    generate_btn.click(
        fn=_on_generate,
        inputs=[state],
        outputs=[status, download_file],
    )

    return {
        "lock_msg":      lock_msg,
        "content":       content,
        "generate_btn":  generate_btn,
        "status":        status,
        "download_file": download_file,
    }


# ── Cascade-reset helper (Pattern H) ──────────────────────────────────────
def clear_state_updates() -> tuple:
    """Reset Tab 6 widgets to default empty state.

    Order matches the app.py chain handler outputs= list:
        (status, download_file)
    """
    return (
        gr.update(value=DEFAULT_STATUS),
        gr.update(visible=False, value=None),
    )



# ── Helpers ───────────────────────────────────────────────────────────────
def _safe_filename(name: str) -> str:
    """Sanitise a string for use as a file/path component."""
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name).strip()
    return out or "subset"


def _filter_df_by_classes(state: dict) -> pd.DataFrame:
    """Apply the user's class selection (Tab 4)."""
    df = state["df"]
    schema = state.get("detected_schema") or {}
    class_col = schema.get("class_col")
    selected = state.get("selected_class_ids") or []
    if class_col and class_col in df.columns and selected:
        return df[df[class_col].isin(selected)].copy()
    return df


def _class_labels_mapping(state: dict) -> dict | None:
    """Extract {class_id: name} or None."""
    schema = state.get("detected_schema") or {}
    mapping = schema.get("class_label_mapping")
    return mapping if isinstance(mapping, dict) else None


def _ready_for_export(state: dict) -> tuple[bool, str]:
    """Return (ready, reason). Reason is empty when ready."""
    if not state.get("tab5_done"):
        return False, "Please confirm results in Step 5 first."
    df = state.get("df")
    if df is None or len(df) == 0:
        return False, "No data — re-run Step 3."
    schema = state.get("detected_schema") or {}
    if not schema.get("class_col"):
        return False, "Class column missing — re-run Step 3."
    if not state.get("subsets"):
        return False, "No subsets defined — re-run Step 4."
    return True, ""


# ── Generate handler ─────────────────────────────────────────────────────
def _on_generate(state: dict):
    """Build the ZIP. Returns (status, download_file) updates."""
    ok, reason = _ready_for_export(state)
    if not ok:
        return (
            gr.update(value=f"<span style='color:{_C_WARN};'>⚠ {reason}</span>"),
            gr.update(visible=False, value=None),
        )

    try:
        zip_path = _build_export_zip(state)
    except Exception as exc:
        return (
            gr.update(value=(
                f"<span style='color:{_C_ERR};'>"
                f"❌ Export failed: {type(exc).__name__}: {exc}</span>"
            )),
            gr.update(visible=False, value=None),
        )

    return (
        gr.update(value=(
            f"<span style='color:{_C_OK};'>"
            f"✅ Export ready. The ZIP contains your results CSV files plus "
            f"an HTML reference guide using the example dataset.</span>"
        )),
        gr.update(visible=True, value=str(zip_path)),
    )



# ── Export pipeline ──────────────────────────────────────────────────────
def _build_export_zip(state: dict) -> Path:
    """Compute everything and write the ZIP. Returns the ZIP path."""
    tmp_root = Path(tempfile.mkdtemp(prefix="jm_export_"))
    bundle_dir = tmp_root / "bundle"
    results_dir = bundle_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1) User results — CSVs ──────────────────────────────────────
    df_filtered = _filter_df_by_classes(state)
    schema = state["detected_schema"] or {}
    class_col = schema["class_col"]
    subsets: dict[str, list[str]] = state["subsets"]
    class_labels = _class_labels_mapping(state)

    summary_df = compute_subset_summary(df_filtered, class_col, subsets)
    summary_df.to_csv(results_dir / "subset_summary.csv", index=False)

    for sname, bands in subsets.items():
        ranked = compute_ranked_pairs(
            df_filtered, class_col, bands,
            class_labels=class_labels,
        )
        ranked.to_csv(
            results_dir / f"ranked_pairs_{_safe_filename(sname)}.csv",
            index=False,
        )

    # ── 2) Example guide — render HTML using bundled example CSV ───
    html_path = bundle_dir / "example_guide.html"
    _render_example_guide(html_path)

    # ── 3) README.txt ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sensor = state.get("preset_name") or "Custom sensor"
    n_classes = len(state.get("selected_class_ids") or [])
    readme = _render_readme(timestamp, sensor, list(subsets.keys()), n_classes)
    (bundle_dir / "README.txt").write_text(readme, encoding="utf-8")

    # ── 4) Zip everything ───────────────────────────────────────────
    ts_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = tmp_root / f"jm_export_{ts_compact}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in bundle_dir.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(bundle_dir))

    return zip_path


def _render_readme(
    timestamp: str, sensor: str, subset_names: list[str], n_classes: int,
) -> str:
    """Return a human-readable README.txt content."""
    return f"""JM Separability Toolbox — Export Bundle
=========================================

Generated: {timestamp}
Sensor:    {sensor}
Subsets:   {", ".join(subset_names)}
Classes:   {n_classes}

Contents
--------
results/
    subset_summary.csv         YOUR computed JM stats per subset.
    ranked_pairs_<subset>.csv  YOUR class pairs ranked worst-first.

example_guide.html
    Interpretation guide. Open in any modern web browser.
    NOTE: This file uses the BUNDLED example dataset
    (spectral_samples.csv — MicaSense Altum-PT, 6997 samples,
    7 classes) to demonstrate how to read the values in your
    results/*.csv files. The plots inside example_guide.html are
    NOT your data — your data lives in the CSV files in this same
    bundle.

How to use
----------
1. Open example_guide.html — read the theory section and walk
   through the reference example.
2. Open results/subset_summary.csv side-by-side and compare your
   mean_jm values against the example values shown in the guide.
3. The 4-bucket interpretation scheme (Poor / Moderate / Good /
   Excellent) is the same regardless of sensor or scene.

Requirements
------------
The HTML guide loads Plotly from a CDN, so internet is required
to render the interactive figures inside it. The CSV files are
fully offline.

Project: https://github.com/Nickkoro21/jm-separability-toolbox
"""



# ── Example guide rendering ───────────────────────────────────────────────
def _render_example_guide(out_path: Path) -> None:
    """Compute the example dataset's full Tab 5 figures + write HTML."""
    if not EXAMPLE_CSV_PATH.exists():
        out_path.write_text(
            "<!DOCTYPE html><html><body>"
            "<h1>Example dataset missing</h1>"
            f"<p>The bundled example CSV was not found at "
            f"<code>{EXAMPLE_CSV_PATH}</code>. The export bundle still "
            f"contains your results in the <code>results/</code> "
            f"folder.</p></body></html>",
            encoding="utf-8",
        )
        return

    df = pd.read_csv(EXAMPLE_CSV_PATH)
    schema = auto_detect_schema(df)
    class_col = schema.class_col
    spectral_bands = list(schema.band_cols)
    non_spectral_bands = list(
        getattr(schema, "non_spectral_cols", None) or [],
    )
    all_band_cols = spectral_bands + non_spectral_bands

    # 4 canonical subsets (matches the demo on Tab 4).
    def _norm(s: str) -> str:
        return (
            str(s).lower().replace("_", "").replace("-", "").replace(" ", "")
        )

    canon = {_norm(b): b for b in all_band_cols}
    subsets: dict[str, list[str]] = {}

    rgb = [canon[k] for k in ("blue", "green", "red") if k in canon]
    if len(rgb) == 3:
        subsets["RGB"] = rgb

    fivems = [canon[k] for k in ("blue", "green", "red", "rededge", "nir")
              if k in canon]
    if len(fivems) == 5:
        subsets["5MS"] = fivems

    # 7D = 5MS + nDSM + Thermal. Search ALL band-like columns (spectral
    # + non-spectral) so nDSM_m / Thermal_C are matched even when
    # auto-detect classifies them as non-spectral.
    ndsm_col = next(
        (b for b in all_band_cols if "ndsm" in _norm(b)), None,
    )
    thermal_col = next(
        (b for b in all_band_cols
         if any(kw in _norm(b) for kw in ("thermal", "tir", "lwir"))),
        None,
    )
    if len(fivems) == 5 and ndsm_col and thermal_col:
        subsets["7D"] = fivems + [ndsm_col, thermal_col]

    subsets["All"] = list(all_band_cols)

    class_labels = (
        schema.class_label_mapping
        if isinstance(schema.class_label_mapping, dict)
        else None
    )
    selected_class_ids = sorted(df[class_col].unique().tolist())
    palette = generate_class_palette(
        selected_class_ids,
        base="tab20" if len(selected_class_ids) > 10 else "tab10",
    )

    # Compute everything
    summary_df = compute_subset_summary(df, class_col, subsets)
    comparative_fig = make_jm_comparative_bar(df, class_col, subsets)
    bucket_fig = make_jm_bucket_distribution(df, class_col, subsets)

    heatmaps: dict[str, Any] = {}
    rankings: dict[str, pd.DataFrame] = {}
    ranked_bars: dict[str, Any] = {}
    for sname, sbands in subsets.items():
        heatmaps[sname] = make_jm_heatmap(
            df, class_col, sbands,
            subset_name=sname, class_labels=class_labels,
        )
        rankings[sname] = compute_ranked_pairs(
            df, class_col, sbands, class_labels=class_labels,
        )
        ranked_bars[sname] = make_ranked_pairs_bar(
            df, class_col, sbands,
            subset_name=sname, class_labels=class_labels,
        )

    # Filter to reflectance bands only — match Tab 5 behaviour. Height
    # (nDSM) and Temperature (Thermal) have incompatible units and are
    # covered in the live Tab 5 view by the boxplot/violin sections.
    reflectance_bands = list(
        group_bands_by_category(all_band_cols, EXAMPLE_WAVELENGTHS).get(
            CATEGORY_REFLECTANCE, [],
        )
    )
    spectral_fig = make_spectral_combined(
        df, class_col, reflectance_bands,
        wavelengths=EXAMPLE_WAVELENGTHS,
        class_labels=class_labels,
        palette=palette,
        show_std=False,
    )

    html = _render_html_template(
        summary_df=summary_df,
        comparative_fig=comparative_fig,
        bucket_fig=bucket_fig,
        heatmaps=heatmaps,
        rankings=rankings,
        ranked_bars=ranked_bars,
        spectral_fig=spectral_fig,
        n_classes=df[class_col].nunique(),
        n_samples=len(df),
    )
    out_path.write_text(html, encoding="utf-8")


def _fig_html(fig: Any, div_id: str, include_plotlyjs: bool = False) -> str:
    """Return inline HTML for a Plotly figure with a stable div id."""
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )



# ── HTML template ─────────────────────────────────────────────────────────
def _render_html_template(
    *,
    summary_df: pd.DataFrame,
    comparative_fig: Any,
    bucket_fig: Any,
    heatmaps: dict[str, Any],
    rankings: dict[str, pd.DataFrame],
    ranked_bars: dict[str, Any],
    spectral_fig: Any,
    n_classes: int,
    n_samples: int,
) -> str:
    """Compose the example_guide.html as a single self-contained string."""
    summary_html = summary_df.to_html(
        index=False, classes="data-table", border=0, float_format="%.4f",
    )

    # First figure loads plotly.js from CDN; subsequent ones reuse it.
    comparative_html = _fig_html(
        comparative_fig, "fig_comparative", include_plotlyjs=True,
    )
    bucket_html = _fig_html(bucket_fig, "fig_bucket")
    spectral_html = _fig_html(spectral_fig, "fig_spectral")

    # Per-subset heatmap blocks
    heatmap_blocks: list[str] = []
    for sname, fig in heatmaps.items():
        heatmap_blocks.append(
            f'<h3 class="subsection-title">{sname}</h3>\n'
            f'{_fig_html(fig, f"fig_heatmap_{_safe_filename(sname)}")}'
        )
    heatmaps_html = "\n".join(heatmap_blocks)

    # Per-subset ranked-pairs blocks
    ranked_blocks: list[str] = []
    for sname, rdf in rankings.items():
        rdf_html = rdf.to_html(
            index=False, classes="data-table", border=0, float_format="%.4f",
        )
        bar_html = _fig_html(
            ranked_bars[sname],
            f"fig_ranked_{_safe_filename(sname)}",
        )
        ranked_blocks.append(
            f'<h3 class="subsection-title">{sname}</h3>\n'
            f'<div class="grid-2col">'
            f'<div>{rdf_html}</div>'
            f'<div>{bar_html}</div>'
            f'</div>'
        )
    ranked_html = "\n".join(ranked_blocks)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    head_section = _html_head_and_style()
    body_section = _html_body(
        timestamp=timestamp,
        n_samples=n_samples,
        n_classes=n_classes,
        summary_html=summary_html,
        comparative_html=comparative_html,
        bucket_html=bucket_html,
        heatmaps_html=heatmaps_html,
        ranked_html=ranked_html,
        spectral_html=spectral_html,
    )

    return head_section + body_section



def _html_head_and_style() -> str:
    """Return <!DOCTYPE> + <head> + opening <body>. Plain string, no f-string."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>JM Separability — Example Guide</title>
<style>
:root {
    --color-primary: #2563eb;
    --color-accent:  #c084fc;
    --color-poor:      #ef4444;
    --color-moderate:  #f59e0b;
    --color-good:      #4ade80;
    --color-excellent: #16a34a;
    --color-text:    #1f2937;
    --color-muted:   #6b7280;
    --color-border:  #e5e7eb;
}
* { box-sizing: border-box; }
body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    color: var(--color-text);
    line-height: 1.6;
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px 24px;
    background: #fff;
}
h1 {
    font-size: 2rem;
    background: linear-gradient(135deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 4px 0;
}
h2 {
    color: var(--color-primary);
    margin-top: 48px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--color-border);
}
h3.subsection-title {
    color: var(--color-text);
    margin-top: 32px;
    font-size: 1.15rem;
}
.tagline {
    color: var(--color-muted);
    font-size: 1rem;
    margin: 0 0 8px 0;
}
.author-line {
    color: var(--color-muted);
    font-size: 0.92rem;
    margin: 0 0 24px 0;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--color-border);
}
.author-line strong { color: var(--color-text); }
.author-line a {
    color: var(--color-primary);
    text-decoration: none;
}
.author-line a:hover { text-decoration: underline; }
.banner-warning {
    background: rgba(245,158,11,0.10);
    border-left: 4px solid var(--color-moderate);
    padding: 16px 20px;
    margin: 24px 0;
    border-radius: 4px;
}
.banner-warning strong { color: #92400e; }
.banner-info {
    background: rgba(96,165,250,0.10);
    border-left: 4px solid var(--color-primary);
    padding: 16px 20px;
    margin: 24px 0;
    border-radius: 4px;
}
.bucket-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin: 16px 0;
}
.bucket-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 12px;
    font-size: 0.9rem;
    border: 1px solid var(--color-border);
}
.bucket-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
}
.data-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.92rem;
    margin: 12px 0;
}
.data-table th, .data-table td {
    border: 1px solid var(--color-border);
    padding: 6px 10px;
    text-align: right;
}
.data-table th {
    background: #f9fafb;
    color: var(--color-text);
    text-align: left;
}
.data-table td:first-child {
    text-align: left;
    font-weight: 500;
}
.grid-2col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
}
@media (max-width: 900px) {
    .grid-2col { grid-template-columns: 1fr; }
}
.equation {
    background: #f9fafb;
    padding: 12px 16px;
    border-left: 3px solid var(--color-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    margin: 12px 0;
    overflow-x: auto;
}
footer {
    margin-top: 64px;
    padding-top: 24px;
    border-top: 1px solid var(--color-border);
    text-align: center;
    color: var(--color-muted);
    font-size: 0.85rem;
}
</style>
</head>
<body>
"""



def _html_body(
    *,
    timestamp: str,
    n_samples: int,
    n_classes: int,
    summary_html: str,
    comparative_html: str,
    bucket_html: str,
    heatmaps_html: str,
    ranked_html: str,
    spectral_html: str,
) -> str:
    """Return the <body> contents. F-string with all interpolations."""
    return f"""
<h1>JM Separability — Example Guide</h1>
<p class="tagline">Reference walkthrough using the bundled example dataset.
Generated {timestamp}.</p>
<p class="author-line">
  Author: <strong>Nikolaos Koroniadis</strong> ·
  MSc Geography &amp; Applied Geoinformatics ·
  <a href="https://www.geo.aegean.gr/geo-en.php" target="_blank" rel="noopener">University of the Aegean</a> ·
  Supervisor: Dr. Christos Vasilakos ·
  <a href="https://github.com/Nickkoro21" target="_blank" rel="noopener">GitHub</a> ·
  <a href="https://www.linkedin.com/in/nick-koroniadis-328962226" target="_blank" rel="noopener">LinkedIn</a>
</p>

<div class="banner-warning">
<p><strong>⚠ This is a REFERENCE EXAMPLE.</strong> The figures and values
shown below are <em>not</em> your data — they were computed from the
bundled <code>spectral_samples.csv</code>
(MicaSense Altum-PT, {n_samples:,} samples, {n_classes} classes).
<strong>Your computed results live in <code>results/*.csv</code></strong>
in the same ZIP. Use this guide to learn how to read and interpret
those files.</p>
</div>

<h2>1. Theory</h2>
<p>The <strong>Jeffries-Matusita (JM) distance</strong> is a probabilistic
measure of class separability used in remote-sensing classification. For
two classes assumed multivariate-normal, it is computed via the
Bhattacharyya distance:</p>
<div class="equation">
B = (1/8)·(μ₁ − μ₂)ᵀ·Σ̄⁻¹·(μ₁ − μ₂) + (1/2)·ln( |Σ̄| / √(|Σ₁|·|Σ₂|) )
&nbsp;&nbsp;where&nbsp;&nbsp;Σ̄ = (Σ₁ + Σ₂)/2
</div>
<div class="equation">
JM = 2·(1 − e<sup>−B</sup>) &nbsp;&nbsp;∈&nbsp;[0, 2]
</div>
<p>JM saturates at <code>2.0</code> for perfectly separable classes and
approaches <code>0</code> for indistinguishable ones. We bin JM values
into four interpretation buckets:</p>
<div class="bucket-legend">
  <span class="bucket-chip"><span class="bucket-dot" style="background:var(--color-poor);"></span>Poor &lt; 1.0</span>
  <span class="bucket-chip"><span class="bucket-dot" style="background:var(--color-moderate);"></span>Moderate 1.0 – 1.5</span>
  <span class="bucket-chip"><span class="bucket-dot" style="background:var(--color-good);"></span>Good 1.5 – 1.9</span>
  <span class="bucket-chip"><span class="bucket-dot" style="background:var(--color-excellent);"></span>Excellent ≥ 1.9</span>
</div>

<h2>2. Subset Summary <span style="color:var(--color-muted); font-weight:normal; font-size:0.9rem;">(EXAMPLE)</span></h2>
<p>Per-subset descriptive statistics. <code>mean_jm</code> averages the
off-diagonal pairs; <code>mean_bucket</code> categorises that mean.</p>
{summary_html}

<h3 class="subsection-title">Comparative bar (mean JM per subset)</h3>
{comparative_html}

<h3 class="subsection-title">Bucket distribution per subset</h3>
{bucket_html}

<h2>3. JM Distance Matrices <span style="color:var(--color-muted); font-weight:normal; font-size:0.9rem;">(EXAMPLE)</span></h2>
<p>One heatmap per subset. Cells use the discrete 4-bucket colour scheme;
the diagonal is masked. Hover any cell to see the exact JM value and the
class pair.</p>
{heatmaps_html}

<h2>4. Ranked Class Pairs <span style="color:var(--color-muted); font-weight:normal; font-size:0.9rem;">(EXAMPLE)</span></h2>
<p>Worst pairs first — these are the bottleneck pairs limiting your
overall separability. Same data shown as a sortable table and a bar
chart per subset.</p>
{ranked_html}

<h2>5. Spectral Signatures <span style="color:var(--color-muted); font-weight:normal; font-size:0.9rem;">(EXAMPLE)</span></h2>
<p>Mean reflectance per class, with bands ordered by wavelength.
<strong>Height (nDSM) and Temperature (Thermal) are intentionally
omitted</strong> from this panel — they have incompatible units and
are instead covered by the Boxplot / Violin sections of the live
app (Step 5), where their unit-specific Y axis is meaningful.</p>
{spectral_html}

<h2>6. Interpreting YOUR results</h2>
<p>Open <code>results/subset_summary.csv</code> alongside this guide and
compare your <code>mean_jm</code> values to the example above:</p>
<ul>
<li><strong>Same sensor, similar scene:</strong> Your values should land
in the same range as this example. Large deviations point to a
data-quality issue or a markedly different scene.</li>
<li><strong>Higher mean_jm than example:</strong> Your scene is more
separable. Good news — the same band subset is likely sufficient for
classification.</li>
<li><strong>Lower mean_jm than example:</strong> Your classes overlap
more in feature space. Consider adding bands, refining class
definitions, or merging confused classes.</li>
<li><strong>One subset much better than others:</strong> Open the
<code>ranked_pairs_&lt;subset&gt;.csv</code> for that subset and check
which class pairs benefit most.</li>
<li><strong>Bucket interpretation:</strong> A subset's
<code>mean_bucket</code> equal to <em>Poor</em> means most class pairs
are not distinguishable; <em>Excellent</em> means most pairs are very
well separated.</li>
</ul>
<div class="banner-info">
<strong>Reminder:</strong> JM thresholds (Poor/Moderate/Good/Excellent)
are sensor-agnostic. The 4-bucket scheme in this guide applies to your
results regardless of which camera or scene you used.</div>

<footer>
  JM Separability Toolbox ·
  <a href="https://github.com/Nickkoro21/jm-separability-toolbox">GitHub</a> ·
  Nikolaos Koroniadis, MSc Geography &amp; Applied Geoinformatics ·
  University of the Aegean
</footer>

</body>
</html>
"""


__all__ = [
    "build",
    "clear_state_updates",
    "DEFAULT_STATUS",
]
