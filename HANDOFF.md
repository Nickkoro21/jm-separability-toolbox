# Spectral Separability Explorer — Handoff Document (v5)

> **Purpose**: Continue development of the JM Separability Toolbox in a new conversation. This document is self-contained — no prior context required.

**Date created**: 2026-04-27
**Last updated**: 2026-04-28 (v5 — Tab 6 implemented, all 6 tabs wired in app.py; live-test of Tab 6 + GitHub/HF deploy pending)
**Project status**: Tabs 1-6 implemented and wired. Tabs 1-5 live-tested end-to-end. **Tab 6 live-test pending.** New `data/media/` folder with 14 thesis figures added for documentation use.
**Next step**: (1) Live-test Tab 6 export flow via `.\run.ps1`. (2) Apply minor fixes if any. (3) GitHub repo init + push (§11 Phase A-B). (4) HF Space + GitHub Actions sync (Phase C). (5) GitHub Pages docs (Phase D / Group 5).

---

## 1. Project identity

| Attribute | Value |
|---|---|
| **App display name** | Spectral Separability Explorer |
| **Repo / Space slug** | `jm-separability-toolbox` |
| **Local folder** | `D:\thesis\media_cowork\High\03_Spectral_Signatures\jm_separability_toolbox\` |
| **GitHub repo** (to create) | `https://github.com/Nickkoro21/jm-separability-toolbox` |
| **HF Space** (to create) | `https://huggingface.co/spaces/NickKoro21/jm-separability-toolbox` |
| **GitHub Pages** (to enable) | `https://nickkoro21.github.io/jm-separability-toolbox/` |
| **License** | MIT |
| **Local launch** | `cd <root>; .\run.ps1` (auto-kills stale ports + activates venv + runs app) |

## 2. User profile & links

| Item | Value |
|---|---|
| Author | Nikolaos Koroniadis |
| LinkedIn | https://www.linkedin.com/in/nick-koroniadis-328962226 |
| GitHub | https://github.com/Nickkoro21 |
| Hugging Face | https://huggingface.co/NickKoro21 |
| Email (uni) | geom24014@geo.aegean.gr |
| University | https://www.geo.aegean.gr/geo-en.php |
| Department | https://geography.aegean.gr/ |
| MSc Programme | https://geography.aegean.gr/geoinformatics/ |
| RSGIS Lab | https://rsgis.geo.aegean.gr/ |
| Supervisor | Dr. Christos Vasilakos |
| Companion repo | https://github.com/Nickkoro21/PostProcessing-Toolbox |
| Companion HF Space | https://huggingface.co/spaces/NickKoro21/spectral-3d-explorer |

## 3. Architectural decisions (frozen)

| # | Decision | Choice |
|---|---|---|
| 1 | UI flow | **Sequential tabs** (each unlocks after previous completes) |
| 2 | Camera presets | 8 sensors, each with confirmation step + ground-truth source URL |
| 3 | Custom wavelengths | User enters **band/wavelength pairs** (no JSON) |
| 4 | Comparative bands | **Auto-detect from preset + manual override** (Option Γ) |
| 5 | Min samples per class | **100** (hard error block) |
| 6 | Validation severity | **Hard error** for <100 samples (no soft warning override) |
| 7 | Visual style | **Modern look** matching Spectral 3D Explorer |
| 8 | Color palette for classes | **Auto-generated** (tab10/tab20), customizable |
| 9 | README language | **English** (broader audience) |
| 10 | Hosting model | **HF Space (Gradio) + GitHub repo + GitHub Pages docs** |
| 11 | Deployment sync | **Option Γ.2** — GitHub Actions auto-deploys to HF Space on push |
| 12 | Demo dataset | Existing `spectral_samples.csv` (6,997 samples × 7 bands × 7 classes) |
| 13 | Python | 3.11+, venv (not conda), HF Space-compatible requirements.txt |
| 14 | Tech stack | Gradio 5.x, Plotly (interactive), Matplotlib (static export), NumPy, SciPy, Pandas |
| 15 | Validation report shape | **Dataclass with `to_dict()`** — type safety + JSON-serializable |
| 16 | JM self-test | Embedded `__main__` block in `src/core/jm.py` for instant numerical verification |
| 17 | JM categorisation | **4-bucket scheme** — Poor<1.0, Moderate 1.0-1.5, Good 1.5-1.9, Excellent ≥1.9 |
| 18 | Tab unlock UX | Each tab N≥2 has a `lock_msg` Group + a hidden `content` Group. The upstream tab's chain handler toggles visibility on confirm. |
| 19 | Visual progress | Confirmed tabs get `✅ <name>` label; unconfirmed get `Nº <name>`. Cascade-reset when upstream re-confirms. |
| 20 | Auto-tab-switch | Every confirm button auto-jumps to the next tab via `gr.update(selected=N+1)` on the `gr.Tabs` container. |
| 21 | Tab 3 / Tab 4 split | Tab 3 = data-ingestion gate; Tab 4 = analysis configurator. Band selection lives in Tab 4, not Tab 3. |
| 22 | Subset definition UI | 2-column `gr.Dataframe`: `[Subset name, Bands]`. Tokens matched case-insensitively against available bands. |
| 23 | Default subset heuristics | `RGB / 5MS / 7D / All` from `available_bands` via canonical-name lookup. |
| 24 | Class checkbox display | Display: `"<class_name> (<count> samples)"`; underlying value: raw class id. Order: by count descending. |
| 25 | Cascade-relock policy | When tab N is re-confirmed, **all** downstream tabs get re-locked. Only tab N+1 gets cleared/populated. |
| **26** | **Group 4 module signature** | Every viz module exposes pure functions with the contract `(df, class_col, bands, *, palette=None, **kwargs) -> go.Figure`. Two extra optional kwargs are universal: `class_labels` (display-name mapping) and `wavelengths` (numeric x-axis when known). |
| **27** | **JM heatmap colour scale** | **Discrete 4-bucket** (NOT continuous). Plotly colorscale uses 8 stops with hard transitions at thresholds normalised by `JM_MAX=2.0` (0.500 / 0.750 / 0.950). Aligns visually with bucket counts elsewhere in Tab 5. |
| **28** | **JM heatmap layout** | Default `show="full"` (full square matrix), diagonal masked with NaN displayed as `"—"`, `scaleanchor="y"` for square cells, `autorange="reversed"` on y so row 0 sits at top (numpy convention). |
| **29** | **Ranked pairs** | Two functions per module: `compute_*` (returns `pd.DataFrame`, useful for export) + `make_*_bar` (visual). Default sort `ascending` so worst pairs sit at the **top** of the chart (worst-first reading order); requires reversing rows before plotting because Plotly horizontal bars place row 0 at the bottom. |
| **30** | **Comparative module** | Three public functions: `compute_subset_summary` (DataFrame), `make_jm_comparative_bar` (mean JM per subset, colour by bucket of mean, optional min/max error bars), `make_jm_bucket_distribution` (stacked or grouped bar of bucket counts; optional percentage normalisation). |
| **31** | **Comparative bar — bucket counts** | Re-uses `src.core.count_buckets(matrix)` which iterates over **unique off-diagonal pairs only** (verified against the BGR self-test reporting Total=21 for 7 classes = 7×6/2). No manual recount needed. |
| **32** | **Adaptive x-axis (spectral plots)** | When **all** bands have a known wavelength, x-axis is numeric and bands are ordered ascending by wavelength. When **any** band lacks a wavelength (e.g. `nDSM`, `Thermal_C` in 7D), x-axis falls back to categorical in input order. |
| **33** | **Per-class spectral grid** | `n_cols = ceil(sqrt(n_classes))` for a balanced grid (3×3 for 7 classes). Shared y-axis range computed manually across all classes (with 5% padding) so amplitude comparisons are meaningful. Empty padding cells get axes hidden. |
| **34** | **Boxplots/Violins layout** | **Faceted by band**, classes side-by-side per subplot (NOT the inverse). Each subplot has independent y-axis because different bands carry different units (reflectance / metres / °C). Legend appears once at the top, classes share `legendgroup` so toggling hides them across all subplots. |
| **35** | **Violin defaults** | `box_visible=True` (inner mini-boxplot ON), `meanline_visible=True`, `points=False` (silent for 100+ samples/class), `spanmode="hard"` (KDE clipped to actual data range — scientific honesty over visual flair). |
| **36** | **Tab 5 layout** | **6 collapsible accordions** (NOT inner `gr.Tabs`). Default open: Subset Summary + JM Heatmaps. Rest closed. Per-subset content (heatmaps, ranked-pair sections) uses **pre-allocated `MAX_SUBSETS=8` slots** with `visible=True/False` toggle — Gradio widgets must exist at layout-time. |
| **37** | **Tab 5 auto-compute** | All visualisations are computed in `populate_state_updates(state)` when Tab 4 confirms. No manual "Compute" button — sub-second for typical sample sizes. |
| **38** | **Shared palette across Tab 5 plots** | The same class always gets the same colour everywhere in Tab 5. The palette is computed once in `populate_state_updates` from `selected_class_ids` and passed to every spectral / box / violin call. JM-derived plots use `JM_BUCKET_COLORS` instead. |
| **39** | **Tab 5 widget ordering contract** | `tab5_results.populate_refs(refs)` returns the exact widget list that matches the populate tuple. `app.py` uses this so the `outputs=[…]` list and the populate tuple stay in lock-step. A startup `assert` checks the count against `_TAB5_POPULATE_COUNT=66`. |
| **40** | **Schema dict keys** | Detected schema (`DetectedSchema.to_dict()`) uses **`class_col` / `band_cols` / `non_spectral_cols` / `xy_cols`** — NOT `class_column` etc. Tab 5 reads `class_col` to avoid a silent `_ready_for_compute` False-return. |
| **41** | **Tab 6 scope (simplified)** | **CSV results + HTML guide + ZIP**, no PNG/SVG export. Rationale: Tab 5 Plotly figures already provide per-plot download via the camera icon — re-rastering them in Tab 6 adds complexity for no gain. |
| **42** | **HTML guide rendering** | f-string template (no `jinja2`), single-file, **Plotly via CDN** (`include_plotlyjs="cdn"` for first fig, `False` for the rest). Internet required to render figures in the HTML; CSV files are fully offline. |
| **43** | **Tab 6 ZIP location** | `tempfile.mkdtemp()` — not persistent. Gradio's `gr.File` widget exposes the path for download. |
| **44** | **Tab 6 compute strategy** | Live recompute of the bundled `data/examples/spectral_samples.csv` on each Generate click. Sub-second; no caching needed. The HTML guide is ALWAYS the example dataset (NOT the user's data) — explicit ⚠ EXAMPLE banner at the top. The user's data lives in the CSV files in `results/` next to the HTML. |
| **45** | **Spectral plot banner integration** | Bands are categorised by physical quantity (Reflectance / Height / Temperature / Other / Index) via `src.core.band_classification`. Display order locked: Reflectance → Height → Temperature → Other → Index. Classification priority: Reflectance → Height → Temperature → Index → Other (specific-first, Other as fallback). 32/32 self-test pass. This raised `_TAB5_POPULATE_COUNT` from 65 to 66. |

## 4. Camera presets data (verified from official sources)

```python
CAMERA_PRESETS = {
    "MicaSense Altum-PT": {
        "bands": [
            ("Blue",     475, 32),
            ("Green",    560, 27),
            ("Red",      668, 14),
            ("RedEdge",  717, 12),
            ("NIR",      842, 57),
        ],
        "non_spectral": [
            ("Pan",      "634 nm (broadband, panchromatic)"),
            ("Thermal",  "LWIR 7.5-13.5 μm"),
            ("nDSM",     "meters (derived)"),
        ],
        "source": "https://support.micasense.com/hc/en-us/articles/214878778",
    },
    # ... 7 more sensors (RedEdge-MX, RedEdge-MX Dual, DJI Phantom 4 MS,
    #     DJI Mavic 3 MS, Parrot Sequoia, Sentinel-2 MSI, Landsat 8/9 OLI+TIRS)
}

SENTINEL_CUSTOM = "Custom / Unknown sensor"
```

Total via `list_preset_names()` = 9 entries (8 sensors + 1 sentinel).

## 5. Files (current state — 2026-04-28, post-Tab 5)

### Project tree

```
D:\thesis\media_cowork\High\03_Spectral_Signatures\jm_separability_toolbox\
├── .gitignore                 Standard Python/IDE/secrets exclusions
├── LICENSE                    MIT (2026, Nikolaos Koroniadis)
├── README.md                  Full English README (HF YAML, theory, mermaid)
├── app.py                     ── Gradio entry point ── ✅ updated v4 (Tab 5 wired)
├── app.py.bak                 (older snapshot — pre-v3)
├── run.ps1                    PowerShell launcher
├── requirements.txt           gradio>=5.0, numpy, pandas, scipy, plotly,
│                              matplotlib, kaleido, jinja2
├── HANDOFF.md                 ← THIS FILE (v5)
├── src/
│   ├── __init__.py
│   ├── core/                  ✅ DONE (Group 2 + v5 banner extension)
│   │   ├── __init__.py        re-exports 41 public names + band_classification
│   │   ├── __init__.py.bak
│   │   ├── jm.py              4-bucket scheme, 17.4 KB
│   │   ├── jm.py.bak
│   │   ├── presets.py         (unchanged)
│   │   ├── validation.py      (unchanged)
│   │   ├── detection.py       (unchanged)
│   │   └── band_classification.py   NEW v5 — 11.7 KB (32/32 tests pass)
│   ├── ui/                    ✅ Tabs 1-6 done (all wired in app.py v5)
│   │   ├── __init__.py        ✅ updated v5 — exports 6 tabs
│   │   ├── tab1_camera.py
│   │   ├── tab2_wavelengths.py
│   │   ├── tab3_upload.py
│   │   ├── tab4_config.py
│   │   ├── tab5_results.py    Updated v5 — banner integration via band_classification
│   │   └── tab6_export.py     NEW v5 — 27.5 KB (CSV + HTML guide + ZIP)
│   └── viz/                   ✅ DONE (Group 4 — all 7 modules)
│       ├── __init__.py        Constants + 5 helpers + 6 re-exports — 9.7 KB
│       ├── jm_matrix.py       make_jm_heatmap                       — 12.9 KB
│       ├── ranked_pairs.py    compute_ranked_pairs + make_*_bar     — 12.2 KB
│       ├── jm_comparative.py  3 functions (summary + 2 plots)       — 15.3 KB
│       ├── spectral_combined.py  make_spectral_combined             — 10.6 KB
│       ├── spectral_per_class.py make_spectral_per_class            — 13.5 KB
│       ├── boxplots.py        make_boxplots                         — 11.8 KB
│       └── violins.py         make_violins                          — 12.7 KB
├── tests/__init__.py          EMPTY — Group 5 fills this
├── data/
│   ├── examples/              holds spectral_samples.csv (6,997 × 7 × 7)
│   └── media/                 NEW v5 — 14 thesis figures (~2.2 MB)
│                              spectral profiles (a1-a7 per class + combined),
│                              separability matrices (RGB / 5MS / 7D),
│                              combined nDSM boxplot, combined Thermal violin,
│                              cumulative_gain. Used for docs/Pages content.
├── docs/                      EMPTY — Group 5 fills this (GitHub Pages)
└── .github/workflows/         EMPTY — Group 5 fills this (CI/CD to HF)
```

### `src.core` API (unchanged — 41 public names)

- **`jm.py` (14)**: `class_statistics`, `bhattacharyya_distance`, `jm_distance`, `jm_matrix`, `interpret_jm`, `bucket_color`, `count_buckets` + constants `JM_THRESHOLD_POOR=1.0`, `JM_THRESHOLD_MODERATE=1.5`, `JM_THRESHOLD_GOOD=1.9`, `JM_MAX=2.0`, `DEFAULT_REGULARISATION=1e-6`, `JM_BUCKETS`, `JM_BUCKET_COLORS`
- **`presets.py` (12)**: `CAMERA_PRESETS`, `SENTINEL_CUSTOM`, `list_preset_names`, `is_custom`, `get_preset`, `get_band_names`, `get_band_wavelengths`, `get_band_fwhm`, `get_non_spectral_bands`, `get_source_url`, `format_preset_summary`
- **`validation.py` (10)**: `ValidationError`, `ValidationReport` (with `to_dict()` and `format_markdown()`), `run_full_validation`, six individual `validate_*` fns + constant `MIN_SAMPLES_PER_CLASS=100`
- **`detection.py` (5)**: `DetectedSchema` (with `to_dict()`, keys: **`class_col`, `band_cols`, `non_spectral_cols`, `xy_cols`, `class_label_mapping`, `suggestions`**), `auto_detect_schema`, `detect_class_column`, `detect_band_columns`, `detect_non_spectral_columns`, `detect_xy_columns`, `suggest_class_label_mapping`

### `src.viz` API (NEW v4 — 9 public plot/data functions + 5 helpers)

```python
from src.viz import (
    # Re-exports from src.core (source of truth for JM)
    JM_BUCKETS, JM_BUCKET_COLORS, JM_MAX,
    JM_THRESHOLD_POOR, JM_THRESHOLD_MODERATE, JM_THRESHOLD_GOOD,
    # Style constants
    DEFAULT_PLOT_TEMPLATE, DEFAULT_PLOT_LAYOUT,
    DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE,
    DEFAULT_TITLE_FONT_SIZE, DEFAULT_AXIS_TITLE_FONT_SIZE,
    DEFAULT_EXPORT_WIDTH, DEFAULT_EXPORT_HEIGHT, DEFAULT_EXPORT_SCALE,
    # Helpers
    generate_class_palette,        # {cid: "#rrggbb"} from tab10/tab20
    order_bands_by_wavelength,     # numeric reorder, fallback to input order
    apply_modern_layout,           # apply DEFAULT_PLOT_LAYOUT to a Figure
    fig_to_png, fig_to_svg,        # static export via kaleido
)

# Plot/data functions live in submodules to keep imports cheap:
from src.viz.jm_matrix         import make_jm_heatmap
from src.viz.ranked_pairs      import compute_ranked_pairs, make_ranked_pairs_bar
from src.viz.jm_comparative    import compute_subset_summary, make_jm_comparative_bar, make_jm_bucket_distribution
from src.viz.spectral_combined import make_spectral_combined
from src.viz.spectral_per_class import make_spectral_per_class
from src.viz.boxplots          import make_boxplots
from src.viz.violins           import make_violins
```

### `src.ui` exposes (after v5): `tab1_camera`, `tab2_wavelengths`, `tab3_upload`, `tab4_config`, `tab5_results`, `tab6_export`

Each `tabN_*` module has the contract:
- `build(state) -> dict` — renders widgets, wires internal events, returns refs.
  - **Always**: `lock_msg`, `content`, `confirm_btn`, `status` (Pattern A invariant).
  - **Tab-specific** refs.
- `clear_state_updates() -> tuple` — Pattern H. Cascade-reset values.
- `populate_state_updates(state) -> tuple` — Pattern I. Tabs N≥4. Auto-fill from upstream state.
- **NEW v4 — Tab 5 only**: `populate_refs(refs) -> list` — returns widgets in the same deterministic order as the populate tuple, so `app.py` can wire `outputs=[…]` without manual ordering.
- Public `DEFAULT_*` constants for `app.py` cascade helpers.

### Per-tab refs returned by `build()` (current snapshot v4)

| Tab | refs returned |
|---|---|
| 1 (Camera) | `lock_msg`, `content`, `dropdown`, `info`, `confirm_btn`, `status` |
| 2 (Wavelengths) | `lock_msg`, `content`, `df`, `confirm_btn`, `status` |
| 3 (Upload) | `lock_msg`, `content`, `file_input`, `schema_preview`, `class_col_dd`, `validation_md`, `suggestions_md`, `confirm_btn`, `status`, `df_state`, `schema_state`, `report_state`, `csv_path_state` |
| 4 (Config) | `lock_msg`, `content`, `class_checkbox`, `detected_bands_hint`, `subset_df`, `validation_status`, `confirm_btn`, `status` |
| **5 (Results) NEW v4** | `lock_msg`, `content`, `confirm_btn`, `status`, `summary_table`, `comparative_bar_plot`, `bucket_distribution_plot`, `heatmap_section_groups[8]`, `heatmap_titles[8]`, `heatmap_plots[8]`, `ranked_section_groups[8]`, `ranked_titles[8]`, `ranked_dfs[8]`, `ranked_bars[8]`, `spectral_combined_plot`, `spectral_per_class_plot`, `boxplots_plot`, `violins_plot` |
| **6 (Export) NEW v5** | `lock_msg`, `content`, `generate_btn`, `status`, `download_file` |

### Canonical hex palette (matches the JM Presentation deliverable)

```python
JM_BUCKET_COLORS = {
    "Poor":      "#ef4444",   # Tailwind red-500
    "Moderate":  "#f59e0b",   # Tailwind amber-500
    "Good":      "#4ade80",   # Tailwind green-400 (light green)
    "Excellent": "#16a34a",   # Tailwind green-600 (dark green)
}
```

These colours flow through every JM-derived visualisation in Tab 5 and the eventual HTML report (Tab 6 export).

## 6. Roadmap (updated v4)

| Group | Status | Files | Description |
|---|:--:|---|---|
| **1. Foundation** | ✅ DONE | README, LICENSE, requirements, .gitignore, app.py skeleton, package init files, run.ps1 | Project scaffolding |
| **2. Core math** | ✅ DONE | `src/core/jm.py`, `presets.py`, `validation.py`, `detection.py`, `__init__.py` | Bhattacharyya + JM, 4-bucket categorisation, 8 presets, validation, auto-detection |
| **3. UI tabs** | ✅ **6 / 6** | All 6 tabs implemented and wired; **live-test of Tab 6 pending** | Sequential workflow with progressive disclosure, ✅ label progress, auto-jump, cascade-relock, terminal Tab 6 export |
| **4. Visualizations** | ✅ **DONE** | `src/viz/__init__.py` + 7 plot modules | 9 public plot/data functions + 5 helpers; Plotly + matplotlib; static export via kaleido |
| **5. Docs + CI/CD** | ⏳ | `docs/index.md`, `methodology.md`, `presets.md`, `troubleshooting.md`, `.github/workflows/sync-hf.yml` | GitHub Pages content, GitHub Actions auto-deploy to HF Space |

## 7. Verifications passed

### Group 1 — Smoke test (2026-04-27)
✅ venv setup, pip install, `python app.py`, hero banner, all 6 tabs visible with placeholders, footer rendered.

### Group 2 — Numerical & integration tests (2026-04-27)
**Run via `python -m src.core.jm`** — three test sections, all ✅:
- `interpret_jm` 4-bucket thresholds (13 cases) ✅
- Numerical regression vs reference `spectral_signatures_v2.py`: BGR=1.288, 5MS=1.551, 7D=1.838 (4-decimal match) ✅
- BGR bucket distribution matches JM Presentation: Poor=5, Moderate=7, Good=9, Excellent=0, Total=21 ✅

### Group 3 — Live UI tests
**Tabs 1-4 (2026-04-28):** all confirmed via screenshots — auto-jump, cascade-reset, cascade-relock, clear/populate helpers working.

### Group 4 — Module imports + smoke render (2026-04-28)
✅ All 7 viz modules import without errors. `make_*` functions return non-empty `go.Figure` for `spectral_samples.csv`.

### **Tab 5 (NEW v4) — End-to-end live test (2026-04-28)**
User confirmed via screenshots after the `class_col` bug fix:
- ✅ Tab 5 lock screen until Tab 4 confirms
- ✅ Auto-populate on Tab 4 confirm — instant render of all 6 accordions
- ✅ Subset Summary table with one row per subset, mean_jm/min/max/std/bucket counts populated
- ✅ Comparative bar + bucket distribution plots render
- ✅ JM heatmap slots: 3 visible (RGB, 5MS, All), 5 hidden — discrete 4-bucket colour scheme as expected
- ✅ Ranked Pairs slots: DataFrame + bar per subset
- ✅ Spectral combined + per-class faceted plots
- ✅ Boxplots and Violins
- ✅ Confirm button enables after compute, status reads "Computed N subsets · M classes · K bands"
- ✅ Tab 5 confirm jumps to Tab 6 placeholder, label switches to ✅ Results

### **Tab 6 (NEW v5) — Implementation complete, live-test pending (2026-04-28)**

**Implemented**:
- ✅ `src/ui/tab6_export.py` — 27.5 KB, written in 8 chunks via `Windows-MCP:FileSystem` mode=`write` with `append=True` (verified working — append mode handles arbitrary chunk sizes, unlike overwrite mode).
- ✅ `app.py` — 14 surgical edits applied via `Filesystem:edit_file` (dryRun-validated first):
  - Imports updated (`tab6_export` added to `from src.ui import (...)` block).
  - Top docstring updated (Tab 6 status: placeholder → implemented).
  - Comment block above chain handlers updated with new output counts.
  - All 5 chain handler functions extended with Tab 6 cascade-relock entries.
  - `_on_tab5_confirm_chain` rewritten to unlock Tab 6 + clear (6 → 10 outputs).
  - Placeholder replaced with `tab6_refs = tab6_export.build(session_state)`.
  - All 5 `.click()` `outputs=[…]` lists updated to include Tab 6 refs in correct order.
- ✅ `src/ui/__init__.py` — `tab6_export` added to imports + `__all__`.

**Live test pending**: User to run `.\run.ps1` and confirm:
1. Gradio launches without assertion errors (`_TAB5_POPULATE_COUNT=66` startup assert passes).
2. Tab 6 shows lock screen until Tab 5 confirms.
3. Tab 5 confirm auto-jumps to Tab 6 with content visible.
4. **📦 Generate export** produces a ZIP downloadable via `gr.File`.
5. ZIP contents: `README.txt` + `results/` (subset_summary.csv + 4 ranked_pairs CSVs) + `example_guide.html`.
6. `example_guide.html` renders all 6 sections with ⚠ EXAMPLE banner and Plotly figures via CDN.
7. Cascade-relock works (re-confirm Tab 1 → Tabs 2-6 all re-locked).

## 8. Architectural patterns established (Patterns A-P)

These patterns must be followed by `tab6_export.py` and any future Tab N modules.

### Pattern A — Tab module structure (unchanged from v3)

```python
# src/ui/tabN_*.py
def build(state: gr.State) -> dict:
    with gr.Group(visible=True) as lock_msg: ...
    with gr.Group(visible=False) as content:
        # tab-specific widgets
        confirm_btn = gr.Button("✓ Confirm …", interactive=False, variant="primary")
        status = gr.Markdown(value="")
    confirm_btn.click(fn=_on_confirm, inputs=[…, state], outputs=[state, status])
    return {"lock_msg": ..., "content": ..., "confirm_btn": ..., "status": ..., ...}
```

### Pattern B — Shared state shape (current as of v4)

```python
def _initial_state() -> dict:
    return {
        # Tab 1
        "preset_name": None, "preset_data": None, "is_custom": False, "tab1_done": False,
        # Tab 2
        "wavelengths": [], "tab2_done": False,    # list[(name, center_nm, fwhm_nm)]
        # Tab 3
        "csv_path": None, "df": None, "detected_schema": None,
        "validation_report": None, "tab3_done": False,
        # Tab 4
        "subsets": {}, "selected_classes": [], "selected_class_ids": [], "tab4_done": False,
        # Tab 5
        "jm_results": {}, "tab5_done": False,
    }
```

### Pattern C — Internal `_on_confirm` updates state + cascade-resets

```python
def _on_confirm(…, state: dict):
    if not _validate(…):
        return state, gr.update(value=err_msg)
    new_state = dict(state)
    new_state["tabN_done"] = True
    for k in ("tabN+1_done", "tabN+2_done", …):
        new_state[k] = False
    return new_state, gr.update(value=success_msg)
```

### Pattern D — Chain handler in `app.py`

For each tab N≥1, a chain handler runs as a *second* click listener on the tab's confirm button. It:
1. Reads the (already-updated) state.
2. Computes derived inputs for tab N+1.
3. Returns a tuple of `gr.update()`s for the unlock pair, populated fields, the tabs container, the cascade-relock pairs for tabs N+2..end, and 5 tab-view label updates.

### Pattern E — Tab labels constants (unchanged)

```python
_LABELS_PENDING: dict[int, str] = {1: "1️⃣ Camera", 2: "2️⃣ Wavelengths",
                                   3: "3️⃣ Upload CSV", 4: "4️⃣ Configure",
                                   5: "5️⃣ Results", 6: "6️⃣ Export"}
_LABELS_DONE:    dict[int, str] = {1: "✅ Camera", 2: "✅ Wavelengths",
                                   3: "✅ Upload CSV", 4: "✅ Configure",
                                   5: "✅ Results"}

def _label_updates(completed_through: int) -> tuple:
    return tuple(
        gr.update(label=_LABELS_DONE[i] if i <= completed_through else _LABELS_PENDING[i])
        for i in range(1, 6)
    )
```

### Pattern F — Tab view capture in `build_app()`

```python
with gr.Tabs() as tabs:
    with gr.Tab(_LABELS_PENDING[1], id=1) as tab1_view:
        tab1_refs = tab1_camera.build(session_state)
    # … each tab captured as tabN_view (Tab 5 included as of v4)

tab_views = (tab1_view, tab2_view, tab3_view, tab4_view, tab5_view)
```

### Pattern G — Local intermediate `gr.State` for staged data

When a tab needs to **stage** computed data before Confirm propagates to shared state, it declares **local** `gr.State` objects inside its `content` Group. Tab 3 uses this for `df_state`, `schema_state`, `report_state`, `csv_path_state`.

### Pattern H — `clear_state_updates()` helper for cascade-reset

Each tab N≥3 exposes a top-level `clear_state_updates() -> tuple`. The order MUST match the order of refs in the upstream `outputs=[…]` list **exactly**.

### Pattern I — `populate_state_updates(state)` helper for next-tab population

Each tab N≥4 exposes a top-level `populate_state_updates(state) -> tuple`. Reads the freshly-confirmed shared state and returns updates pre-filling the tab.

### Pattern J — Cascade-relock policy

When tab N is re-confirmed, **all** downstream tabs (N+1, N+2, …) get re-locked. Only tab N+1 gets its fields cleared/auto-populated. Deeper tabs stay locked.

### **Pattern K — Pure-function viz module (NEW v4)**

Every module in `src/viz` exposes one or more **pure functions** with the contract:

```python
make_<plot>(
    df: pd.DataFrame,
    class_col: str,
    bands: Sequence[str],
    *,
    palette: dict | None = None,            # {class_id: hex}
    class_labels: dict | None = None,       # {class_id: display_name}
    wavelengths: dict | None = None,        # {band_name: nm}
    # plot-specific kwargs
) -> go.Figure
```

Rules:
- **No I/O, no global state** — same input → same output.
- **Defensive guards**: invalid inputs return an "empty figure with message" (`_empty_figure(msg)`), never raise.
- **Source of truth for JM colours**: `JM_BUCKET_COLORS` re-exported via `src.viz`. Never hard-code hex codes.
- **Static export helpers**: every figure can be exported via the shared `fig_to_png` / `fig_to_svg` from `src.viz`.

### **Pattern L — Adaptive x-axis for spectral plots (NEW v4)**

```python
use_numeric_x = bool(wavelengths) and all(b in wavelengths for b in bands)
if use_numeric_x:
    ordered_bands = order_bands_by_wavelength(bands, wavelengths)
    x_values = [float(wavelengths[b]) for b in ordered_bands]
    # tick labels: "<band>\n<wl> nm"
else:
    ordered_bands = list(bands)        # preserve input order
    x_values = list(ordered_bands)     # categorical
```

Rationale: the moment a single non-spectral band (`nDSM_m`, `Thermal_C`) joins the subset, the wavelength axis becomes meaningless. Falling back to categorical preserves usefulness for mixed subsets without sacrificing scientific correctness when all bands are spectral.

### **Pattern M — Pre-allocated max-N slots for variable-count UI (NEW v4)**

When a tab's content depends on a variable count of upstream items (e.g. number of subsets), pre-allocate a fixed maximum (`MAX_SUBSETS=8` in Tab 5) and toggle `visible=True/False` on populate. Gradio widgets must exist at layout-time, so this avoids dynamic widget creation entirely.

```python
heatmap_section_groups: list = []
for i in range(MAX_SUBSETS):
    with gr.Group(visible=False) as g:
        title_md = gr.Markdown(value=f"### Subset {i + 1}")
        plot = gr.Plot(label=None)
    heatmap_section_groups.append(g)
    # ...
```

In `populate_state_updates`, iterate the actual subsets and emit `visible=True` for filled slots, `visible=False` for the rest.

### **Pattern N — Deterministic widget ordering helper (NEW v4)**

When a tab's `populate_state_updates` returns a large tuple (Tab 5: 65 items), pair it with a public `populate_refs(refs)` helper that returns the widget list in the **same** order. This decouples the populate-tuple ordering from the chain-handler `outputs=[…]` wiring:

```python
def populate_refs(refs: dict) -> list:
    out = []
    out.append(refs["summary_table"])
    out.append(refs["comparative_bar_plot"])
    # ... (in the SAME order as populate_state_updates)
    return out
```

`app.py` then wires:
```python
*tab5_populate_widgets,    # = tab5_results.populate_refs(tab5_refs)
```

A startup `assert len(tab5_populate_widgets) == _TAB5_POPULATE_COUNT` catches drift early — if either side changes, the app refuses to launch.

### **Pattern O — Schema dict key naming (NEW v4)**

The detected schema (from `auto_detect_schema().to_dict()`) uses these keys:

```python
{
    "class_col": str | None,
    "band_cols": list[str],
    "non_spectral_cols": list[str],
    "xy_cols": list[str | None],   # always length 2
    "class_label_mapping": dict,   # {class_id: display_name}
    "suggestions": list[str],
}
```

**Use `class_col`, NOT `class_column`.** Reading `schema.get("class_column")` returns `None`, which silently routes through `_ready_for_compute` returning `False` and triggering `clear_state_updates()` — empty Tab 5. This bug bit Tab 5 development on first integration; the fix (commit 2026-04-28) is mandatory for any future tab that reads from the schema.

### **Pattern P — Terminal tab structure (NEW v5)**

The last tab (currently Tab 6 Export) has no downstream targets, so its module signature is reduced:

- ✅ `build(state) -> dict` — same contract as Pattern A, but `confirm_btn` is replaced by an action-specific button (`generate_btn`).
- ✅ `clear_state_updates() -> tuple` — for cascade-reset by upstream confirm (Pattern H).
- ❌ NO `populate_state_updates(state)` — nothing to auto-populate (terminal tab).
- ❌ NO `populate_refs(refs)` — same reason.
- ❌ NO confirm_btn / `tabN_done` flag — replaced by an action button (`generate_btn`).
- ❌ NOT included in `_LABELS_DONE` — there is no completion state to mark.
- ❌ NOT included in `_label_updates` (`range(1, 6)` stays unchanged).

The terminal tab has its own internal action handler (`_on_generate`), which is purely local — no chain handler in `app.py`. The shared session state is read-only from this tab's perspective (the tab does not modify `tabN_done` flags).

Cascade-relock from upstream tabs (1-4) only toggles `lock_msg` / `content` visibility on the terminal tab — fields are NOT cleared from upstream chains because they're hidden behind the lock screen anyway. Only the immediately-upstream chain (Tab 5 → Tab 6) clears Tab 6 fields, per Pattern J.

### Chain handler output counts (cumulative reference for the v5 wiring)

| Chain handler | Outputs | Composition |
|---|---:|---|
| `_on_tab1_confirm_chain` | **17** | `tab2.df` + `tab2 lock/content` (2) + `tab3 lock/content` (2) + `tab4 lock/content` (2) + `tab5 lock/content` (2) + **`tab6 lock/content` (2)** + `tabs` + 5 labels |
| `_on_tab2_confirm_chain` | **25** | `tabs` + 5 labels + `tab3 lock/content` (2) + 11 tab3 clears + `tab4 lock/content` (2) + `tab5 lock/content` (2) + **`tab6 lock/content` (2)** |
| `_on_tab3_confirm_chain` | **18** | `tabs` + 5 labels + `tab4 lock/content` (2) + 6 tab4 populates + `tab5 lock/content` (2) + **`tab6 lock/content` (2)** |
| `_on_tab4_confirm_chain` | **76** | `tabs` + 5 labels + `tab5 lock/content` (2) + **66 tab5 populates** + **`tab6 lock/content` (2)** |
| `_on_tab5_confirm_chain` | **10** | `tabs` + 5 labels + **`tab6 lock/content` (2)** + **`tab6_export.clear_state_updates()` (2)** |

Tab 6 has no chain handler (terminal tab — Pattern P). Its `_on_generate` is wired directly to `generate_btn.click()` with `inputs=[state]` and `outputs=[status, download_file]`.

## 9. Critical Gradio 5.x compatibility notes

- ✅ Keep `theme` and `css` in `gr.Blocks(...)` — `launch()` doesn't accept them yet in 5.x.
- ⚠️ DeprecationWarnings are cosmetic and harmless until Gradio 6.0.
- `gr.State(value=_initial_state())` is safe — Gradio deepcopies non-callable values per session.
- `gr.Tab(label, id=N)` accepts `as varname` to capture the tab object for later `gr.update(label=…)`.
- `gr.Dataframe(row_count=(N, "dynamic"))` provides drag-handles for add/remove rows.
- Multiple `.click()` listeners on the same button run **sequentially**; subsequent listeners see state updates from previous listeners.
- `gr.File(type='filepath')` returns plain `str` path in Gradio 5.x.
- `gr.Markdown` supports inline HTML — used heavily for colour-coded status boxes.
- `gr.CheckboxGroup(choices=[(label, value), …])` uses `value` for the underlying selection.
- **`gr.Plot(label=None)`** valid for unlabeled plots.
- **`gr.Accordion(open=True/False)`** for the collapsible sections in Tab 5.

## 10. Tooling / MCP gotchas (lessons learned, updated v4)

| Tool | Status | Notes |
|---|---|---|
| `Filesystem:read_file` / `read_text_file` | ✅ Reliable | Use for D: drive reads. Supports `head` / `tail` for partial reads. |
| `Filesystem:read_multiple_files` | ✅ Reliable | **Preferred** for batch context-loading at session start. |
| `Filesystem:edit_file` | ✅ Reliable when loaded | May require `tool_search` to load in some sessions. Surgical line-based edits. |
| `Filesystem:write_file` | ✅ Reliable | Use for **new** files OR **full overwrites** of any size. Verified on files up to 24 KB. |
| `Windows-MCP:FileSystem` mode=`read` | ✅ Reliable | Backup option for reads. Supports offset/limit pagination. |
| `Windows-MCP:FileSystem` mode=`write`, `overwrite=True` | ⚠️ Times out (>~2-3KB) | Use only for tiny initial-file creation. |
| `Windows-MCP:FileSystem` mode=`write`, `append=True` | ✅ Reliable for any size | **Verified v5**: 8 chunks averaging ~3 KB each, totaling 27.5 KB, all written successfully. Use for chunked writes when `Filesystem:write_file` is unavailable in the loaded toolset. |
| `Windows-MCP:FileSystem` mode=`list` | ✅ Reliable | Quick directory listing for verification. |
| `Windows-MCP:FileSystem` mode=`copy` | ✅ Reliable | Use for backup creation (`*.bak` snapshots). |
| `Windows-MCP:PowerShell` | ⚠️ Fragile | Avoid for D: file writes. Reliable for one-shot diagnostics. |

**If Filesystem MCP becomes unresponsive mid-session**:
1. Quit Claude Desktop (Ctrl+Q on Windows) and relaunch — restarts MCP servers.
2. Or apply small patches manually using diff blocks Claude provides.

## 11. Deployment plan (post-development, unchanged from v3)

### Phase A — Local Git initialization
```powershell
cd "D:\thesis\media_cowork\High\03_Spectral_Signatures\jm_separability_toolbox"
git init -b main
git add .
git commit -m "Initial commit: app skeleton + core math + UI + visualizations + Tab 5 results"
```

### Phase B — GitHub repository
1. Create public repo `jm-separability-toolbox` (no README/LICENSE/gitignore from UI).
2. Topics: `multispectral`, `remote-sensing`, `jeffries-matusita`, `gradio`, `huggingface`.
3. `git remote add origin https://github.com/Nickkoro21/jm-separability-toolbox.git`.
4. `git push -u origin main`.

### Phase C — Hugging Face Space (Option Γ.2 — GitHub Actions sync)
1. Create HF Space `NickKoro21/jm-separability-toolbox` (Gradio SDK, CPU basic, public).
2. Add HF write token to **GitHub Secrets** as `HF_TOKEN`.
3. Group 5 will provide `.github/workflows/sync-hf.yml`.

### Phase D — GitHub Pages
1. Settings → Pages → Source: Deploy from a branch → main / `/docs`.

## 12. Critical security notes (unchanged)

- HF token stored in Windows User environment variable `HF_TOKEN`.
- `.gitignore` excludes `*.token`, `*token*.txt`, `*.env`, `*.key`, `*.pem`.
- **Rule**: credentials never travel through chat, even partial.

## 13. User preferences (project-level)

- Concise instructions in **Greek** for strategic discussion; code/files in **English**.
- Step-by-step confirmation between stages — strategy before code, always.
- Complete `.py` files (not snippets or notebook cells).
- Proactive identification of issues without being asked.
- File delivery via `Filesystem MCP` (`edit_file` / `write_file`) for D: drive.
- Modern look matching Spectral 3D Explorer aesthetic.
- Badges similar style to PostProcessing-Toolbox repo.

## 14. Theory reference

**Bhattacharyya distance** between two Gaussian classes:
```
B = (1/8) (μ₁ - μ₂)ᵀ Σ̄⁻¹ (μ₁ - μ₂) + (1/2) ln(|Σ̄| / √(|Σ₁|·|Σ₂|))
```
where `Σ̄ = (Σ₁ + Σ₂) / 2`.

**Jeffries-Matusita transformation** (saturates at 2):
```
JM = 2 · (1 - exp(-B))
```

**Numerical stability** (in `src/core/jm.py`):
- Add `1e-6 · I` to each covariance matrix before inversion.
- Use `np.linalg.slogdet` for log-determinants.
- Use `np.linalg.solve` instead of explicit `np.linalg.inv`.
- Use `2.0 * (-np.expm1(-B))` instead of `2.0 * (1 - np.exp(-B))`.
- Detect rank-deficiency via determinant sign / `LinAlgError`, return `nan` gracefully.
- Clamp final JM to `[0.0, JM_MAX]`.

**4-bucket interpretation thresholds:**

| Bucket | Range | Hex colour | Tailwind class |
|---|---|---|---|
| Poor | 0.0 ≤ JM < 1.0 | `#ef4444` | red-500 |
| Moderate | 1.0 ≤ JM < 1.5 | `#f59e0b` | amber-500 |
| Good | 1.5 ≤ JM < 1.9 | `#4ade80` | green-400 |
| Excellent | 1.9 ≤ JM ≤ 2.0 | `#16a34a` | green-600 |

**Reference implementation** (used as ground truth for verification):
- `D:\thesis\models\deeplab_50_101\spectral_analysis\spectral_signatures_v2.py`
- Source CSV: `D:\thesis\models\deeplab_50_101\spectral_analysis\data\spectral_samples.csv`
- Verified working: BGR=1.288, 5MS=1.551, 7D=1.838 mean JM (4-decimal match)

## 15. Local launch — `run.ps1`

```powershell
cd "D:\thesis\media_cowork\High\03_Spectral_Signatures\jm_separability_toolbox"
.\run.ps1
```

Behaviour: resolves project root → sweeps ports 7860-7870 + kills listeners → activates `.venv` → runs `python app.py`.

First-time setup:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
```

## 16. Default subset heuristic algorithm — known caveat (NEW v4)

The function `_suggest_default_subsets(available_bands)` in `src/ui/tab4_config.py` matches via case-insensitive normalised lookup. The matching collapses whitespace/dashes/underscores and lower-cases — so `RedEdge`, `red_edge`, `RED-EDGE`, `redEDGE` all collapse to the key `rededge`.

```
Lookup keys: blue, green, red, rededge, nir, ndsm, thermal

Generated rows (only if all required bands are present):
    1. RGB                    if blue + green + red
    2. 5MS = RGB + RedEdge + NIR
    3. 7D = 5MS + nDSM + Thermal
    4. All                    if len(available_bands) >= 4
                              (or fallback when nothing else matched)
```

**Caveat (observed live 2026-04-28)**: the demo CSV uses band names `nDSM_m` and `Thermal_C` (with unit suffixes). After normalisation these become `ndsmm` and `thermalc`, which do **not** match the canonical keys `ndsm` and `thermal`. The 7D row therefore is NOT auto-generated for that CSV — only RGB, 5MS, and All appear.

**Workaround**: the user can either (a) use the auto-generated `All` subset (functionally equivalent to 7D for that CSV), or (b) manually add a `7D` row in Tab 4's Dataframe.

**Future fix candidate**: extend the `_norm` lookup to also try suffix-stripped canonical keys, e.g. `ndsmm.startswith("ndsm")` → match. Defer to a later session; not blocking.

## 17. Live test checklist + post-implementation pipeline (NEW v5)

### Step 1 — Live test of Tab 6 (immediate)

```powershell
cd D:\thesis\media_cowork\High\03_Spectral_Signatures\jm_separability_toolbox
.\run.ps1
```

Confirm each item before proceeding to Step 2:

1. ✅ Gradio launches without assertion errors (the `_TAB5_POPULATE_COUNT=66` startup assert passes).
2. ✅ Walk through Tabs 1→5 with `data/examples/spectral_samples.csv` — all 4 subsets (RGB, 5MS, 7D, All) compute and display.
3. ✅ Tab 6 shows lock screen until Tab 5 confirms.
4. ✅ Tab 5 confirm → auto-switch to Tab 6 with content visible.
5. ✅ Click **📦 Generate export** → green status banner `✅ Export ready.` + download file appears.
6. ✅ Download ZIP. Contents:
   - `README.txt` (sensor + subsets + classes + how-to-use)
   - `results/subset_summary.csv` (4 rows: RGB, 5MS, 7D, All)
   - `results/ranked_pairs_RGB.csv`, `ranked_pairs_5MS.csv`, `ranked_pairs_7D.csv`, `ranked_pairs_All.csv`
   - `example_guide.html`
7. ✅ Open `example_guide.html` in browser (internet required for Plotly CDN). Verify:
   - ⚠ EXAMPLE banner at top
   - Theory section (Bhattacharyya + JM equations + 4-bucket legend)
   - Section 2 — Subset Summary table + comparative bar + bucket distribution
   - Section 3 — JM heatmaps (4 subsets × interactive Plotly)
   - Section 4 — Ranked pairs (table + bar per subset, grid-2col layout)
   - Section 5 — Spectral signatures (grouped layout: Reflectance / Height / Temperature panels)
   - Section 6 — Interpreting YOUR results (paragraph + bullets + reminder banner)
   - Footer with GitHub link
8. ✅ Cascade-relock test — re-confirm Tab 1, Tab 2, Tab 3, or Tab 4 → all downstream tabs (including Tab 6) re-locked.

### Step 2 — Minor fixes (if any surface from Step 1)

User indicated some minor fixes will be applied after live test. Common candidates:
- Greek micro-copy adjustments in any user-facing string.
- Visual polish on the HTML guide (spacing, colours, banner styling).
- README.txt wording tweaks.
- Edge cases not covered by current `_ready_for_export` checks.
- Minor CSS tweaks in `_html_head_and_style()`.

Each fix should follow the standard pattern: strategy first in Greek, surgical `Filesystem:edit_file`, dryRun if uncertain, live re-test.

### Step 3 — GitHub repo init + push (§11 Phase A-B)

Once Tab 6 is verified working and any fixes applied:

```powershell
cd "D:\thesis\media_cowork\High\03_Spectral_Signatures\jm_separability_toolbox"
git init -b main
git add .
git commit -m "Initial commit: 6-tab JM Separability Toolbox (Tabs 1-6 complete)"
```

**Pre-push checklist**:
- ✅ `.gitignore` excludes `.venv/`, `__pycache__/`, `*.bak`, `*.token`, `*.env`.
- ✅ `data/media/` figures committed (~2.2 MB) — these are project assets for docs/Pages.
- ✅ `data/examples/spectral_samples.csv` committed (test fixture).
- ✅ No HF token / credentials in code.

Then create the public GitHub repo `jm-separability-toolbox` (no README/LICENSE/gitignore from UI), set topics, and push:

```powershell
git remote add origin https://github.com/Nickkoro21/jm-separability-toolbox.git
git push -u origin main
```

### Step 4 — HF Space + GitHub Actions sync (Phase C, Decision #11 — Option Γ.2)

1. Create HF Space `NickKoro21/jm-separability-toolbox` (Gradio SDK, CPU basic, public).
2. Generate HF write token → store as **GitHub Secrets** key `HF_TOKEN` (NEVER paste in chat).
3. Add `.github/workflows/sync-hf.yml` with the standard `huggingface/huggingface_hub` push action.

The README already has the `app_file: app.py` YAML front-matter for Gradio Spaces; first push should auto-build.

### Step 5 — GitHub Pages docs (Group 5)

1. Settings → Pages → Source: Deploy from branch → main / `/docs`.
2. Populate `docs/` with `index.md`, `methodology.md`, `presets.md`, `troubleshooting.md`.
3. The `data/media/` figures can be referenced inline in `methodology.md` (e.g., separability matrices, spectral profiles, nDSM boxplot, thermal violin) for the academic walkthrough.

## 18. Next conversation kickoff prompt (paste verbatim)

> Continue development of the JM Separability Toolbox at `D:\thesis\media_cowork\High\03_Spectral_Signatures\jm_separability_toolbox\`. Read `HANDOFF.md` (v5) first — it contains every decision, file written, verification, and architectural pattern from previous groups.
>
> **Status**: All 6 tabs implemented and wired in `app.py`. Tabs 1-5 live-tested end-to-end. Tab 6 (Export) implemented in v5 — 27.5 KB module producing CSV results + HTML interpretation guide + ZIP bundle. **Tab 6 live-test still pending.** New `data/media/` folder with 14 thesis figures (spectral profiles a1-a7 + combined, separability matrices RGB/5MS/7D, combined nDSM boxplot, combined Thermal violin, cumulative_gain) added for documentation purposes.
>
> **Patterns established (A-P, all in production)**: sequential lock pattern, ✅ label progress, auto-tab-switch, cascade-reset, cascade-relock, `clear_state_updates()` (Pattern H), `populate_state_updates()` (Pattern I), `populate_refs()` (Pattern N), schema-key naming convention (Pattern O), terminal-tab structure (Pattern P).
>
> **Next steps** (in order):
> 1. **Live-test Tab 6** (HANDOFF §17 Step 1) — `.\run.ps1`, walk through Tabs 1→5, generate export, verify ZIP contents and HTML guide.
> 2. **Apply minor fixes** if any surface (HANDOFF §17 Step 2).
> 3. **GitHub repo init + push** (Phase A-B of §11).
> 4. **HF Space + GitHub Actions sync** (Phase C, Decision #11 — Option Γ.2).
> 5. **GitHub Pages docs** (Phase D / Group 5) — reference figures from `data/media/`.
>
> **Constraints** (must follow):
> - **Strategy first** in Greek before code. Confirm the design with me before writing any `.py`.
> - Follow the patterns documented in HANDOFF.md §8 (Pattern A through P) — they are non-negotiable.
> - Use `Filesystem:edit_file` for surgical edits on existing files. Use `Filesystem:write_file` (when available) OR `Windows-MCP:FileSystem` mode=`write` with **`append=True`** for chunked writes (the append mode handles arbitrary sizes; the overwrite mode times out >~2-3KB).
> - Use the existing `src.core` and `src.viz` APIs only — no re-implementation of math, presets, validation, detection, plot factories, or band classification.
> - **Schema dict keys: `class_col` / `band_cols` / `non_spectral_cols` / `xy_cols`** — NEVER `class_column` etc. (Pattern O).
> - 4-bucket palette `JM_BUCKET_COLORS` from `src.core` is the source of truth for any colour mapping in plots.
> - For credentials: `HF_TOKEN` lives in **GitHub Secrets** / Windows User env vars — NEVER in chat or commits.
> - Test live in the browser via `.\run.ps1` after each step.

---

**End of handoff document v5.**
