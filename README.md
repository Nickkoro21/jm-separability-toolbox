---
title: Spectral Separability Explorer
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.20.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Sensor-agnostic JM separability for multispectral data
---

<div align="center">

# 🛰️ Spectral Separability Explorer

### Sensor-agnostic Jeffries–Matusita separability analysis for multispectral data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.0+-F97316?logo=gradio&logoColor=white)](https://gradio.app/)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-FFD21E)](https://huggingface.co/spaces/NickKoro21/jm-separability-toolbox)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Documentation-181717?logo=github)](https://nickkoro21.github.io/jm-separability-toolbox/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](#)

[**▶ Try the live app**](https://huggingface.co/spaces/NickKoro21/jm-separability-toolbox) ·
[**📖 Documentation**](https://nickkoro21.github.io/jm-separability-toolbox/) ·
[**📊 Source code**](https://github.com/Nickkoro21/jm-separability-toolbox)

</div>

---

## Overview

**Spectral Separability Explorer** is a browser-based, sensor-agnostic tool that quantifies how well land-cover (or any other) classes can be distinguished from one another given a chosen subset of spectral bands. It accepts any CSV with per-sample band values and a class label, and produces:

- Per-class **spectral signatures** (mean reflectance, all classes overlaid)
- **Boxplots** and **violin plots** per band (units-aware: reflectance, height in metres, temperature in °C)
- **Jeffries–Matusita (JM) distance matrices** with discrete 4-bucket colour coding (Poor / Moderate / Good / Excellent)
- **Comparative analysis** across band subsets (e.g. RGB vs 5MS vs 7D = RGB+RedEdge+NIR+nDSM+Thermal)
- **Ranked separability table** for pair-by-pair drill-down
- **ZIP export bundle** with your results as CSV plus an HTML interpretation guide

The app guides you through a six-step sequential workflow with hard-error validation, progressive disclosure, and explanatory feedback at every stage.

> **Built as a complementary deliverable to my MSc thesis** at the University of the Aegean, where it supports the band-selection rationale of an urban semantic-segmentation pipeline. Released open-source so other researchers can apply the same analysis to **their** sensors and datasets.

---

## Why this tool exists

Adding more spectral bands to a remote-sensing pipeline is rarely a free lunch. Each band increases data volume, processing cost, and (in deep learning) the dimensionality of the input tensor. Before committing to a sensor or band combination, it pays to ask a quantitative question: *how much extra class separability does each additional band actually buy?*

The classical answer is the **Jeffries–Matusita distance** — a statistical measure of separability between two Gaussian-distributed classes that ranges from 0 (identical distributions) to 2 (perfectly separable). It is widely used in feature-selection literature ([Bruzzone & Serpico 2000](https://doi.org/10.1080/014311600210740); [Herold et al. 2004](https://doi.org/10.1016/j.rse.2004.02.013)) but its practical computation requires building per-class covariance matrices, handling singular cases, and visualizing pairwise results — friction that often discourages its routine use.

This app removes that friction.

---

## Theory in 60 seconds

For each pair of classes *(i, j)*, given samples in a *d*-dimensional feature space:

**1. Estimate per-class statistics** — mean vector **μ** and covariance matrix **Σ** for each class.

**2. Compute the Bhattacharyya distance** *B*:

```math
B = \tfrac{1}{8} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \, \bar{\boldsymbol{\Sigma}}^{-1} \, (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) \;+\; \tfrac{1}{2} \ln\!\left( \frac{|\bar{\boldsymbol{\Sigma}}|}{\sqrt{|\boldsymbol{\Sigma}_1|\,|\boldsymbol{\Sigma}_2|}} \right)
```

where $\bar{\boldsymbol{\Sigma}} = \tfrac{1}{2}(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2)$. The first term measures separation of the means weighted by pooled variance (Mahalanobis-like). The second term captures divergence in class shape — two classes can share a mean but differ in spread, and this term detects exactly that.

**3. Transform to JM distance**, bounded in [0, 2]:

```math
JM = 2 \left( 1 - e^{-B} \right)
```

The exponential transform saturates at 2.0, preventing outlier-driven distortion and yielding a stable interpretation:

| JM range | Bucket | Interpretation |
|:--------:|:------:|:---------------|
| 0.0 – 1.0 | **Poor** | classes substantially overlap in feature space |
| 1.0 – 1.5 | **Moderate** | partial separability; classifier-dependent results |
| 1.5 – 1.9 | **Good** | most pairs distinguishable under the Gaussian assumption |
| 1.9 – 2.0 | **Excellent** | near-complete separability |

For numerical stability, a small regularization term (`1e-6 · I`) is added to each covariance matrix before inversion, and `slogdet` is used to compute log-determinants of large matrices safely.

**Caveat**: JM assumes class-conditional Gaussianity. Real-world spectral distributions are often multimodal or skewed (especially mixed-pixel classes like *Shadow* or *Soil*). Treat JM as a **first-order indicator** of separability — useful for ranking and feature selection, not a substitute for empirical classifier evaluation.

---

## Six-step workflow

```mermaid
flowchart LR
    A[1. Camera<br/>preset] --> B[2. Confirm<br/>wavelengths]
    B --> C[3. Upload<br/>CSV]
    C --> D[4. Select bands<br/>& classes]
    D --> E[5. Visualize<br/>results]
    E --> F[6. Export<br/>artifacts]
    style A fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style B fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style C fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style D fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style E fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style F fill:#1e3a5f,stroke:#60a5fa,color:#fff
```

Each tab unlocks only when the previous step is complete. Validation errors appear with clear messages — no silent failures.

---

## Supported camera presets

The app ships with built-in band metadata for eight popular sensors. Each preset includes center wavelengths, FWHM bandwidths, and an official source link for ground-truth verification.

| Sensor | Bands | Source |
|---|:--:|:--:|
| **MicaSense Altum-PT** | 5 MS + Pan + Thermal | [📄](https://support.micasense.com/hc/en-us/articles/214878778) |
| **MicaSense RedEdge-MX** | 5 MS | [📄](https://support.micasense.com/hc/en-us/articles/214878778) |
| **MicaSense RedEdge-MX Dual** | 10 MS | [📄](https://support.micasense.com/hc/en-us/articles/214878778) |
| **DJI Phantom 4 Multispectral** | 5 MS + RGB | [📄](https://ag.dji.com/p4-multispectral/specs) |
| **DJI Mavic 3 Multispectral** | 4 MS + RGB | [📄](https://enterprise.dji.com/mavic-3-m/specs) |
| **Parrot Sequoia / Sequoia+** | 4 MS + RGB | [📄](https://www.parrot.com/assets/s3fs-public/2021-09/bd_sequoia_integration_manual_en_0.pdf) |
| **Sentinel-2 MSI** | 13 bands | [📄](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial) |
| **Landsat 8/9 OLI+TIRS** | 11 bands | [📄](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites) |

You can also enter **custom** wavelengths or **skip** the wavelength step entirely — the analysis runs identically; only the x-axis of spectral signature plots changes.

---

## Input format

A CSV file with one row per sample and the following structure:

| Column | Required | Type | Description |
|---|:--:|:---:|:---|
| `class` | ✅ | int / str | Class identifier (e.g. `1` or `"Tree"`) |
| `class_name` | optional | str | Human-readable label (overrides `class` for display) |
| `<band_1>` … `<band_n>` | ✅ (≥ 2) | float | Per-band reflectance, DN, or any numeric feature |
| `x`, `y`, `sample_id` | optional | any | Reserved metadata columns (excluded from analysis) |

**Detection** of class column and non-spectral bands is automatic, with manual override available if heuristics fail.

### Validation rules (hard errors, block execution)

- ❌ Fewer than **2 numeric bands**
- ❌ Fewer than **2 classes**
- ❌ Any class with fewer than **100 samples** (insufficient for stable covariance estimation)
- ❌ Number of samples in any class ≤ number of selected bands + 1 (singular covariance)

### Soft warnings (proceed with confirmation)

- ⚠️ NaN values present (rows auto-dropped)
- ⚠️ Mixed scales detected (e.g. one band 0–1, another 0–255)
- ⚠️ Class imbalance > 10:1

---

## Output artifacts

The live app produces six visualization types in **Step 5** and one downloadable export bundle in **Step 6**:

| Visualization | Purpose |
|---|---|
| **Subset summary table** | Per-subset stats: mean / min / max / std JM, bucket counts, mean-bucket category |
| **Comparative bar + bucket distribution** | Cross-subset comparison of mean JM and bucket-count breakdown |
| **JM distance matrices** | One heatmap per subset, discrete 4-bucket colour scheme, masked diagonal |
| **Ranked class pairs** | Sortable table + bar chart per subset, worst pairs first |
| **Spectral signatures** | All classes overlaid, mean reflectance only, x-axis ordered by wavelength |
| **Boxplots & Violins per band** | Distribution shape, IQR, outliers; faceted by band with unit-aware Y axis (reflectance / metres / °C) |

### Export bundle (Step 6)

Click **📦 Generate export** and download a ZIP containing:

```
jm_export_<timestamp>.zip
├── README.txt                          — sensor + subsets + classes + how-to-use
├── results/
│   ├── subset_summary.csv             — your per-subset JM stats
│   └── ranked_pairs_<subset>.csv      — worst-first pair ranking, one CSV per subset
└── example_guide.html                 — interactive interpretation guide
```

> 💡 The `example_guide.html` uses the **bundled example dataset** to demonstrate how to read your results — it is clearly marked with an EXAMPLE banner. Your computed values live in `results/*.csv`. For individual high-resolution PNGs of any plot, use the camera icon on each Plotly chart in Step 5.

---

## Quick start (local installation)

**Prerequisites**: Python 3.11 or newer.

```bash
git clone https://github.com/Nickkoro21/jm-separability-toolbox.git
cd jm-separability-toolbox
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python app.py
```

The Gradio interface opens at `http://localhost:7860`.

---

## Project structure

```
jm-separability-toolbox/
├── app.py                       Entry point (Gradio root)
├── requirements.txt             Python dependencies
├── README.md                    This file (also serves as HF Space landing page)
├── LICENSE                      MIT
├── run.ps1                      PowerShell launcher (Windows)
├── src/
│   ├── core/
│   │   ├── jm.py                JM math: Bhattacharyya, regularised covariance, 4-bucket scheme
│   │   ├── presets.py           Camera band metadata + verification URLs (8 sensors)
│   │   ├── validation.py        Hard-error and soft-warning checks (min 100 samples/class, ...)
│   │   ├── detection.py         Auto-detection of class column, spectral / non-spectral bands, x/y
│   │   └── band_classification.py  Group bands by physical quantity (Reflectance / Height / Temperature / Index)
│   ├── ui/
│   │   ├── tab1_camera.py       Step 1 — sensor selection from 8 presets or custom
│   │   ├── tab2_wavelengths.py  Step 2 — confirm or override centre wavelengths
│   │   ├── tab3_upload.py       Step 3 — CSV upload, schema detection, validation
│   │   ├── tab4_config.py       Step 4 — class filter and band-subset definition
│   │   ├── tab5_results.py      Step 5 — 6 collapsible accordions, all viz auto-computed
│   │   └── tab6_export.py       Step 6 — ZIP export with results CSV + HTML interpretation guide
│   └── viz/
│       ├── spectral_combined.py  All classes overlaid, mean reflectance only
│       ├── boxplots.py           Per-band, classes side-by-side, unit-aware Y
│       ├── violins.py            Per-band KDE shape with inner mini-boxplot
│       ├── jm_matrix.py          Discrete 4-bucket heatmap, masked diagonal
│       ├── jm_comparative.py     Subset summary + comparative bar + bucket distribution
│       └── ranked_pairs.py       Worst-first sortable ranking per subset
├── data/
│   ├── examples/
│   │   └── spectral_samples.csv      Demo dataset (MicaSense Altum-PT, 6 997 samples × 7 bands × 7 classes)
│   └── media/                    14 thesis figures (spectral profiles, separability matrices, etc.)
├── docs/                                GitHub Pages source
└── tests/                               Unit tests for core math and validation
```

---

## Roadmap

- [x] Project skeleton and HF YAML configuration
- [x] Core JM math module with numerical-stability tests
- [x] Eight built-in camera presets
- [x] Six-tab Gradio UI with progressive disclosure
- [x] Six visualization types (Plotly, units-aware Y axis)
- [x] Multi-subset comparative mode
- [x] CSV results + HTML interpretation guide ZIP export
- [x] Demo dataset bundle (`spectral_samples.csv`)
- [x] Thesis-figure documentation gallery (`data/media/`)
- [ ] GitHub Pages documentation site
- [ ] Continuous deployment to Hugging Face Space (GitHub Actions)
- [ ] Unit-test suite under `tests/`

---

## Documentation gallery

The `data/media/` folder ships with **14 reference figures** generated for the MSc thesis using the bundled example dataset. They illustrate every visualisation type the toolbox produces, plus a comparative cumulative-gain plot used in the band-selection chapter:

| Group | Files | What they show |
|---|---|---|
| **Per-class spectral profiles** | `fig_a1_tree_profile.png` … `fig_a7_shadow_noise_profile.png` | One panel per land-cover class — mean ± 1σ across the 5 MS bands plus nDSM and thermal |
| **Combined spectral profile** | `fig_a_spectral_profile.png` | All classes overlaid on a shared reflectance axis |
| **Boxplot — nDSM (height)** | `combined_fig_b_ndsm_boxplot.png` | Class-wise distribution of normalised-DSM values (metres) |
| **Violin — Thermal** | `combined_fig_c_thermal_violin.png` | Class-wise KDE of thermal radiance (°C) |
| **JM separability matrices** | `seperability_matrix_RGB.png`, `..._5MS.png`, `..._7D.png` | The 4-bucket heatmap for each band subset — same plot the live app produces |
| **Cumulative gain** | `cumulative_gain.png` | How mean JM grows as bands are added in increasing-information order |

These files are committed to the repo so the upcoming GitHub Pages documentation site (and any external citations) can reference them with stable URLs.

---

## Citation

If you use this tool in academic work, please cite:

```bibtex
@software{koroniadis2026spectral,
  author    = {Koroniadis, Nikolaos},
  title     = {{Spectral Separability Explorer:
              Sensor-agnostic Jeffries--Matusita analysis for multispectral data}},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/spaces/NickKoro21/jm-separability-toolbox},
  note      = {MSc thesis deliverable, University of the Aegean}
}
```

---

## Author & affiliations

**Nikolaos Koroniadis**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nick_Koroniadis-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nick-koroniadis-328962226)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-NickKoro21-FFD21E)](https://huggingface.co/NickKoro21)
[![GitHub](https://img.shields.io/badge/GitHub-Nickkoro21-181717?logo=github&logoColor=white)](https://github.com/Nickkoro21)

MSc Geography and Applied Geoinformatics
Department of Geography, [University of the Aegean](https://www.geo.aegean.gr/geo-en.php)
[Remote Sensing & GIS Research Group (RSGIS Lab)](https://rsgis.geo.aegean.gr/)

**Thesis Supervisor**: Dr. Christos Vasilakos

### Related links

- [MSc Programme — Geography and Applied Geoinformatics](https://geography.aegean.gr/geoinformatics/)
- [Department of Geography](https://geography.aegean.gr/)
- [University of the Aegean](https://www.geo.aegean.gr/geo-en.php)
- [RSGIS Lab](https://rsgis.geo.aegean.gr/)
- Companion project: [PostProcessing Toolbox](https://github.com/Nickkoro21/PostProcessing-Toolbox) — ArcGIS Pro toolbox for vectorizing semantic segmentation outputs
- Companion app: [3D Spectral Feature Space Explorer](https://huggingface.co/spaces/NickKoro21/spectral-3d-explorer) — interactive 3D visualization of class separability

---

## Acknowledgments

- **Dr. Christos Vasilakos** for thesis supervision and guidance.
- **University of the Aegean RSGIS Lab** for computational resources and academic environment.
- **Anthropic Claude** for AI-assisted development during the thesis project.
- **Hugging Face** and **GitHub** for free, open-source-friendly hosting infrastructure that makes deliverables like this possible.

---

## License

Released under the [MIT License](LICENSE) — free for academic, commercial, and personal use.

---

<div align="center">
<sub>Made with ☕, 🛰️, and a lot of <code>NumPy</code> in Mytilene, Greece.</sub>
</div>
