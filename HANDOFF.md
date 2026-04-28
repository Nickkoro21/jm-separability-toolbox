# JM Separability Toolbox — HANDOFF

> **Self-contained handoff for the JM Separability Toolbox project.**
> Replaces all earlier conversations about this project. After reading
> this document, a fresh Claude session has everything it needs to
> continue work without further context transfer.

---

## 0. TL;DR — current state

| Surface | Status | URL |
|---|---|---|
| GitHub repo | ✅ Live | https://github.com/Nickkoro21/jm-separability-toolbox |
| Hugging Face Space | 🟢 Running | https://huggingface.co/spaces/NickKoro21/jm-separability-toolbox |
| GitHub Actions auto-sync | 🟢 Working | Triggers on push to `main` |
| GitHub Pages | ❌ Not enabled yet | **This is Phase D** — the next deliverable |

The app works end-to-end on HF Space. The 6-tab Gradio workflow
(Camera → Wavelengths → Upload CSV → Configure → Results → Export)
is fully functional and deployed.

---

## 1. Project identity

| Field | Value |
|---|---|
| Owner | Nikolaos Koroniadis |
| Affiliation | MSc Geography & Applied Geoinformatics, University of the Aegean |
| Supervisor | Dr. Christos Vasilakos (RSGIS Lab) |
| Local path | `D:\thesis\media_cowork\High\03_Spectral_Signatures\jm_separability_toolbox` |
| GitHub user | `Nickkoro21` |
| HF user | `NickKoro21` (case-sensitive) |
| Companion repo | https://github.com/Nickkoro21/PostProcessing-Toolbox |
| Companion HF Space | https://huggingface.co/spaces/NickKoro21/spectral-3d-explorer |

The toolbox is a **sensor-agnostic Jeffries–Matusita (JM) separability
analyser** for any multispectral CSV. It is the validated, distributable
artefact derived from the spectral-signature analysis chapter of
Nikolaos's MSc thesis on UAV multispectral urban land-cover
segmentation. The thesis itself uses the bundled MicaSense Altum-PT
dataset (6 997 samples × 7 classes) as its reference scene.

---

## 2. Repository file map

This is the **authoritative map of every tracked file**, what it does,
and where to look when making small changes. Use this section as your
first stop before any modification.

### 2.1 Top-level

| Path | Purpose |
|---|---|
| `app.py` | Gradio entry point. Builds the 6-tab `Blocks` shell, defines tab-transition handlers, mounts the footer, calls `app.launch()`. **The footer HTML lives here** (lines ~55–80). |
| `README.md` | Two roles: (a) GitHub README rendered on the repo home, (b) **HF Space metadata block** (the YAML frontmatter at the top configures `sdk_version`, `python_version`, emoji, colours, etc. — HF reads this on every push). |
| `requirements.txt` | Python pinning. **Critical for HF Space stability** — see §5 for the full version-pin saga. |
| `LICENSE` | MIT. |
| `.gitignore` | Excludes `.venv/`, `__pycache__/`, `*.bak`, `*.orig`, `.env`, `*.token`, build artefacts. |
| `run.ps1` | Local launcher (PowerShell). Activates `.venv` and runs `python app.py`. Not used on HF — there `app.py` is the entry. |
| `pyproject.toml` | (if present) build config; not load-bearing for the app. |

### 2.2 `src/` — application code

```
src/
├── __init__.py
├── core/                  # Pure logic. No Gradio imports here.
│   ├── __init__.py        # Re-exports: auto_detect_schema,
│   │                      # group_bands_by_category,
│   │                      # CATEGORY_REFLECTANCE / _HEIGHT / _THERMAL,
│   │                      # get_unrecognised_bands, presets, JM funcs.
│   ├── band_classification.py  # Routes column names into Reflectance/
│   │                      # Height/Thermal/Other based on wavelengths
│   │                      # and name heuristics. Used by Tab 5 to keep
│   │                      # spectral plot reflectance-only.
│   ├── detection.py       # auto_detect_schema(df) — finds class column,
│   │                      # spectral bands, and non-spectral cols
│   │                      # (nDSM_m, Thermal_C, etc.).
│   ├── jm.py              # The numerical heart. Bhattacharyya distance
│   │                      # via slogdet, regularised covariance,
│   │                      # rank-deficiency fallback, expm1 conversion
│   │                      # to JM. 4-bucket scheme: Poor/Moderate/
│   │                      # Good/Excellent (cutoffs 1.0 / 1.5 / 1.9).
│   ├── presets.py         # 8 built-in sensors + Custom. Each preset is
│   │                      # a dataclass with name, bands, wavelengths,
│   │                      # source URL.
│   └── validation.py      # CSV schema and value-range checks used by
│                          # Tab 3 upload.
│
├── ui/                    # All Gradio glue. One module per tab.
│   ├── __init__.py
│   ├── tab1_camera.py     # Step 1. Dropdown of presets, info card,
│   │                      # Confirm button.
│   ├── tab2_wavelengths.py# Step 2. Editable wavelength table per
│   │                      # band.
│   ├── tab3_upload.py     # Step 3. CSV upload, auto-schema preview,
│   │                      # validation banner.
│   ├── tab4_config.py     # Step 4. Subset selection (RGB / 5MS / 7D
│   │                      # / All), class picker, Confirm to populate
│   │                      # Tab 5.
│   ├── tab5_results.py    # Step 5. The big one. Renders subset
│   │                      # summary, comparative bar, bucket
│   │                      # distribution, JM heatmaps per subset,
│   │                      # ranked class pairs, spectral signatures
│   │                      # (REFLECTANCE BANDS ONLY, no ±1σ),
│   │                      # boxplots, violins. _TAB5_POPULATE_COUNT
│   │                      # in app.py MUST equal len(populate_refs).
│   └── tab6_export.py     # Step 6. ZIP packager + the
│                          # `example_guide.html` template (single big
│                          # f-string in this file). The HF/GitHub
│                          # author line and the section-5 description
│                          # both live here.
│
└── viz/                   # Plot factories. Plotly + matplotlib palette.
    ├── __init__.py        # Re-exports + generate_class_palette()
    │                      # (matplotlib tab10/tab20 → hex dict).
    ├── boxplots.py        # Per-band boxplots, faceted by class.
    ├── jm_comparative.py  # Subset summary table, comparative bar,
    │                      # bucket distribution stacked bar.
    ├── jm_matrix.py       # Single JM heatmap (used per subset).
    ├── ranked_pairs.py    # Sorted class-pair JM ranking (table+bar).
    ├── spectral_combined.py # All-classes-overlaid mean line plot.
    │                      # show_std=False is the default since Tab 5
    │                      # cleanup. Filters to reflectance bands only.
    ├── spectral_per_class.py # FACETED per-class spectral plot.
    │                      # NOTE: still in the codebase as a module,
    │                      # but NOT WIRED into Tab 5 anymore (removed
    │                      # in the final UX pass — see §3).
    └── violins.py         # Per-band violin plots.
```

### 2.3 `data/`

```
data/
├── examples/
│   └── spectral_samples.csv     # The bundled demo dataset.
│                                # MicaSense Altum-PT, 6 997 rows,
│                                # 7 classes (Tree, Building, Road,
│                                # Vehicle, Grass, Bare Soil,
│                                # Shadow-Noise). 7 bands: Blue, Green,
│                                # Red, RedEdge, NIR, nDSM_m, Thermal_C.
└── media/                       # 14 PNG figures used by GitHub README
    │                            # gallery and (later) GitHub Pages.
    │                            # IMPORTANT: this folder is GITHUB-ONLY.
    │                            # The CI workflow strips it before
    │                            # pushing to HF Space (HF Hub rejects
    │                            # binary files unless via Xet).
    ├── fig_a_spectral_profile.png        # Combined mean-line plot
    ├── fig_a1_tree_profile.png           # Per-class spectral panels
    ├── fig_a2_building_profile.png
    ├── fig_a3_road_profile.png
    ├── fig_a4_vehicle_profile.png
    ├── fig_a5_grass_profile.png
    ├── fig_a6_bare_soil_profile.png
    ├── fig_a7_shadow_noise_profile.png
    ├── combined_fig_b_ndsm_boxplot.png   # nDSM height distributions
    ├── combined_fig_c_thermal_violin.png # Thermal °C violins
    ├── seperability_matrix_RGB.png       # JM heatmaps per subset
    ├── seperability_matrix_5MS.png       # (note: filename has the
    ├── seperability_matrix_7D.png        #  typo "seperability" — keep
    └── cumulative_gain.png               #  it; renaming would break
                                          #  any external links)
```

### 2.4 `.github/workflows/`

```
.github/workflows/
└── sync-hf.yml          # The GitHub Actions auto-deploy workflow.
                         # Trigger: push to main + manual dispatch.
                         # Strategy: orphan-branch squash (see §4).
                         # Required secret: HF_TOKEN (Write role).
                         # Uses actions/checkout@v5 (Node.js 24).
```

### 2.5 What is **not** in the repo (intentionally)

- `.venv/` — local virtualenv, gitignored.
- Python 3.12 / 3.13 wheels caches — not relevant.
- `*.bak`, `*.orig` — earlier scratch backups, gitignored.
- HF Space–specific `app.py` — there is no separate file. The same
  `app.py` runs everywhere; HF reads it via the YAML frontmatter.

---

## 3. UX decisions worth remembering

These were made during the final cleanup pass before deployment. Do
**not** revert them without a strong reason.

| # | Decision | Why | Where |
|---|---|---|---|
| D1 | **Spectral Signatures = mean line only**, no ±1σ shaded band | Visual noise; makes class trends harder to read | `tab5_results.py` passes `show_std=False`; `tab6_export.py` does the same when building the HTML guide |
| D2 | **Spectral Per-Class panel removed** from Tab 5 | Redundant with the combined plot once boxplots/violins are present below | Removed from `tab5_results.py` widget list, populate refs, and clear updates. `_TAB5_POPULATE_COUNT` in `app.py` was decremented `66 → 65`. The `spectral_per_class.py` module still exists — left in place for future opt-in but not wired |
| D3 | **Spectral Signatures filtered to reflectance bands only** | nDSM (m) and Thermal (°C) have incompatible Y-axis units; covered properly by the per-band Boxplot/Violin panels below | `tab5_results.py` and `tab6_export.py` both call `group_bands_by_category(...).get(CATEGORY_REFLECTANCE, [])` before passing bands to `make_spectral_combined` |
| D4 | **HTML export covers RGB / 5MS / 7D / All** | Earlier code only built 7D when `len(bands)==7`; this missed it because `nDSM_m` and `Thermal_C` are detected as non-spectral. Fix: search both `band_cols` and `non_spectral_cols` for the 7D match | `tab6_export.py`, the `subsets` dict construction near the top of `_render_example_guide` |
| D5 | **HTML export has author line in header** | Attribution + supervisor + GitHub/LinkedIn links | `tab6_export.py` — `<p class="author-line">` block in the f-string template; matching CSS rule `.author-line { ... }` |
| D6 | **f-string templates, no Jinja2** | Single source of truth, fewer deps; HANDOFF Decision #42 from earlier sessions | `requirements.txt` has no `jinja2` line. Don't add one |

If you're tempted to add `jinja2` for "cleaner templating": don't.
The HTML guide is small enough that f-strings are clearer and the
removal was deliberate.

---

## 4. Deployment pipeline (the painful bits)

This section captures the issues hit during Phase C (HF Space
deployment). Future you will save hours by reading it.

### 4.1 The auto-sync workflow

The single source of truth for deployment is `.github/workflows/sync-hf.yml`.
Its job is: every push to `main` on GitHub → force-push the same
state to the HF Space `main` branch.

**Strategy: orphan-branch squash.** This is non-negotiable. Naive
`git push --force` fails because:

1. HF Hub validates the **full git history** for binary files, not
   just the current state.
2. The 14 PNGs in `data/media/` exist in the initial GitHub commit.
3. Removing them in a follow-up commit doesn't help — the offending
   blobs are still in history.
4. HF rejects with `pre-receive hook declined: Your push was rejected
   because it contains binary files. Please use https://huggingface.co/docs/hub/xet`.

The workflow handles this by:

```yaml
- Build clean orphan branch:
    git checkout --orphan hf-deploy-tmp
    git rm -rf --cached data/media
    rm -rf data/media
    git add -A
    git commit -m "Deploy to HF Space (squashed from main)"
- Force-push:
    git push --force https://NickKoro21:$HF_TOKEN@huggingface.co/spaces/NickKoro21/jm-separability-toolbox hf-deploy-tmp:main
```

The same trick works locally if you ever need to push by hand:

```powershell
$token = [Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
git remote add hf "https://NickKoro21:$token@huggingface.co/spaces/NickKoro21/jm-separability-toolbox"
git checkout --orphan hf-deploy-tmp
git rm -rf --cached data/media; Remove-Item -Recurse -Force "data\media"
git commit -m "Deploy to HF Space"
git push hf hf-deploy-tmp:main --force
git checkout main
git branch -D hf-deploy-tmp
# Local data/media/ is restored automatically when checking out main.
```

### 4.2 Secrets

| Secret | Where | Value |
|---|---|---|
| `HF_TOKEN` | Windows env var (User scope), local | Write-role HF token, 37 chars, starts `hf_n...` |
| `HF_TOKEN` | **GitHub repo Secrets** (Settings → Secrets and variables → Actions) | **Same token** as above. Required by the workflow. **If you rotate the HF token, rotate it in both places.** |

The token was generated at https://huggingface.co/settings/tokens with
"Write" role.

### 4.3 The five upstream issues (for the changelog)

In order of discovery, with the fix that finally stuck:

| # | Symptom | Cause | Final fix |
|---|---|---|---|
| 1 | HF push rejected, `binary files` error | Full-history binary check | Orphan-branch squash in workflow |
| 2 | HF Space build error: `short_description must be ≤60 chars` | README YAML had 73 chars | Trimmed to "Sensor-agnostic JM separability for multispectral data" (54 chars) |
| 3 | Runtime error: `ModuleNotFoundError: audioop` | Python 3.13 (HF default) removed `audioop`; pydub still imports it via Gradio | Pin `python_version: "3.11"` in README YAML |
| 4 | Runtime error: `cannot import name 'HfFolder' from 'huggingface_hub'` | Gradio <5.10 used the removed `HfFolder` | Bump `gradio>=5.50` (the version that works locally) |
| 5 | UI loaded but `Confirm camera` stayed disabled; logs showed `gradio_client` `TypeError: argument of type 'bool' is not iterable` | gradio_client schema introspection bug + dropdown change events broke in Gradio 5.20 | Same bump: `gradio>=5.50`, `gradio_client>=1.14`, `huggingface_hub>=1.0,<2.0` |

The current `requirements.txt` reflects the locally-verified stack
(`gradio 5.50.0`, `gradio_client 1.14.0`, `huggingface_hub 1.12.0`).
**If you ever change pinning, validate locally first** with the same
Python 3.11 and run through Tabs 1–6 end-to-end before pushing.

### 4.4 The PowerShell stderr quirk

`git push` writes informational output to stderr. PowerShell
interprets that as an error and reports `Status Code: 1`. **This is
not a real failure.** Look at the actual git output: if you see
`abc123..def456  main -> main`, the push succeeded. The exit code is
misleading.

### 4.5 What the YAML frontmatter actually controls

The block at the top of `README.md`:

```yaml
---
title: Spectral Separability Explorer
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.50.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Sensor-agnostic JM separability for multispectral data
---
```

| Field | Effect |
|---|---|
| `sdk_version` | HF pre-installs this Gradio version into the image. **Should match what `requirements.txt` allows** to avoid pip resolving away from it on the second install pass |
| `python_version` | Picks the base Docker image. Stay on 3.11 unless audioop comes back |
| `short_description` | Card text on the HF profile. **Hard cap: 60 chars** — HF rejects the build if longer |
| `app_file` | Entry point. Must point to a file that calls `gr.Blocks().launch()` |

---

## 5. Phase status overview

| Phase | Description | Status |
|---|---|---|
| A | Local Git init + first commit | ✅ Complete (`65fd54e`) |
| B | GitHub repo + push | ✅ Complete |
| C | HF Space + Actions auto-sync | ✅ Complete after 5 deployment fixes |
| **D** | **GitHub Pages docs site** | **⏳ Next — see §6** |
| E | (optional) HANDOFF.md update for thesis chapters | Separate workflow |

---

## 6. Phase D — GitHub Pages docs (the goal of this handoff)

The README currently links to `https://nickkoro21.github.io/jm-separability-toolbox/`
which 404s because GitHub Pages is not enabled yet. Phase D fixes that
by publishing a small static documentation site built from Markdown
files in a `docs/` folder.

### 6.1 What to build

A 4-page Jekyll-rendered site under `docs/`, themed with the built-in
[Cayman](https://github.com/pages-themes/cayman) or
[Minimal](https://github.com/pages-themes/minimal) theme:

| Page | File | Purpose | Source material |
|---|---|---|---|
| Landing | `docs/index.md` | Project intro, screenshots of HF Space, 1-paragraph overview of each of the 6 tabs, links to repo and HF Space | Pull from current README §"Workflow" + screenshots from `data/media/` |
| Methodology | `docs/methodology.md` | Academic walkthrough of JM separability: Bhattacharyya equation, regularisation, the 4-bucket scheme. Embed `seperability_matrix_*.png` and `cumulative_gain.png` | Source material in `src/core/jm.py` docstrings + the thesis-internal Bhattacharyya derivation. Reusable for thesis Chapter 6.2 |
| Presets | `docs/presets.md` | Reference table of all 8 built-in sensors with bands, wavelengths, and citation links | Pull from `src/core/presets.py` — the dataclass instances are already structured for this |
| Troubleshooting | `docs/troubleshooting.md` | Common errors, CSV format expectations, missing classes, large-file behaviour, HF Space cold-start latency | Mostly new content; some can be lifted from validation messages in `src/core/validation.py` |

Plus three **infrastructure files**:

| File | Purpose |
|---|---|
| `docs/_config.yml` | Jekyll config: `theme: jekyll-theme-cayman`, site title, navigation links |
| `docs/assets/img/` | A copy or symlink of `data/media/` for use in the Markdown pages (Pages can serve from `data/media/` directly too — pick whichever feels cleaner) |
| `.github/workflows/pages.yml` *(optional)* | Custom Pages build. **Not needed if using a built-in theme** — GitHub auto-builds from the `docs/` folder when Pages is enabled with "Branch: main, Folder: /docs" |

### 6.2 Concrete next steps

These are in order. Each step is small enough to verify before moving
to the next.

1. **Create `docs/_config.yml`** with the Cayman theme + title + nav.
2. **Create `docs/index.md`** — landing page; lift the high-quality
   parts of the current README's intro and Workflow sections.
3. **Create `docs/methodology.md`** — the longest page; this is the
   one that doubles as thesis-section draft material.
4. **Create `docs/presets.md`** — generate from `src/core/presets.py`.
   Keep it auto-generation-friendly so future preset additions only
   need a regen step.
5. **Create `docs/troubleshooting.md`** — short, FAQ-style.
6. **Decide on figure path**: easiest is to add `docs/assets/img/` and
   commit copies of the 4–6 figures actually referenced in the docs
   (full set stays in `data/media/`).
7. **Push**, then in repo settings: **Settings → Pages → Source: Deploy
   from a branch → Branch: main, Folder: /docs → Save**.
8. **Wait 1–2 min**, then verify https://nickkoro21.github.io/jm-separability-toolbox/
   loads.
9. **Smoke test all internal links** — the README in §6.1 mentions
   the Pages URL and the cross-page nav must work.

### 6.3 Things to watch out for

- **Theme defaults to `_layouts/default.html` from the gem.** You
  don't need to copy it locally unless you want to override.
- **Front matter on every page**: even a minimal one helps Jekyll —
  `---\nlayout: default\ntitle: Methodology\n---` at the top of each
  `.md`.
- **Image paths**: Pages serves from `/jm-separability-toolbox/` so
  use root-relative paths like `/jm-separability-toolbox/docs/assets/img/foo.png`,
  or — easier — relative paths from the page itself like `assets/img/foo.png`.
- **The `data/media/` strip in `sync-hf.yml` does not affect Pages.**
  Pages reads from the GitHub `main` branch where the figures still
  live. Don't move them out of `data/media/` thinking it'll help the
  HF deploy — that would actually break the Pages build.
- **No need to edit the workflow.** Pages is a separate
  GitHub-managed pipeline; the auto-sync workflow stays as-is.

### 6.4 Estimated effort

~30–45 min for a clean first cut: 5 min infra + 20 min content +
10 min iteration + 5 min build/verify.

---

## 7. Working preferences (Nikolaos)

Standing rules for any future session, copied from earlier handoffs:

- **Strategy before code.** Discuss approach and flow before diving
  into implementation. Avoid premature `.py` dumps.
- **Step-by-step confirmation.** One module at a time, verify, then
  next. Split scripts over monoliths.
- **Greek for discussion, English for code.** Explanations in Greek,
  variable names / comments / docstrings in English.
- **Complete files when delivering code**, not snippets. Outputs go
  to logical subdirs.
- **Self-contained HTML artefacts.** Dark academic style, sticky nav,
  inline SVG. No external image deps.
- **Proactive fixes welcomed**: "αν μου διαφεύγει κάτι μη διστάσεις
  να διορθώσεις".
- **Paths**: hardcoded for now; argparse later if needed.

### Tooling notes

| Tool | Reliability for this project |
|---|---|
| `Filesystem` MCP (`read_text_file`, `edit_file`) | ✅ Reliable for D: drive reads/edits |
| `Windows-MCP:FileSystem` mode `read`/`list` | ✅ Reliable |
| `Windows-MCP:FileSystem` mode `write` | ⚠️ Times out on writes >2 KB. Use `Filesystem:edit_file` or full-file `Windows-MCP:FileSystem write` only for small files |
| `Windows-MCP:PowerShell` | ✅ Reliable for navigation, git, file globs. Status Code 1 on `git push` is a stderr quirk, not a real failure |
| `present_files` tool | ✅ Best for delivering large HTML/output files to Nikolaos |

---

## 8. Quick command reference

### Local dev

```powershell
cd D:\thesis\media_cowork\High\03_Spectral_Signatures\jm_separability_toolbox
.\.venv\Scripts\Activate.ps1
python app.py
# Or:
.\run.ps1
```

### Manual HF push (orphan strategy, when CI is unavailable)

```powershell
$token = [Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
git remote add hf "https://NickKoro21:$token@huggingface.co/spaces/NickKoro21/jm-separability-toolbox" 2>$null
git checkout --orphan hf-deploy-tmp
git rm -rf --cached data/media | Out-Null
Remove-Item -Recurse -Force "data\media" -ErrorAction SilentlyContinue
git commit -m "Deploy to HF Space"
git push hf hf-deploy-tmp:main --force
git checkout main
git branch -D hf-deploy-tmp
```

### Trigger CI manually

GitHub → Actions → "Sync to Hugging Face Space" → Run workflow → main.

### Verify both remotes match

```powershell
git rev-parse HEAD                                        # local main
git rev-parse origin/main                                 # GitHub
git ls-remote "https://NickKoro21:$token@huggingface.co/spaces/NickKoro21/jm-separability-toolbox"  # HF
```

---

## 9. Glossary

- **JM** — Jeffries–Matusita distance. Bounded [0, √2 ≈ 1.414] in the
  original definition; this toolbox uses the [0, 2] version derived
  from `2(1 − e^(-B))` where `B` is Bhattacharyya distance.
- **4-bucket scheme** — Poor (<1.0), Moderate ([1.0, 1.5)), Good
  ([1.5, 1.9)), Excellent (≥1.9). Cutoffs are conventions, not
  absolutes; documented in `methodology.md` (to be written).
- **Subset** — A named selection of bands. Built-ins: RGB (3 bands),
  5MS (5 bands incl. RedEdge+NIR), 7D (5MS + nDSM + Thermal), All.
- **Spectral signature** — Mean reflectance per class across bands,
  ordered by wavelength.
- **MicaSense Altum-PT** — The thesis sensor. 7-band: Blue, Green,
  Red, RedEdge, NIR, plus nDSM_m (height in metres) and Thermal_C
  (LWIR temperature in °C).
- **Companion repo** — `PostProcessing-Toolbox`, the ArcGIS Pro
  vectorisation toolbox; same author. Not affected by anything in
  this repo.

---

## 10. Closing checklist for the next session

When the next Claude session opens, it should be able to answer all
of the following from this document alone:

- [ ] Where does the footer HTML live? → `app.py`, lines ~55–80.
- [ ] Where does `_TAB5_POPULATE_COUNT` live and what value should it
      have? → `app.py`, currently 65.
- [ ] How do I push to HF Space without going through GitHub?
      → §8, "Manual HF push" snippet.
- [ ] Why was Gradio pinned to 5.50? → §4.3 #4 and #5.
- [ ] Why is `data/media/` only on GitHub and not HF? → §4.1.
- [ ] What's blocking https://nickkoro21.github.io/jm-separability-toolbox/ ?
      → Phase D is not done; see §6.
- [ ] What's the next concrete deliverable? → Phase D (§6.2 list).

If any of these can't be answered from this file, the file is
incomplete and should be updated before further work proceeds.

---

*End of HANDOFF. Generated 2026-04-28 after Phases A/B/C completion
and the Tab 5 / Tab 6 UX cleanup pass. Companion documents:
`README.md` (user-facing), `Project_Instructions_v3.md` (thesis-wide
context).*
