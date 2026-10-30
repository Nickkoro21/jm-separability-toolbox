"""
Tab 4 — Class filter + Band subset configuration.

Sequential workflow: STEP 4 of 6.

Locked until Tab 3 (CSV upload + validation) is confirmed. On unlock, the
tab auto-populates from upstream state:

    * **Class checkbox**: every class from ``state['validation_report']
      ['class_counts']`` is offered with its sample count, sorted by count
      descending. Display labels combine class name (from
      ``schema['class_label_mapping']``) with their sample count, e.g.
      "Tree (1,234 samples)". The underlying value is the raw class id.
    * **Subset Dataframe**: 2 columns — *Subset name*, *Bands (comma-sep)*.
      Heuristic defaults aligned with the thesis terminology:
        - ``RGB``  ← Blue + Green + Red
        - ``5MS``  ← RGB + RedEdge + NIR
        - ``7D``   ← 5MS + nDSM + Thermal
      Plus an ``All`` row when ≥ 4 bands are available, or as a fallback
      when none of the named heuristics matches (e.g. Sentinel-2).

Two analysis modes (radio):
    * **Manual subsets** (default) — the user names band subsets in the
      Dataframe described above; each becomes one JM matrix in Step 5.
    * **Exhaustive search** — the user picks candidate bands (≤ MAX_BANDS)
      and a minimum subset size k; on confirm, ``src.core.exhaustive``
      evaluates *every* combination of ≥ k bands, stores the full ranking
      in ``state['exhaustive_result']`` and auto-fills ``state['subsets']``
      with the best combination at each k (``Best-3``, ``Best-4``, …) so
      every downstream visualisation works unchanged.

Validation rules (live, hard-error) — manual mode:
    1. ≥ 2 classes selected
    2. ≥ 1 subset row (after empty-row stripping)
    3. Subset names: non-empty, unique
    4. Each subset: ≥ 2 distinct bands
    5. Each token in a subset's band list must match an available band
       (case- / whitespace- / dash- / underscore-insensitive). Unknown
       tokens are reported with a sample of the available band names.

Exhaustive mode:
    1. ≥ 2 classes selected
    2. min_k ≤ number of candidate bands ≤ MAX_BANDS (20)

Public API
----------
build(state) -> dict
    Render the tab and return refs needed by app.py for chain wiring.

populate_state_updates(state) -> tuple
    Compute updates that fill the tab from Tab 3 confirm state.

clear_state_updates() -> tuple
    Updates that reset every Tab 4 field to its default.

Constants exposed for re-use:
    ``DEFAULT_BANDS_HINT``, ``DEFAULT_VALIDATION``, ``DEFAULT_STATUS``,
    ``MODE_MANUAL``, ``MODE_EXHAUSTIVE``, ``POPULATE_COUNT``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import gradio as gr
import pandas as pd

from src.core import (
    DEFAULT_MIN_K,
    MAX_BANDS,
    count_combinations,
    exhaustive_search,
)
from src.ui.tab5_results import MAX_SUBSETS


# ─── Colour tokens (aligned with Tabs 2 + 3) ──────────────────────────────────
_C_OK    = "#16a34a"   # green-600
_C_WARN  = "#d97706"   # amber-600
_C_ERR   = "#ef4444"   # red-500
_C_INFO  = "#8896b3"   # slate
_C_BLUE  = "#60a5fa"   # blue-400


# ─── Public defaults (re-used by app.py cascade-reset) ────────────────────────
DEFAULT_BANDS_HINT: str = (
    f"<span style='color:{_C_INFO};'>"
    f"ℹ️ Available bands will appear here once Step 3 is confirmed.</span>"
)

DEFAULT_VALIDATION: str = (
    f"<span style='color:{_C_INFO};'>"
    f"ℹ️ Validation feedback appears here as you edit.</span>"
)

DEFAULT_STATUS: str = ""

#: Radio labels for the analysis mode.
MODE_MANUAL: str = "Manual subsets"
MODE_EXHAUSTIVE: str = "Exhaustive search (all band combinations)"
MODE_CHOICES: tuple[str, str] = (MODE_MANUAL, MODE_EXHAUSTIVE)

#: Rough per-combination cost used only for the on-screen time estimate.
_SECONDS_PER_COMBO: float = 60e-6

#: Number of widgets filled by :func:`populate_state_updates` /
#: :func:`clear_state_updates` (app.py wires exactly this many outputs).
POPULATE_COUNT: int = 12


# ─── Normalisation (mirrors detection._norm — kept local to avoid leaking
#     a private symbol from a sibling package) ──────────────────────────────
_NORM_RE = re.compile(r"[\s_\-]+")

def _norm(s: Any) -> str:
    """Lower-case, strip, collapse whitespace / underscores / dashes."""
    return _NORM_RE.sub("", str(s).strip().lower())


# ─── Default subset heuristics (RGB / 5MS / 7D / All) ─────────────────────────

def _suggest_default_subsets(available_bands: list[str]) -> list[list[str]]:
    """Build heuristic default subset rows aligned with thesis terminology.

    Matching is case- and separator-insensitive (so ``RedEdge``, ``red_edge``,
    ``red-edge`` all collapse to the same key).
    """
    norm_to_orig: dict[str, str] = {}
    for b in available_bands:
        norm_to_orig.setdefault(_norm(b), b)

    def _prefix(key: str) -> str | None:
        """Exact match first, then the first band whose name starts with
        ``key`` — so ``nDSM_m`` / ``Thermal_C`` (unit suffixes) match."""
        if key in norm_to_orig:
            return norm_to_orig[key]
        for nb, orig in norm_to_orig.items():
            if nb.startswith(key):
                return orig
        return None

    blue    = norm_to_orig.get("blue")
    green   = norm_to_orig.get("green")
    red     = norm_to_orig.get("red")
    re_band = norm_to_orig.get("rededge")
    nir     = norm_to_orig.get("nir")
    ndsm    = _prefix("ndsm")
    thermal = _prefix("thermal")

    rows: list[list[str]] = []

    # RGB
    if blue and green and red:
        rows.append(["RGB", ", ".join([blue, green, red])])

    # 5MS = RGB + RedEdge + NIR
    if blue and green and red and re_band and nir:
        rows.append(
            ["5MS", ", ".join([blue, green, red, re_band, nir])]
        )

    # 7D = 5MS + nDSM + Thermal
    if blue and green and red and re_band and nir and ndsm and thermal:
        rows.append(
            ["7D", ", ".join(
                [blue, green, red, re_band, nir, ndsm, thermal]
            )]
        )

    # "All" row when ≥ 4 bands; also as fallback if nothing else matched
    if len(available_bands) >= 4:
        rows.append(["All", ", ".join(available_bands)])
    elif not rows and len(available_bands) >= 2:
        rows.append(["All", ", ".join(available_bands)])

    return rows


# ─── Exhaustive-mode helpers ──────────────────────────────────────────────────

def _coerce_min_k(min_k: Any) -> int:
    """Slider value → int, falling back to the default on junk input."""
    try:
        return int(min_k)
    except (TypeError, ValueError):
        return DEFAULT_MIN_K


def _check_exhaustive_inputs(n_bands: int, min_k: int) -> str | None:
    """Single source of truth for the exhaustive-mode input guards.

    Returns an error message (without the ❌ prefix) or ``None`` when the
    inputs are acceptable. Used by both the live estimate line and the
    validation line so the two can never disagree.
    """
    if n_bands < 2:
        return "Select at least 2 candidate bands."
    if n_bands > MAX_BANDS:
        return (
            f"{n_bands} candidate bands exceed the limit of {MAX_BANDS}. "
            f"Deselect {n_bands - MAX_BANDS} band(s)."
        )
    if min_k > n_bands:
        return (
            f"Minimum subset size k = {min_k} exceeds the {n_bands} "
            f"candidate bands."
        )
    return None


def _format_combo_estimate(n_bands: int, min_k: int) -> str:
    """Info line: number of combinations + rough runtime for the search."""
    err = _check_exhaustive_inputs(n_bands, min_k)
    if err is not None:
        color = _C_INFO if n_bands < 2 else _C_ERR
        icon = "ℹ️" if n_bands < 2 else "❌"
        return f"<span style='color:{color};'>{icon} {err}</span>"
    n_combos = count_combinations(n_bands, min_k)
    est = n_combos * _SECONDS_PER_COMBO
    if est < 1:
        est_txt = "under a second"
    elif est < 90:
        est_txt = f"≈ {est:.0f} s"
    else:
        est_txt = f"≈ {est / 60:.1f} min"
    color = _C_WARN if n_combos > 100_000 else _C_BLUE
    return (
        f"<span style='color:{color};'>🔎 <b>{n_combos:,}</b> combinations "
        f"of {min_k}…{n_bands} bands will be evaluated "
        f"({est_txt}).</span>"
    )


def _validate_exhaustive(
    class_selection: list | None,
    candidate_bands: list[str] | None,
    min_k: Any,
) -> tuple[bool, str]:
    """Validate the exhaustive-mode inputs. Returns ``(ok, message)``."""
    n_classes = len(class_selection or [])
    if n_classes < 2:
        return False, f"❌ Need at least 2 classes selected (got {n_classes})."
    bands = list(candidate_bands or [])
    k = _coerce_min_k(min_k)
    err = _check_exhaustive_inputs(len(bands), k)
    if err is not None:
        return False, f"❌ {err}"
    n_combos = count_combinations(len(bands), k)
    return (
        True,
        f"✓ Exhaustive search · {n_classes} classes · {len(bands)} bands · "
        f"{n_combos:,} combinations (k ≥ {k})",
    )


# ─── Live validation ──────────────────────────────────────────────────────────

def _normalise_subset_rows(rows: Any) -> list[list]:
    """Coerce gr.Dataframe input to a stripped list of non-empty rows."""
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        rows = rows.values.tolist()
    if not isinstance(rows, (list, tuple)):
        return []

    cleaned: list[list] = []
    for row in rows:
        if row is None:
            continue
        non_empty = [
            c for c in row
            if c not in (None, "")
            and not (isinstance(c, float) and pd.isna(c))
        ]
        if not non_empty:
            continue
        cleaned.append(list(row))
    return cleaned


def _validate(
    class_selection: list | None,
    subset_rows: Any,
    available_bands: list[str],
) -> tuple[bool, str, dict[str, list[str]] | None]:
    """Validate the configuration. Returns ``(ok, message, parsed_subsets)``.

    ``parsed_subsets`` is a dict mapping subset name → list of *normalised*
    band names (matched against ``available_bands`` via case-insensitive
    lookup) on success, ``None`` on failure.
    """
    n_classes = len(class_selection or [])
    if n_classes < 2:
        return (
            False,
            f"❌ Need at least 2 classes selected (got {n_classes}).",
            None,
        )

    cleaned = _normalise_subset_rows(subset_rows)
    if not cleaned:
        return False, "❌ Need at least 1 subset row.", None

    if not available_bands:
        return (
            False,
            "❌ No bands available — re-run Step 3 with a valid CSV.",
            None,
        )

    norm_to_orig: dict[str, str] = {}
    for b in available_bands:
        norm_to_orig.setdefault(_norm(b), b)

    subsets: dict[str, list[str]] = {}
    seen_names: set[str] = set()

    for i, row in enumerate(cleaned, start=1):
        if len(row) < 2:
            return (
                False,
                f"❌ Subset {i}: missing column(s) "
                f"(need *Subset name*, *Bands*).",
                None,
            )

        name = str(row[0] or "").strip()
        if not name:
            return False, f"❌ Subset {i}: name is empty.", None
        if name in seen_names:
            return False, f"❌ Duplicate subset name: **`{name}`**.", None
        seen_names.add(name)

        bands_str = str(row[1] or "").strip()
        if not bands_str:
            return (
                False,
                f"❌ Subset **`{name}`**: bands cell is empty.",
                None,
            )

        raw_tokens = [t.strip() for t in bands_str.split(",") if t.strip()]
        if not raw_tokens:
            return (
                False,
                f"❌ Subset **`{name}`**: no bands listed.",
                None,
            )

        matched: list[str] = []
        unknown: list[str] = []
        seen_bands: set[str] = set()
        for tok in raw_tokens:
            hit = norm_to_orig.get(_norm(tok))
            if hit is None:
                unknown.append(tok)
                continue
            if hit in seen_bands:
                # Silent dedupe — common when user copy-pastes
                continue
            matched.append(hit)
            seen_bands.add(hit)

        if unknown:
            avail_preview = ", ".join(available_bands[:10])
            if len(available_bands) > 10:
                avail_preview += ", …"
            return (
                False,
                f"❌ Subset **`{name}`**: unknown band(s) "
                f"`{', '.join(unknown)}`. "
                f"Available: <code>[{avail_preview}]</code>",
                None,
            )

        if len(matched) < 2:
            return (
                False,
                f"❌ Subset **`{name}`**: need ≥ 2 distinct bands "
                f"(got {len(matched)}).",
                None,
            )

        subsets[name] = matched

    n_subsets = len(subsets)
    total_bands = sum(len(v) for v in subsets.values())
    return (
        True,
        f"✓ {n_subsets} subset{'s' if n_subsets != 1 else ''} · "
        f"{n_classes} classes · {total_bands} bands total",
        subsets,
    )


def _format_validation_md(ok: bool, msg: str) -> str:
    """Coloured inline span for the live validation indicator."""
    color = _C_OK if ok else _C_ERR
    return f"<span style='color:{color};'>{msg}</span>"


def _format_bands_hint(available_bands: list[str]) -> str:
    """Info card listing every available band name."""
    if not available_bands:
        return (
            f"<span style='color:{_C_ERR};'>"
            f"❌ No available bands. Re-run Step 3 with a valid CSV.</span>"
        )
    return (
        f"<div style='padding:10px 14px;"
        f" background:rgba(96,165,250,0.06);"
        f" border:1px solid rgba(96,165,250,0.3);"
        f" border-left:4px solid {_C_BLUE};"
        f" border-radius:6px;'>"
        f"<b style='color:{_C_BLUE};'>💡 Available bands</b> "
        f"<span style='color:{_C_INFO}; font-size:0.9em;'>"
        f"(case-insensitive, comma-separated):</span><br/>"
        f"<code style='font-size:0.95em;'>"
        + ", ".join(available_bands)
        + "</code></div>"
    )


# ─── Helpers for upstream state extraction ────────────────────────────────────

def _available_bands_from_state(state: dict) -> list[str]:
    """Spectral + non-spectral, in CSV order, from Tab 3's detected schema."""
    schema = state.get("detected_schema") or {}
    spectral     = list(schema.get("band_cols", []) or [])
    non_spectral = list(schema.get("non_spectral_cols", []) or [])
    # Preserve order; dedupe just in case
    seen: set[str] = set()
    out: list[str] = []
    for b in spectral + non_spectral:
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def _class_choices_from_state(state: dict) -> list[tuple[str, Any]]:
    """Build ``[(label, raw_id), …]`` for the class checkbox.

    Order: matches ``class_counts`` (already sorted descending by count in
    ``validation.run_full_validation``).
    """
    report = state.get("validation_report") or {}
    schema = state.get("detected_schema") or {}
    class_counts: dict = report.get("class_counts", {}) or {}
    label_map:    dict = schema.get("class_label_mapping", {}) or {}

    choices: list[tuple[str, Any]] = []
    for raw_id, count in class_counts.items():
        # The mapping key may be the raw id, or its string form (JSON
        # round-trip flattens int keys to str). Try both.
        name = (
            label_map.get(raw_id)
            or label_map.get(str(raw_id))
            or str(raw_id)
        )
        try:
            count_str = f"{int(count):,}"
        except (TypeError, ValueError):
            count_str = str(count)
        label = f"{name} ({count_str} samples)"
        choices.append((label, raw_id))
    return choices


# ─── Event handlers ───────────────────────────────────────────────────────────

def _on_live_validate(
    mode: str | None,
    class_selection: list | None,
    subset_rows: Any,
    candidate_bands: list[str] | None,
    min_k: Any,
    state: dict,
):
    """Live validation feedback while editing — does NOT mutate shared state.

    Outputs (2): validation_status, confirm_btn
    """
    if mode == MODE_EXHAUSTIVE:
        ok, msg = _validate_exhaustive(class_selection, candidate_bands, min_k)
    else:
        available = _available_bands_from_state(state)
        ok, msg, _ = _validate(class_selection, subset_rows, available)
    return (
        gr.update(value=_format_validation_md(ok, msg)),
        gr.update(interactive=ok),
    )


def _on_mode_change(
    mode: str | None,
    class_selection: list | None,
    subset_rows: Any,
    candidate_bands: list[str] | None,
    min_k: Any,
    state: dict,
):
    """Toggle manual / exhaustive widget groups and re-validate.

    Outputs (4): manual_group, exhaustive_group, validation_status,
    confirm_btn
    """
    is_exh = mode == MODE_EXHAUSTIVE
    status, btn = _on_live_validate(
        mode, class_selection, subset_rows, candidate_bands, min_k, state,
    )
    return (
        gr.update(visible=not is_exh),
        gr.update(visible=is_exh),
        status,
        btn,
    )


def _on_exhaustive_inputs_change(
    candidate_bands: list[str] | None,
    min_k: Any,
):
    """Refresh the combination-count estimate. Outputs (1): exh_estimate."""
    k = _coerce_min_k(min_k)
    return gr.update(value=_format_combo_estimate(len(candidate_bands or []), k))


def _run_exhaustive(
    state: dict,
    class_selection: list,
    candidate_bands: list[str],
    min_k: int,
    progress: gr.Progress | None,
):
    """Run the exhaustive search on the class-filtered data.

    Returns ``(result, error_message)``; exactly one of them is ``None``.
    """
    df = state.get("df")
    schema = state.get("detected_schema") or {}
    class_col = schema.get("class_col")
    if df is None or not class_col or class_col not in df.columns:
        return None, "❌ Data or class column missing — re-run Step 3."
    sub = df[df[class_col].isin(class_selection)]
    label_map = schema.get("class_label_mapping") or None

    def _cb(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, desc=f"Exhaustive JM search — {msg}")

    try:
        result = exhaustive_search(
            sub, class_col, candidate_bands,
            min_k=min_k, class_labels=label_map, progress=_cb,
        )
    except ValueError as exc:
        return None, f"❌ Exhaustive search failed: {exc}"
    except MemoryError:
        return None, (
            "❌ Exhaustive search ran out of memory. Deselect some candidate "
            "bands or raise the minimum subset size k."
        )
    except Exception as exc:  # noqa: BLE001 — surface inline, never a toast
        logging.getLogger(__name__).exception("Exhaustive search crashed")
        return None, (
            f"❌ Exhaustive search failed: {type(exc).__name__}: {exc}. "
            f"Check that the class column has a single consistent type."
        )
    return result, None


def _on_confirm(
    mode: str | None,
    class_selection: list | None,
    subset_rows: Any,
    candidate_bands: list[str] | None,
    min_k: Any,
    state: dict,
    progress: gr.Progress = gr.Progress(),
):
    """Persist confirmed config to shared state, cascade-reset Tab 5.

    In exhaustive mode this also runs the search (with a progress bar) and
    derives the ``Best-k`` subsets from its result.

    Outputs (2): state, status
    """
    is_exh = mode == MODE_EXHAUSTIVE
    exhaustive_result = None
    parsed_subsets: dict[str, list[str]] | None = None

    if is_exh:
        ok, msg = _validate_exhaustive(class_selection, candidate_bands, min_k)
        if not ok:
            return state, gr.update(
                value=f"<span style='color:{_C_ERR};'>{msg}</span>"
            )
        exhaustive_result, err = _run_exhaustive(
            state, list(class_selection or []), list(candidate_bands or []),
            _coerce_min_k(min_k), progress,
        )
        if err is not None or exhaustive_result is None:
            return state, gr.update(
                value=f"<span style='color:{_C_ERR};'>{err}</span>"
            )
        parsed_subsets = exhaustive_result.subsets_for_tab5(
            max_subsets=MAX_SUBSETS,
        )
    else:
        available = _available_bands_from_state(state)
        ok, msg, parsed_subsets = _validate(
            class_selection, subset_rows, available
        )
        if not ok or parsed_subsets is None:
            return state, gr.update(
                value=f"<span style='color:{_C_ERR};'>{msg}</span>"
            )

    # Build display labels for the confirmation summary
    schema    = state.get("detected_schema") or {}
    label_map = schema.get("class_label_mapping", {}) or {}
    display_classes: list[str] = []
    for cid in (class_selection or []):
        nm = label_map.get(cid) or label_map.get(str(cid)) or str(cid)
        display_classes.append(str(nm))

    new_state = dict(state)
    new_state["selected_classes"]    = display_classes
    new_state["selected_class_ids"]  = list(class_selection or [])
    new_state["subsets"]              = parsed_subsets
    new_state["analysis_mode"]        = "exhaustive" if is_exh else "manual"
    new_state["exhaustive_result"]    = exhaustive_result
    new_state["tab4_done"]            = True
    new_state["tab5_done"]            = False  # cascade

    n_classes = len(display_classes)
    n_subsets = len(parsed_subsets)
    if is_exh and exhaustive_result is not None:
        r = exhaustive_result
        dropped_note = ""
        if r.n_rows_dropped:
            dropped_note = (
                f"</span><br/><span style='color:{_C_WARN};'>⚠ "
                f"{r.n_rows_dropped:,} row(s) with a non-finite value in some "
                f"candidate band were excluded; Step 5 and the export use the "
                f"same {len(r.kept_index):,} rows so all numbers agree."
            )
        summary = (
            f"<span style='color:{_C_OK};'>"
            f"✅ <b>Exhaustive search complete.</b> "
            f"{r.n_combinations:,} combinations of {r.min_k}…{r.max_k} bands "
            f"evaluated in {r.elapsed_s:.1f} s · {n_classes} classes. "
            f"The best combination at each size was added as a subset "
            f"(<code>{'</code>, <code>'.join(parsed_subsets)}</code>). "
            f"Proceed to <b>Step 5</b> for the full ranking."
            f"{dropped_note}</span>"
        )
    else:
        summary = (
            f"<span style='color:{_C_OK};'>"
            f"✅ <b>Configuration confirmed.</b> "
            f"{n_subsets} subset{'s' if n_subsets != 1 else ''} · "
            f"{n_classes} classes. Proceed to <b>Step 5</b> for results.</span>"
        )
    return new_state, gr.update(value=summary)


# ─── Public helpers used by app.py chain handlers ─────────────────────────────

def populate_state_updates(state: dict) -> tuple:
    """Compute updates that fill Tab 4 with derived inputs from Tab 3.

    Used by ``_on_tab3_confirm_chain`` in app.py.

    Outputs (``POPULATE_COUNT`` = 12) in this exact order:
        mode_radio, manual_group, exhaustive_group,
        class_checkbox, detected_bands_hint, subset_df,
        exh_bands, exh_min_k, exh_estimate,
        validation_status, confirm_btn, status
    """
    available = _available_bands_from_state(state)
    choices   = _class_choices_from_state(state)
    default_classes = [c[1] for c in choices]
    default_subsets = _suggest_default_subsets(available)

    # Live-validate the defaults so user sees the green status immediately
    ok, msg, _ = _validate(default_classes, default_subsets, available)

    # Exhaustive defaults: every available band (capped at MAX_BANDS),
    # k from DEFAULT_MIN_K up to the number of candidates.
    exh_default = list(available[:MAX_BANDS])
    n_exh = len(exh_default)
    min_k_default = min(DEFAULT_MIN_K, max(2, n_exh))

    return (
        gr.update(value=MODE_MANUAL),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(
            choices=choices,
            value=default_classes,
            interactive=True,
        ),
        gr.update(value=_format_bands_hint(available)),
        gr.update(value=default_subsets),
        gr.update(choices=list(available), value=exh_default, interactive=True),
        gr.update(
            minimum=2, maximum=max(2, n_exh), value=min_k_default, step=1,
            interactive=True,
        ),
        gr.update(value=_format_combo_estimate(n_exh, min_k_default)),
        gr.update(value=_format_validation_md(ok, msg)),
        gr.update(interactive=ok),
        gr.update(value=DEFAULT_STATUS),
    )


def clear_state_updates() -> tuple:
    """Reset every Tab 4 widget back to its default (cascade-reset).

    Outputs (``POPULATE_COUNT`` = 12) in this exact order:
        mode_radio, manual_group, exhaustive_group,
        class_checkbox, detected_bands_hint, subset_df,
        exh_bands, exh_min_k, exh_estimate,
        validation_status, confirm_btn, status
    """
    return (
        gr.update(value=MODE_MANUAL),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(choices=[], value=[], interactive=False),
        gr.update(value=DEFAULT_BANDS_HINT),
        gr.update(value=[]),
        gr.update(choices=[], value=[], interactive=False),
        gr.update(minimum=2, maximum=MAX_BANDS, value=DEFAULT_MIN_K, step=1),
        gr.update(value=""),
        gr.update(value=DEFAULT_VALIDATION),
        gr.update(interactive=False),
        gr.update(value=DEFAULT_STATUS),
    )


# ─── Public builder ───────────────────────────────────────────────────────────

def build(state: gr.State) -> dict:
    """Render Tab 4 widgets and wire internal events.

    Parameters
    ----------
    state : gr.State
        Shared session state object, owned by ``app.py``.

    Returns
    -------
    dict
        Refs needed by ``app.py`` for chain wiring & cascade-reset.
    """
    # ── Lock screen ──
    with gr.Group(visible=True) as lock_msg:
        gr.Markdown(
            f"""
            <div style="text-align:center; padding:32px 24px;
                        background:rgba(245,158,11,0.08);
                        border:1px solid rgba(245,158,11,0.3);
                        border-radius:12px; color:{_C_WARN};">
              <h3 style="margin:0 0 8px 0;">🔒 Locked</h3>
              <p style="margin:0;">Please complete <b>Step 3 — Upload CSV</b>
              first.<br/>The class list and band subsets will populate
              automatically once a valid CSV is confirmed.</p>
            </div>
            """,
        )

    # ── Tab content ──
    with gr.Group(visible=False) as content:
        gr.Markdown(
            """
            ### <span class="step-badge" style="background:rgba(96,165,250,0.15); color:#60a5fa;">STEP 4</span> Configure analysis

            Pick which **classes** participate in the JM separability
            computation and define one or more **band subsets** to compare
            side-by-side (e.g. `RGB` vs `5MS` vs `7D`). Each subset will
            yield its own JM matrix in Step 5, so subset names should be
            short and descriptive.
            """,
        )

        gr.Markdown("#### 🎯 Class filter")
        class_checkbox = gr.CheckboxGroup(
            label="Classes to include in the analysis",
            info="At least 2 classes are required.",
            choices=[],
            value=[],
            interactive=False,
        )

        gr.Markdown("#### 🎚️ Band subsets")
        mode_radio = gr.Radio(
            choices=list(MODE_CHOICES),
            value=MODE_MANUAL,
            label="Analysis mode",
            info=(
                "Manual: you name the subsets. Exhaustive: every "
                f"combination of ≥ k bands is evaluated (max {MAX_BANDS} "
                "candidate bands) and ranked by JM separability."
            ),
            interactive=True,
        )

        with gr.Group(visible=True) as manual_group:
            detected_bands_hint = gr.Markdown(value=DEFAULT_BANDS_HINT)

            subset_df = gr.Dataframe(
                headers=["Subset name", "Bands (comma-separated)"],
                datatype=["str", "str"],
                row_count=(1, "dynamic"),
                col_count=(2, "fixed"),
                interactive=True,
                label=(
                    "📊 Subset definitions — one row per JM matrix to be "
                    "computed. Use the row controls to add / remove rows."
                ),
                wrap=True,
            )

        with gr.Group(visible=False) as exhaustive_group:
            exh_bands = gr.CheckboxGroup(
                label="Candidate bands",
                info=(
                    f"Every combination of these bands is evaluated. "
                    f"Limit: {MAX_BANDS} bands."
                ),
                choices=[],
                value=[],
                interactive=False,
            )
            exh_min_k = gr.Slider(
                minimum=2, maximum=MAX_BANDS, value=DEFAULT_MIN_K, step=1,
                label="Minimum subset size (k)",
                info="Combinations smaller than k bands are skipped.",
                interactive=True,
            )
            exh_estimate = gr.Markdown(value="")

        validation_status = gr.Markdown(value=DEFAULT_VALIDATION)

        with gr.Row():
            confirm_btn = gr.Button(
                "✓ Confirm configuration",
                variant="primary",
                interactive=False,
                size="lg",
            )

        status = gr.Markdown(value=DEFAULT_STATUS)

    # ── Internal event wiring ──
    # Live re-validation on either input change. Both inputs are sent to
    # the same handler; the shared state is read-only so no mutation risk.
    live_inputs = [
        mode_radio, class_checkbox, subset_df, exh_bands, exh_min_k, state,
    ]
    for widget in (class_checkbox, subset_df, exh_bands, exh_min_k):
        widget.change(
            fn=_on_live_validate,
            inputs=live_inputs,
            outputs=[validation_status, confirm_btn],
        )
    for widget in (exh_bands, exh_min_k):
        widget.change(
            fn=_on_exhaustive_inputs_change,
            inputs=[exh_bands, exh_min_k],
            outputs=[exh_estimate],
        )
    mode_radio.change(
        fn=_on_mode_change,
        inputs=live_inputs,
        outputs=[manual_group, exhaustive_group, validation_status, confirm_btn],
    )

    confirm_btn.click(
        fn=_on_confirm,
        inputs=[
            mode_radio, class_checkbox, subset_df, exh_bands, exh_min_k, state,
        ],
        outputs=[state, status],
    )

    return {
        "lock_msg":            lock_msg,
        "content":             content,
        "mode_radio":          mode_radio,
        "manual_group":        manual_group,
        "exhaustive_group":    exhaustive_group,
        "class_checkbox":      class_checkbox,
        "detected_bands_hint": detected_bands_hint,
        "subset_df":           subset_df,
        "exh_bands":           exh_bands,
        "exh_min_k":           exh_min_k,
        "exh_estimate":        exh_estimate,
        "validation_status":   validation_status,
        "confirm_btn":         confirm_btn,
        "status":              status,
    }


def populate_refs(refs: dict) -> list:
    """Widgets in the order of :func:`populate_state_updates` (12 items).

    ``app.py`` passes this list as the Tab 3 → Tab 4 chain outputs so the
    ordering is defined in exactly one place.
    """
    return [
        refs["mode_radio"],
        refs["manual_group"],
        refs["exhaustive_group"],
        refs["class_checkbox"],
        refs["detected_bands_hint"],
        refs["subset_df"],
        refs["exh_bands"],
        refs["exh_min_k"],
        refs["exh_estimate"],
        refs["validation_status"],
        refs["confirm_btn"],
        refs["status"],
    ]
