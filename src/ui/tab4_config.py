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

Validation rules (live, hard-error):
    1. ≥ 2 classes selected
    2. ≥ 1 subset row (after empty-row stripping)
    3. Subset names: non-empty, unique
    4. Each subset: ≥ 2 distinct bands
    5. Each token in a subset's band list must match an available band
       (case- / whitespace- / dash- / underscore-insensitive). Unknown
       tokens are reported with a sample of the available band names.

Public API
----------
build(state) -> dict
    Render the tab and return refs needed by app.py for chain wiring.

populate_state_updates(state) -> tuple
    Compute updates that fill the tab from Tab 3 confirm state.

clear_state_updates() -> tuple
    Updates that reset every Tab 4 field to its default.

Constants exposed for re-use:
    ``DEFAULT_BANDS_HINT``, ``DEFAULT_VALIDATION``, ``DEFAULT_STATUS``.
"""

from __future__ import annotations

import re
from typing import Any

import gradio as gr
import pandas as pd


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

    blue    = norm_to_orig.get("blue")
    green   = norm_to_orig.get("green")
    red     = norm_to_orig.get("red")
    re_band = norm_to_orig.get("rededge")
    nir     = norm_to_orig.get("nir")
    ndsm    = norm_to_orig.get("ndsm")
    thermal = norm_to_orig.get("thermal")

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
        + f"</code></div>"
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
    class_selection: list | None,
    subset_rows: Any,
    state: dict,
):
    """Live validation feedback while editing — does NOT mutate shared state.

    Outputs (2): validation_status, confirm_btn
    """
    available = _available_bands_from_state(state)
    ok, msg, _ = _validate(class_selection, subset_rows, available)
    return (
        gr.update(value=_format_validation_md(ok, msg)),
        gr.update(interactive=ok),
    )


def _on_confirm(
    class_selection: list | None,
    subset_rows: Any,
    state: dict,
):
    """Persist confirmed config to shared state, cascade-reset Tab 5.

    Outputs (2): state, status
    """
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
    new_state["tab4_done"]            = True
    new_state["tab5_done"]            = False  # cascade

    n_classes = len(display_classes)
    n_subsets = len(parsed_subsets)
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

    Outputs (6) in this exact order:
        class_checkbox, detected_bands_hint, subset_df,
        validation_status, confirm_btn, status
    """
    available = _available_bands_from_state(state)
    choices   = _class_choices_from_state(state)
    default_classes = [c[1] for c in choices]
    default_subsets = _suggest_default_subsets(available)

    # Live-validate the defaults so user sees the green status immediately
    ok, msg, _ = _validate(default_classes, default_subsets, available)

    return (
        gr.update(
            choices=choices,
            value=default_classes,
            interactive=True,
        ),
        gr.update(value=_format_bands_hint(available)),
        gr.update(value=default_subsets),
        gr.update(value=_format_validation_md(ok, msg)),
        gr.update(interactive=ok),
        gr.update(value=DEFAULT_STATUS),
    )


def clear_state_updates() -> tuple:
    """Reset every Tab 4 widget back to its default (cascade-reset).

    Outputs (6) in this exact order:
        class_checkbox, detected_bands_hint, subset_df,
        validation_status, confirm_btn, status
    """
    return (
        gr.update(choices=[], value=[], interactive=False),
        gr.update(value=DEFAULT_BANDS_HINT),
        gr.update(value=[]),
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
    class_checkbox.change(
        fn=_on_live_validate,
        inputs=[class_checkbox, subset_df, state],
        outputs=[validation_status, confirm_btn],
    )
    subset_df.change(
        fn=_on_live_validate,
        inputs=[class_checkbox, subset_df, state],
        outputs=[validation_status, confirm_btn],
    )

    confirm_btn.click(
        fn=_on_confirm,
        inputs=[class_checkbox, subset_df, state],
        outputs=[state, status],
    )

    return {
        "lock_msg":            lock_msg,
        "content":             content,
        "class_checkbox":      class_checkbox,
        "detected_bands_hint": detected_bands_hint,
        "subset_df":           subset_df,
        "validation_status":   validation_status,
        "confirm_btn":         confirm_btn,
        "status":              status,
    }
