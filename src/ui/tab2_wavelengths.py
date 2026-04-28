"""
Tab 2 — Wavelength Confirmation / Customization.

Sequential workflow: STEP 2 of 6.

Locked until Tab 1 (camera selection) is confirmed. On unlock, the dataframe
is auto-populated from the chosen preset's band layout (or with two empty
template rows if the user picked the ``Custom / Unknown sensor`` sentinel).

The user can:
    * accept the prefilled values and click Confirm directly, OR
    * edit any cell, OR
    * add/remove rows via the dataframe's native handles.

Validation runs on Confirm:
    * ≥ 2 bands required (JM is pairwise — fewer is meaningless)
    * Band names: non-empty, unique
    * Center wavelength: positive number (> 0)
    * FWHM: non-negative number (≥ 0)

Public API
----------
build(state) -> dict
    Render the tab and return refs for app-level event wiring.
    Keys: ``lock_msg``, ``content``, ``df``, ``confirm_btn``, ``status``.

bands_to_rows(state) -> list[list]
    Helper used by ``app.py`` to populate the dataframe when Tab 1 confirms.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd


# Number of empty rows shown when entering the Custom-sensor path
_CUSTOM_TEMPLATE_ROWS = 2


# ─── Public helpers ───────────────────────────────────────────────────────────

def bands_to_rows(state: dict) -> list[list[Any]]:
    """Convert ``state['preset_data']`` into ``gr.Dataframe`` rows.

    Returns
    -------
    list[list]
        Each inner list is ``[band_name, center_nm, fwhm_nm]``.
        For the custom-sensor path, returns ``_CUSTOM_TEMPLATE_ROWS``
        rows of empty placeholders so the user has a starting grid.
    """
    if state.get("is_custom") or state.get("preset_data") is None:
        return [["", None, None] for _ in range(_CUSTOM_TEMPLATE_ROWS)]

    bands = state["preset_data"].get("bands", [])
    return [[name, float(center), float(fwhm)] for name, center, fwhm in bands]


# ─── Validation ───────────────────────────────────────────────────────────────

def _validate_rows(
    rows: Any,
) -> tuple[bool, str, list[tuple[str, float, float]] | None]:
    """Validate the dataframe contents.

    Accepts either a ``pd.DataFrame`` (Gradio's native type) or a
    list-of-lists (when called programmatically).

    Returns
    -------
    (ok, msg, normalised_bands)
        ``normalised_bands`` is a list of ``(name, center, fwhm)`` tuples
        on success, ``None`` on failure.
    """
    # Normalise input shape
    if rows is None:
        return False, "❌ No data — please fill in the band table.", None
    if isinstance(rows, pd.DataFrame):
        rows = rows.values.tolist()
    if not isinstance(rows, (list, tuple)):
        return False, f"❌ Unexpected data type: {type(rows).__name__}.", None

    # Drop fully-empty trailing rows (common when user removes via UI)
    cleaned: list[list] = []
    for row in rows:
        if row is None:
            continue
        # Treat row as empty if all cells are empty/NaN
        non_empty_cells = [c for c in row if c not in (None, "") and not (isinstance(c, float) and pd.isna(c))]
        if not non_empty_cells:
            continue
        cleaned.append(list(row))

    if len(cleaned) < 2:
        return False, (
            "❌ Need **at least 2 bands** for separability analysis "
            f"(got {len(cleaned)})."
        ), None

    bands: list[tuple[str, float, float]] = []
    seen_names: set[str] = set()

    for i, row in enumerate(cleaned, start=1):
        if len(row) < 3:
            return False, f"❌ Row {i}: missing column(s) (need Band, λ, FWHM).", None

        # Band name
        name_raw = row[0]
        name = str(name_raw).strip() if name_raw is not None else ""
        if not name:
            return False, f"❌ Row {i}: band name cannot be empty.", None
        if name in seen_names:
            return False, f"❌ Duplicate band name: **`{name}`**.", None
        seen_names.add(name)

        # Center wavelength
        try:
            center = float(row[1])
        except (TypeError, ValueError):
            return False, (
                f"❌ Row {i} (`{name}`): center wavelength must be a number "
                f"(got `{row[1]!r}`)."
            ), None
        if not (center > 0):
            return False, (
                f"❌ Row {i} (`{name}`): center wavelength must be a positive "
                f"number (got `{center}`)."
            ), None

        # FWHM
        try:
            fwhm = float(row[2])
        except (TypeError, ValueError):
            return False, (
                f"❌ Row {i} (`{name}`): FWHM must be a number "
                f"(got `{row[2]!r}`)."
            ), None
        if fwhm < 0:
            return False, (
                f"❌ Row {i} (`{name}`): FWHM must be ≥ 0 (got `{fwhm}`)."
            ), None

        bands.append((name, center, fwhm))

    return True, f"✅ {len(bands)} bands valid.", bands


# ─── Event handlers ───────────────────────────────────────────────────────────

def _on_validate(rows: Any) -> tuple[Any, Any]:
    """Live validation feedback while editing — does NOT mutate state."""
    ok, msg, _ = _validate_rows(rows)
    if ok:
        feedback = f"<span style='color:#16a34a;'>{msg}</span>"
    else:
        feedback = f"<span style='color:#ef4444;'>{msg}</span>"
    return (
        gr.update(value=feedback),
        gr.update(interactive=ok),  # confirm button enabled only when valid
    )


def _on_confirm(rows: Any, state: dict) -> tuple[dict, Any]:
    """Persist confirmed wavelengths into state, cascade-reset downstream."""
    ok, msg, normalised = _validate_rows(rows)
    if not ok:
        return state, gr.update(value=f"<span style='color:#ef4444;'>{msg}</span>")

    new_state = dict(state)
    new_state["wavelengths"] = normalised
    new_state["tab2_done"] = True
    # Cascade-reset downstream
    for k in ("tab3_done", "tab4_done", "tab5_done"):
        new_state[k] = False

    summary = (
        f"<span style='color:#16a34a;'>✅ <b>{len(normalised)} bands confirmed.</b> "
        f"Proceed to <b>Step 3</b> to upload your CSV.</span>"
    )
    return new_state, gr.update(value=summary)


# ─── Public builder ───────────────────────────────────────────────────────────

def build(state: gr.State) -> dict:
    """Render Tab 2 widgets and wire internal events.

    Parameters
    ----------
    state : gr.State
        Shared session state object.

    Returns
    -------
    dict
        Keys:
            * ``lock_msg``  — gr.Group, visible when Tab 1 not yet confirmed
            * ``content``   — gr.Group, hidden until Tab 1 confirms
            * ``df``        — gr.Dataframe with the editable band table
            * ``confirm_btn`` — gr.Button (enabled when validation passes)
            * ``status``    — gr.Markdown with confirmation message
    """
    # ── Lock screen (visible until Tab 1 unlocks this tab) ──
    with gr.Group(visible=True) as lock_msg:
        gr.Markdown(
            """
            <div style="text-align:center; padding:32px 24px;
                        background:rgba(245,158,11,0.08);
                        border:1px solid rgba(245,158,11,0.3);
                        border-radius:12px; color:#d97706;">
              <h3 style="margin:0 0 8px 0;">🔒 Locked</h3>
              <p style="margin:0;">Please complete <b>Step 1 — Camera</b> first.<br/>
              The wavelength table will populate automatically once a camera is confirmed.</p>
            </div>
            """,
        )

    # ── Tab content (hidden until Tab 1 unlocks) ──
    with gr.Group(visible=False) as content:
        gr.Markdown(
            """
            ### <span class="step-badge" style="background:rgba(96,165,250,0.15); color:#60a5fa;">STEP 2</span> Confirm or customize band wavelengths

            Review the band layout loaded from your camera preset, edit any
            cell if needed, or add/remove rows. JM separability does not
            depend on wavelength values themselves — they are used only for
            plot labels and band ordering. Click **Confirm** when ready.
            """,
        )

        df = gr.Dataframe(
            headers=["Band", "λ Center (nm)", "FWHM (nm)"],
            datatype=["str", "number", "number"],
            row_count=(2, "dynamic"),
            col_count=(3, "fixed"),
            interactive=True,
            label="📊 Band table — click any cell to edit",
            wrap=True,
        )

        validation_status = gr.Markdown(
            value="<span style='color:#8896b3;'>ℹ️ Edit the table; "
                  "validation feedback appears here.</span>",
        )

        with gr.Row():
            confirm_btn = gr.Button(
                "✓ Confirm wavelengths",
                variant="primary",
                interactive=False,  # enabled by live validation
                size="lg",
            )

        status = gr.Markdown(value="")

    # ── Internal event wiring ──
    # Live validation on every cell edit
    df.change(
        fn=_on_validate,
        inputs=[df],
        outputs=[validation_status, confirm_btn],
    )

    confirm_btn.click(
        fn=_on_confirm,
        inputs=[df, state],
        outputs=[state, status],
    )

    return {
        "lock_msg":    lock_msg,
        "content":     content,
        "df":          df,
        "confirm_btn": confirm_btn,
        "status":      status,
    }
