"""
Tab 1 — Camera Selection.

Sequential workflow: STEP 1 of 6 (entry point — always unlocked).

The user picks one of eight built-in camera presets or selects the sentinel
``Custom / Unknown sensor`` to enter band layout manually in Step 2.

On confirmation:
    * The chosen preset is persisted in the shared session state.
    * Downstream ``tabN_done`` flags are cascade-reset to ``False``.
    * The parent ``app.py`` wires the confirm button to auto-switch to Tab 2.

Public API
----------
build(state) -> dict
    Render the tab and return refs for app-level event wiring.
    Returned keys:
        - "confirm_btn": gr.Button — caller wires .click() to switch tabs
        - "dropdown":    gr.Dropdown — exposed for completeness/testing
"""

from __future__ import annotations

import gradio as gr

from src.core import (
    CAMERA_PRESETS,
    SENTINEL_CUSTOM,
    format_preset_summary,
    get_preset,
    is_custom,
    list_preset_names,
)


# ─── UI helpers ───────────────────────────────────────────────────────────────

def _initial_info_card() -> str:
    """Default markdown shown before any preset is selected."""
    return (
        f"ℹ️ **Pick a camera from the dropdown above.**\n\n"
        f"{len(CAMERA_PRESETS)} verified built-in presets are available "
        f"(MicaSense, DJI, Parrot, Sentinel-2, Landsat 8/9), plus a "
        f"**`{SENTINEL_CUSTOM}`** option for sensors not in the list."
    )


def _render_preset_card(name: str | None) -> str:
    """Render the markdown info card for the currently selected preset."""
    if not name:
        return _initial_info_card()
    # format_preset_summary handles both real presets and the custom sentinel
    return format_preset_summary(name)


# ─── Event handlers ───────────────────────────────────────────────────────────

def _on_select(name: str | None):
    """Update the info card and toggle the confirm button when selection changes."""
    return (
        gr.update(value=_render_preset_card(name)),
        gr.update(interactive=bool(name)),
    )


def _on_confirm(name: str | None, state: dict):
    """Persist the choice into state, cascade-reset downstream, return message."""
    if not name:
        return (
            state,
            gr.update(value="❌ **Please select a camera first.**"),
        )

    is_cust = is_custom(name)
    preset = None if is_cust else get_preset(name)

    # Shallow copy to avoid mutating gr.State in place
    new_state = dict(state)
    new_state["preset_name"] = name
    new_state["preset_data"] = preset
    new_state["is_custom"] = is_cust
    new_state["tab1_done"] = True
    # Cascade-reset downstream — any tab1 change invalidates later steps
    for k in ("tab2_done", "tab3_done", "tab4_done", "tab5_done"):
        new_state[k] = False

    if is_cust:
        msg = (
            "✅ **Custom sensor confirmed.** "
            "Proceed to **Step 2** to enter band names and wavelengths manually."
        )
    else:
        n_bands = len(preset["bands"])
        non_spec = preset.get("non_spectral", [])
        ns_note = (
            f" + {len(non_spec)} non-spectral channel"
            f"{'s' if len(non_spec) != 1 else ''}"
            if non_spec
            else ""
        )
        msg = (
            f"✅ **`{name}` confirmed** — "
            f"{n_bands} spectral band{'s' if n_bands != 1 else ''}{ns_note}. "
            "Proceed to **Step 2** to confirm wavelengths."
        )

    return new_state, gr.update(value=msg)


# ─── Public builder ───────────────────────────────────────────────────────────

def build(state: gr.State) -> dict:
    """Render Tab 1 widgets and wire internal events.

    Parameters
    ----------
    state : gr.State
        Shared session state object (created in ``app.py`` once and passed
        to every tab builder).

    Returns
    -------
    dict
        ``"confirm_btn"`` and ``"dropdown"`` references — used by ``app.py``
        to chain tab-switching and (later) downstream tab unlocking.
    """
    with gr.Column():
        gr.Markdown(
            f"""
            ### <span class="step-badge" style="background:rgba(96,165,250,0.15); color:#60a5fa;">STEP 1</span> Select your camera

            Choose the multispectral / thermal sensor that produced your data,
            or pick **`{SENTINEL_CUSTOM}`** if your sensor is not in the list.
            JM separability does **not** depend on wavelength values themselves —
            they are used only for plot labels and band ordering.
            """,
        )

        preset_dropdown = gr.Dropdown(
            choices=list_preset_names(),
            value=None,
            label="📷 Camera preset",
            info=f"{len(CAMERA_PRESETS)} verified sensors + Custom",
            interactive=True,
            allow_custom_value=False,
        )

        info_card = gr.Markdown(value=_initial_info_card())

        with gr.Row():
            confirm_btn = gr.Button(
                "✓ Confirm camera",
                variant="primary",
                interactive=False,  # enabled once a preset is selected
                size="lg",
            )

        status = gr.Markdown(value="")

    # ── Internal event wiring ──
    preset_dropdown.change(
        fn=_on_select,
        inputs=[preset_dropdown],
        outputs=[info_card, confirm_btn],
    )

    confirm_btn.click(
        fn=_on_confirm,
        inputs=[preset_dropdown, state],
        outputs=[state, status],
    )

    return {
        "confirm_btn": confirm_btn,
        "dropdown": preset_dropdown,
    }
