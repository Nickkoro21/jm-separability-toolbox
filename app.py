"""
Spectral Separability Explorer — Gradio entry point.

Six-tab sequential workflow:
    1. Camera preset selection             ← implemented
    2. Wavelength confirmation             ← implemented
    3. CSV upload + validation             ← implemented
    4. Band subset + class selection       ← implemented
    5. Visualizations (spectral, boxplots, violins, JM matrices, ranked
       pairs, comparative bar + bucket distribution)
                                           ← implemented
    6. Export (CSV results + HTML guide + ZIP) ← implemented

A single ``gr.State`` dict is shared across tabs and holds all results from
previous steps (preset choice, wavelengths, dataframe, detected schema,
validation report, subsets, JM matrices, …).

Sequential unlock + cascade-reset model
---------------------------------------
Tabs 2..6 each render a ``lock_msg`` group and a hidden ``content`` group.
When the upstream tab confirms, the chain handler in ``app.py``:

    * unlocks the next tab (lock_msg → invisible, content → visible),
    * populates the next tab with derived inputs,
    * switches the active tab pointer,
    * **cascade-relocks** every tab further downstream so stale views never
      leak across re-confirm cycles, and
    * **cascade-clears** the immediately-next tab's fields (only the next
      one — deeper tabs stay locked so their stale state is invisible).

This guarantees: every time the user lands on Tab N via a confirm chain,
Tab N is in a fresh state — no leftover CSV / class column / validation
from a previous run.
"""

from __future__ import annotations

import gradio as gr
import socket

from src.ui import (
    tab1_camera,
    tab2_wavelengths,
    tab3_upload,
    tab4_config,
    tab5_results,
    tab6_export,
)


APP_TITLE = "🛰️ Spectral Separability Explorer"
APP_TAGLINE = (
    "Sensor-agnostic Jeffries–Matusita separability for any "
    "multispectral CSV"
)

FOOTER_HTML = """
<div style="text-align:center; padding:24px 16px; color:#8896b3; font-size:0.85rem; line-height:1.6;">
  <a href="https://www.linkedin.com/in/nick-koroniadis-328962226"
     target="_blank" rel="noopener"
     style="color:inherit; text-decoration:none; font-weight:700;">
    Nikolaos Koroniadis
  </a>
  &nbsp;·&nbsp; MSc Geography &amp; Applied Geoinformatics &nbsp;·&nbsp;
  <a href="https://geography.aegean.gr/geoinformatics/index.php" target="_blank" rel="noopener"
     style="color:#2563eb; text-decoration:none;">University of the Aegean</a>
  &nbsp;·&nbsp;
  <a href="https://rsgis.aegean.gr/" target="_blank" rel="noopener"
     style="color:#9333ea; text-decoration:none;">RSGIS Lab</a>
  &nbsp;·&nbsp; Supervised by Dr. Christos Vasilakos
  <br/>
  <a href="https://github.com/Nickkoro21/jm-separability-toolbox" target="_blank"
     rel="noopener" style="color:#16a34a; text-decoration:none;">GitHub</a>
  &nbsp;·&nbsp;
  <a href="https://nickkoro21.github.io/jm-separability-toolbox/" target="_blank"
     rel="noopener" style="color:#16a34a; text-decoration:none;">Documentation</a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/spaces/NickKoro21/spectral-3d-explorer" target="_blank"
     rel="noopener" style="color:#d97706; text-decoration:none;">Companion: 3D Explorer</a>
  &nbsp;·&nbsp; MIT License
</div>
"""


# ─── Theme — modern look matching Spectral 3D Explorer ─────────────────────
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)

CUSTOM_CSS = """
/* Hide default Gradio footer */
footer { display: none !important; }

/* Tighten tab spacing */
.tab-nav { padding: 6px 12px !important; }

/* Custom hero block */
.hero-banner {
    text-align: center;
    padding: 20px 16px 8px;
    background: linear-gradient(135deg, rgba(96,165,250,0.06), rgba(192,132,252,0.06));
    border-radius: 12px;
    margin-bottom: 8px;
}
.hero-banner h1 {
    font-size: 1.7rem !important;
    background: linear-gradient(135deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px !important;
}
.hero-banner p {
    color: #8896b3;
    font-size: 0.95rem;
    margin: 0;
}

/* Step badges */
.step-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    margin-right: 8px;
}
"""


# ─── Tab labels — pending vs completed ─────────────────────────────────────
# When a tab confirms, its label switches from numbered (pending) to ✅
# (completed). When an upstream tab is re-confirmed, downstream labels
# cascade-reset to pending.
_LABELS_PENDING: dict[int, str] = {
    1: "1️⃣ Camera",
    2: "2️⃣ Wavelengths",
    3: "3️⃣ Upload CSV",
    4: "4️⃣ Configure",
    5: "5️⃣ Results",
    6: "6️⃣ Export",
}

_LABELS_DONE: dict[int, str] = {
    1: "✅ Camera",
    2: "✅ Wavelengths",
    3: "✅ Upload CSV",
    4: "✅ Configure",
    5: "✅ Results",
}


def _label_updates(completed_through: int) -> tuple:
    """Return ``gr.update`` label changes for tabs 1-5.

    Tabs ≤ ``completed_through`` get the ``✅ <name>`` label.
    Tabs > ``completed_through`` are reset to the pending ``Nº <name>`` label.
    Tab 6 (Export) has no completed state and is not included.
    """
    return tuple(
        gr.update(
            label=_LABELS_DONE[i] if i <= completed_through
            else _LABELS_PENDING[i]
        )
        for i in range(1, 6)
    )


# ─── Initial session state ─────────────────────────────────────────────────
def _initial_state() -> dict:
    """Fresh session state shared by every tab."""
    return {
        # ── Tab 1 outputs ──
        "preset_name":  None,
        "preset_data":  None,
        "is_custom":    False,
        "tab1_done":    False,
        # ── Tab 2 outputs ──
        "wavelengths":  [],
        "tab2_done":    False,
        # ── Tab 3 outputs ──
        "csv_path":          None,
        "df":                None,
        "detected_schema":   None,
        "validation_report": None,
        "tab3_done":    False,
        # ── Tab 4 outputs ──
        "subsets":             {},
        "selected_classes":    [],
        "selected_class_ids":  [],
        "tab4_done":    False,
        # ── Tab 5 outputs ──
        "jm_results":   {},
        "tab5_done":    False,
    }


# ─── Tab placeholders (will be replaced as each tab gets implemented) ──────
def _placeholder_tab(step_num: int, title: str, description: str) -> None:
    """Render a temporary placeholder for tabs not yet implemented."""
    gr.Markdown(
        f"""
        ### <span class="step-badge" style="background:rgba(96,165,250,0.15); color:#60a5fa;">STEP {step_num}</span> {title}

        {description}

        > 🚧 **Under construction** — this tab will be implemented in
        > upcoming development steps. The skeleton you see now defines the
        > navigation structure; the logic and widgets will be wired in next.
        """
    )


# ─── Chain handlers — fire when an upstream tab confirms ───────────────────
#
# Output shape per chain (must match the .click(...) outputs= list 1:1):
#
#   _on_tab1_confirm_chain  → 17 outputs
#       tab2.df, tab2.lock_msg, tab2.content,
#       tab3.lock_msg, tab3.content,                       (relock Tab 3)
#       tab4.lock_msg, tab4.content,                       (relock Tab 4)
#       tab5.lock_msg, tab5.content,                       (relock Tab 5)
#       tab6.lock_msg, tab6.content,                       (relock Tab 6)
#       tabs, tab1_view..tab5_view
#
#   _on_tab2_confirm_chain  → 25 outputs
#       tabs, tab1_view..tab5_view,
#       tab3.lock_msg, tab3.content,                       (unlock Tab 3)
#       *tab3_upload.clear_state_updates()                 (11 fresh-state)
#       tab4.lock_msg, tab4.content,                       (relock Tab 4)
#       tab5.lock_msg, tab5.content,                       (relock Tab 5)
#       tab6.lock_msg, tab6.content,                       (relock Tab 6)
#
#   _on_tab3_confirm_chain  → 18 outputs
#       tabs, tab1_view..tab5_view,
#       tab4.lock_msg, tab4.content,                       (unlock Tab 4)
#       *tab4_config.populate_state_updates(state),        (6 populated)
#       tab5.lock_msg, tab5.content,                       (relock Tab 5)
#       tab6.lock_msg, tab6.content,                       (relock Tab 6)
#
#   _on_tab4_confirm_chain  → 75 outputs
#       tabs, tab1_view..tab5_view,
#       tab5.lock_msg, tab5.content,                       (unlock Tab 5)
#       *tab5_results.populate_state_updates(state)        (65 populated)
#       tab6.lock_msg, tab6.content,                       (relock Tab 6)
#
#   _on_tab5_confirm_chain  → 10 outputs
#       tabs, tab1_view..tab5_view,
#       tab6.lock_msg, tab6.content,                       (unlock Tab 6)
#       *tab6_export.clear_state_updates()                 (2 fresh-state)


_TAB5_POPULATE_COUNT = 65   # Must match len(tab5_results.populate_refs(refs))


def _on_tab1_confirm_chain(state: dict):
    """After Tab 1 confirm: prep Tab 2, relock Tabs 3+4+5+6, update labels."""
    if not state.get("tab1_done"):
        return tuple(gr.update() for _ in range(17))

    rows = tab2_wavelengths.bands_to_rows(state)
    return (
        gr.update(value=rows),       # tab2 dataframe
        gr.update(visible=False),    # tab2 lock_msg → unlock
        gr.update(visible=True),     # tab2 content  → unlock
        gr.update(visible=True),     # tab3 lock_msg → relock
        gr.update(visible=False),    # tab3 content  → relock
        gr.update(visible=True),     # tab4 lock_msg → relock
        gr.update(visible=False),    # tab4 content  → relock
        gr.update(visible=True),     # tab5 lock_msg → relock
        gr.update(visible=False),    # tab5 content  → relock
        gr.update(visible=True),     # tab6 lock_msg → relock
        gr.update(visible=False),    # tab6 content  → relock
        gr.update(selected=2),       # tabs container → Tab 2
        *_label_updates(completed_through=1),
    )


def _on_tab2_confirm_chain(state: dict):
    """After Tab 2 confirm: switch to Tab 3, unlock+clear Tab 3, relock Tabs 4+5+6."""
    if not state.get("tab2_done"):
        return tuple(gr.update() for _ in range(25))

    return (
        gr.update(selected=3),                        # tabs → Tab 3
        *_label_updates(completed_through=2),
        gr.update(visible=False),                     # tab3 lock_msg → unlock
        gr.update(visible=True),                      # tab3 content  → unlock
        *tab3_upload.clear_state_updates(),           # 11 cascade clears
        gr.update(visible=True),                      # tab4 lock_msg → relock
        gr.update(visible=False),                     # tab4 content  → relock
        gr.update(visible=True),                      # tab5 lock_msg → relock
        gr.update(visible=False),                     # tab5 content  → relock
        gr.update(visible=True),                      # tab6 lock_msg → relock
        gr.update(visible=False),                     # tab6 content  → relock
    )


def _on_tab3_confirm_chain(state: dict):
    """After Tab 3 confirm: switch to Tab 4, unlock + populate it, relock Tabs 5+6."""
    if not state.get("tab3_done"):
        return tuple(gr.update() for _ in range(18))

    return (
        gr.update(selected=4),                        # tabs → Tab 4
        *_label_updates(completed_through=3),
        gr.update(visible=False),                     # tab4 lock_msg → unlock
        gr.update(visible=True),                      # tab4 content  → unlock
        *tab4_config.populate_state_updates(state),   # 6 populated values
        gr.update(visible=True),                      # tab5 lock_msg → relock
        gr.update(visible=False),                     # tab5 content  → relock
        gr.update(visible=True),                      # tab6 lock_msg → relock
        gr.update(visible=False),                     # tab6 content  → relock
    )


def _on_tab4_confirm_chain(state: dict):
    """After Tab 4 confirm: switch to Tab 5, unlock + populate it, relock Tab 6.

    The populate step computes every visualisation in ``src.viz`` for the
    confirmed configuration and emits ``_TAB5_POPULATE_COUNT`` updates,
    one per Tab 5 widget (in the order defined by
    ``tab5_results.populate_refs``).
    """
    no_op_count = 6 + 2 + _TAB5_POPULATE_COUNT + 2
    if not state.get("tab4_done"):
        return tuple(gr.update() for _ in range(no_op_count))

    return (
        gr.update(selected=5),                          # tabs → Tab 5
        *_label_updates(completed_through=4),
        gr.update(visible=False),                       # tab5 lock_msg → unlock
        gr.update(visible=True),                        # tab5 content  → unlock
        *tab5_results.populate_state_updates(state),    # 65 populated values
        gr.update(visible=True),                        # tab6 lock_msg → relock
        gr.update(visible=False),                       # tab6 content  → relock
    )


def _on_tab5_confirm_chain(state: dict):
    """After Tab 5 confirm: switch to Tab 6, unlock it + reset its widgets."""
    if not state.get("tab5_done"):
        return tuple(gr.update() for _ in range(10))
    return (
        gr.update(selected=6),                       # tabs → Tab 6
        *_label_updates(completed_through=5),
        gr.update(visible=False),                    # tab6 lock_msg → unlock
        gr.update(visible=True),                     # tab6 content  → unlock
        *tab6_export.clear_state_updates(),          # 2 fresh-state values
    )


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(
        title="Spectral Separability Explorer",
        theme=THEME,
        css=CUSTOM_CSS,
        analytics_enabled=False,
    ) as demo:
        # ─── Hero ──────────────────────────────────────────────────────────
        gr.HTML(
            f"""
            <div class="hero-banner">
                <h1>{APP_TITLE}</h1>
                <p>{APP_TAGLINE}</p>
            </div>
            """
        )

        # ─── Shared session state ──────────────────────────────────────────
        session_state = gr.State(value=_initial_state())

        # ─── Sequential tabs ───────────────────────────────────────────────
        with gr.Tabs() as tabs:
            with gr.Tab(_LABELS_PENDING[1], id=1) as tab1_view:
                tab1_refs = tab1_camera.build(session_state)

            with gr.Tab(_LABELS_PENDING[2], id=2) as tab2_view:
                tab2_refs = tab2_wavelengths.build(session_state)

            with gr.Tab(_LABELS_PENDING[3], id=3) as tab3_view:
                tab3_refs = tab3_upload.build(session_state)

            with gr.Tab(_LABELS_PENDING[4], id=4) as tab4_view:
                tab4_refs = tab4_config.build(session_state)

            with gr.Tab(_LABELS_PENDING[5], id=5) as tab5_view:
                tab5_refs = tab5_results.build(session_state)

            with gr.Tab(_LABELS_PENDING[6], id=6) as tab6_view:
                tab6_refs = tab6_export.build(session_state)

        # Group tab view refs for compact wiring
        tab_views = (tab1_view, tab2_view, tab3_view, tab4_view, tab5_view)

        # Pre-compute Tab 5 populate-output ordering (used by Tab 4 chain).
        tab5_populate_widgets = tab5_results.populate_refs(tab5_refs)

        # Defensive: fail fast if the populate ordering and the constant
        # used by the chain handler ever drift apart.
        assert len(tab5_populate_widgets) == _TAB5_POPULATE_COUNT, (
            f"Tab 5 populate ordering returned "
            f"{len(tab5_populate_widgets)} widgets, "
            f"expected {_TAB5_POPULATE_COUNT}. Update either "
            f"_TAB5_POPULATE_COUNT in app.py or "
            f"tab5_results.populate_refs() so they agree."
        )

        # ─── App-level event chains ────────────────────────────────────────
        # Tab 1 confirm → unlock Tab 2 + relock Tabs 3+4+5 + update labels.
        tab1_refs["confirm_btn"].click(
            fn=_on_tab1_confirm_chain,
            inputs=[session_state],
            outputs=[
                tab2_refs["df"],
                tab2_refs["lock_msg"],
                tab2_refs["content"],
                tab3_refs["lock_msg"],
                tab3_refs["content"],
                tab4_refs["lock_msg"],
                tab4_refs["content"],
                tab5_refs["lock_msg"],
                tab5_refs["content"],
                tab6_refs["lock_msg"],
                tab6_refs["content"],
                tabs,
                *tab_views,
            ],
        )

        # Tab 2 confirm → unlock+clear Tab 3 + relock Tabs 4+5+6 + update labels.
        tab2_refs["confirm_btn"].click(
            fn=_on_tab2_confirm_chain,
            inputs=[session_state],
            outputs=[
                tabs,
                *tab_views,
                tab3_refs["lock_msg"],
                tab3_refs["content"],
                # Order MUST match tab3_upload.clear_state_updates():
                tab3_refs["file_input"],
                tab3_refs["schema_preview"],
                tab3_refs["class_col_dd"],
                tab3_refs["validation_md"],
                tab3_refs["suggestions_md"],
                tab3_refs["status"],
                tab3_refs["confirm_btn"],
                tab3_refs["df_state"],
                tab3_refs["schema_state"],
                tab3_refs["report_state"],
                tab3_refs["csv_path_state"],
                tab4_refs["lock_msg"],
                tab4_refs["content"],
                tab5_refs["lock_msg"],
                tab5_refs["content"],
                tab6_refs["lock_msg"],
                tab6_refs["content"],
            ],
        )

        # Tab 3 confirm → unlock Tab 4 + populate it + relock Tabs 5+6.
        tab3_refs["confirm_btn"].click(
            fn=_on_tab3_confirm_chain,
            inputs=[session_state],
            outputs=[
                tabs,
                *tab_views,
                tab4_refs["lock_msg"],
                tab4_refs["content"],
                # Order MUST match tab4_config.populate_state_updates():
                tab4_refs["class_checkbox"],
                tab4_refs["detected_bands_hint"],
                tab4_refs["subset_df"],
                tab4_refs["validation_status"],
                tab4_refs["confirm_btn"],
                tab4_refs["status"],
                tab5_refs["lock_msg"],
                tab5_refs["content"],
                tab6_refs["lock_msg"],
                tab6_refs["content"],
            ],
        )

        # Tab 4 confirm → unlock Tab 5 + populate it (compute all viz), relock Tab 6.
        tab4_refs["confirm_btn"].click(
            fn=_on_tab4_confirm_chain,
            inputs=[session_state],
            outputs=[
                tabs,
                *tab_views,
                tab5_refs["lock_msg"],
                tab5_refs["content"],
                # Order MUST match tab5_results.populate_refs(tab5_refs):
                *tab5_populate_widgets,
                tab6_refs["lock_msg"],
                tab6_refs["content"],
            ],
        )

        # Tab 5 confirm → unlock Tab 6 + reset its widgets.
        tab5_refs["confirm_btn"].click(
            fn=_on_tab5_confirm_chain,
            inputs=[session_state],
            outputs=[
                tabs,
                *tab_views,
                tab6_refs["lock_msg"],
                tab6_refs["content"],
                tab6_refs["status"],
                tab6_refs["download_file"],
            ],
        )

        # ─── Footer ────────────────────────────────────────────────────────
        gr.HTML(FOOTER_HTML)

    return demo


def _find_free_port(
    host: str = "0.0.0.0", start: int = 7860, end: int = 7900
) -> int:
    """First available TCP port in [start, end]. Falls back to ``start``."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return start


# ─── Module entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_app()
    port = _find_free_port()
    if port != 7860:
        print(f"[app] WARNING  Port 7860 busy - using port {port} instead.")
    else:
        print(f"[app] Launching on port {port}.")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )
