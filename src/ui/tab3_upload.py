"""
Tab 3 — CSV Upload, Schema Auto-Detection & Hard-Error Validation.

Sequential workflow: STEP 3 of 6.

Locked until Tab 2 (wavelength confirmation) is confirmed. On unlock, the
upload area accepts a CSV with one row per sample, one column per band,
and a class label column.

Pipeline (auto-on-upload):
    1. Read CSV (UTF-8 / utf-8-sig / latin-1 / cp1252 fallbacks).
    2. ``auto_detect_schema(df, preset_name=state['preset_name'])``.
    3. ``run_full_validation(df, schema.class_col, schema.band_cols)``.
    4. Render schema preview + validation report (colour-coded box).
    5. Class column is exposed as a Dropdown override; its change re-runs
       validation only (the schema itself stays cached — band detection is
       deliberately NOT redone, so a wrong-class fix doesn't perturb bands).
    6. Confirm button is gated on ``report.ok``.

Public API
----------
build(state) -> dict
    Render the tab and return refs for app-level event wiring.
    Keys (full list — needed by app.py for cascade-clear on Tab 2 re-confirm):
        ``lock_msg``, ``content``, ``file_input``, ``schema_preview``,
        ``class_col_dd``, ``validation_md``, ``suggestions_md``,
        ``confirm_btn``, ``status``,
        ``df_state``, ``schema_state``, ``report_state``, ``csv_path_state``.

clear_state_updates() -> tuple
    Returns 11 ``gr.update()`` values + ``None`` placeholders that reset every
    visible field + every local ``gr.State`` to its default value. Used by
    ``app.py`` chain handlers when cascade-resetting after upstream re-confirm.

Constants exposed for re-use by app.py and any future test harness:
    ``DEFAULT_SCHEMA_PREVIEW``, ``DEFAULT_VALIDATION``,
    ``DEFAULT_SUGGESTIONS``, ``DEFAULT_STATUS``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.core import (
    DetectedSchema,
    ValidationReport,
    auto_detect_schema,
    run_full_validation,
)


# ─── Colour tokens (aligned with Tab 2 + 4-bucket palette) ────────────────────
_C_OK    = "#16a34a"   # green-600
_C_WARN  = "#d97706"   # amber-600
_C_ERR   = "#ef4444"   # red-500
_C_INFO  = "#8896b3"   # slate
_C_BLUE  = "#60a5fa"   # blue-400


# ─── Public default messages (re-used in cascade-reset by app.py) ─────────────
DEFAULT_SCHEMA_PREVIEW: str = (
    f"<span style='color:{_C_INFO};'>"
    f"ℹ️ Upload a CSV to see schema detection results.</span>"
)

DEFAULT_VALIDATION: str = (
    f"<span style='color:{_C_INFO};'>"
    f"ℹ️ Validation report will appear here after upload.</span>"
)

DEFAULT_SUGGESTIONS: str = ""
DEFAULT_STATUS:      str = ""


# ─── CSV reading with encoding fallbacks ──────────────────────────────────────

def _read_csv_safely(
    path: str | Path,
) -> tuple[pd.DataFrame | None, str | None]:
    """Read a CSV trying common encodings; return ``(df, error)``.

    ``error`` is ``None`` on success, otherwise a human-readable message
    suitable for direct display. UTF-8 is tried first (with and without
    BOM), then Latin-1, then CP1252 (common for Excel-exported CSVs on
    Greek Windows).
    """
    p = Path(path)
    if not p.exists():
        return None, f"File not found: {p}"
    if p.stat().st_size == 0:
        return None, "File is empty (0 bytes)."

    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(p, encoding=enc)
            if df.shape[0] == 0:
                return None, "CSV has zero rows."
            if df.shape[1] == 0:
                return None, "CSV has zero columns."
            return df, None
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except pd.errors.EmptyDataError:
            return None, "CSV is empty."
        except pd.errors.ParserError as e:
            return None, f"CSV parsing error: {e}"
        except Exception as e:  # noqa: BLE001
            return None, f"Could not read CSV: {type(e).__name__}: {e}"

    return None, (
        f"Could not decode CSV with any common encoding "
        f"(utf-8 / utf-8-sig / latin-1 / cp1252). "
        f"Last error: {last_err}"
    )


# ─── Markdown formatters ──────────────────────────────────────────────────────

def _format_schema_preview(df: pd.DataFrame, schema: DetectedSchema) -> str:
    """Pretty schema preview (Markdown + inline HTML accents)."""
    lines = [
        "### 📋 Schema preview",
        "",
        f"- **Rows × columns**: `{df.shape[0]:,} × {df.shape[1]}`",
    ]

    if schema.band_cols:
        lines.append(
            f"- **Detected band columns** ({len(schema.band_cols)}): "
            + ", ".join(f"`{c}`" for c in schema.band_cols)
        )
    else:
        lines.append(
            f"- <span style='color:{_C_ERR};'>"
            f"**No band columns detected.**</span> "
            f"Check that your CSV has numeric spectral columns."
        )

    if schema.non_spectral_cols:
        lines.append(
            f"- **Non-spectral columns** "
            f"({len(schema.non_spectral_cols)}): "
            + ", ".join(f"`{c}`" for c in schema.non_spectral_cols)
            + f" <span style='color:{_C_INFO};font-size:0.85em;'>"
            f"(excluded from spectral analysis by default)</span>"
        )

    xy_x, xy_y = schema.xy_cols
    if xy_x or xy_y:
        lines.append(
            f"- **X / Y coordinates**: "
            f"`{xy_x or '—'}` / `{xy_y or '—'}`"
        )

    return "\n".join(lines)


def _format_suggestions(schema: DetectedSchema) -> str:
    """Bulleted list of schema suggestions; empty string if none."""
    if not schema.suggestions:
        return ""
    lines = ["### 💡 Suggestions", ""]
    for s in schema.suggestions:
        lines.append(f"- {s}")
    return "\n".join(lines)


def _wrap_validation_md(report: ValidationReport) -> str:
    """Wrap ``ValidationReport.format_markdown()`` inside a coloured box."""
    body = report.format_markdown()
    if report.ok:
        bg, border, accent = (
            "rgba(22,163,74,0.06)",
            "rgba(22,163,74,0.3)",
            _C_OK,
        )
    else:
        bg, border, accent = (
            "rgba(239,68,68,0.06)",
            "rgba(239,68,68,0.3)",
            _C_ERR,
        )
    return (
        f"<div style='padding:14px 18px;"
        f" background:{bg};"
        f" border:1px solid {border};"
        f" border-left:4px solid {accent};"
        f" border-radius:8px;'>\n\n"
        f"{body}\n\n"
        f"</div>"
    )


def _read_error_box(msg: str) -> str:
    """Red error box rendered when CSV reading itself fails."""
    return (
        f"<div style='padding:14px 18px;"
        f" background:rgba(239,68,68,0.08);"
        f" border:1px solid rgba(239,68,68,0.3);"
        f" border-left:4px solid {_C_ERR};"
        f" border-radius:8px;'>"
        f"<b style='color:{_C_ERR};'>❌ Cannot read CSV.</b><br/>"
        f"<span style='color:{_C_INFO};'>{msg}</span>"
        f"</div>"
    )


# ─── gr.File output → str path resolver ───────────────────────────────────────

def _resolve_path(file_obj: Any) -> str | None:
    """Extract a string path from whatever ``gr.File`` returned.

    With ``type='filepath'`` Gradio 5.x returns ``str`` directly, but some
    object-style fallbacks expose ``.name`` — handle both gracefully.
    """
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return file_obj.name
    return None


# ─── Event handlers ───────────────────────────────────────────────────────────

def _on_upload(file_obj: Any, state: dict):
    """File uploaded: read → detect → validate → render everything.

    Outputs (10):
        df_state, schema_state, report_state, csv_path_state,
        schema_preview, class_col_dd, validation_md, suggestions_md,
        confirm_btn, status
    """
    # File cleared by user (×) → reset to defaults
    if file_obj is None:
        return (
            None, None, None, None,
            gr.update(value=DEFAULT_SCHEMA_PREVIEW),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(value=DEFAULT_VALIDATION),
            gr.update(value=DEFAULT_SUGGESTIONS),
            gr.update(interactive=False),
            gr.update(value=DEFAULT_STATUS),
        )

    path = _resolve_path(file_obj)
    if path is None:
        return (
            None, None, None, None,
            gr.update(value=DEFAULT_SCHEMA_PREVIEW),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(value=_read_error_box(
                "Could not resolve uploaded file path."
            )),
            gr.update(value=DEFAULT_SUGGESTIONS),
            gr.update(interactive=False),
            gr.update(value=DEFAULT_STATUS),
        )

    df, err = _read_csv_safely(path)
    if err is not None:
        return (
            None, None, None, None,
            gr.update(value=DEFAULT_SCHEMA_PREVIEW),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(value=_read_error_box(err)),
            gr.update(value=DEFAULT_SUGGESTIONS),
            gr.update(interactive=False),
            gr.update(value=DEFAULT_STATUS),
        )

    # Use the camera preset chosen in Tab 1 to bias band-name matching
    preset_name = (
        state.get("preset_name") if isinstance(state, dict) else None
    )
    schema = auto_detect_schema(df, preset_name=preset_name)

    # Build a validation report. If detection failed to find class col or
    # band cols, synthesise a report so the user sees actionable errors
    # in the same green/red box rather than a generic Python traceback.
    if schema.class_col and schema.band_cols:
        report = run_full_validation(
            df, schema.class_col, schema.band_cols
        )
    else:
        problems: list[str] = []
        if not schema.class_col:
            problems.append(
                "Class column could not be auto-detected. "
                "Please pick the correct one from the dropdown below."
            )
        if not schema.band_cols:
            problems.append(
                "No band columns could be auto-detected. "
                "Check that your CSV has numeric spectral columns."
            )
        report = ValidationReport(
            ok=False,
            errors=problems,
            warnings=[],
            n_samples=0,
            class_counts={},
        )

    return (
        df,
        schema.to_dict(),
        report.to_dict(),
        path,
        gr.update(value=_format_schema_preview(df, schema)),
        gr.update(
            choices=list(df.columns),
            value=schema.class_col,
            interactive=True,
        ),
        gr.update(value=_wrap_validation_md(report)),
        gr.update(value=_format_suggestions(schema)),
        gr.update(interactive=report.ok),
        gr.update(value=DEFAULT_STATUS),
    )


def _on_class_col_change(
    class_col: str | None,
    df: pd.DataFrame | None,
    schema_dict: dict | None,
):
    """Class-column override: re-run validation only.

    Outputs (4):
        report_state, validation_md, confirm_btn, status

    Note
    ----
    The status is also cleared here — if the user had already confirmed
    earlier and now changes ``class_col``, the stale "✅ confirmed"
    message is misleading. The shared ``state`` is NOT mutated here;
    only Confirm propagates to shared state.
    """
    if df is None or schema_dict is None:
        return (
            None,
            gr.update(value=DEFAULT_VALIDATION),
            gr.update(interactive=False),
            gr.update(value=DEFAULT_STATUS),
        )

    if class_col is None or class_col == "":
        warn = (
            f"<span style='color:{_C_WARN};'>"
            f"⚠️ Please pick a class column.</span>"
        )
        return (
            None,
            gr.update(value=warn),
            gr.update(interactive=False),
            gr.update(value=DEFAULT_STATUS),
        )

    band_cols = schema_dict.get("band_cols", [])
    if not band_cols:
        report = ValidationReport(
            ok=False,
            errors=["No band columns detected — cannot validate."],
            warnings=[],
            n_samples=0,
            class_counts={},
        )
    else:
        report = run_full_validation(df, class_col, band_cols)

    return (
        report.to_dict(),
        gr.update(value=_wrap_validation_md(report)),
        gr.update(interactive=report.ok),
        gr.update(value=DEFAULT_STATUS),
    )


def _on_confirm(
    df: pd.DataFrame | None,
    schema_dict: dict | None,
    report_dict: dict | None,
    csv_path: str | None,
    class_col: str | None,
    state: dict,
):
    """Persist confirmed CSV state, cascade-reset downstream tabs.

    Outputs (2): ``state``, ``status``.
    """
    if (
        df is None
        or report_dict is None
        or not report_dict.get("ok", False)
    ):
        msg = (
            f"<span style='color:{_C_ERR};'>"
            f"❌ Cannot confirm — validation has not passed.</span>"
        )
        return state, gr.update(value=msg)

    # Persist the (possibly-overridden) class_col into the saved schema
    final_schema = dict(schema_dict or {})
    if class_col:
        final_schema["class_col"] = class_col

    new_state = dict(state)
    new_state["csv_path"]            = csv_path
    new_state["df"]                  = df
    new_state["detected_schema"]     = final_schema
    new_state["validation_report"]   = report_dict
    new_state["tab3_done"]           = True
    # Cascade-reset downstream
    for k in ("tab4_done", "tab5_done"):
        new_state[k] = False

    n_samples = report_dict.get("n_samples", 0)
    n_classes = len(report_dict.get("class_counts", {}) or {})
    n_bands   = len(final_schema.get("band_cols", []) or [])

    summary = (
        f"<span style='color:{_C_OK};'>"
        f"✅ <b>CSV confirmed.</b> "
        f"{n_samples:,} samples · {n_classes} classes · {n_bands} bands. "
        f"Proceed to <b>Step 4</b> to configure band subsets.</span>"
    )
    return new_state, gr.update(value=summary)


# ─── Cascade-reset helper (called by app.py from Tab 2 chain handler) ─────────

def clear_state_updates() -> tuple:
    """Return updates that reset every Tab 3 widget + local state to defaults.

    Used by the ``_on_tab2_confirm_chain`` in ``app.py`` so re-confirming
    Tab 2 always presents Tab 3 as a fresh slate (no stale CSV results).

    Order (must match the ``outputs=[...]`` list in app.py exactly):
        file_input,
        schema_preview,
        class_col_dd,
        validation_md,
        suggestions_md,
        status,
        confirm_btn,
        df_state,
        schema_state,
        report_state,
        csv_path_state
    """
    return (
        gr.update(value=None),                                 # file_input
        gr.update(value=DEFAULT_SCHEMA_PREVIEW),               # schema_preview
        gr.update(choices=[], value=None, interactive=False),  # class_col_dd
        gr.update(value=DEFAULT_VALIDATION),                   # validation_md
        gr.update(value=DEFAULT_SUGGESTIONS),                  # suggestions_md
        gr.update(value=DEFAULT_STATUS),                       # status
        gr.update(interactive=False),                          # confirm_btn
        None,                                                  # df_state
        None,                                                  # schema_state
        None,                                                  # report_state
        None,                                                  # csv_path_state
    )


# ─── Public builder ───────────────────────────────────────────────────────────

def build(state: gr.State) -> dict:
    """Render Tab 3 widgets and wire internal events.

    Parameters
    ----------
    state : gr.State
        Shared session state object, owned by ``app.py``.

    Returns
    -------
    dict
        Refs needed by ``app.py`` for chain wiring & cascade-reset:

        =================== ==========================================
        Key                 Purpose
        =================== ==========================================
        lock_msg            ``gr.Group`` — visible while Tab 2 not done
        content             ``gr.Group`` — hidden until Tab 2 unlocks
        file_input          ``gr.File`` — CSV uploader
        schema_preview      ``gr.Markdown`` — auto-detected schema
        class_col_dd        ``gr.Dropdown`` — class-column override
        validation_md       ``gr.Markdown`` — validation report (boxed)
        suggestions_md      ``gr.Markdown`` — schema suggestions list
        confirm_btn         ``gr.Button`` — gated on report.ok
        status              ``gr.Markdown`` — confirmation feedback
        df_state            ``gr.State`` — pd.DataFrame | None
        schema_state        ``gr.State`` — DetectedSchema.to_dict()|None
        report_state        ``gr.State`` — ValidationReport.to_dict()|None
        csv_path_state      ``gr.State`` — uploaded file path|None
        =================== ==========================================
    """
    # ── Lock screen (visible until Tab 2 unlocks this tab) ────────────────
    with gr.Group(visible=True) as lock_msg:
        gr.Markdown(
            f"""
            <div style="text-align:center; padding:32px 24px;
                        background:rgba(245,158,11,0.08);
                        border:1px solid rgba(245,158,11,0.3);
                        border-radius:12px; color:{_C_WARN};">
              <h3 style="margin:0 0 8px 0;">🔒 Locked</h3>
              <p style="margin:0;">Please complete <b>Step 2 — Wavelengths</b>
              first.<br/>Once confirmed, the upload area below will unlock
              automatically.</p>
            </div>
            """,
        )

    # ── Tab content (hidden until Tab 2 confirms) ─────────────────────────
    with gr.Group(visible=False) as content:
        gr.Markdown(
            """
            ### <span class="step-badge" style="background:rgba(96,165,250,0.15); color:#60a5fa;">STEP 3</span> Upload your data file

            Drop a **CSV** with one row per sample, one column per band, and
            a class label column. The app **auto-detects schema** (class
            column, band columns, non-spectral channels, X/Y coordinates)
            and runs **hard-error validation**:

            - **≥ 2 classes** in the chosen class column,
            - **≥ 100 samples per class** (Decision #5 — non-overridable),
            - **finite numeric features** in every band column.

            If detection picks the wrong class column, override it with the
            dropdown below — validation re-runs instantly.
            """,
        )

        file_input = gr.File(
            label="📁 Upload CSV",
            file_types=[".csv"],
            file_count="single",
            type="filepath",
        )

        schema_preview = gr.Markdown(value=DEFAULT_SCHEMA_PREVIEW)

        class_col_dd = gr.Dropdown(
            label="Class column",
            info=(
                "Auto-detected from your CSV. "
                "Override here if the wrong column was picked."
            ),
            choices=[],
            value=None,
            interactive=False,
            allow_custom_value=False,
        )

        validation_md  = gr.Markdown(value=DEFAULT_VALIDATION)
        suggestions_md = gr.Markdown(value=DEFAULT_SUGGESTIONS)

        with gr.Row():
            confirm_btn = gr.Button(
                "✓ Confirm CSV",
                variant="primary",
                interactive=False,
                size="lg",
            )

        status = gr.Markdown(value=DEFAULT_STATUS)

        # ── Local intermediate state ──────────────────────────────────────
        # gr.State doesn't render anything; placing it inside `content` is
        # purely for code-locality. It's still session-scoped.
        df_state       = gr.State(value=None)
        schema_state   = gr.State(value=None)
        report_state   = gr.State(value=None)
        csv_path_state = gr.State(value=None)

    # ── Internal event wiring ─────────────────────────────────────────────
    file_input.change(
        fn=_on_upload,
        inputs=[file_input, state],
        outputs=[
            df_state, schema_state, report_state, csv_path_state,
            schema_preview, class_col_dd,
            validation_md, suggestions_md,
            confirm_btn, status,
        ],
    )

    class_col_dd.change(
        fn=_on_class_col_change,
        inputs=[class_col_dd, df_state, schema_state],
        outputs=[report_state, validation_md, confirm_btn, status],
    )

    confirm_btn.click(
        fn=_on_confirm,
        inputs=[
            df_state, schema_state, report_state, csv_path_state,
            class_col_dd, state,
        ],
        outputs=[state, status],
    )

    return {
        "lock_msg":       lock_msg,
        "content":        content,
        "file_input":     file_input,
        "schema_preview": schema_preview,
        "class_col_dd":   class_col_dd,
        "validation_md":  validation_md,
        "suggestions_md": suggestions_md,
        "confirm_btn":    confirm_btn,
        "status":         status,
        "df_state":       df_state,
        "schema_state":   schema_state,
        "report_state":   report_state,
        "csv_path_state": csv_path_state,
    }
