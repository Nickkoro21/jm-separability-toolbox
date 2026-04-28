"""
Auto-detection of CSV schema (Option Γ from HANDOFF §3).

Detects the most likely class column, spectral band columns, non-spectral
companions (Pan / Thermal / nDSM / DEM …), X/Y coordinate columns, and a
class-id → class-name mapping. The user can override every result later in
the UI; this module exists to make the first interaction effortless.

Strategy (deterministic, testable):
    1. Walk a priority list of canonical column names — case-insensitive,
       whitespace/underscore/dash tolerant.
    2. If a camera preset is supplied, attempt name-based matching of band
       columns to the preset's band list.
    3. Fall back to "all numeric columns minus excluded".
    4. Non-spectral companions are inferred from the preset's
       ``non_spectral`` list combined with substring heuristics.

The detector never raises — it returns a :class:`DetectedSchema` carrying
whatever it could infer plus a list of human-readable suggestions for the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import presets as _presets


# ─── Heuristic priority lists ────────────────────────────────────────────────

_CLASS_COL_CANDIDATES: tuple[str, ...] = (
    "class_id", "class_name", "class",
    "label", "category",
    "gt_class", "gt", "target",
    "y_true", "ground_truth", "ground truth",
)

_X_COL_CANDIDATES: tuple[str, ...] = (
    "x", "lon", "longitude", "easting", "utm_x",
)
_Y_COL_CANDIDATES: tuple[str, ...] = (
    "y", "lat", "latitude", "northing", "utm_y",
)

# Substrings indicating a non-spectral / derived channel
_NON_SPECTRAL_SUBSTRINGS: tuple[str, ...] = (
    "ndsm", "dsm", "dtm", "dem", "height", "elev",
    "thermal", "lwir", "tir", "temperature", "temp",
    "pan", "panchro",
)


# ─── Output dataclass ────────────────────────────────────────────────────────

@dataclass
class DetectedSchema:
    """Result of :func:`auto_detect_schema`.

    Every field can be overridden by the user in the UI. ``suggestions`` is a
    list of human-readable notes the UI can render under the schema preview.
    """
    class_col: str | None = None
    band_cols: list[str] = field(default_factory=list)
    non_spectral_cols: list[str] = field(default_factory=list)
    xy_cols: tuple[str | None, str | None] = (None, None)
    class_label_mapping: dict = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "class_col": self.class_col,
            "band_cols": list(self.band_cols),
            "non_spectral_cols": list(self.non_spectral_cols),
            "xy_cols": list(self.xy_cols),
            "class_label_mapping": dict(self.class_label_mapping),
            "suggestions": list(self.suggestions),
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lower-case, strip, collapse whitespace / underscores / dashes."""
    return re.sub(r"[\s_\-]+", "", str(s).strip().lower())


def _find_first(
    columns: list[str], candidates: tuple[str, ...]
) -> str | None:
    """Return the first column matching any candidate.

    Phase 1: exact normalised match.
    Phase 2: substring match (column contains candidate).
    Both case-insensitive after normalisation.
    """
    norm_to_orig: dict[str, str] = {}
    for c in columns:
        norm_to_orig.setdefault(_norm(c), c)

    # Phase 1: exact
    for cand in candidates:
        key = _norm(cand)
        if key in norm_to_orig:
            return norm_to_orig[key]

    # Phase 2: substring
    for cand in candidates:
        key = _norm(cand)
        if not key:
            continue
        for ncol, orig in norm_to_orig.items():
            if key in ncol:
                return orig
    return None


def _is_numeric(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


# ─── Public detection functions ───────────────────────────────────────────────

def detect_class_column(df: pd.DataFrame) -> str | None:
    """Best-effort class column detection.

    Prefers ``class_id`` / ``class_name`` / ``class`` then ``label`` etc.
    """
    return _find_first(list(df.columns), _CLASS_COL_CANDIDATES)


def detect_xy_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Best-effort (X, Y) coordinate column detection."""
    cols = list(df.columns)
    return (
        _find_first(cols, _X_COL_CANDIDATES),
        _find_first(cols, _Y_COL_CANDIDATES),
    )


def detect_non_spectral_columns(
    df: pd.DataFrame,
    preset_name: str | None = None,
) -> list[str]:
    """Find numeric columns that look like Pan / Thermal / nDSM.

    Combines the preset's non-spectral entries (when applicable) with
    generic substring heuristics. Returns columns in original DataFrame
    order.
    """
    numeric_cols = [c for c in df.columns if _is_numeric(df, c)]
    candidates: set[str] = set()

    # Preset-driven matches
    if preset_name and not _presets.is_custom(preset_name):
        try:
            for ns_name, _desc in _presets.get_non_spectral_bands(preset_name):
                match = _find_first(numeric_cols, (ns_name,))
                if match:
                    candidates.add(match)
        except KeyError:
            pass

    # Substring sweep
    for col in numeric_cols:
        ncol = _norm(col)
        for sub in _NON_SPECTRAL_SUBSTRINGS:
            if sub in ncol:
                candidates.add(col)
                break

    # Preserve original DataFrame order
    return [c for c in df.columns if c in candidates]


def detect_band_columns(
    df: pd.DataFrame,
    preset_name: str | None = None,
    excluded_cols: list[str] | None = None,
) -> list[str]:
    """Detect spectral band columns.

    If ``preset_name`` is provided (and not the sentinel), tries to match
    each preset band name to a CSV column by exact/substring rules.
    Otherwise returns "all numeric columns minus excluded" in DataFrame
    order.
    """
    excluded = set(excluded_cols or [])
    numeric_cols = [
        c for c in df.columns
        if _is_numeric(df, c) and c not in excluded
    ]

    if preset_name and not _presets.is_custom(preset_name):
        try:
            matched: list[str] = []
            for band_name in _presets.get_band_names(preset_name):
                hit = _find_first(numeric_cols, (band_name,))
                if hit and hit not in matched:
                    matched.append(hit)
            if matched:
                return matched
        except KeyError:
            pass

    return numeric_cols


def suggest_class_label_mapping(
    df: pd.DataFrame,
    class_col: str,
) -> dict:
    """Build a class_id → label mapping when possible.

    Scenarios handled:
        * ``class_col`` is integer AND a sibling ``*_name`` column exists →
          use the most frequent name per integer.
        * ``class_col`` is integer with no sibling → ``{i: f"Class {i}"}``.
        * ``class_col`` is string → identity ``{v: v}``.
    """
    if class_col not in df.columns:
        return {}

    series = df[class_col].dropna()
    if series.empty:
        return {}

    # String labels — identity
    if not pd.api.types.is_numeric_dtype(series):
        unique = series.astype(str).unique().tolist()
        return {v: v for v in unique}

    # Integer labels — search for a sibling *_name column
    base_norm = _norm(class_col)
    sibling: str | None = None
    for c in df.columns:
        if c == class_col:
            continue
        nc = _norm(c)
        # Heuristic: contains "name" AND shares lineage with the class column
        if "name" in nc and (
            base_norm.replace("id", "") in nc or "class" in nc
        ):
            sibling = c
            break

    try:
        unique_ints = sorted(series.astype(int).unique().tolist())
    except (TypeError, ValueError):
        # Numeric but not cleanly int-castable — fall back to stringification
        return {v: str(v) for v in series.unique().tolist()}

    if sibling is not None:
        mapping: dict = {}
        for i in unique_ints:
            sub = (
                df.loc[df[class_col] == i, sibling]
                .dropna()
                .astype(str)
            )
            mapping[i] = sub.mode().iat[0] if not sub.empty else f"Class {i}"
        return mapping

    return {i: f"Class {i}" for i in unique_ints}


# ─── Aggregator ───────────────────────────────────────────────────────────────

def auto_detect_schema(
    df: pd.DataFrame,
    preset_name: str | None = None,
) -> DetectedSchema:
    """Run every detector and return a populated :class:`DetectedSchema`."""
    suggestions: list[str] = []

    class_col = detect_class_column(df)
    if class_col is None:
        suggestions.append(
            "Class column could not be auto-detected. Please pick one manually."
        )

    xy = detect_xy_columns(df)
    non_spectral = detect_non_spectral_columns(df, preset_name=preset_name)

    excluded = list(filter(None, [class_col, *xy])) + list(non_spectral)
    band_cols = detect_band_columns(
        df, preset_name=preset_name, excluded_cols=excluded
    )

    if preset_name and not _presets.is_custom(preset_name):
        try:
            expected = _presets.get_band_names(preset_name)
            if len(band_cols) < len(expected):
                matched_lc = {b.lower() for b in band_cols}
                missing = [
                    name for name in expected
                    if name.lower() not in matched_lc
                ]
                if missing:
                    suggestions.append(
                        f"Preset {preset_name!r} expects {len(expected)} "
                        f"bands; only {len(band_cols)} matched. "
                        f"Unmatched band names: {missing}"
                    )
        except KeyError:
            pass

    label_mapping = (
        suggest_class_label_mapping(df, class_col) if class_col else {}
    )

    if non_spectral:
        suggestions.append(
            f"Detected {len(non_spectral)} non-spectral column(s): "
            f"{non_spectral}. They are excluded from the band list by default."
        )

    return DetectedSchema(
        class_col=class_col,
        band_cols=band_cols,
        non_spectral_cols=non_spectral,
        xy_cols=xy,
        class_label_mapping=label_mapping,
        suggestions=suggestions,
    )
