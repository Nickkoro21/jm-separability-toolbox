"""
Hard-error validation gate for the JM Separability workflow.

Decisions #5 / #6 in HANDOFF.md are frozen: a class with fewer than 100
samples blocks the pipeline. This module never converts errors to warnings.

Design contract
---------------
* Every individual ``validate_*`` function returns ``None`` on success and
  raises :class:`ValidationError` on failure.
* :func:`run_full_validation` aggregates the checks into a single
  :class:`ValidationReport` so the UI can present every problem at once
  rather than failing on the first one.
* No I/O. The caller passes a ``pandas.DataFrame``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

# Frozen by Decision #5 in HANDOFF.md
MIN_SAMPLES_PER_CLASS: int = 100


# ─── Exceptions & data containers ─────────────────────────────────────────────

class ValidationError(ValueError):
    """Blocking error raised when input data fails a hard check."""


@dataclass
class ValidationReport:
    """Aggregated outcome of running every check.

    Attributes
    ----------
    ok : bool
        True iff no errors were collected.
    errors : list[str]
        Hard errors that block downstream computation.
    warnings : list[str]
        Non-blocking notes (currently always empty by Decision #6, kept
        for future-proofing).
    n_samples : int
        Total rows after dropping rows with non-finite features. If band
        columns were missing, falls back to count of non-NaN class labels.
    class_counts : dict
        Sample count per class label, sorted descending.
    """
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_samples: int = 0
    class_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "n_samples": self.n_samples,
            "class_counts": dict(self.class_counts),
        }

    def format_markdown(self) -> str:
        """Human-readable report suitable for a Gradio Markdown component."""
        if self.ok:
            head = "### ✅ Validation passed"
        else:
            head = f"### ❌ Validation failed — {len(self.errors)} error(s)"
        lines = [head, "", f"- Samples (with finite features): **{self.n_samples}**"]
        if self.class_counts:
            lines.append("- Class counts:")
            for label, count in self.class_counts.items():
                marker = "✅" if count >= MIN_SAMPLES_PER_CLASS else "❌"
                lines.append(f"    - {marker} `{label}` → {count}")
        if self.errors:
            lines += ["", "**Errors**"]
            for err in self.errors:
                lines.append(f"- {err}")
        if self.warnings:
            lines += ["", "**Warnings**"]
            for w in self.warnings:
                lines.append(f"- {w}")
        return "\n".join(lines)


# ─── Individual checks ────────────────────────────────────────────────────────

def validate_dataframe_not_empty(df: pd.DataFrame) -> None:
    if df is None:
        raise ValidationError("No DataFrame provided.")
    if df.shape[0] == 0:
        raise ValidationError("Uploaded CSV has no rows.")
    if df.shape[1] == 0:
        raise ValidationError("Uploaded CSV has no columns.")


def validate_class_column_exists(df: pd.DataFrame, class_col: str) -> None:
    if class_col not in df.columns:
        raise ValidationError(
            f"Class column {class_col!r} not found. "
            f"Available columns: {list(df.columns)}"
        )
    if df[class_col].isna().all():
        raise ValidationError(f"Class column {class_col!r} is entirely NaN.")


def validate_band_columns_exist(
    df: pd.DataFrame, band_cols: Iterable[str]
) -> None:
    band_cols = list(band_cols)
    if not band_cols:
        raise ValidationError("No band columns specified.")
    missing = [b for b in band_cols if b not in df.columns]
    if missing:
        raise ValidationError(
            f"Band columns missing from CSV: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_band_columns_numeric(
    df: pd.DataFrame, band_cols: Iterable[str]
) -> None:
    non_numeric = [
        b for b in band_cols
        if not pd.api.types.is_numeric_dtype(df[b])
    ]
    if non_numeric:
        raise ValidationError(
            f"Band columns must be numeric; these are not: {non_numeric}"
        )


def validate_at_least_two_classes(
    df: pd.DataFrame, class_col: str
) -> None:
    n_classes = df[class_col].dropna().nunique()
    if n_classes < 2:
        raise ValidationError(
            f"JM separability needs ≥2 classes; only {n_classes} found "
            f"in column {class_col!r}."
        )


def validate_min_samples_per_class(
    df: pd.DataFrame,
    class_col: str,
    min_samples: int = MIN_SAMPLES_PER_CLASS,
) -> None:
    counts = df[class_col].value_counts(dropna=True)
    deficient = counts[counts < min_samples]
    if not deficient.empty:
        details = ", ".join(
            f"{label!r}={count}" for label, count in deficient.items()
        )
        raise ValidationError(
            f"Each class must have ≥{min_samples} samples. "
            f"Below threshold: {details}"
        )


def validate_finite_features(
    df: pd.DataFrame, band_cols: Iterable[str]
) -> None:
    band_cols = list(band_cols)
    finite_mask = np.isfinite(df[band_cols].to_numpy()).all(axis=1)
    n_valid = int(finite_mask.sum())
    if n_valid == 0:
        raise ValidationError(
            "No rows have finite values across all selected band columns "
            "(every row contains NaN or Inf)."
        )


# ─── Aggregate runner ─────────────────────────────────────────────────────────

def _safe(check, *args, **kwargs) -> str | None:
    """Run a check; return its message on failure, ``None`` on success."""
    try:
        check(*args, **kwargs)
        return None
    except ValidationError as exc:
        return str(exc)


def run_full_validation(
    df: pd.DataFrame,
    class_col: str,
    band_cols: Iterable[str],
    min_samples: int = MIN_SAMPLES_PER_CLASS,
) -> ValidationReport:
    """Run the full pipeline and return an aggregated report.

    Even when one check fails, the others still run (where meaningful) so the
    user sees every blocking issue at once. Some checks short-circuit when a
    prerequisite fails — e.g. counting samples per class is skipped if the
    class column itself is missing.
    """
    band_cols = list(band_cols)
    errors: list[str] = []

    # Foundational check — bail out completely if no usable DataFrame
    err = _safe(validate_dataframe_not_empty, df)
    if err:
        return ValidationReport(ok=False, errors=[err])

    err_class_col = _safe(validate_class_column_exists, df, class_col)
    if err_class_col:
        errors.append(err_class_col)

    err_bands_exist = _safe(validate_band_columns_exist, df, band_cols)
    if err_bands_exist:
        errors.append(err_bands_exist)

    # Numeric / finite checks only meaningful if band cols exist
    if not err_bands_exist:
        err = _safe(validate_band_columns_numeric, df, band_cols)
        if err:
            errors.append(err)
        err = _safe(validate_finite_features, df, band_cols)
        if err:
            errors.append(err)

    # Class-count checks only meaningful if class column exists
    class_counts: dict = {}
    n_samples = 0
    if not err_class_col:
        err = _safe(validate_at_least_two_classes, df, class_col)
        if err:
            errors.append(err)

        err = _safe(
            validate_min_samples_per_class, df, class_col, min_samples
        )
        if err:
            errors.append(err)

        # Always build class_counts — informative even when a check failed
        class_counts = (
            df[class_col]
            .dropna()
            .value_counts()
            .sort_values(ascending=False)
            .to_dict()
        )

        if not err_bands_exist:
            try:
                finite_mask = np.isfinite(
                    df[band_cols].to_numpy()
                ).all(axis=1)
                n_samples = int(finite_mask.sum())
            except (TypeError, ValueError):
                # Non-numeric band cols: fall back to class label count
                n_samples = int(df[class_col].notna().sum())
        else:
            n_samples = int(df[class_col].notna().sum())

    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=[],
        n_samples=n_samples,
        class_counts=class_counts,
    )
