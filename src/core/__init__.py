"""Core math, presets, validation, and detection — sensor-agnostic.

Everything in this package is independent of UI and visualisation layers
(which live in ``src.ui`` and ``src.viz``). Importing from this namespace
should be enough for any downstream tab module:

    from src.core import (
        jm_matrix, interpret_jm, count_buckets, bucket_color,
        list_preset_names, get_band_names,
        run_full_validation,
        auto_detect_schema,
    )
"""

from .jm import (
    DEFAULT_REGULARISATION,
    JM_BUCKETS,
    JM_BUCKET_COLORS,
    JM_MAX,
    JM_THRESHOLD_GOOD,
    JM_THRESHOLD_MODERATE,
    JM_THRESHOLD_POOR,
    bhattacharyya_distance,
    bucket_color,
    class_statistics,
    count_buckets,
    interpret_jm,
    jm_distance,
    jm_matrix,
)

from .presets import (
    CAMERA_PRESETS,
    SENTINEL_CUSTOM,
    format_preset_summary,
    get_band_fwhm,
    get_band_names,
    get_band_wavelengths,
    get_non_spectral_bands,
    get_preset,
    get_source_url,
    is_custom,
    list_preset_names,
)

from .validation import (
    MIN_SAMPLES_PER_CLASS,
    ValidationError,
    ValidationReport,
    run_full_validation,
    validate_at_least_two_classes,
    validate_band_columns_exist,
    validate_band_columns_numeric,
    validate_class_column_exists,
    validate_dataframe_not_empty,
    validate_finite_features,
    validate_min_samples_per_class,
)

from .detection import (
    DetectedSchema,
    auto_detect_schema,
    detect_band_columns,
    detect_class_column,
    detect_non_spectral_columns,
    detect_xy_columns,
    suggest_class_label_mapping,
)

from .band_classification import (
    CATEGORY_DISPLAY,
    CATEGORY_HEIGHT,
    CATEGORY_INDEX,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    CATEGORY_OTHER,
    CATEGORY_REFLECTANCE,
    CATEGORY_TEMPERATURE,
    classify_band,
    get_unrecognised_bands,
    group_bands_by_category,
)

__all__ = [
    # ── jm ────────────────────────────────────────────────────────────────
    "DEFAULT_REGULARISATION",
    "JM_BUCKETS",
    "JM_BUCKET_COLORS",
    "JM_MAX",
    "JM_THRESHOLD_GOOD",
    "JM_THRESHOLD_MODERATE",
    "JM_THRESHOLD_POOR",
    "bhattacharyya_distance",
    "bucket_color",
    "class_statistics",
    "count_buckets",
    "interpret_jm",
    "jm_distance",
    "jm_matrix",
    # ── presets ───────────────────────────────────────────────────────────
    "CAMERA_PRESETS",
    "SENTINEL_CUSTOM",
    "format_preset_summary",
    "get_band_fwhm",
    "get_band_names",
    "get_band_wavelengths",
    "get_non_spectral_bands",
    "get_preset",
    "get_source_url",
    "is_custom",
    "list_preset_names",
    # ── validation ────────────────────────────────────────────────────────
    "MIN_SAMPLES_PER_CLASS",
    "ValidationError",
    "ValidationReport",
    "run_full_validation",
    "validate_at_least_two_classes",
    "validate_band_columns_exist",
    "validate_band_columns_numeric",
    "validate_class_column_exists",
    "validate_dataframe_not_empty",
    "validate_finite_features",
    "validate_min_samples_per_class",
    # ── detection ─────────────────────────────────────────────────────────
    "DetectedSchema",
    "auto_detect_schema",
    "detect_band_columns",
    "detect_class_column",
    "detect_non_spectral_columns",
    "detect_xy_columns",
    "suggest_class_label_mapping",
    # ── band_classification ───────────────────────────────────────────────
    "CATEGORY_DISPLAY",
    "CATEGORY_HEIGHT",
    "CATEGORY_INDEX",
    "CATEGORY_LABELS",
    "CATEGORY_ORDER",
    "CATEGORY_OTHER",
    "CATEGORY_REFLECTANCE",
    "CATEGORY_TEMPERATURE",
    "classify_band",
    "get_unrecognised_bands",
    "group_bands_by_category",
]
