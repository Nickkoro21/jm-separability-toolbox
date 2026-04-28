"""
Band classification utilities — group spectral and non-spectral bands by
physical quantity for unit-aware visualisation.

Categories (in display order — this is the panel order top-to-bottom in figures):
  1. Reflectance — unitless, 0-1 range (Blue, Green, Red, NIR, SWIR, ...)
  2. Height       — metres (nDSM, DSM, DTM, CHM, ...)
  3. Temperature  — degrees Celsius (Thermal, LWIR, TIR, ...)
  4. Other        — fallback for unrecognised bands
  5. Index        — derived indices (NDVI, NDWI, SAVI, ...)

Classification priority (most-specific-first, with Other as the fallback):
  Reflectance → Height → Temperature → Index → Other

Note: the *display* order (above) and the *priority* order (here) differ
deliberately. Display order is a UX choice. Priority order ensures Other
catches only genuinely unknown bands, since Other is a fallback — placing it
before Index in the priority chain would make Index unreachable.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence


# ----------------------------------------------------------------------------
# Category constants
# ----------------------------------------------------------------------------

CATEGORY_REFLECTANCE = "reflectance"
CATEGORY_HEIGHT      = "height"
CATEGORY_TEMPERATURE = "temperature"
CATEGORY_OTHER       = "other"
CATEGORY_INDEX       = "index"

# Display order (panel order in figures, top to bottom).
# This is the order returned by group_bands_by_category().
CATEGORY_ORDER: List[str] = [
    CATEGORY_REFLECTANCE,
    CATEGORY_HEIGHT,
    CATEGORY_TEMPERATURE,
    CATEGORY_OTHER,
    CATEGORY_INDEX,
]

# Y-axis labels per category (used as panel y-axis titles).
CATEGORY_LABELS: Dict[str, str] = {
    CATEGORY_REFLECTANCE: "Reflectance (0–1)",
    CATEGORY_HEIGHT:      "Height (m)",
    CATEGORY_TEMPERATURE: "Temperature (°C)",
    CATEGORY_OTHER:       "Value",
    CATEGORY_INDEX:       "Index value",
}

# Human-readable display names (used in subplot titles, banners, etc.).
CATEGORY_DISPLAY: Dict[str, str] = {
    CATEGORY_REFLECTANCE: "Reflectance",
    CATEGORY_HEIGHT:      "Height",
    CATEGORY_TEMPERATURE: "Temperature",
    CATEGORY_OTHER:       "Other",
    CATEGORY_INDEX:       "Index",
}

# Wavelength range (nm) considered as "optical reflectance" for fallback
# classification when name keywords don't match.
WAVELENGTH_MIN_NM = 300.0
WAVELENGTH_MAX_NM = 2500.0


# ----------------------------------------------------------------------------
# Keyword tables (substring match against canonicalised band names)
# ----------------------------------------------------------------------------

_INDEX_KEYWORDS = (
    "ndvi", "ndwi", "savi", "evi", "gci", "ndmi",
    "mndwi", "ndbi", "ndsi", "msavi", "arvi", "gndvi",
)

_TEMPERATURE_KEYWORDS = (
    "thermal", "lwir", "tir", "temp",
)

_HEIGHT_KEYWORDS = (
    "ndsm", "dsm", "dtm", "chm",
    "elevation", "height", "altitude",
)

_REFLECTANCE_KEYWORDS = (
    "blue", "green", "red", "rededge", "nir",
    "swir", "swir1", "swir2", "pan", "panchro",
    "coastal", "aerosol", "cirrus", "vapour", "vapor",
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _canonical(name: str) -> str:
    """
    Lowercase and strip whitespace/dashes/underscores. Mirrors the `_norm()`
    helper in `src/ui/tab4_config.py` so that nDSM_m → 'ndsmm', RED-EDGE →
    'rededge', etc.
    """
    if name is None:
        return ""
    return (
        str(name).strip()
                  .lower()
                  .replace(" ", "")
                  .replace("_", "")
                  .replace("-", "")
    )


def _matches_any(canonical_name: str, keywords: Sequence[str]) -> bool:
    """True if any keyword is a substring of the canonical name."""
    return any(kw in canonical_name for kw in keywords)


def _wavelength_in_optical_range(wavelength_nm: Optional[float]) -> bool:
    """True if wavelength is a finite number inside [300, 2500] nm."""
    if wavelength_nm is None:
        return False
    try:
        wl = float(wavelength_nm)
    except (TypeError, ValueError):
        return False
    return WAVELENGTH_MIN_NM <= wl <= WAVELENGTH_MAX_NM


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def classify_band(name: str, wavelength_nm: Optional[float] = None) -> str:
    """
    Classify a single band into one of the five categories.

    Priority chain (specific-first, Other always last as fallback):
        Reflectance → Height → Temperature → Index → Other

    Reflectance is checked first because it has both a name-based AND a
    wavelength-based criterion (the most specific signal). Index is checked
    after the physical-quantity categories because index names (NDVI, NDWI,
    ...) are unambiguous and do not collide with reflectance / height /
    temperature keywords.

    Edge case: a band literally named e.g. ``"thermal_anomaly_index"`` will be
    classified as Temperature first (substring "thermal"). If the user intends
    a derived index, they should use a canonical index name (NDVI etc.).

    Args:
        name: raw band name as written by the user / sensor preset.
        wavelength_nm: known central wavelength in nanometres, or None.

    Returns:
        One of the CATEGORY_* constants.
    """
    canonical = _canonical(name)

    # 1. Reflectance — name keyword OR wavelength in optical range.
    if _matches_any(canonical, _REFLECTANCE_KEYWORDS):
        return CATEGORY_REFLECTANCE
    if _wavelength_in_optical_range(wavelength_nm):
        return CATEGORY_REFLECTANCE

    # 2. Height keywords.
    if _matches_any(canonical, _HEIGHT_KEYWORDS):
        return CATEGORY_HEIGHT

    # 3. Temperature keywords.
    if _matches_any(canonical, _TEMPERATURE_KEYWORDS):
        return CATEGORY_TEMPERATURE

    # 4. Index keywords.
    if _matches_any(canonical, _INDEX_KEYWORDS):
        return CATEGORY_INDEX

    # 5. Fallback.
    return CATEGORY_OTHER


def group_bands_by_category(
    bands: Sequence[str],
    wavelengths: Optional[Dict[str, float]] = None,
) -> "OrderedDict[str, List[str]]":
    """
    Group bands by category, preserving CATEGORY_ORDER (display order).

    Empty categories are omitted from the result. Within each category, bands
    appear in their original input order — the caller can re-sort by
    wavelength later if needed (Decision #32: numeric x-axis when all bands
    in a panel have a known wavelength).

    Args:
        bands: list of band names as used in the dataframe / subset.
        wavelengths: optional dict {band_name → wavelength_nm}. Bands not in
            this dict (or with None value) are classified by name only.

    Returns:
        OrderedDict where keys are CATEGORY_* constants in CATEGORY_ORDER
        and values are non-empty lists of band names.
    """
    wavelengths = wavelengths or {}

    # First pass: classify every band.
    by_category: Dict[str, List[str]] = {cat: [] for cat in CATEGORY_ORDER}
    for band in bands:
        cat = classify_band(band, wavelengths.get(band))
        by_category[cat].append(band)

    # Second pass: emit in CATEGORY_ORDER, skipping empties.
    result: "OrderedDict[str, List[str]]" = OrderedDict()
    for cat in CATEGORY_ORDER:
        if by_category[cat]:
            result[cat] = by_category[cat]
    return result


def get_unrecognised_bands(
    bands: Sequence[str],
    wavelengths: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Return the list of bands that fall into the Other category.

    Convenience wrapper used by the Tab 5 banner logic.
    """
    grouped = group_bands_by_category(bands, wavelengths)
    return list(grouped.get(CATEGORY_OTHER, []))


# ----------------------------------------------------------------------------
# Self-test (run this module directly to verify the rules)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cases = [
        # (band_name, wavelength, expected_category)
        # --- Reflectance: canonical names with wavelength ---
        ("Blue",       475,  CATEGORY_REFLECTANCE),
        ("Green",      560,  CATEGORY_REFLECTANCE),
        ("Red",        668,  CATEGORY_REFLECTANCE),
        ("RedEdge",    717,  CATEGORY_REFLECTANCE),
        ("NIR",        842,  CATEGORY_REFLECTANCE),
        ("SWIR1",     1610,  CATEGORY_REFLECTANCE),
        ("SWIR2",     2200,  CATEGORY_REFLECTANCE),
        ("Pan",        634,  CATEGORY_REFLECTANCE),
        ("Coastal",    443,  CATEGORY_REFLECTANCE),
        # --- Reflectance: wavelength-only fallback (no name keyword) ---
        ("B12",       2202,  CATEGORY_REFLECTANCE),    # Sentinel-2 SWIR2
        ("Band_5",     842,  CATEGORY_REFLECTANCE),    # Landsat 8 NIR
        # --- Height ---
        ("nDSM_m",     None, CATEGORY_HEIGHT),
        ("DSM",        None, CATEGORY_HEIGHT),
        ("DTM",        None, CATEGORY_HEIGHT),
        ("CHM",        None, CATEGORY_HEIGHT),
        ("Elevation",  None, CATEGORY_HEIGHT),
        ("height_m",   None, CATEGORY_HEIGHT),
        # --- Temperature ---
        ("Thermal_C",  None, CATEGORY_TEMPERATURE),
        ("LWIR_band",  None, CATEGORY_TEMPERATURE),
        ("TIR",        None, CATEGORY_TEMPERATURE),
        ("temp_K",     None, CATEGORY_TEMPERATURE),
        # --- Index ---
        ("NDVI",       None, CATEGORY_INDEX),
        ("ndwi_mean",  None, CATEGORY_INDEX),
        ("SAVI",       None, CATEGORY_INDEX),
        ("EVI",        None, CATEGORY_INDEX),
        # --- Other ---
        ("mystery_field", None, CATEGORY_OTHER),
        ("foo_bar",       None, CATEGORY_OTHER),
        ("xyz123",        None, CATEGORY_OTHER),
        # --- Edge cases (priority verification) ---
        ("thermal_anomaly_index", None, CATEGORY_TEMPERATURE),  # 'thermal' wins
        ("ndsm_height",           None, CATEGORY_HEIGHT),       # both Height keywords
        ("RED-EDGE",              None, CATEGORY_REFLECTANCE),  # canonicalisation
        ("red edge",              None, CATEGORY_REFLECTANCE),  # whitespace
    ]

    failures = 0
    print("classify_band() self-test:")
    print("-" * 70)
    for name, wl, expected in cases:
        got = classify_band(name, wl)
        status = "OK  " if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"  [{status}] {name!r:30s} wl={wl!s:5s} -> {got:12s} (exp {expected})")

    print()
    if failures:
        print(f"  ✗ {failures} / {len(cases)} cases failed")
        sys.exit(1)
    print(f"  ✓ All {len(cases)} cases passed")

    # Group test — mirrors the live demo CSV
    print("\ngroup_bands_by_category() — demo CSV (7D):")
    print("-" * 70)
    bands_7d = ["Blue", "Green", "Red", "RedEdge", "NIR", "nDSM_m", "Thermal_C"]
    wavelengths = {
        "Blue": 475, "Green": 560, "Red": 668, "RedEdge": 717, "NIR": 842,
    }
    grouped = group_bands_by_category(bands_7d, wavelengths)
    for cat, members in grouped.items():
        print(f"  {cat:13s} → {members}")

    # Group test — RGB only
    print("\ngroup_bands_by_category() — RGB only:")
    print("-" * 70)
    grouped = group_bands_by_category(
        ["Red", "Green", "Blue"],
        {"Red": 668, "Green": 560, "Blue": 475},
    )
    for cat, members in grouped.items():
        print(f"  {cat:13s} → {members}")

    # Group test — with unknown band
    print("\ngroup_bands_by_category() — with unknown band:")
    print("-" * 70)
    grouped = group_bands_by_category(
        ["Blue", "Green", "mystery_field", "Thermal_C"],
        {"Blue": 475, "Green": 560},
    )
    for cat, members in grouped.items():
        print(f"  {cat:13s} → {members}")
    print()
