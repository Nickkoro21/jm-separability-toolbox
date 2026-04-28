"""
Camera preset registry — eight verified multispectral / thermal sensors.

All wavelength data is documented in HANDOFF.md §4 with a citation URL per
preset. The schema is deliberately frozen:

    "<preset name>": {
        "bands":        [(name, center_nm, fwhm_nm), ...],   # spectral only
        "non_spectral": [(name, description), ...],          # may be []
        "source":       "<verification URL>",
    }

The UI flow uses :func:`list_preset_names` for the dropdown (always includes
the sentinel ``Custom / Unknown sensor`` at the end). All accessors raise
``KeyError`` if asked about the sentinel — call :func:`is_custom` first.
"""

from __future__ import annotations

from typing import Final

# Sentinel returned at the end of list_preset_names() for the custom path
SENTINEL_CUSTOM: Final[str] = "Custom / Unknown sensor"


# ─── The data table — verbatim from HANDOFF §4 ────────────────────────────────

CAMERA_PRESETS: Final[dict] = {
    "MicaSense Altum-PT": {
        "bands": [
            ("Blue",     475, 32),
            ("Green",    560, 27),
            ("Red",      668, 14),
            ("RedEdge",  717, 12),
            ("NIR",      842, 57),
        ],
        "non_spectral": [
            ("Pan",      "634 nm (broadband, panchromatic)"),
            ("Thermal",  "LWIR 7.5-13.5 μm"),
            ("nDSM",     "meters (derived)"),
        ],
        "source": "https://support.micasense.com/hc/en-us/articles/214878778",
    },
    "MicaSense RedEdge-MX": {
        "bands": [
            ("Blue",     475, 32),
            ("Green",    560, 27),
            ("Red",      668, 14),
            ("RedEdge",  717, 12),
            ("NIR",      842, 57),
        ],
        "non_spectral": [],
        "source": "https://support.micasense.com/hc/en-us/articles/214878778",
    },
    "MicaSense RedEdge-MX Dual": {
        "bands": [
            ("Blue",         475, 32),
            ("Green",        560, 27),
            ("Red",          668, 14),
            ("RedEdge",      717, 12),
            ("NIR",          842, 57),
            ("Coastal Blue", 444, 28),
            ("Green-531",    531, 14),
            ("Red-650",      650, 16),
            ("RedEdge-705",  705, 10),
            ("RedEdge-740",  740, 18),
        ],
        "non_spectral": [],
        "source": "https://support.micasense.com/hc/en-us/articles/214878778",
    },
    "DJI Phantom 4 Multispectral": {
        "bands": [
            ("Blue",     450, 32),
            ("Green",    560, 32),
            ("Red",      650, 32),
            ("RedEdge",  730, 32),
            ("NIR",      840, 52),
        ],
        "non_spectral": [],
        "source": "https://ag.dji.com/p4-multispectral/specs",
    },
    "DJI Mavic 3 Multispectral": {
        "bands": [
            ("Green",    560, 32),
            ("Red",      650, 32),
            ("RedEdge",  730, 32),
            ("NIR",      860, 52),
        ],
        "non_spectral": [],
        "source": "https://enterprise.dji.com/mavic-3-m/specs",
    },
    "Parrot Sequoia / Sequoia+": {
        "bands": [
            ("Green",    550, 80),
            ("Red",      660, 80),
            ("RedEdge",  735, 20),
            ("NIR",      790, 80),
        ],
        "non_spectral": [],
        "source": "https://www.parrot.com/assets/s3fs-public/2021-09/bd_sequoia_integration_manual_en_0.pdf",
    },
    "Sentinel-2 MSI": {
        "bands": [
            ("B1 Coastal", 443,   20),
            ("B2 Blue",    490,   65),
            ("B3 Green",   560,   35),
            ("B4 Red",     665,   30),
            ("B5 RE1",     705,   15),
            ("B6 RE2",     740,   15),
            ("B7 RE3",     783,   20),
            ("B8 NIR",     842,  115),
            ("B8a NIR2",   865,   20),
            ("B9 WV",      940,   20),
            ("B10 Cirrus", 1375,   30),
            ("B11 SWIR1",  1610,   90),
            ("B12 SWIR2",  2190,  180),
        ],
        "non_spectral": [],
        "source": "https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial",
    },
    "Landsat 8/9 OLI+TIRS": {
        "bands": [
            ("B1 Coastal",  443,    16),
            ("B2 Blue",     482,    60),
            ("B3 Green",    562,    57),
            ("B4 Red",      655,    38),
            ("B5 NIR",      865,    28),
            ("B6 SWIR1",   1610,    85),
            ("B7 SWIR2",   2200,   187),
            ("B8 Pan",      590,   172),
            ("B9 Cirrus",  1375,    20),
            ("B10 TIRS1", 10895,   590),
            ("B11 TIRS2", 12005,  1010),
        ],
        "non_spectral": [],
        "source": "https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites",
    },
}


# ─── Public API ───────────────────────────────────────────────────────────────

def list_preset_names() -> list[str]:
    """Ordered preset names + the sentinel ``Custom / Unknown sensor``."""
    return list(CAMERA_PRESETS.keys()) + [SENTINEL_CUSTOM]


def is_custom(name: str) -> bool:
    """True if ``name`` is the sentinel for the custom / unknown path."""
    return name == SENTINEL_CUSTOM


def get_preset(name: str) -> dict | None:
    """Return the raw preset dict, or ``None`` for the sentinel."""
    if is_custom(name):
        return None
    return CAMERA_PRESETS.get(name)


def _require(name: str) -> dict:
    """Internal — raise on unknown name or sentinel."""
    if is_custom(name):
        raise KeyError("Operation not allowed on the custom-sensor sentinel")
    if name not in CAMERA_PRESETS:
        raise KeyError(f"Unknown preset: {name!r}")
    return CAMERA_PRESETS[name]


def get_band_names(name: str) -> list[str]:
    """Spectral band names in declared order."""
    return [b[0] for b in _require(name)["bands"]]


def get_band_wavelengths(name: str) -> list[float]:
    """Center wavelengths in nm, in declared order."""
    return [float(b[1]) for b in _require(name)["bands"]]


def get_band_fwhm(name: str) -> list[float]:
    """FWHM in nm, in declared order. Same length as get_band_names()."""
    return [float(b[2]) for b in _require(name)["bands"]]


def get_non_spectral_bands(name: str) -> list[tuple[str, str]]:
    """Non-spectral entries (Pan, Thermal, nDSM, etc.). Empty list if absent."""
    return list(_require(name).get("non_spectral", []))


def get_source_url(name: str) -> str:
    """Verification URL for the preset's wavelength data."""
    return _require(name)["source"]


def format_preset_summary(name: str) -> str:
    """Markdown block suitable for the camera-confirmation UI tab."""
    if is_custom(name):
        return (
            "**Custom / Unknown sensor**\n\n"
            "Wavelength data will be entered manually in the next step. "
            "JM separability does not require wavelength values; they are used "
            "only for plotting and for ordering bands sensibly."
        )

    p = _require(name)
    lines = [f"### {name}", "", "**Spectral bands**", ""]
    lines.append("| Band | Center (nm) | FWHM (nm) |")
    lines.append("|---|---:|---:|")
    for bname, center, fwhm in p["bands"]:
        lines.append(f"| {bname} | {center} | {fwhm} |")

    if p.get("non_spectral"):
        lines += ["", "**Non-spectral channels**", ""]
        for ns_name, ns_desc in p["non_spectral"]:
            lines.append(f"- **{ns_name}** — {ns_desc}")

    lines += ["", f"_Source: {p['source']}_"]
    return "\n".join(lines)
