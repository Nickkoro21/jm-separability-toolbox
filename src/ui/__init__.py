"""User-interface tab modules — sequential workflow.

Each ``tabN_*`` module exposes a ``build(state)`` function that renders its
widgets, wires its internal events, and returns a dict of references the
parent ``app.py`` uses for tab-switching and downstream unlocking.
"""

from . import (
    tab1_camera,
    tab2_wavelengths,
    tab3_upload,
    tab4_config,
    tab5_results,
    tab6_export,
)

__all__ = [
    "tab1_camera",
    "tab2_wavelengths",
    "tab3_upload",
    "tab4_config",
    "tab5_results",
    "tab6_export",
]
