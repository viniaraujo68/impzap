"""
Visual style shared across all defense-slide figure scripts.

All scripts must import palette/typography/output helpers from this module.
Do not redefine colors, fonts, or DPI in individual scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# Text and structure
INK = "#1A1A1A"          # primary text
INK_SOFT = "#555555"     # secondary text, axis labels
RULE = "#999999"         # thin divider lines, reference lines

# Accents
NAVY = "#1F3864"         # primary fill (titles, main bars, Python layer)
NAVY_LIGHT = "#E8F1FA"
GOLD = "#B8860B"         # decisions, CGO bridge
GOLD_LIGHT = "#FFF4D6"
GREEN = "#2E7D32"        # Go layer, safe path
GREEN_LIGHT = "#E6F4EA"
RED = "#C0504D"          # exploitation path, alerts (use sparingly)
RED_LIGHT = "#FBE5D6"

# Functional
BG = "#FFFFFF"           # plot background (kept for explicit fills if needed)
GRID = "#E8E8E8"         # gridlines


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

_FONT_PREFERENCE = ["Calibri", "Arial", "DejaVu Sans"]


def _resolve_font_family() -> str:
    """Pick the first font from _FONT_PREFERENCE that is installed."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _FONT_PREFERENCE:
        if name in available:
            return name
    return _FONT_PREFERENCE[-1]


FONT_FAMILY = _resolve_font_family()

TITLE_PT = 16
AXIS_PT = 13
TICK_PT = 11
ANNOT_PT = 11


# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------

def apply_matplotlib_style() -> None:
    """Apply the project's matplotlib rcParams. Call once at script entry."""
    plt.rcdefaults()
    matplotlib.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": TICK_PT,
        "axes.titlesize": TITLE_PT,
        "axes.labelsize": AXIS_PT,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titleweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": TICK_PT,
        "savefig.dpi": 200,
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
    })


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def save_fig(fig: Any, name: str) -> Path:
    """Save fig to slides/figures/out/{name}.png with project defaults.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        File stem (no extension).

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(
        path,
        dpi=200,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    return path


def graphviz_attrs() -> dict[str, str]:
    """Default node/edge attributes for graphviz diagrams (used by slides 5/6/16).

    Each script can extend or override these per-node.
    """
    return {
        "fontname": FONT_FAMILY,
        "fontsize": str(TICK_PT),
        "color": INK_SOFT,
        "fontcolor": INK,
    }
