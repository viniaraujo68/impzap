"""Slide 8 — 2x2 paradigm matrix for the four agent families.

Axes:
  - Rows  (top -> bottom): Static, Adaptive
  - Cols  (left -> right): No lookahead, Search-based

Quadrants:
  - Static + No lookahead   : REINFORCE (navy_light)
  - Static + Search         : MCTS      (green_light)
  - Adaptive + No lookahead : HMM       (gold_light)
  - Adaptive + Search       : CFR       (navy solid, white text — highlighted)

Implemented in matplotlib for clean grid layout.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from _style import (
    NAVY,
    NAVY_LIGHT,
    GREEN_LIGHT,
    GOLD_LIGHT,
    INK,
    INK_SOFT,
    apply_matplotlib_style,
    save_fig,
)


# (column, row) -> (fill, edge, agent label, subtitle)
QUADRANTS = {
    (0, 0): (NAVY_LIGHT, INK_SOFT, "REINFORCE", "Policy gradient"),
    (1, 0): (GREEN_LIGHT, INK_SOFT, "MCTS", "PIMC tree search"),
    (0, 1): (GOLD_LIGHT, INK_SOFT, "HMM", "Opponent inference"),
    (1, 1): (NAVY,        NAVY,    "CFR\nHMM+CFR", "Nash + adaptation"),
}

COL_LABELS = ["No lookahead", "Search-based"]
ROW_LABELS = ["Static", "Adaptive"]


def main() -> None:
    apply_matplotlib_style()
    fig, ax = plt.subplots(figsize=(9.6, 6.4))

    # Hide axes; we draw everything as patches.
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylim(-0.05, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Column / row headers ----
    for col, label in enumerate(COL_LABELS):
        ax.text(
            col + 0.5, 2.05, label,
            ha="center", va="bottom",
            fontsize=13, color=INK_SOFT, fontweight="regular",
        )
    for row, label in enumerate(ROW_LABELS):
        # Rows: index 0 top, 1 bottom
        y = 1.5 - row
        ax.text(
            -0.15, y, label,
            ha="right", va="center",
            fontsize=13, color=INK_SOFT, rotation=0,
        )

    # ---- Quadrant tiles ----
    for (col, row), (fill, edge, name, subtitle) in QUADRANTS.items():
        # Place tiles so y=0 is bottom row (Adaptive); flip with row index.
        x = col
        y = 1 - row
        rect = patches.FancyBboxPatch(
            (x + 0.04, y + 0.04),
            0.92, 0.92,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.5,
            edgecolor=edge,
            facecolor=fill,
            zorder=2,
        )
        ax.add_patch(rect)

        text_color = "white" if fill == NAVY else INK
        title_lines = name.count("\n") + 1
        # Centre the title block; if it has 2 lines, raise it slightly so
        # the subtitle still has its own breathing room below.
        title_y = y + 0.65 if title_lines == 1 else y + 0.68
        subtitle_y = y + 0.30 if title_lines == 1 else y + 0.22
        ax.text(
            x + 0.5, title_y, name,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color=text_color,
            zorder=3,
        )
        ax.text(
            x + 0.5, subtitle_y, subtitle,
            ha="center", va="center",
            fontsize=10,
            color=text_color,
            zorder=3,
        )

    save_fig(fig, "slide_08_matriz_paradigmas")
    plt.close(fig)
    print("wrote slide_08_matriz_paradigmas.png")


if __name__ == "__main__":
    main()
