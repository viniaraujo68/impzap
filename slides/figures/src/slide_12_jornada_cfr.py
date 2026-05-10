"""Slide 12 — CFR abstraction journey v3 -> v9.

Two horizontal-bar panels sharing the y-axis:
- Left: win rate vs HeuristicAgent. v8 shown as a 45-48% range with whiskers.
- Right: information-set count, log-scaled x-axis.

Numeric values come from sessions.md tournament tables, algorithms.md, and
the implementation chapter of the thesis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import (
    NAVY,
    NAVY_LIGHT,
    INK,
    INK_SOFT,
    RULE,
    GRID,
    apply_matplotlib_style,
    save_fig,
)


VERSION_LABELS = ["v3 (5 buckets)", "v4 (8 buckets)", "v8 (round history)", "v9 (score-aware)"]
ITER_LABELS = ["1M iters", "1M iters", "2M iters", "6M iters"]

# Win rate vs HeuristicAgent
WIN_RATE_POINTS = [41.0, 50.0, None, 50.1]  # None = use range
WIN_RATE_RANGE = [None, None, (45.0, 48.0), None]

# Information-set count
INFO_SETS = [183_000, 1_970_000, 84_000, 423_000]


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M".rstrip("0").rstrip(".")
    return f"{n / 1_000:.0f}K"


def main() -> None:
    apply_matplotlib_style()
    fig, (ax_win, ax_sets) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.35}
    )

    n = len(VERSION_LABELS)
    y = np.arange(n)

    # --- Panel A: win rate vs HeuristicAgent ---
    bar_height = 0.55
    for i in range(n):
        if WIN_RATE_POINTS[i] is not None:
            ax_win.barh(y[i], WIN_RATE_POINTS[i], color=NAVY, height=bar_height, zorder=3)
            ax_win.text(
                WIN_RATE_POINTS[i] + 1.0,
                y[i],
                f"{WIN_RATE_POINTS[i]:.1f}%",
                va="center",
                ha="left",
                color=INK,
                fontsize=11,
            )
        else:
            lo, hi = WIN_RATE_RANGE[i]
            mid = (lo + hi) / 2
            ax_win.barh(y[i], mid, color=NAVY_LIGHT, edgecolor=NAVY, height=bar_height, zorder=3)
            # Whisker showing the range
            ax_win.errorbar(
                [mid], [y[i]],
                xerr=[[mid - lo], [hi - mid]],
                fmt="none",
                ecolor=NAVY,
                elinewidth=1.4,
                capsize=4,
                capthick=1.4,
                zorder=4,
            )
            ax_win.text(
                hi + 1.0,
                y[i],
                f"{lo:.0f}–{hi:.0f}%",
                va="center",
                ha="left",
                color=INK,
                fontsize=11,
            )

    # 50% reference line
    ax_win.axvline(50, color=RULE, linestyle="--", linewidth=1.0, zorder=2)
    ax_win.text(
        50, -0.85, "50% parity",
        color=INK_SOFT, fontsize=10, ha="center", va="center",
    )

    ax_win.set_xlim(0, 65)
    ax_win.set_ylim(-1.1, n - 0.4)
    ax_win.set_yticks(y)
    ax_win.set_yticklabels(VERSION_LABELS)
    ax_win.invert_yaxis()
    ax_win.set_xlabel("Win rate vs HeuristicAgent (%)")
    ax_win.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_win.grid(axis="x", color=GRID, zorder=1)

    # Iteration footnote inside the bar area, right-aligned next to each label
    for i in range(n):
        ax_win.text(
            -1.5, y[i] + 0.32, ITER_LABELS[i],
            va="center", ha="right", color=INK_SOFT, fontsize=9,
        )

    # --- Panel B: information-set count, log-scale ---
    ax_sets.barh(y, INFO_SETS, color=NAVY_LIGHT, edgecolor=NAVY, height=bar_height, zorder=3)
    for i, count in enumerate(INFO_SETS):
        ax_sets.text(
            count * 1.08,
            y[i],
            _format_count(count),
            va="center",
            ha="left",
            color=INK,
            fontsize=11,
        )

    ax_sets.set_xscale("log")
    ax_sets.set_xlim(5e4, 1e7)
    ax_sets.set_ylim(-1.1, n - 0.4)
    ax_sets.set_yticks(y)
    ax_sets.set_yticklabels([])  # share with left panel
    ax_sets.invert_yaxis()
    ax_sets.set_xlabel("Information sets (log scale)")
    ax_sets.grid(axis="x", color=GRID, which="both", zorder=1)

    save_fig(fig, "slide_12_jornada_cfr")
    plt.close(fig)
    print("wrote slide_12_jornada_cfr.png")


if __name__ == "__main__":
    main()
