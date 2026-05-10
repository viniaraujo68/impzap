"""Slide 13 — CFR v9 (6M iterations) head-to-head against each opponent.

Vertical bar chart with the four canonical opponents on the x-axis and the
v9 win rate on the y-axis. A horizontal reference line marks the 50%
parity threshold.

Numeric values come from algorithms.md and the thesis Final Benchmark
Results section.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import (
    NAVY,
    INK,
    INK_SOFT,
    RULE,
    GRID,
    apply_matplotlib_style,
    save_fig,
)


OPPONENTS = ["HeuristicAgent", "MCTS", "REINFORCE", "RandomAgent"]
WIN_RATES = [50.1, 61.3, 75.2, 88.5]
SAMPLES = ["n=5000", "n=300", "n=500", "n=5000"]


def main() -> None:
    apply_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    x = np.arange(len(OPPONENTS))
    bar_width = 0.55

    bars = ax.bar(x, WIN_RATES, width=bar_width, color=NAVY, zorder=3)

    # 50% parity reference line
    ax.axhline(50, color=RULE, linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        len(OPPONENTS) - 0.35,
        51.5,
        "50% parity",
        color=INK_SOFT,
        fontsize=10,
        ha="right",
    )

    # Numeric annotation above each bar
    for xi, rate, sample in zip(x, WIN_RATES, SAMPLES):
        ax.text(
            xi,
            rate + 1.5,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            xi,
            rate + 7.0,
            sample,
            ha="center",
            va="bottom",
            color=INK_SOFT,
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(OPPONENTS)
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", color=GRID, zorder=1)

    save_fig(fig, "slide_13_cfr_v9_oponentes")
    plt.close(fig)
    print("wrote slide_13_cfr_v9_oponentes.png")


if __name__ == "__main__":
    main()
