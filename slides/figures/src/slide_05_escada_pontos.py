"""Slide 5 — Truco point-stake ladder (1 -> 3 -> 6 -> 9 -> 12).

Five horizontally-arranged boxes connected by directed edges. The first
four boxes share a soft-navy fill; the last box (12 points, "max stake")
is filled solid navy with white text to mark it as the terminal level.

Labels are in English to match the rest of the figure set (intended for
reuse in the thesis).
"""

from __future__ import annotations

from pathlib import Path

import graphviz

from _style import (
    NAVY,
    NAVY_LIGHT,
    INK,
    INK_SOFT,
    OUT_DIR,
    FONT_FAMILY,
    TICK_PT,
)


STAGES = [
    ("base", "Base hand\n1 point", False),
    ("truco", "Truco\n3 points", False),
    ("six", "Six\n6 points", False),
    ("nine", "Nine\n9 points", False),
    ("twelve", "Twelve\n12 points\n(max stake)", True),
]


def main() -> None:
    g = graphviz.Digraph(
        "escada",
        format="png",
        graph_attr={
            "rankdir": "LR",
            "bgcolor": "transparent",
            "ranksep": "0.45",
            "nodesep": "0.35",
            "margin": "0.10",
        },
        node_attr={
            "shape": "box",
            "style": "filled,rounded",
            "fontname": FONT_FAMILY,
            "fontsize": str(TICK_PT + 1),
            "color": NAVY,
            "penwidth": "1.4",
            "margin": "0.18,0.12",
        },
        edge_attr={
            "color": INK_SOFT,
            "penwidth": "1.4",
            "arrowsize": "0.8",
            "fontname": FONT_FAMILY,
            "fontsize": str(TICK_PT - 1),
            "fontcolor": INK_SOFT,
        },
    )

    for node_id, label, is_terminal in STAGES:
        if is_terminal:
            g.node(
                node_id, label,
                fillcolor=NAVY,
                fontcolor="white",
                color=NAVY,
            )
        else:
            g.node(
                node_id, label,
                fillcolor=NAVY_LIGHT,
                fontcolor=INK,
            )

    for (src, _, _), (dst, _, _) in zip(STAGES, STAGES[1:]):
        g.edge(src, dst, label="raise")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = OUT_DIR / "slide_05_escada_pontos"
    g.attr(dpi="200")
    g.render(filename=str(out_stem), cleanup=True)
    # graphviz renders to <stem>.png; we keep the same file name.
    print(f"wrote {out_stem}.png")


if __name__ == "__main__":
    main()
