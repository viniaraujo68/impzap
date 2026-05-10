"""Slide 6 — three-layer system architecture diagram.

Vertical stack of three clusters:
  - Python (top, navy): environment / agents / training
  - CGO bridge (middle, gold): typed FFI bindings
  - Go (bottom, green): rules engine, simulator, CFR traversal

Inter-cluster edges:
  - Solid: function/method calls (descend the stack)
  - Dashed: data flowing back up (states, payoffs, tensors)

All labels in English for thesis reuse.
"""

from __future__ import annotations

import graphviz

from _style import (
    NAVY,
    NAVY_LIGHT,
    GOLD,
    GOLD_LIGHT,
    GREEN,
    GREEN_LIGHT,
    INK,
    INK_SOFT,
    OUT_DIR,
    FONT_FAMILY,
    TICK_PT,
)


def main() -> None:
    g = graphviz.Digraph(
        "arch",
        format="png",
        graph_attr={
            "bgcolor": "transparent",
            "rankdir": "TB",
            "compound": "true",
            "splines": "ortho",
            "ranksep": "0.55",
            "nodesep": "0.40",
            "margin": "0.10",
        },
        node_attr={
            "shape": "box",
            "style": "filled,rounded",
            "fontname": FONT_FAMILY,
            "fontsize": str(TICK_PT),
            "penwidth": "1.3",
            "margin": "0.15,0.10",
        },
        edge_attr={
            "fontname": FONT_FAMILY,
            "fontsize": str(TICK_PT - 2),
            "fontcolor": INK_SOFT,
            "color": INK_SOFT,
            "penwidth": "1.2",
            "arrowsize": "0.7",
        },
    )

    # ------------------------------------------------------------------
    # Python cluster (top)
    # ------------------------------------------------------------------
    with g.subgraph(name="cluster_python") as c:
        c.attr(
            label="Python layer — agents and training",
            style="rounded,filled",
            color=NAVY,
            fillcolor=NAVY_LIGHT,
            fontname=FONT_FAMILY,
            fontsize=str(TICK_PT + 1),
            fontcolor=NAVY,
            penwidth="1.3",
            margin="14",
        )
        c.node(
            "py_env",
            "Gymnasium environment\n164-d observation · 9 actions",
            fillcolor="white", color=NAVY, fontcolor=INK,
        )
        c.node(
            "py_agents",
            "Agents\nREINFORCE · MCTS · CFR\nHMM · HMM+CFR",
            fillcolor="white", color=NAVY, fontcolor=INK,
        )
        c.node(
            "py_train",
            "Training and benchmark\ntournaments · metrics · logging",
            fillcolor="white", color=NAVY, fontcolor=INK,
        )

    # ------------------------------------------------------------------
    # Bridge cluster (middle)
    # ------------------------------------------------------------------
    with g.subgraph(name="cluster_bridge") as c:
        c.attr(
            label="CGO / ctypes bridge",
            style="rounded,filled",
            color=GOLD,
            fillcolor=GOLD_LIGHT,
            fontname=FONT_FAMILY,
            fontsize=str(TICK_PT + 1),
            fontcolor=GOLD,
            penwidth="1.3",
            margin="14",
        )
        c.node(
            "bridge",
            "Typed bindings\nstate · action · reward",
            fillcolor="white", color=GOLD, fontcolor=INK,
        )

    # ------------------------------------------------------------------
    # Go cluster (bottom)
    # ------------------------------------------------------------------
    with g.subgraph(name="cluster_go") as c:
        c.attr(
            label="Go layer — high-performance core",
            style="rounded,filled",
            color=GREEN,
            fillcolor=GREEN_LIGHT,
            fontname=FONT_FAMILY,
            fontsize=str(TICK_PT + 1),
            fontcolor=GREEN,
            penwidth="1.3",
            margin="14",
        )
        c.node(
            "go_rules",
            "Truco Paulista\nrules engine",
            fillcolor="white", color=GREEN, fontcolor=INK,
        )
        c.node(
            "go_sim",
            "Simulator / rollouts\n~98 µs per rollout",
            fillcolor="white", color=GREEN, fontcolor=INK,
        )
        c.node(
            "go_cfr",
            "Tree traversal\nExternal Sampling CFR",
            fillcolor="white", color=GREEN, fontcolor=INK,
        )

    # ------------------------------------------------------------------
    # Inter-cluster connections
    # ------------------------------------------------------------------
    g.edge("py_agents", "bridge", label="FFI calls")
    g.edge("bridge", "go_rules", label="native primitives")
    g.edge("go_sim", "bridge", style="dashed", label="state / payoff")
    g.edge("bridge", "py_train", style="dashed", label="tensors / dicts")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = OUT_DIR / "slide_06_arquitetura"
    g.attr(dpi="200")
    g.render(filename=str(out_stem), cleanup=True)
    print(f"wrote {out_stem}.png")


if __name__ == "__main__":
    main()
