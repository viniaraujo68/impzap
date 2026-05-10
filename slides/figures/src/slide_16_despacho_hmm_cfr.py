"""Slide 16 — HMM+CFR dispatch flowchart.

Top-down flow:
  1. Start (oval, navy fill, white text)
  2. Update HMM belief (process)
  3. Decision: dominant state with sufficient confidence?
       - No / ambiguous     -> Neutral mode
       - Aggressive (p>=tau) -> Neutral mode
       - Bluffing (p>=tau)   -> Bluff-aware CFR
       - Passive (p>=tau)    -> Probe gate
  4. Decision: probe window confirms fold-rate?
       - Yes                 -> Exploit mode
       - No / inconsistent   -> Neutral mode
  5. Three terminal modes converge into:
     "Action distribution + illegal-action mask" -> Sample action -> End

All labels in English for thesis reuse.
"""

from __future__ import annotations

import graphviz

from _style import (
    NAVY,
    NAVY_LIGHT,
    GOLD,
    GOLD_LIGHT,
    GREEN_LIGHT,
    RED_LIGHT,
    INK,
    INK_SOFT,
    OUT_DIR,
    FONT_FAMILY,
    TICK_PT,
)


def main() -> None:
    g = graphviz.Digraph(
        "dispatch",
        format="png",
        graph_attr={
            "bgcolor": "transparent",
            "rankdir": "TB",
            "ranksep": "0.45",
            "nodesep": "0.35",
            "margin": "0.10",
            "splines": "ortho",
        },
        node_attr={
            "fontname": FONT_FAMILY,
            "fontsize": str(TICK_PT),
            "penwidth": "1.3",
            "margin": "0.18,0.10",
        },
        edge_attr={
            "fontname": FONT_FAMILY,
            "fontsize": str(TICK_PT - 2),
            "fontcolor": INK_SOFT,
            "color": INK_SOFT,
            "arrowsize": "0.7",
            "penwidth": "1.1",
        },
    )

    # --- Terminals ---
    g.node(
        "start",
        "Decision step",
        shape="oval", style="filled", fillcolor=NAVY,
        fontcolor="white", color=NAVY,
    )
    g.node(
        "end",
        "Execute action",
        shape="oval", style="filled", fillcolor=NAVY,
        fontcolor="white", color=NAVY,
    )

    # --- Process steps (rounded rectangles, white) ---
    g.node(
        "belief",
        "Update HMM belief\n(forward algorithm)",
        shape="box", style="filled,rounded",
        fillcolor="white", color=INK_SOFT, fontcolor=INK,
    )

    # --- Decisions (diamonds, gold) ---
    g.node(
        "dec_state",
        "Dominant state\nwith confidence ≥ τ ?",
        shape="diamond", style="filled",
        fillcolor=GOLD_LIGHT, color=GOLD, fontcolor=INK,
    )
    g.node(
        "dec_probe",
        "Probe window:\n3 hands confirm\nhigh fold-rate?",
        shape="diamond", style="filled",
        fillcolor=GOLD_LIGHT, color=GOLD, fontcolor=INK,
    )

    # --- Mode boxes ---
    g.node(
        "neutral",
        "Neutral mode\nHMM heuristic policy",
        shape="box", style="filled,rounded",
        fillcolor=NAVY_LIGHT, color=NAVY, fontcolor=INK,
    )
    g.node(
        "bluff_cfr",
        "Bluff-aware CFR\nreweight: call ×2.0, fold ×0.5",
        shape="box", style="filled,rounded",
        fillcolor=GREEN_LIGHT, color=NAVY, fontcolor=INK,
    )
    g.node(
        "exploit",
        "Exploit mode\naggressive HMM raise/fold",
        shape="box", style="filled,rounded",
        fillcolor=RED_LIGHT, color=NAVY, fontcolor=INK,
    )

    # --- Convergence node ---
    g.node(
        "merge",
        "Action distribution\n+ illegal-action mask\n+ sampling",
        shape="box", style="filled,rounded",
        fillcolor="white", color=INK_SOFT, fontcolor=INK,
    )

    # --- Edges ---
    g.edge("start", "belief")
    g.edge("belief", "dec_state")

    g.edge("dec_state", "neutral", label="None / ambiguous")
    g.edge("dec_state", "neutral", label="Aggressive (p ≥ τ)")
    g.edge("dec_state", "bluff_cfr", label="Bluffing (p ≥ τ)")
    g.edge("dec_state", "dec_probe", label="Passive (p ≥ τ)")

    g.edge("dec_probe", "exploit", label="Yes")
    g.edge("dec_probe", "neutral", label="No / inconsistent")

    g.edge("neutral", "merge")
    g.edge("bluff_cfr", "merge")
    g.edge("exploit", "merge")
    g.edge("merge", "end")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = OUT_DIR / "slide_16_despacho_hmm_cfr"
    g.attr(dpi="200")
    g.render(filename=str(out_stem), cleanup=True)
    print(f"wrote {out_stem}.png")


if __name__ == "__main__":
    main()
