"""
AlwaysRaiseAgent — extreme Bluffing archetype for benchmarking.

Raises whenever a raise option is available. Accepts all incoming raises
(never folds). Plays the strongest legal card otherwise.
Used to isolate the HMM's Bluffing-state exploitation signal.
"""

from __future__ import annotations

from typing import Any, Dict, List


class AlwaysRaiseAgent:
    """
    Raises on every turn where action 3 is legal.
    Accepts all raises (action 4). Accepts mao-de-onze.
    Plays the strongest card on card-play turns.
    """

    name: str = "AlwaysRaise"

    def act(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        legal_actions: List[int] = info["legal_actions"]

        # Always raise when possible (action 3 = truco/six/nine/twelve).
        if 3 in legal_actions:
            return 3

        # Accept all raises and mao-de-onze (action 4).
        if 4 in legal_actions:
            return 4

        # Card play: choose the strongest face-up card.
        play_actions = [a for a in legal_actions if 0 <= a <= 2]
        if play_actions:
            hand: List[str] = state.get("hand", [])
            vira: str = state.get("vira", "")
            from agents.card_utils import card_strength
            return max(play_actions, key=lambda a: card_strength(hand[a], vira))

        return legal_actions[0]
