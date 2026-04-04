"""
AlwaysFoldAgent — extreme Passive archetype for benchmarking.

Folds whenever a fold option is available. Never raises proactively.
Plays the weakest legal card otherwise.
Used to isolate the HMM's Passive-state exploitation signal.
"""

from __future__ import annotations

from typing import Any, Dict, List


class AlwaysFoldAgent:
    """
    Folds to every raise (action 5 when available).
    Refuses mao-de-onze. Never initiates a raise.
    Plays the weakest card on card-play turns.
    """

    name: str = "AlwaysFold"

    def act(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        legal_actions: List[int] = info["legal_actions"]

        # Fold / refuse mao-de-onze whenever available.
        if 5 in legal_actions:
            return 5

        # Card play: choose the weakest face-up card.
        play_actions = [a for a in legal_actions if 0 <= a <= 2]
        if play_actions:
            hand: List[str] = state.get("hand", [])
            vira: str = state.get("vira", "")
            from agents.card_utils import card_strength
            return min(play_actions, key=lambda a: card_strength(hand[a], vira))

        # Fallback: accept (only reachable if the only options are accept/raise
        # with no fold available — should not happen in normal Truco rules).
        return legal_actions[0]
