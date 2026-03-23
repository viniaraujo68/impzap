import random
from typing import Any, Dict, List

from agents.card_utils import STRONG_CARD_THRESHOLD, card_strength


class HeuristicAgent:
    """
    Rule-based agent for Truco Paulista.

    Bet decisions: accept when holding a strong card (manilha or 3), fold
    otherwise. Card play: always plays the strongest card face-up. Requests
    truco with 50% probability when holding a strong card.
    """

    def _has_strong_card(self, hand: List[str], vira: str) -> bool:
        """Return True if the hand contains at least one manilha or a 3."""
        return any(card_strength(c, vira) >= STRONG_CARD_THRESHOLD for c in hand)

    def act(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select an action given the current observable state.

        Bet response is inferred from legal_actions: if the only options are
        accept (4) and fold (5) — with an optional raise (3) — the agent is
        in a bet-response situation. This covers both mao-de-onze and truco
        responses without relying on state keys absent from the Go View.

        Parameters
        ----------
        state : Dict[str, Any]
            Observable state dict from TrucoEnv.
        info : Dict[str, Any]
            Info dict from TrucoEnv. Must contain 'legal_actions'.

        Returns
        -------
        int
            A legal action integer.
        """
        legal_actions: List[int] = info["legal_actions"]
        hand: List[str] = state.get("hand", [])
        vira: str = state.get("vira", "")

        # Detect bet-response or mao-de-onze situations by checking whether
        # there are no play actions (0-2) in the legal set.
        play_actions: List[int] = [a for a in legal_actions if 0 <= a <= 2]
        is_bet_decision: bool = not play_actions

        if is_bet_decision:
            if self._has_strong_card(hand, vira) and 4 in legal_actions:
                return 4
            return 5 if 5 in legal_actions else random.choice(legal_actions)

        if 3 in legal_actions and self._has_strong_card(hand, vira):
            if random.random() > 0.5:
                return 3

        return max(play_actions, key=lambda a: card_strength(hand[a], vira))
