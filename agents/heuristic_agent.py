import random
from typing import Any, Dict, List, Tuple

from agents.card_utils import STRONG_CARD_THRESHOLD, card_strength


class HeuristicAgent:
    """
    Rule-based agent for Truco Paulista.

    Strategy overview:
    - Card play: conserve strong cards when possible. Play weakest winning
      card, or sacrifice the weakest card when no card can win.
    - Bet requests: raise with 2+ strong cards (or any manilha), probability
      scales with hand quality.
    - Bet responses: accept with strong hands, re-raise with very strong
      hands, fold weak hands.
    - Mao-de-onze: accept with any strong card.
    """

    name: str = "HeuristicAgent"

    def _hand_quality(
        self, hand: List[str], vira: str
    ) -> Tuple[int, int, int]:
        """
        Evaluate hand quality.

        Returns
        -------
        Tuple[int, int, int]
            (num_strong, num_manilha, max_strength)
            strong = strength >= STRONG_CARD_THRESHOLD (3s and manilhas)
            manilha = strength >= 10
        """
        strengths = [card_strength(c, vira) for c in hand if c]
        if not strengths:
            return 0, 0, 0
        num_strong = sum(1 for s in strengths if s >= STRONG_CARD_THRESHOLD)
        num_manilha = sum(1 for s in strengths if s >= 10)
        return num_strong, num_manilha, max(strengths)

    def _pick_card(
        self,
        play_actions: List[int],
        hand: List[str],
        vira: str,
        table_cards: List[str],
    ) -> int:
        """
        Choose which card to play.

        When responding to an opponent's card on the table: play the weakest
        card that beats it (conserve strong cards). If no card beats it,
        sacrifice the weakest card.

        When leading (empty table): play a medium-strength card to probe.
        Save the strongest for later rounds.
        """
        strengths = {a: card_strength(hand[a], vira) for a in play_actions}

        # Determine opponent's card strength on the table (if any).
        opp_card = None
        for tc in table_cards:
            if tc and tc != "FACEDOWN":
                opp_card = tc
                break

        if opp_card is not None:
            opp_strength = card_strength(opp_card, vira)
            # Find weakest card that beats opponent.
            winners = [
                a for a in play_actions if strengths[a] > opp_strength
            ]
            if winners:
                return min(winners, key=lambda a: strengths[a])
            # No card beats opponent — sacrifice weakest.
            return min(play_actions, key=lambda a: strengths[a])

        # Leading: if only one card left, play it.
        if len(play_actions) == 1:
            return play_actions[0]

        # Leading with multiple cards: play the strongest to take the round.
        # Exception: in round 0 with 3 cards, lead with a medium card
        # to save the strongest for later.
        if len(play_actions) == 3:
            sorted_actions = sorted(play_actions, key=lambda a: strengths[a])
            # Play the middle card (not weakest, not strongest).
            return sorted_actions[1]

        return max(play_actions, key=lambda a: strengths[a])

    def act(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        legal_actions: List[int] = info["legal_actions"]
        hand: List[str] = state.get("hand", [])
        vira: str = state.get("vira", "")
        table_cards: List[str] = state.get("table_cards", [])

        play_actions: List[int] = [a for a in legal_actions if 0 <= a <= 2]
        is_bet_decision: bool = not play_actions

        num_strong, num_manilha, max_str = self._hand_quality(hand, vira)

        # --- Bet response (truco/raise or mao-de-onze) ---
        if is_bet_decision:
            is_mao = state.get("waiting_for_mao_de_onze", False)
            if is_mao:
                # Accept mao-de-onze with any strong card.
                if num_strong >= 1 and 4 in legal_actions:
                    return 4
                return 5 if 5 in legal_actions else random.choice(legal_actions)

            # Re-raise with very strong hand (2+ strong or any manilha).
            if 3 in legal_actions and (num_strong >= 2 or num_manilha >= 1):
                return 3

            # Accept with at least one strong card.
            if num_strong >= 1 and 4 in legal_actions:
                return 4

            return 5 if 5 in legal_actions else random.choice(legal_actions)

        # --- Proactive raise ---
        if 3 in legal_actions:
            # Raise with 2+ strong cards (70%) or manilha (80%).
            if num_manilha >= 1 and random.random() < 0.8:
                return 3
            if num_strong >= 2 and random.random() < 0.7:
                return 3

        # --- Card play ---
        return self._pick_card(play_actions, hand, vira, table_cards)
