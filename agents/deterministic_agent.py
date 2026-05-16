"""
DeterministicAgent — third-party academic baseline.

Faithful port of the "Agente Determinista" specified in:

    Filevich, J. P. (2023).
    "Aproximacion de equilibrios de Nash en juegos de informacion
     imperfecta: el caso del Truco Uruguayo."
    Master's thesis, Facultad de Ingenieria, Universidad de la Republica
    (Udelar), Uruguay.
    Section 4.11.2 "Agente determinista", Algorithms 7, 9 and 10
    (pages 75-78).
    https://www.colibri.udelar.edu.uy/jspui/bitstream/20.500.12008/39789/1/Fi23.pdf

Filevich proposes this deterministic agent as the second baseline (after
the random agent) against which his CFR / Deep Monte Carlo agents are
validated. The agent is explicitly described by the author as a minimal
deterministic strategy intended to beat the random baseline with the
least possible effort, providing a reproducible reference between
independent investigations (Filevich, 2023, p. 78).

Port to Truco Paulista (1v1)
----------------------------
The original spec is written for the multiplayer Uruguayan Truco, which
includes the Envido and Flor side-games and uses the Truco / Retruco /
Vale-4 raise ladder. Truco Paulista does not include Envido or Flor and
uses a longer raise ladder (1 -> 3 -> 6 -> 9 -> 12), so the port is
restricted to:

  - the card-play routine (Algorithm 10, Filevich p. 68); and
  - the Truco-response routine (Algorithm 9, Filevich p. 68), with the
    "pieza" threshold and the bet ladder mapped to the project's
    STRONG_CARD_THRESHOLD and Truco Paulista raise levels respectively.

Mapping notes
-------------
- "Pieza" (a strong card in the Uruguayan classification: manilha or one
  of the special-rank cards) maps to "strong card" in this project, i.e.
  any card with strength >= STRONG_CARD_THRESHOLD (a "3" or a manilha).
- "Truco / Retruco / Vale-4" (1 / 2 / 3 piezas required to accept) maps
  to bet levels 3 / 6 / 9 / 12 (1 / 2 / 3 / 3 strong cards required,
  capped at three because a Truco Paulista hand has only three cards).
- The agent never proposes a raise and never re-raises (faithful to
  Filevich's Algorithm 7, in which Truco only appears on the response
  side: "if el truco acepta respuesta then jugar el truco").
- Mao-de-onze is a Truco-Paulista-specific subgame with no Uruguayan
  analogue. We apply the most conservative deterministic rule consistent
  with the agent's philosophy: accept only if the hand contains at least
  one strong card.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.card_utils import (
    STRONG_CARD_THRESHOLD,
    card_strength,
)


_BET_STRONG_REQUIREMENT: Dict[int, int] = {
    3: 1,
    6: 2,
    9: 3,
    12: 3,
}

_NEXT_BET_IN_LADDER: Dict[int, int] = {
    1: 3,
    3: 6,
    6: 9,
    9: 12,
}


class DeterministicAgent:
    """
    Deterministic Strategist baseline (Filevich, 2023, Sec. 4.11.2).

    Never bluffs, never proposes a raise. Card play follows Filevich's
    Algorithm 10 (three explicit branches keyed on whether we are
    currently winning the hand). Truco response follows Algorithm 9
    with the bet-level-dependent "pieza" threshold.
    """

    name: str = "Deterministic"

    def _winning_hand_so_far(
        self, round_winners: List[int], me: int
    ) -> Optional[bool]:
        """
        Inspect resolved sub-rounds and return the current player's
        status within the current hand.

        The Go engine encodes round_winners as a length-3 list with the
        winner of each sub-round (0 or 1) or -1 if the sub-round has
        not been resolved yet.

        Returns True if the current player has won more resolved
        sub-rounds than the opponent, False if they have lost more,
        and None for the tied/no-data case. Tied sub-rounds (encoded by
        the engine with the same logic that drives "parda" handling)
        contribute to neither count.
        """
        wins = 0
        losses = 0
        for w in round_winners:
            if w == me:
                wins += 1
            elif w == 1 - me and w >= 0:
                losses += 1
        if wins > losses:
            return True
        if losses > wins:
            return False
        return None

    def _opponent_card_in_round(
        self, table_cards: List[str]
    ) -> Optional[str]:
        for tc in table_cards:
            if tc and tc != "FACEDOWN" and not tc.startswith("FD:"):
                return tc
        return None

    def _pick_card(
        self,
        play_actions: List[int],
        hand: List[str],
        vira: str,
        table_cards: List[str],
        round_winners: List[int],
        me: int,
    ) -> int:
        strengths: Dict[int, int] = {
            a: card_strength(hand[a], vira) for a in play_actions
        }
        weakest = min(play_actions, key=lambda a: strengths[a])
        strongest = max(play_actions, key=lambda a: strengths[a])

        status = self._winning_hand_so_far(round_winners, me)

        # Branch 1 (Filevich Alg. 10, line 1): winning the hand so far,
        # conserve by throwing the lowest card.
        if status is True:
            return weakest

        opp_card = self._opponent_card_in_round(table_cards)

        # Branch 2 (Filevich Alg. 10, lines 2-6): losing or tied AND
        # the opponent has played in the current sub-round.
        if opp_card is not None:
            if strengths[strongest] > card_strength(opp_card, vira):
                return strongest
            return weakest

        # Branch 3 (Filevich Alg. 10, lines 7-8): no comparable opponent
        # card in the current sub-round; play the highest card.
        return strongest

    def _count_strong_cards(self, hand: List[str], vira: str) -> int:
        return sum(
            1
            for c in hand
            if c and card_strength(c, vira) >= STRONG_CARD_THRESHOLD
        )

    def _accept_truco(self, hand: List[str], vira: str, bet: int) -> bool:
        needed = _BET_STRONG_REQUIREMENT.get(bet, 3)
        return self._count_strong_cards(hand, vira) >= needed

    def act(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        legal_actions: List[int] = info["legal_actions"]
        hand: List[str] = state.get("hand", [])
        vira: str = state.get("vira", "")
        table_cards: List[str] = state.get("table_cards", [])
        round_winners: List[int] = state.get("round_winners", [-1, -1, -1])
        me: int = state.get("current_player", 0)

        play_actions: List[int] = [a for a in legal_actions if 0 <= a <= 2]

        if play_actions:
            return self._pick_card(
                play_actions, hand, vira, table_cards, round_winners, me
            )

        # Mao-de-onze: Truco-Paulista-specific; no Filevich analogue.
        # Apply the most conservative deterministic rule consistent with
        # the agent's philosophy.
        if state.get("waiting_for_mao_de_onze", False):
            if self._count_strong_cards(hand, vira) >= 1 and 4 in legal_actions:
                return 4
            return 5 if 5 in legal_actions else legal_actions[0]

        # Truco response (Filevich Alg. 9): the threshold is keyed on the
        # bet value under negotiation, i.e. the next ladder step above the
        # currently-accepted bet. The Truco-Paulista engine does not
        # expose PendingBet to the player view, but during a response
        # decision the pending value is always the next ladder step.
        current_bet = state.get("current_bet_value", 1)
        pending_bet = _NEXT_BET_IN_LADDER.get(current_bet, current_bet)
        if self._accept_truco(hand, vira, pending_bet) and 4 in legal_actions:
            return 4
        if 5 in legal_actions:
            return 5
        return legal_actions[0]
