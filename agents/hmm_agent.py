"""
HMM (Hidden Markov Model) opponent modeling agent for Truco Paulista (1v1).

Standalone agent that infers opponent behavioral state (Aggressive, Passive,
Bluffing) using a 3-state HMM with per-hand observations, then selects
actions to exploit the inferred tendency.

Observation signals are extracted from opponent actions within each hand:
whether they raised, folded, and whether they won or lost. The belief
vector is updated after each completed hand using the forward algorithm.

Action selection modifies heuristic-style thresholds based on the dominant
inferred opponent state and belief confidence.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.card_utils import STRONG_CARD_THRESHOLD, card_strength


# ---------------------------------------------------------------------------
# Hidden states
# ---------------------------------------------------------------------------
STATE_AGGRESSIVE = 0
STATE_PASSIVE = 1
STATE_BLUFFING = 2
NUM_STATES = 3

# ---------------------------------------------------------------------------
# Observation symbols (per completed hand)
# ---------------------------------------------------------------------------
OBS_FOLD = 0         # Opponent folded to a raise
OBS_PASSIVE_LOSS = 1 # Opponent never raised, lost
OBS_PASSIVE_WIN = 2  # Opponent never raised, won
OBS_RAISE_WIN = 3    # Opponent raised, won
OBS_RAISE_LOSS = 4   # Opponent raised, lost (bluff signal)
NUM_OBS = 5

# Minimum confidence to exploit (below this, play neutral heuristic)
CONFIDENCE_THRESHOLD = 0.45


class HMMModel:
    """
    3-state HMM for opponent behavior inference.

    Maintains a belief vector over hidden states and updates it after each
    completed hand using the forward algorithm with hand-tuned transition
    and emission matrices.
    """

    def __init__(self) -> None:
        # Initial state distribution.
        self.pi = np.array([0.40, 0.40, 0.20])

        # Transition matrix A[i][j] = P(state_j | state_i).
        self.transition = np.array([
            [0.70, 0.15, 0.15],  # Aggressive -> ...
            [0.15, 0.70, 0.15],  # Passive -> ...
            [0.25, 0.25, 0.50],  # Bluffing -> ...
        ])

        # Emission matrix B[s][o] = P(observation_o | state_s).
        #              FOLD  P_LOSS P_WIN  R_WIN  R_LOSS
        self.emission = np.array([
            [0.05, 0.10, 0.15, 0.45, 0.25],  # Aggressive
            [0.30, 0.30, 0.25, 0.10, 0.05],  # Passive
            [0.10, 0.10, 0.10, 0.30, 0.40],  # Bluffing
        ])

        self.belief: np.ndarray = self.pi.copy()

    def reset(self) -> None:
        """Reset belief to prior distribution (between games)."""
        self.belief = self.pi.copy()

    def update(self, observation: int) -> None:
        """
        Bayesian belief update after observing one hand outcome.
        new_belief[s] = B[s][obs] * sum_over_s'(belief[s'] * A[s'][s])
        Then normalize.
        """
        predicted = self.belief @ self.transition  # shape (3,)
        likelihood = self.emission[:, observation]  # shape (3,)
        unnormalized = likelihood * predicted
        total = unnormalized.sum()
        if total > 0:
            self.belief = unnormalized / total
        else:
            self.belief = self.pi.copy()

    def dominant_state(self) -> Tuple[int, float]:
        """Return (state_id, confidence) for the most likely hidden state."""
        state_id = int(np.argmax(self.belief))
        return state_id, float(self.belief[state_id])


class HMMAgent:
    """
    Opponent-modeling agent using a Hidden Markov Model.

    Tracks opponent behavior hand-by-hand, infers their behavioral mode,
    and adapts action selection to exploit detected tendencies.
    """

    name: str = "HMM"

    def __init__(self, perspective: int = 0) -> None:
        """
        Parameters
        ----------
        perspective : int
            Which player this agent controls (0 or 1). Used to determine
            which reward signal indicates hand outcome.
        """
        self._perspective = perspective
        self._model = HMMModel()

        # Per-hand tracking.
        self._opponent_raised = False
        self._we_raised = False
        self._prev_score: Optional[List[int]] = None
        self._prev_bet: int = 1

    def reset(self) -> None:
        """Reset between games. Clears belief and hand tracking."""
        self._model.reset()
        self._reset_hand_tracking()
        self._prev_score = None

    def _reset_hand_tracking(self) -> None:
        self._opponent_raised = False
        self._we_raised = False
        self._prev_bet = 1

    def _hand_quality(
        self, hand: List[str], vira: str
    ) -> Tuple[int, int, int]:
        """
        Evaluate hand quality.
        Returns (num_strong, num_manilha, max_strength).
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
        Card selection logic. Identical to HeuristicAgent — the HMM's
        exploitation happens at the raise/call/fold level, not card play.
        """
        strengths = {a: card_strength(hand[a], vira) for a in play_actions}

        opp_card = None
        for tc in table_cards:
            if tc and tc != "FACEDOWN":
                opp_card = tc
                break

        if opp_card is not None:
            opp_strength = card_strength(opp_card, vira)
            winners = [
                a for a in play_actions if strengths[a] > opp_strength
            ]
            if winners:
                return min(winners, key=lambda a: strengths[a])
            return min(play_actions, key=lambda a: strengths[a])

        if len(play_actions) == 1:
            return play_actions[0]

        if len(play_actions) == 3:
            sorted_actions = sorted(play_actions, key=lambda a: strengths[a])
            return sorted_actions[1]

        return max(play_actions, key=lambda a: strengths[a])

    def _detect_hand_end(
        self, state: Dict[str, Any], info: Dict[str, Any]
    ) -> Optional[int]:
        """
        Detect whether a hand just ended by comparing scores to previous.
        Returns the observation ID if a hand ended, None otherwise.
        """
        current_score = state.get("score", [0, 0])
        if self._prev_score is None:
            self._prev_score = list(current_score)
            return None

        opp = 1 - self._perspective
        my_score_delta = current_score[self._perspective] - self._prev_score[self._perspective]
        opp_score_delta = current_score[opp] - self._prev_score[opp]

        if my_score_delta == 0 and opp_score_delta == 0:
            return None

        # A hand ended. Determine observation.
        self._prev_score = list(current_score)

        # Detect fold: we gained points and we had raised (opponent folded
        # to our raise). Played cards will be sparse but we can't reliably
        # count them from the View, so use the raise tracking.
        opponent_folded = my_score_delta > 0 and self._we_raised and not self._opponent_raised
        opp_won = opp_score_delta > 0

        if opponent_folded:
            obs = OBS_FOLD
        elif self._opponent_raised:
            obs = OBS_RAISE_WIN if opp_won else OBS_RAISE_LOSS
        else:
            obs = OBS_PASSIVE_WIN if opp_won else OBS_PASSIVE_LOSS

        return obs

    def _track_opponent_action(
        self, state: Dict[str, Any], info: Dict[str, Any]
    ) -> None:
        """
        Infer what the opponent did since our last call. We check if the
        bet level changed (opponent raised) or if we're now facing a bet
        decision (opponent raised into us).
        """
        current_bet = state.get("current_bet_value", 1)
        waiting_for_bet = (
            4 in info.get("legal_actions", [])
            and not state.get("waiting_for_mao_de_onze", False)
        )

        # If we're being asked to accept/fold, opponent raised.
        if waiting_for_bet and current_bet >= self._prev_bet:
            self._opponent_raised = True

        self._prev_bet = current_bet

    def act(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select an action using opponent-model-informed heuristics.

        Parameters
        ----------
        state : Dict[str, Any]
            Observable state dict from TrucoEnv (View format).
        info : Dict[str, Any]
            Info dict with 'legal_actions'.

        Returns
        -------
        int
            A legal action.
        """
        legal_actions: List[int] = info["legal_actions"]
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Detect hand boundaries and update HMM belief.
        obs = self._detect_hand_end(state, info)
        if obs is not None:
            self._model.update(obs)
            self._reset_hand_tracking()

        # Track opponent behavior within the current hand.
        self._track_opponent_action(state, info)

        hand: List[str] = state.get("hand", [])
        vira: str = state.get("vira", "")
        table_cards: List[str] = state.get("table_cards", [])

        play_actions: List[int] = [a for a in legal_actions if 0 <= a <= 2]
        is_bet_decision: bool = not play_actions

        num_strong, num_manilha, max_str = self._hand_quality(hand, vira)

        opp_state, confidence = self._model.dominant_state()
        exploiting = confidence >= CONFIDENCE_THRESHOLD

        # --- Bet response (truco/raise or mao-de-onze) ---
        if is_bet_decision:
            action = self._bet_response(
                legal_actions, state, num_strong, num_manilha,
                opp_state, exploiting,
            )
            if action == 3:
                self._we_raised = True
            return action

        # --- Proactive raise ---
        if 3 in legal_actions:
            raise_action = self._maybe_raise(
                num_strong, num_manilha, opp_state, exploiting,
            )
            if raise_action is not None:
                self._we_raised = True
                return raise_action

        # --- Card play ---
        return self._pick_card(play_actions, hand, vira, table_cards)

    def _bet_response(
        self,
        legal_actions: List[int],
        state: Dict[str, Any],
        num_strong: int,
        num_manilha: int,
        opp_state: int,
        exploiting: bool,
    ) -> int:
        """Handle bet response decisions, adjusted by opponent model."""
        is_mao = state.get("waiting_for_mao_de_onze", False)
        if is_mao:
            if num_strong >= 1 and 4 in legal_actions:
                return 4
            return 5 if 5 in legal_actions else random.choice(legal_actions)

        if exploiting:
            if opp_state == STATE_BLUFFING:
                # Opponent likely bluffing — call with weaker hands,
                # re-raise with anything decent.
                if 3 in legal_actions and num_strong >= 1:
                    return 3
                if 4 in legal_actions:
                    return 4
                return 5 if 5 in legal_actions else random.choice(legal_actions)

            if opp_state == STATE_PASSIVE:
                # Passive opponent rarely raises — when they do, respect it.
                # Only call with genuinely strong hands.
                if 3 in legal_actions and (num_strong >= 2 and num_manilha >= 1):
                    return 3
                if num_strong >= 2 and 4 in legal_actions:
                    return 4
                return 5 if 5 in legal_actions else random.choice(legal_actions)

            if opp_state == STATE_AGGRESSIVE:
                # Aggressive opponent raises often — their raises are less
                # informative. Call with decent hands, re-raise with strong.
                if 3 in legal_actions and (num_strong >= 2 or num_manilha >= 1):
                    return 3
                if num_strong >= 1 and 4 in legal_actions:
                    return 4
                return 5 if 5 in legal_actions else random.choice(legal_actions)

        # Neutral / low confidence — standard heuristic.
        if 3 in legal_actions and (num_strong >= 2 or num_manilha >= 1):
            return 3
        if num_strong >= 1 and 4 in legal_actions:
            return 4
        return 5 if 5 in legal_actions else random.choice(legal_actions)

    def _maybe_raise(
        self,
        num_strong: int,
        num_manilha: int,
        opp_state: int,
        exploiting: bool,
    ) -> Optional[int]:
        """Decide whether to proactively raise, adjusted by opponent model."""
        if exploiting:
            if opp_state == STATE_PASSIVE:
                # Passive opponents fold easily — raise more aggressively.
                if num_manilha >= 1 and random.random() < 0.90:
                    return 3
                if num_strong >= 1 and random.random() < 0.60:
                    return 3
                # Bluff raise with medium hands against passive opponents.
                if random.random() < 0.20:
                    return 3
                return None

            if opp_state == STATE_BLUFFING:
                # Don't raise into a bluffer — let them raise, then call.
                # Only raise with genuinely strong hands.
                if num_manilha >= 1 and random.random() < 0.70:
                    return 3
                if num_strong >= 2 and random.random() < 0.50:
                    return 3
                return None

            if opp_state == STATE_AGGRESSIVE:
                # Against aggressive, raise with strong hands (they'll call).
                if num_manilha >= 1 and random.random() < 0.85:
                    return 3
                if num_strong >= 2 and random.random() < 0.75:
                    return 3
                return None

        # Neutral — standard heuristic thresholds.
        if num_manilha >= 1 and random.random() < 0.80:
            return 3
        if num_strong >= 2 and random.random() < 0.70:
            return 3
        return None
