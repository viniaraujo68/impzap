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
    completed hand using the forward algorithm. Emission and transition
    matrices are calibrated from match data and optionally adapt online
    via exponential moving average soft counts.
    """

    # Online adaptation rate for the dominant state's emission row.
    # Only the highest-responsibility state adapts, preventing row collapse.
    ADAPT_RATE = 0.003
    # Minimum emission probability — prevents any observation from being
    # driven to zero, preserving state distinguishability.
    EMISSION_FLOOR = 0.02
    # Minimum responsibility to trigger adaptation for a state.
    ADAPT_RESPONSIBILITY_MIN = 0.50

    def __init__(self, adapt: bool = True) -> None:
        self._adapt = adapt

        # Initial state distribution.
        self.pi = np.array([0.40, 0.40, 0.20])

        # Transition matrix A[i][j] = P(state_j | state_i).
        self.transition = np.array([
            [0.70, 0.15, 0.15],  # Aggressive -> ...
            [0.15, 0.70, 0.15],  # Passive -> ...
            [0.25, 0.25, 0.50],  # Bluffing -> ...
        ])

        # Emission matrix B[s][o] = P(observation_o | state_s).
        # Calibrated from 2000-game tournaments against known agents.
        # Rows maintain clear separation on the key distinguishing signals:
        # - Aggressive: moderate FOLD (they do fold weak hands), near-zero
        #   R_LOSS (disciplined — only raise with real cards)
        # - Passive: high FOLD, very low raise frequency
        # - Bluffing: low FOLD, high raise frequency, high R_LOSS
        #              FOLD   P_LOSS  P_WIN  R_WIN  R_LOSS
        self.emission = np.array([
            [0.20, 0.27, 0.30, 0.20, 0.03],  # Aggressive
            [0.40, 0.25, 0.20, 0.10, 0.05],  # Passive
            [0.08, 0.15, 0.12, 0.35, 0.30],  # Bluffing
        ])

        self.belief: np.ndarray = self.pi.copy()
        self._prev_belief: np.ndarray = self.pi.copy()

    def reset(self) -> None:
        """Reset belief to prior distribution (between games).
        Matrices are NOT reset — adaptation carries across games."""
        self.belief = self.pi.copy()
        self._prev_belief = self.pi.copy()

    def update(self, observation: int) -> None:
        """
        Bayesian belief update after observing one hand outcome.
        new_belief[s] = B[s][obs] * sum_over_s'(belief[s'] * A[s'][s])
        Then normalize. Optionally adapts emission and transition matrices.
        """
        predicted = self.belief @ self.transition  # shape (3,)
        likelihood = self.emission[:, observation]  # shape (3,)
        unnormalized = likelihood * predicted
        total = unnormalized.sum()
        if total > 0:
            new_belief = unnormalized / total
        else:
            new_belief = self.pi.copy()

        if self._adapt:
            self._adapt_matrices(new_belief, observation)

        self._prev_belief = self.belief.copy()
        self.belief = new_belief

    def _adapt_matrices(
        self, new_belief: np.ndarray, observation: int
    ) -> None:
        """
        Online adaptation of emission and transition matrices.

        Only the dominant state (highest responsibility, above threshold)
        has its emission row adapted. Non-dominant rows stay fixed, which
        prevents all rows from collapsing toward the same observed
        distribution when playing a single opponent type.

        A minimum emission floor ensures no observation probability is
        driven to zero, preserving the model's ability to distinguish states.
        """
        alpha = self.ADAPT_RATE
        dominant = int(np.argmax(new_belief))

        # Only adapt emission for the dominant state.
        if new_belief[dominant] >= self.ADAPT_RESPONSIBILITY_MIN:
            target = np.zeros(NUM_OBS)
            target[observation] = 1.0
            self.emission[dominant] = (
                (1.0 - alpha) * self.emission[dominant]
                + alpha * target
            )
            # Apply floor and re-normalize.
            self.emission[dominant] = np.maximum(
                self.emission[dominant], self.EMISSION_FLOOR
            )
            self.emission[dominant] /= self.emission[dominant].sum()

        # Adapt transition: only the dominant previous state's row.
        prev_dominant = int(np.argmax(self._prev_belief))
        if self._prev_belief[prev_dominant] >= self.ADAPT_RESPONSIBILITY_MIN:
            target_trans = new_belief / new_belief.sum()
            self.transition[prev_dominant] = (
                (1.0 - alpha) * self.transition[prev_dominant]
                + alpha * target_trans
            )
            self.transition[prev_dominant] /= self.transition[prev_dominant].sum()

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

    def __init__(self, perspective: int = 0, adapt: bool = False) -> None:
        """
        Parameters
        ----------
        perspective : int
            Which player this agent controls (0 or 1). Used to determine
            which reward signal indicates hand outcome.
        adapt : bool
            Whether the HMM matrices adapt online during play.
        """
        self._perspective = perspective
        self._model = HMMModel(adapt=adapt)

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
                legal_actions, state, num_strong, num_manilha, max_str,
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
        max_str: int,
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
                # Opponent raises with weak hands but never folds our
                # re-raises. Keep re-raise threshold strict (same as
                # neutral), but widen calling range since their raise
                # doesn't signal real strength.
                if 3 in legal_actions and (num_strong >= 2 or num_manilha >= 1):
                    return 3
                if 4 in legal_actions and (num_strong >= 1 or max_str >= 6):
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
                # Aggressive but disciplined (R_LOSS ~1%). Their strategy is
                # too balanced to exploit at the raise/fold level — fall
                # through to neutral play below.
                pass

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
                # Against bluffers, don't need to raise proactively as much
                # since they'll raise themselves. Fall through to neutral.
                pass

            if opp_state == STATE_AGGRESSIVE:
                # Aggressive but disciplined — fall through to neutral play.
                pass

        # Neutral — standard heuristic thresholds.
        if num_manilha >= 1 and random.random() < 0.80:
            return 3
        if num_strong >= 2 and random.random() < 0.70:
            return 3
        return None
