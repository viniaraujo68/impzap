"""
HMM+CFR combined agent for Truco Paulista (1v1).

Uses CFR average strategy (Nash equilibrium approximation) as the base
policy and exploits opponent behavioral state inferred by an HMM.

Dispatch rules (applied only when HMM confidence >= 0.45):
  vs Passive  : hybrid dispatch — HMM explicit raise/fold logic replaces
                CFR sampling. CFR Nash assigns low raise probability from
                self-play; multiplicative reweighting cannot overcome this.
                HMM explicit logic raises at 60-90% vs Passive, matching
                the standalone HMM's exploitation ceiling.
  vs Bluffing : CFR base strategy with multiplicative reweight (call x2.0,
                fold x0.5). CFR already assigns reasonable call probability
                so the tilt is effective here.
  vs Aggressive / low confidence : pure CFR.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from agents.cfr_agent import CFRAgent, _build_action_maps, _get_hand_strengths_view
from agents.hmm_agent import (
    HMMAgent,
    HMMModel,
    STATE_AGGRESSIVE,
    STATE_PASSIVE,
    STATE_BLUFFING,
    CONFIDENCE_THRESHOLD,
    OBS_FOLD,
    OBS_PASSIVE_WIN,
    OBS_PASSIVE_LOSS,
    OBS_RAISE_WIN,
    OBS_RAISE_LOSS,
)


# ---------------------------------------------------------------------------
# Reweighting factors for Bluffing state (abstract action IDs)
# abstract 3 = raise, 4 = accept/call, 5 = fold
# ---------------------------------------------------------------------------
_CALL_BOOST_VS_BLUFFING: float = 2.0
_FOLD_REDUCTION_VS_BLUFFING: float = 0.5

# Passive dispatch uses the same threshold as the Bluffing reweight.
# A single FOLD observation moves Passive belief to ~0.62, above this
# threshold, so the dispatch fires reliably from hand 2 onward vs pure
# Passive opponents. The small cost vs mixed opponents (e.g. Random) is
# acceptable given the large gain vs exploitable Passive archetypes.
_PASSIVE_DISPATCH_THRESHOLD: float = 0.45


class HMMCFRAgent:
    """
    Combined HMM opponent-modeling + CFR strategy agent.

    The CFR table is the base policy. The HMM infers opponent behavioral
    state (Aggressive / Passive / Bluffing) from per-hand observations.

    vs Passive: delegates raise/fold decisions to HMMAgent's explicit
    threshold logic, which is calibrated to exploit passive behavior at
    60-90% raise rates. CFR handles card play.

    vs Bluffing: stays on CFR with a call-boost / fold-reduction reweight.

    vs Aggressive / low confidence: pure CFR.
    """

    name: str = "HMM+CFR"

    def __init__(
        self,
        cfr_model_path: str = "models/cfr_v4_8buck_11M.json.gz",
        perspective: int = 0,
        adapt: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        cfr_model_path : str
            Path to the gzip-compressed CFR table produced by Go training.
        perspective : int
            Which player this agent controls (0 or 1).
        adapt : bool
            Whether the HMM emission matrix adapts online.
        """
        self._perspective = perspective
        self._cfr = CFRAgent()
        self._cfr.load(cfr_model_path)
        self._model = HMMModel(adapt=adapt)
        # HMMAgent instance used only as a stateless delegate for its
        # explicit action-selection helpers (_maybe_raise, _bet_response,
        # _pick_card, _hand_quality). Tracking and HMM updates are managed
        # by this class.
        self._hmm_delegate = HMMAgent(perspective=perspective, adapt=False)

        # Per-hand tracking.
        self._opponent_raised: bool = False
        self._we_raised: bool = False
        self._prev_score: Optional[List[int]] = None
        self._prev_bet: int = 1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset between games. HMM belief reset; CFR table unchanged."""
        self._model.reset()
        self._reset_hand_tracking()
        self._prev_score = None

    def _reset_hand_tracking(self) -> None:
        self._opponent_raised = False
        self._we_raised = False
        self._prev_bet = 1

    # ------------------------------------------------------------------
    # Hand boundary detection
    # ------------------------------------------------------------------

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
        my_score_delta = (
            current_score[self._perspective] - self._prev_score[self._perspective]
        )
        opp_score_delta = current_score[opp] - self._prev_score[opp]

        if my_score_delta == 0 and opp_score_delta == 0:
            return None

        self._prev_score = list(current_score)

        opponent_folded = (
            my_score_delta > 0 and self._we_raised and not self._opponent_raised
        )
        opp_won = opp_score_delta > 0

        if opponent_folded:
            return OBS_FOLD
        if self._opponent_raised:
            return OBS_RAISE_WIN if opp_won else OBS_RAISE_LOSS
        return OBS_PASSIVE_WIN if opp_won else OBS_PASSIVE_LOSS

    def _track_opponent_action(
        self, state: Dict[str, Any], info: Dict[str, Any]
    ) -> None:
        """Track whether the opponent raised during the current hand."""
        current_bet = state.get("current_bet_value", 1)
        waiting_for_bet = (
            4 in info.get("legal_actions", [])
            and not state.get("waiting_for_mao_de_onze", False)
        )
        if waiting_for_bet and current_bet >= self._prev_bet:
            self._opponent_raised = True
        self._prev_bet = current_bet

    # ------------------------------------------------------------------
    # Action selection paths
    # ------------------------------------------------------------------

    def _act_passive_exploit(
        self, state: Dict[str, Any], info: Dict[str, Any]
    ) -> int:
        """
        Full HMM explicit exploitation vs Passive opponent.
        Delegates raise/fold/card decisions to HMMAgent's helper methods,
        which are calibrated to exploit Passive behavior aggressively.
        CFR is bypassed entirely for this decision.
        """
        legal_actions: List[int] = info["legal_actions"]
        hand: List[str] = state.get("hand", [])
        vira: str = state.get("vira", "")
        table_cards: List[str] = state.get("table_cards", [])
        play_actions = [a for a in legal_actions if 0 <= a <= 2]
        is_bet_decision = not play_actions

        num_strong, num_manilha, max_str = self._hmm_delegate._hand_quality(hand, vira)

        if is_bet_decision:
            return self._hmm_delegate._bet_response(
                legal_actions, state, num_strong, num_manilha, max_str,
                STATE_PASSIVE, True,
            )

        if 3 in legal_actions:
            raise_action = self._hmm_delegate._maybe_raise(
                num_strong, num_manilha, STATE_PASSIVE, True,
            )
            if raise_action is not None:
                return raise_action

        return self._hmm_delegate._pick_card(play_actions, hand, vira, table_cards)

    def _act_cfr(
        self,
        state: Dict[str, Any],
        info: Dict[str, Any],
        opp_state: int,
        exploiting: bool,
    ) -> int:
        """
        CFR-based action selection with optional Bluffing reweight.
        """
        legal_actions: List[int] = info["legal_actions"]
        hand_strengths = _get_hand_strengths_view(state)
        _, a2r, abstract_actions = _build_action_maps(legal_actions, hand_strengths)

        info_key = CFRAgent._info_set_key_from_view(state, info)
        strategy = self._cfr._get_average_strategy(info_key, abstract_actions)

        if exploiting and opp_state == STATE_BLUFFING:
            strategy = self._reweight_bluffing(strategy, abstract_actions)

        abs_action = random.choices(
            abstract_actions,
            weights=[strategy[a] for a in abstract_actions],
        )[0]
        return a2r[abs_action]

    def _reweight_bluffing(
        self,
        strategy: Dict[int, float],
        abstract_actions: List[int],
    ) -> Dict[int, float]:
        """Call boost + fold reduction vs Bluffing, then renormalize."""
        weights = {a: strategy[a] for a in abstract_actions}
        if 4 in weights:
            weights[4] *= _CALL_BOOST_VS_BLUFFING
        if 5 in weights:
            weights[5] *= _FOLD_REDUCTION_VS_BLUFFING
        total = sum(weights.values())
        if total <= 0:
            return strategy
        return {a: weights[a] / total for a in abstract_actions}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def act(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select an action using the hybrid HMM+CFR dispatch.

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

        opp_state, confidence = self._model.dominant_state()
        exploiting = confidence >= CONFIDENCE_THRESHOLD

        if opp_state == STATE_PASSIVE and confidence >= _PASSIVE_DISPATCH_THRESHOLD:
            action = self._act_passive_exploit(state, info)
        else:
            action = self._act_cfr(state, info, opp_state, exploiting)

        if action == 3:
            self._we_raised = True
        return action
