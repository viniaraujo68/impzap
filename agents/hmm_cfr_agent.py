"""
HMM+CFR combined agent for Truco Paulista (1v1).

Uses CFR average strategy (Nash equilibrium approximation) as the base
policy and tilts the action distribution based on opponent behavioral
state inferred by an HMM. The CFR strategy anchors decisions; the HMM
layer exploits detectable opponent tendencies without abandoning the
game-theoretic foundation.

Reweighting rules (applied only when HMM confidence >= 0.45):
  vs Passive : raise x2.0, fold x0.5  -- they fold to raises
  vs Bluffing: call  x2.0, fold x0.5  -- their raises are weak
  vs Aggressive: no adjustment         -- too balanced to exploit
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from agents.cfr_agent import CFRAgent, _build_action_maps, _get_hand_strengths_view
from agents.hmm_agent import (
    HMMModel,
    STATE_AGGRESSIVE,
    STATE_PASSIVE,
    STATE_BLUFFING,
    CONFIDENCE_THRESHOLD,
    OBS_FOLD,
    OBS_PASSIVE_LOSS,
    OBS_PASSIVE_WIN,
    OBS_RAISE_WIN,
    OBS_RAISE_LOSS,
)


# ---------------------------------------------------------------------------
# Reweighting factors (applied to abstract action IDs)
# abstract 3 = raise, 4 = accept/call, 5 = fold
# ---------------------------------------------------------------------------
_RAISE_BOOST_VS_PASSIVE: float = 2.0
_FOLD_REDUCTION_VS_PASSIVE: float = 0.5
_CALL_BOOST_VS_PASSIVE: float = 2.0

_CALL_BOOST_VS_BLUFFING: float = 2.0
_FOLD_REDUCTION_VS_BLUFFING: float = 0.5


class HMMCFRAgent:
    """
    Combined HMM opponent-modeling + CFR strategy agent.

    The CFR table provides the base mixed strategy for each info set.
    The HMM infers opponent behavioral state (Aggressive / Passive /
    Bluffing) from per-hand observations. When confidence is sufficient,
    the CFR probability vector is reweighted to exploit the inferred
    tendency before sampling.
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

        # Per-hand tracking (mirrors HMMAgent).
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
    # Hand boundary detection (identical to HMMAgent)
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
    # CFR strategy lookup
    # ------------------------------------------------------------------

    def _get_cfr_strategy(
        self,
        state: Dict[str, Any],
        info: Dict[str, Any],
        abstract_actions: List[int],
    ) -> Dict[int, float]:
        """Return the CFR average strategy over abstract actions."""
        info_key = CFRAgent._info_set_key_from_view(state, info)
        return self._cfr._get_average_strategy(info_key, abstract_actions)

    # ------------------------------------------------------------------
    # Reweighting
    # ------------------------------------------------------------------

    def _reweight(
        self,
        strategy: Dict[int, float],
        abstract_actions: List[int],
        opp_state: int,
    ) -> Dict[int, float]:
        """
        Apply opponent-state-conditioned reweighting to the CFR strategy.
        Only abstract actions 3 (raise), 4 (call), 5 (fold) are modified;
        play actions are left unchanged.
        """
        weights = {a: strategy[a] for a in abstract_actions}

        if opp_state == STATE_PASSIVE:
            if 3 in weights:
                weights[3] *= _RAISE_BOOST_VS_PASSIVE
            if 4 in weights:
                weights[4] *= _CALL_BOOST_VS_PASSIVE
            if 5 in weights:
                weights[5] *= _FOLD_REDUCTION_VS_PASSIVE

        elif opp_state == STATE_BLUFFING:
            if 4 in weights:
                weights[4] *= _CALL_BOOST_VS_BLUFFING
            if 5 in weights:
                weights[5] *= _FOLD_REDUCTION_VS_BLUFFING

        # STATE_AGGRESSIVE: no adjustment

        total = sum(weights.values())
        if total <= 0:
            return strategy
        return {a: weights[a] / total for a in abstract_actions}

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select an action by sampling from the (possibly reweighted) CFR
        average strategy.

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

        # Build abstract action mapping.
        hand_strengths = _get_hand_strengths_view(state)
        _, a2r, abstract_actions = _build_action_maps(legal_actions, hand_strengths)

        # CFR base strategy.
        strategy = self._get_cfr_strategy(state, info, abstract_actions)

        # Reweight based on inferred opponent state.
        opp_state, confidence = self._model.dominant_state()
        if confidence >= CONFIDENCE_THRESHOLD and opp_state != STATE_AGGRESSIVE:
            strategy = self._reweight(strategy, abstract_actions, opp_state)

        # Sample and map back to real action.
        abs_action = random.choices(
            abstract_actions,
            weights=[strategy[a] for a in abstract_actions],
        )[0]

        real_action = a2r[abs_action]
        if real_action == 3:
            self._we_raised = True
        return real_action
