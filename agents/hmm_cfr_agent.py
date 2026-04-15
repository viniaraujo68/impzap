"""
HMM+CFR combined agent for Truco Paulista (1v1).

Dispatch rules:
  Opponent hasn't raised yet:
                Passive exploit mode — HMM explicit raise/fold logic applied
                from game start. AlwaysFold-type opponents are exploited from
                the first hand (no cold-start delay). This is the highest-value
                default: an opponent that folds to every raise is maximally
                exploited before any inference is needed.
  Opponent has raised at least once (or probe window expired):
                HMM base strategy with CFR reserved for Bluffing exploitation.
                HMM neutral (heuristic-style) outperforms CFR vs balanced
                opponents (Heuristic: ~50% vs ~46%). CFR is only called when
                Bluffing state is confidently detected (confidence >= 0.45),
                where the Bluffing reweight (call x2.0, fold x0.5) improves
                on HMM's simpler x2 call boost.

  Passive confirmation (fold-rate):
                After the initial 3-hand window, Passive exploit continues
                only if fold_rate >= 0.70 over at least 2 raise hands.
                Single-fold confirmation was a false positive: Heuristic
                folds ~25% of hands when raised, locking the agent into
                Passive exploit vs a non-Passive opponent.
                AlwaysFold: fold_rate=1.0 → confirmed immediately.
                Heuristic: fold_rate≈0.25 → never confirmed → HMM neutral.

Rationale for the flipped default:
  The previous architecture defaulted to CFR and switched to Passive exploit
  after enough FOLD observations accumulated. But FOLD observations require
  raising, and CFR's Nash strategy raises ~15% of hands. This created a
  feedback loop: CFR doesn't raise → AlwaysFold never folds → no FOLD
  observations → dispatch never fires. Benchmarks showed HMM+CFR at 85%
  vs AlwaysFold while standalone HMM achieved 96% and "always exploit"
  reached 98.5%.

  Flipping the default to Passive exploit breaks the loop. The cost vs
  non-Passive opponents is bounded: the switch to CFR fires after the
  opponent's first raise, typically by hand 2-5 depending on opponent type.
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


class HMMCFRAgent:
    """
    Combined HMM opponent-modeling + CFR strategy agent.

    The default policy is Passive exploitation: raises aggressively from
    game start. Once the opponent raises (demonstrating they are not
    AlwaysFold-class), the agent permanently switches to CFR with
    HMM-guided reweighting.
    """

    name: str = "HMM+CFR"

    def __init__(
        self,
        cfr_model_path: str = "models/cfr_v8_fullbucket_2M.json.gz",
        perspective: int = 0,
        adapt: bool = True,
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
        # Set permanently to True the first time OBS_RAISE_WIN or
        # OBS_RAISE_LOSS is observed. Triggers switch from Passive exploit
        # to CFR dispatch.
        self._opp_has_raised: bool = False
        # Fold-rate tracking for Passive confirmation. A single OBS_FOLD is
        # unreliable (Heuristic folds ~25% of hands when raised). We require
        # fold_rate >= _PASSIVE_FOLD_RATE_THRESHOLD over at least
        # _PASSIVE_MIN_RAISE_HANDS to confirm a Passive archetype.
        # AlwaysFold: fold_rate=1.0 → confirmed after 2 hands.
        # Heuristic: fold_rate≈0.25 → never confirmed → switches to CFR.
        self._raise_hands: int = 0   # hands in which we raised
        self._fold_hands: int = 0    # OBS_FOLD outcomes (opponent folded our raise)
        # Number of completed hands in the current game.
        self._hands_played: int = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset between games. HMM belief reset; CFR table unchanged."""
        self._model.reset()
        self._reset_hand_tracking()
        self._prev_score = None
        self._opp_has_raised = False
        self._raise_hands = 0
        self._fold_hands = 0
        self._hands_played = 0

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

    def _act_hmm(
        self,
        state: Dict[str, Any],
        info: Dict[str, Any],
        opp_state: int,
        exploiting: bool,
    ) -> int:
        """
        HMM action selection for a given opponent state and exploiting flag.
        Delegates to HMMAgent's action helpers, bypassing CFR entirely.
        Used both for Passive exploit (opp_state=STATE_PASSIVE, exploiting=True)
        and for neutral/Aggressive fallback (where HMM neutral outperforms CFR
        against balanced opponents like Heuristic).
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
                opp_state, exploiting,
            )

        if 3 in legal_actions:
            raise_action = self._hmm_delegate._maybe_raise(
                num_strong, num_manilha, opp_state, exploiting,
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

        # Fold-rate thresholds for Passive archetype confirmation.
        # AlwaysFold fold_rate=1.0 → confirmed after _MIN_RAISE_HANDS.
        # Heuristic fold_rate≈0.25 → never confirmed → exits to CFR.
        _PASSIVE_EXPLOIT_WINDOW = 3
        _PASSIVE_MIN_RAISE_HANDS = 2
        _PASSIVE_FOLD_RATE_THRESHOLD = 0.70

        # Detect hand boundaries and update HMM belief.
        obs = self._detect_hand_end(state, info)
        if obs is not None:
            self._hands_played += 1
            if obs in (OBS_RAISE_WIN, OBS_RAISE_LOSS):
                self._opp_has_raised = True
            if self._we_raised:
                self._raise_hands += 1
                if obs == OBS_FOLD:
                    self._fold_hands += 1
            self._model.update(obs)
            self._reset_hand_tracking()

        # Track opponent behavior within the current hand.
        self._track_opponent_action(state, info)

        # Passive confirmation via fold-rate (requires _MIN_RAISE_HANDS raise
        # hands with fold_rate >= threshold). Single-fold confirmation was a
        # false positive: Heuristic folds ~25% of hands when raised, which
        # was incorrectly locking the agent into Passive exploit mode.
        passive_confirmed = (
            self._raise_hands >= _PASSIVE_MIN_RAISE_HANDS
            and (self._fold_hands / self._raise_hands) >= _PASSIVE_FOLD_RATE_THRESHOLD
        )

        # Passive exploit conditions:
        # - Opponent hasn't raised (this hand or previously)
        # - AND: either within the initial window (first 3 hands) where we
        #   probe aggressively to generate fold-rate evidence, or fold-rate
        #   has confirmed a genuine Passive archetype.
        use_passive_exploit = (
            not self._opp_has_raised
            and not self._opponent_raised
            and (self._hands_played < _PASSIVE_EXPLOIT_WINDOW or passive_confirmed)
        )

        opp_state, confidence = self._model.dominant_state()
        exploiting = confidence >= CONFIDENCE_THRESHOLD

        if use_passive_exploit:
            # Probe or confirmed Passive: raise aggressively via HMM Passive mode.
            action = self._act_hmm(state, info, STATE_PASSIVE, True)
        elif opp_state == STATE_BLUFFING and exploiting:
            # CFR with Bluffing reweight outperforms HMM's simple x2 call boost
            # because it integrates the reweight into the full strategy distribution.
            action = self._act_cfr(state, info, opp_state, exploiting)
        else:
            # HMM neutral (Aggressive or low confidence): heuristic-style play.
            # HMM neutral outperforms CFR Nash vs balanced opponents (Heuristic:
            # ~50% vs ~46%), and handles Passive residual if misclassified.
            action = self._act_hmm(state, info, opp_state, exploiting)

        if action == 3:
            self._we_raised = True
        return action
