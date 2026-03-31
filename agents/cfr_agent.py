"""
Counterfactual Regret Minimization (CFR) agent for Truco Paulista (1v1).

Algorithm: External Sampling CFR (Chance Sampling variant).
- Traversing player's turns: iterate over all legal actions, recurse, accumulate regrets.
- Opponent's turns: sample one action from the current strategy, recurse.
- Chance nodes (card deals): handled by sampling a fresh game per iteration.

Converges to a Nash Equilibrium strategy through self-play.
Uses TrucoEnv.step_from_state() for all state transitions (stateless, no global mutation).
"""

from __future__ import annotations

import gzip
import json
import pickle
import random
import sys
import time
from typing import Any, Dict, List, Optional

from agents.card_utils import RANKS, SUITS, card_strength, go_to_card
from truco_env.env import TrucoEnv

# Ensure recursion limit is high enough for deep game trees.
sys.setrecursionlimit(10_000)

_BET_LADDER: List[int] = [1, 3, 6, 9, 12]


def _go_card_to_str(card: Dict[str, Any], observer_owns: bool) -> str:
    """
    Convert a Go card dict to a canonical string from the observer's POV.
    If the card is facedown and the observer does NOT own it, return "FACEDOWN".
    If the card is facedown but the observer owns it, show the actual card.
    """
    if card.get("facedown", False) and not observer_owns:
        return "FACEDOWN"
    return f"{RANKS[card['rank']]}_{SUITS[card['suit']]}"


def _strength_bucket(strength: int) -> int:
    """
    Bucket card strength into 8 categories for finer card-play distinctions.
    -1 = facedown/unknown,
     0 = weak trash (4,5 — strength 0-1),
     1 = strong trash (6,7 — strength 2-3),
     2 = low (Q,J — strength 4-5),
     3 = mid (K — strength 6),
     4 = mid-high (A — strength 7),
     5 = high (2 — strength 8),
     6 = top (3 — strength 9),
     7 = manilha (strength 10+).
    """
    if strength < 0:
        return -1
    if strength <= 1:
        return 0
    if strength <= 3:
        return 1
    if strength <= 5:
        return 2
    if strength == 6:
        return 3
    if strength == 7:
        return 4
    if strength == 8:
        return 5
    if strength == 9:
        return 6
    return 7




def _build_action_maps(
    legal_actions: List[int],
    hand_strengths: List[int],
) -> tuple:
    """
    Build bidirectional maps between real actions and abstract rank-ordered
    actions. Play actions (0-2 face-up, 6-8 face-down) are remapped so that
    abstract action 0 = play weakest, 1 = play middle, 2 = play strongest.
    Non-play actions (3=raise, 4=accept, 5=fold) keep their identity.

    Parameters
    ----------
    legal_actions : List[int]
        Legal actions from the engine (real action IDs).
    hand_strengths : List[int]
        Card strength for each hand index (hand[0], hand[1], hand[2]).

    Returns
    -------
    (real_to_abstract, abstract_to_real, abstract_actions)
        real_to_abstract: dict mapping real action -> abstract action
        abstract_to_real: dict mapping abstract action -> real action
        abstract_actions: list of abstract actions (same length as legal_actions)
    """
    # Rank hand indices by strength: rank 0 = weakest, rank N = strongest.
    indexed = list(enumerate(hand_strengths))
    indexed.sort(key=lambda x: x[1])
    # index_to_rank[hand_idx] = strength rank (0=weakest)
    index_to_rank = {}
    for rank, (idx, _) in enumerate(indexed):
        index_to_rank[idx] = rank

    real_to_abstract: Dict[int, int] = {}
    abstract_to_real: Dict[int, int] = {}

    for a in legal_actions:
        if 0 <= a <= 2:
            abstract_a = index_to_rank.get(a, a)
            real_to_abstract[a] = abstract_a
            abstract_to_real[abstract_a] = a
        elif 6 <= a <= 8:
            hand_idx = a - 6
            abstract_a = 6 + index_to_rank.get(hand_idx, hand_idx)
            real_to_abstract[a] = abstract_a
            abstract_to_real[abstract_a] = a
        else:
            # Non-play actions stay the same.
            real_to_abstract[a] = a
            abstract_to_real[a] = a

    abstract_actions = [real_to_abstract[a] for a in legal_actions]
    return real_to_abstract, abstract_to_real, abstract_actions


def _get_hand_strengths_full(
    state: Dict[str, Any], player: int
) -> List[int]:
    """Get card strengths for each hand index from the full GameState."""
    vira_str = go_to_card(state["vira"])
    return [
        card_strength(_go_card_to_str(c, observer_owns=True), vira_str)
        for c in state["hands"][player]
    ]


def _get_hand_strengths_view(state: Dict[str, Any]) -> List[int]:
    """Get card strengths for each hand index from the View state."""
    vira = state.get("vira", "")
    hand = state.get("hand", [])
    return [card_strength(c, vira) for c in hand]


class CFRAgent:
    """
    External Sampling CFR agent for Truco Paulista.

    Maintains cumulative regret and cumulative strategy tables keyed by
    information set strings. At play time, uses the average strategy
    (converged Nash Equilibrium approximation).
    """

    name: str = "CFR"

    def __init__(self, env: Optional[TrucoEnv] = None) -> None:
        self._env: Optional[TrucoEnv] = env
        self.regret_sum: Dict[str, Dict[int, float]] = {}
        self.strategy_sum: Dict[str, Dict[int, float]] = {}
        self._iterations: int = 0

    # ------------------------------------------------------------------
    # Strategy computation
    # ------------------------------------------------------------------

    def _get_strategy(
        self, info_key: str, legal_actions: List[int]
    ) -> Dict[int, float]:
        """Current strategy via regret matching."""
        regrets = self.regret_sum.get(info_key, {})
        positive = {a: max(regrets.get(a, 0.0), 0.0) for a in legal_actions}
        total = sum(positive.values())
        if total > 0:
            return {a: positive[a] / total for a in legal_actions}
        return {a: 1.0 / len(legal_actions) for a in legal_actions}

    def _get_average_strategy(
        self, info_key: str, legal_actions: List[int]
    ) -> Dict[int, float]:
        """Average strategy (converged output)."""
        sums = self.strategy_sum.get(info_key, {})
        total = sum(sums.get(a, 0.0) for a in legal_actions)
        if total > 0:
            return {a: sums.get(a, 0.0) / total for a in legal_actions}
        return {a: 1.0 / len(legal_actions) for a in legal_actions}

    # ------------------------------------------------------------------
    # Information set key (full GameState — training)
    # ------------------------------------------------------------------

    @staticmethod
    def _info_set_key(state: Dict[str, Any], player: int) -> str:
        """
        Build an information set key from the full GameState for the given
        player. Uses bucketed card strength abstraction to keep the info set
        space tractable. Produces the same key format as _info_set_key_from_view.
        """
        vira_str = go_to_card(state["vira"])

        # Player's hand as sorted bucketed strength values.
        hand_buckets = sorted(
            _strength_bucket(card_strength(
                _go_card_to_str(c, observer_owns=True), vira_str
            ))
            for c in state["hands"][player]
        )

        # Table cards as bucketed strength tuple.
        table_buckets = tuple(
            _strength_bucket(card_strength(go_to_card(c), vira_str))
            for c in state.get("table_cards", [])
        )

        # Played cards from round_history, bucketed.
        round_starters = state.get("round_starter", [-1, -1, -1])
        round_history_raw = state.get("round_history", [[], [], []])
        played_buckets: List[int] = []
        for r_idx, rnd in enumerate(round_history_raw):
            starter = round_starters[r_idx]
            for c_idx, card in enumerate(rnd):
                owner = starter if c_idx == 0 else 1 - starter
                card_str = _go_card_to_str(
                    card, observer_owns=(owner == player)
                )
                played_buckets.append(
                    _strength_bucket(card_strength(card_str, vira_str))
                )

        current_bet = state.get("current_bet_value", 1)
        pending_bet = state.get("pending_bet", 0)
        current_round = state.get("current_round", 0)

        info_tuple = (
            tuple(hand_buckets),
            table_buckets,
            tuple(played_buckets),
            current_bet,
            pending_bet,
            current_round,
        )
        return str(info_tuple)

    # ------------------------------------------------------------------
    # Information set key (View state — play time)
    # ------------------------------------------------------------------

    @staticmethod
    def _info_set_key_from_view(
        state: Dict[str, Any], info: Dict[str, Any]
    ) -> str:
        """
        Build an info set key from a View state (play time).
        Uses the same bucketed card strength abstraction as _info_set_key.
        """
        vira = state.get("vira", "")
        hand = state.get("hand", [])
        hand_buckets = sorted(
            _strength_bucket(card_strength(c, vira)) for c in hand if c
        )

        # Filter empty strings: the View pads table_cards to 2 entries.
        table_cards_raw = [c for c in state.get("table_cards", []) if c]
        table_buckets = tuple(
            _strength_bucket(card_strength(c, vira)) for c in table_cards_raw
        )

        played_cards = state.get("played_cards", [])
        played_buckets = tuple(
            _strength_bucket(card_strength(c, vira)) for c in played_cards
        )

        current_bet = state.get("current_bet_value", 1)

        # Infer pending_bet from legal actions and bet state.
        pending_bet = 0
        waiting_mao = state.get("waiting_for_mao_de_onze", False)
        legal_actions = info.get("legal_actions", [])
        waiting_for_bet = (
            (4 in legal_actions or 5 in legal_actions) and not waiting_mao
        )
        if waiting_for_bet:
            for i, val in enumerate(_BET_LADDER):
                if val == current_bet and i < len(_BET_LADDER) - 1:
                    pending_bet = _BET_LADDER[i + 1]
                    break

        current_round = len(played_cards) // 2

        info_tuple = (
            tuple(hand_buckets),
            table_buckets,
            played_buckets,
            current_bet,
            pending_bet,
            current_round,
        )
        return str(info_tuple)

    # ------------------------------------------------------------------
    # CFR tree traversal
    # ------------------------------------------------------------------

    def _cfr(
        self,
        state: Dict[str, Any],
        traversing_player: int,
        reach_p0: float,
        reach_p1: float,
    ) -> float:
        """
        External sampling CFR traversal over a single hand.
        Returns utility for the traversing player.
        Stops when a hand resolves (non-zero reward) or game terminates.
        """
        reward = state.get("reward", [0.0, 0.0])
        has_reward = reward[0] != 0.0 or reward[1] != 0.0
        is_terminal = state.get("is_terminal", False)

        if is_terminal or has_reward:
            return float(reward[traversing_player])

        current_player: int = state["current_player"]
        legal_actions: List[int] = state["legal_actions"]

        if not legal_actions:
            return 0.0

        # Build abstract action mapping (rank-ordered play actions).
        hand_strengths = _get_hand_strengths_full(state, current_player)
        r2a, a2r, abstract_actions = _build_action_maps(
            legal_actions, hand_strengths
        )

        info_key = self._info_set_key(state, current_player)
        strategy = self._get_strategy(info_key, abstract_actions)

        if current_player == traversing_player:
            # Traverse all actions for the traversing player.
            # Regret pruning: skip actions with very negative cumulative regret
            # (they've been proven bad). Always keep at least one action.
            PRUNE_THRESHOLD = -300.0
            regrets = self.regret_sum.get(info_key, {})
            explore_actions = [
                a for a in abstract_actions
                if regrets.get(a, 0.0) > PRUNE_THRESHOLD
            ]
            if not explore_actions:
                explore_actions = abstract_actions

            action_values: Dict[int, float] = {}
            for abs_a in explore_actions:
                real_a = a2r[abs_a]
                next_state = self._env.step_from_state(state, real_a)
                if current_player == 0:
                    action_values[abs_a] = self._cfr(
                        next_state, traversing_player,
                        reach_p0 * strategy[abs_a], reach_p1,
                    )
                else:
                    action_values[abs_a] = self._cfr(
                        next_state, traversing_player,
                        reach_p0, reach_p1 * strategy[abs_a],
                    )
            # Pruned actions get node_value as their counterfactual value
            # (equivalent to assuming they perform at average).

            # Node value uses explored actions weighted by strategy.
            # Pruned actions contribute strategy weight * node_value (neutral).
            explored_value = sum(
                strategy[a] * action_values[a] for a in explore_actions
            )
            explored_weight = sum(strategy[a] for a in explore_actions)
            if explored_weight > 0 and explored_weight < 1.0:
                node_value = explored_value / explored_weight
            else:
                node_value = explored_value

            # Update regrets weighted by opponent reach probability.
            opponent_reach = reach_p1 if current_player == 0 else reach_p0
            if info_key not in self.regret_sum:
                self.regret_sum[info_key] = {}
            for abs_a in explore_actions:
                regret = opponent_reach * (action_values[abs_a] - node_value)
                self.regret_sum[info_key][abs_a] = (
                    self.regret_sum[info_key].get(abs_a, 0.0) + regret
                )

            return node_value

        else:
            # Opponent node: sample one action, update strategy sum.
            my_reach = reach_p0 if current_player == 0 else reach_p1
            if info_key not in self.strategy_sum:
                self.strategy_sum[info_key] = {}
            for a in abstract_actions:
                self.strategy_sum[info_key][a] = (
                    self.strategy_sum[info_key].get(a, 0.0)
                    + my_reach * strategy[a]
                )

            abs_action = random.choices(
                abstract_actions,
                weights=[strategy[a] for a in abstract_actions],
            )[0]
            real_action = a2r[abs_action]
            next_state = self._env.step_from_state(state, real_action)
            if current_player == 0:
                return self._cfr(
                    next_state, traversing_player,
                    reach_p0 * strategy[abs_action], reach_p1,
                )
            return self._cfr(
                next_state, traversing_player,
                reach_p0, reach_p1 * strategy[abs_action],
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, num_iterations: int = 10_000) -> None:
        """
        Run CFR training for num_iterations. Each iteration traverses twice:
        once as player 0 and once as player 1.
        """
        if self._env is None:
            raise RuntimeError("CFRAgent requires a TrucoEnv for training.")

        log_interval = max(1, num_iterations // 20)
        start_time = time.time()

        for i in range(1, num_iterations + 1):
            # Sample a fresh game (chance node) — one hand traversal.
            state = self._env.init_game_full()

            # Traverse as player 0, then as player 1.
            self._cfr(state, 0, 1.0, 1.0)
            self._cfr(state, 1, 1.0, 1.0)

            self._iterations += 1

            if i % log_interval == 0:
                elapsed = time.time() - start_time
                info_sets = len(self.regret_sum)
                print(
                    f"[CFR] Iteration {i}/{num_iterations} "
                    f"({elapsed:.1f}s) | Info sets: {info_sets}"
                )

        elapsed = time.time() - start_time
        print(
            f"[CFR] Training complete. {num_iterations} iterations in "
            f"{elapsed:.1f}s. Info sets: {len(self.regret_sum)}"
        )

    # ------------------------------------------------------------------
    # Play interface
    # ------------------------------------------------------------------

    def act(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Select an action using the converged average strategy.

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

        hand_strengths = _get_hand_strengths_view(state)
        _, a2r, abstract_actions = _build_action_maps(
            legal_actions, hand_strengths
        )

        info_key = self._info_set_key_from_view(state, info)
        strategy = self._get_average_strategy(info_key, abstract_actions)

        abs_action = random.choices(
            abstract_actions,
            weights=[strategy[a] for a in abstract_actions],
        )[0]
        return a2r[abs_action]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save regret and strategy tables to a pickle file."""
        data = {
            "regret_sum": self.regret_sum,
            "strategy_sum": self.strategy_sum,
            "iterations": self._iterations,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[CFR] Saved to {filepath} ({len(self.regret_sum)} info sets)")

    def load(self, filepath: str) -> None:
        """Load regret and strategy tables from pickle or gzip JSON."""
        if filepath.endswith(".gz") or filepath.endswith(".json.gz"):
            self._load_json(filepath)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.regret_sum = data["regret_sum"]
            self.strategy_sum = data["strategy_sum"]
            self._iterations = data.get("iterations", 0)
        print(
            f"[CFR] Loaded from {filepath} "
            f"({len(self.regret_sum)} info sets, {self._iterations} iterations)"
        )

    def _load_json(self, filepath: str) -> None:
        """Load from gzip-compressed JSON (Go CFR training output)."""
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            data = json.load(f)
        self._iterations = data.get("iterations", 0)
        self.regret_sum = {}
        for key, actions in data.get("regret_sum", {}).items():
            self.regret_sum[key] = {int(a): v for a, v in actions.items()}
        self.strategy_sum = {}
        for key, actions in data.get("strategy_sum", {}).items():
            self.strategy_sum[key] = {int(a): v for a, v in actions.items()}
