"""
Monte Carlo Tree Search agent for Truco Paulista (1v1).

Strategy: Perfect Information Monte Carlo (PIMC-MCTS), also known as
determinization. For each call to act(), K worlds are sampled from the
space of opponent hands consistent with the current observation. A
separate UCT tree is built per world, and root-level visit counts are
aggregated across all worlds. The action with the highest total visits
is returned.

Forward model: All rollout simulations and tree expansions call the Go
engine via TrucoEnv.step_from_state(), which wraps the stateless
StepFromState CGO export. The pure-Python forward model has been removed.

Backpropagation: Negamax convention. The value stored at each node is
the expected return from the perspective of the player who acts at that
node. UCT always maximizes, which is correct under negamax.

Return signal: +1.0 if perspective_player wins the game, -1.0 if the
opponent wins. The scale is already in [-1, +1] (equivalent to
dividing by the maximum possible score of 12, since the terminal
condition is reaching exactly 12 points).
"""

from __future__ import annotations

import math
import random  # used in _determinize (random.sample) and _expand (randrange)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents.card_utils import (
    ALL_CARDS as _ALL_CARDS,
    card_to_go,
    compare_cards,
)
from truco_env.env import TrucoEnv

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_BET_LADDER: List[int] = [1, 3, 6, 9, 12]

# Rollout policy IDs — must match rollout.go constants.
ROLLOUT_POLICY_RANDOM: int = 0
ROLLOUT_POLICY_HEURISTIC: int = 1


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """
    One node in the UCT search tree.

    Under the negamax convention, total_value and q_value represent the
    expected return from the perspective of the player who acts AT this
    node (current_player). UCT always maximizes q_value, which is correct
    regardless of which player is at the node.

    state stores the Go GameState dict at this node. Reading state directly
    avoids replaying the path on every UCT iteration, which is both faster
    and required for correctness (new hand deals are random).
    """

    parent: Optional["MCTSNode"]
    action_taken: Optional[int]           # action from parent that created this node
    current_player: int = 0               # player who acts at this node
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0                  # N(node)
    total_value: float = 0.0             # W(node): sum of returns from acting player's POV
    legal_actions: List[int] = field(default_factory=list)
    untried_actions: List[int] = field(default_factory=list)
    state: Optional[Dict[str, Any]] = None  # Go GameState dict AT this node

    @property
    def q_value(self) -> float:
        """Mean value estimate from the acting player's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def uct_score(
        self,
        parent_visits: int,
        exploration_constant: float,
        parent_player: int,
    ) -> float:
        """
        UCB1: Q_adjusted + c * sqrt(ln(N_parent) / N(node)).

        Q is stored from this node's acting player's perspective. However,
        selection is performed from the PARENT's acting player's perspective.
        When the two players differ (the normal case in a two-player game),
        the Q term must be negated so that 'maximize UCT' always means
        'maximize return for the player making the choice'.

        Returns +inf for unvisited nodes to guarantee they are explored first.
        """
        if self.visit_count == 0:
            return float("inf")
        adjusted_q = (
            self.q_value
            if self.current_player == parent_player
            else -self.q_value
        )
        return (
            adjusted_q
            + exploration_constant
            * math.sqrt(math.log(parent_visits) / self.visit_count)
        )

    def best_child(self, exploration_constant: float) -> "MCTSNode":
        """
        Return the child with the highest UCT score from this node's
        acting player's perspective.
        """
        return max(
            self.children.values(),
            key=lambda c: c.uct_score(
                self.visit_count, exploration_constant, self.current_player
            ),
        )

    def is_fully_expanded(self) -> bool:
        """True when every legal action has an associated child node."""
        return len(self.untried_actions) == 0


# ---------------------------------------------------------------------------
# MCTSAgent
# ---------------------------------------------------------------------------

class MCTSAgent:
    """
    Imperfect-information MCTS agent using PIMC-MCTS (determinization).

    Parameters
    ----------
    env : TrucoEnv
        The game environment. Used to call step_from_state() for
        Go-speed tree expansion and rollout simulation.
    n_simulations : int
        Total number of UCT iterations to run across all determinizations.
    n_determinizations : int
        Number of opponent-hand samples to draw per act() call.
        Each determinization receives n_simulations // n_determinizations
        iterations.
    exploration_constant : float
        UCB1 exploration constant c in Q + c * sqrt(ln N_parent / N).
    perspective_player : int
        The player index (0 or 1) that this agent controls.
    """

    def __init__(
        self,
        env: TrucoEnv,
        n_simulations: int = 200,
        n_determinizations: int = 10,
        exploration_constant: float = math.sqrt(2),
        perspective_player: int = 0,
        rollout_policy: int = ROLLOUT_POLICY_HEURISTIC,
    ) -> None:
        self._env = env
        self.n_simulations = n_simulations
        self.n_determinizations = n_determinizations
        self.exploration_constant = exploration_constant
        self.perspective_player = perspective_player
        self.rollout_policy = rollout_policy
        self._sims_per_det: int = max(1, n_simulations // n_determinizations)
        self.name: str = "MCTSAgent"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, state: dict, info: dict) -> int:
        """
        Select an action given the current observable state and info dict.

        Runs PIMC-MCTS: for each of the n_determinizations worlds, a UCT
        tree is built and searched. Root-level visit counts are accumulated
        across all worlds. The action with the highest total visit count is
        returned.

        The returned action is guaranteed to be in info['legal_actions'].

        Parameters
        ----------
        state : dict
            The observable View dict returned by TrucoEnv.
        info : dict
            The info dict from TrucoEnv (must contain 'legal_actions').

        Returns
        -------
        int
            A legal action integer.
        """
        legal_actions: List[int] = info["legal_actions"]
        if len(legal_actions) == 1:
            return legal_actions[0]

        action_visit_counts: Dict[int, int] = {a: 0 for a in legal_actions}

        for _ in range(self.n_determinizations):
            det_state = self._determinize(state, info)
            root = MCTSNode(
                parent=None,
                action_taken=None,
                current_player=det_state["current_player"],
                legal_actions=list(det_state["legal_actions"]),
                untried_actions=list(det_state["legal_actions"]),
            )
            self._run_uct(root, det_state, self._sims_per_det, action_visit_counts)

        return max(legal_actions, key=lambda a: action_visit_counts.get(a, 0))

    # ------------------------------------------------------------------
    # Determinization
    # ------------------------------------------------------------------

    def _determinize(self, state: dict, info: dict) -> Dict[str, Any]:
        """
        Sample one world consistent with the current observation and return
        a full Go GameState dict suitable for step_from_state().

        Unknown cards (not in own hand, not the vira, not in played_cards
        or table_cards as face-up) are collected into a pool. The opponent's
        hand slots are filled by sampling without replacement from that pool.

        Parameters
        ----------
        state : dict
            Observable View dict from TrucoEnv.
        info : dict
            Info dict from TrucoEnv.

        Returns
        -------
        Dict[str, Any]
            A fully observable Go GameState dict for one sampled world.
        """
        own_hand: List[str] = list(state.get("hand", []))
        vira: str = state.get("vira", "")
        table_cards_raw: List[str] = [
            c for c in state.get("table_cards", []) if c
        ]
        played_cards_raw: List[str] = list(state.get("played_cards", []))
        current_player: int = info.get("current_player", 0)
        score: List[int] = list(state.get("score", [0, 0]))
        current_bet: int = state.get("current_bet_value", 1)
        waiting_mao: bool = bool(state.get("waiting_for_mao_de_onze", False))
        legal_actions: List[int] = list(info.get("legal_actions", []))

        # Infer waiting_for_bet and associated bet state.
        waiting_for_bet: bool = (
            (4 in legal_actions or 5 in legal_actions) and not waiting_mao
        )
        pending_bet: int = 0
        truco_holder: int = -1
        original_turn: int = -1
        if waiting_for_bet:
            for i, val in enumerate(_BET_LADDER):
                if val == current_bet and i < len(_BET_LADDER) - 1:
                    pending_bet = _BET_LADDER[i + 1]
                    break
            truco_holder = 1 - current_player
            original_turn = 1 - current_player

        # Build the pool of cards unknown to the current player.
        unknown_pool = self._build_unknown_card_pool(
            own_hand, vira, table_cards_raw, played_cards_raw
        )

        # Sample opponent hand.
        opp_hand_size = max(
            0,
            min(len(own_hand) - len(table_cards_raw), len(unknown_pool)),
        )
        sampled_opp: List[str] = random.sample(unknown_pool, opp_hand_size)

        hands: List[List[str]] = [[], []]
        hands[current_player] = own_hand
        hands[1 - current_player] = sampled_opp

        # Reconstruct round history from played_cards.
        (
            round_wins,
            round_winners,
            round_starters,
            current_round,
            hand_starter,
            round_history,
        ) = self._reconstruct_round_history(
            played_cards_raw,
            table_cards_raw,
            vira,
            current_player,
        )

        return {
            "is_terminal": False,
            "current_player": current_player,
            "score": score,
            "hands": [
                [card_to_go(c) for c in hands[0]],
                [card_to_go(c) for c in hands[1]],
            ],
            "vira": card_to_go(vira),
            "table_cards": [card_to_go(c) for c in table_cards_raw],
            "current_bet_value": current_bet,
            "pending_bet": pending_bet,
            "waiting_for_bet": waiting_for_bet,
            "waiting_for_mao_de_onze": waiting_mao,
            "truco_holder": truco_holder,
            "original_turn": original_turn,
            "round_wins": round_wins,
            "current_round": current_round,
            "round_history": round_history,
            "round_starter": round_starters,
            "round_winners": round_winners,
            "reward": [0.0, 0.0],
            "winner": -1,
            "legal_actions": legal_actions,
            "hand_just_ended": False,
            "reset_reward_flag": False,
            "hand_starter": hand_starter,
        }

    def _build_unknown_card_pool(
        self,
        own_hand: List[str],
        vira: str,
        table_cards: List[str],
        played_cards: List[str],
    ) -> List[str]:
        """
        Return the list of cards that are unobservable to the searching
        player. Face-down cards ("FACEDOWN") are excluded from the known
        set since their identity is hidden.
        """
        known: set = set()
        known.update(c for c in own_hand if c)
        if vira:
            known.add(vira)
        known.update(c for c in table_cards if c and c != "FACEDOWN")
        known.update(c for c in played_cards if c and c != "FACEDOWN")
        return [c for c in _ALL_CARDS if c not in known]

    def _reconstruct_round_history(
        self,
        played_cards: List[str],
        table_cards: List[str],
        vira: str,
        current_player: int,
    ) -> Tuple[List[int], List[int], List[int], int, int, List[List[Dict[str, Any]]]]:
        """
        Reconstruct round_wins, round_winners, round_starters, current_round,
        hand_starter, and round_history from the observable played_cards list.

        round_history entries are Go card dicts for use with step_from_state().
        "FACEDOWN" entries in played_cards are encoded with facedown=True
        (rank/suit set to 0 as placeholder; Go's Compare() checks facedown
        first so the placeholder values do not affect round resolution).
        """
        completed_rounds = len(played_cards) // 2
        current_round = completed_rounds

        round_wins: List[int] = [0, 0]
        round_winners: List[int] = [-1, -1, -1]
        round_starters: List[int] = [-1, -1, -1]
        round_history: List[List[Dict[str, Any]]] = [[], [], []]

        if completed_rounds == 0 and not table_cards:
            hand_starter = current_player
        elif completed_rounds == 0 and table_cards:
            hand_starter = 1 - current_player
        else:
            hand_starter = 0

        round_starter = hand_starter
        for r in range(completed_rounds):
            c0 = played_cards[2 * r]
            c1 = (
                played_cards[2 * r + 1]
                if 2 * r + 1 < len(played_cards)
                else "FACEDOWN"
            )
            round_starters[r] = round_starter
            round_history[r] = [card_to_go(c0), card_to_go(c1)]

            result = compare_cards(c0, c1, vira)
            if result > 0:
                round_winner = round_starter
            elif result < 0:
                round_winner = 1 - round_starter
            else:
                round_winner = -1

            round_winners[r] = round_winner
            if round_winner != -1:
                round_wins[round_winner] += 1
                round_starter = round_winner

        if table_cards:
            round_starters[current_round] = 1 - current_player
        else:
            round_starters[current_round] = round_starter

        return (
            round_wins,
            round_winners,
            round_starters,
            current_round,
            hand_starter,
            round_history,
        )

    # ------------------------------------------------------------------
    # UCT tree search
    # ------------------------------------------------------------------

    def _run_uct(
        self,
        root: MCTSNode,
        root_state: Dict[str, Any],
        n_iterations: int,
        action_visit_counts: Dict[int, int],
    ) -> None:
        """
        Run n_iterations of UCT on the given determinized root state.
        Root-level child visit counts are accumulated into action_visit_counts
        (mutated in place).
        """
        root.state = root_state
        for _ in range(n_iterations):
            node, state, path = self._select(root)

            if not state["is_terminal"] and not node.is_fully_expanded():
                node, state = self._expand(node, state)
                path.append(node)

            value = self._simulate(state)
            self._backpropagate(path, value)

        for action, child in root.children.items():
            action_visit_counts[action] = (
                action_visit_counts.get(action, 0) + child.visit_count
            )

    def _select(
        self,
        node: MCTSNode,
    ) -> Tuple[MCTSNode, Dict[str, Any], List[MCTSNode]]:
        """
        Tree policy: descend using best_child() (UCT) until reaching a node
        that is either not fully expanded, has no children, or is terminal.

        State is read directly from node.state (O(1) per step).
        """
        assert node.state is not None
        state: Dict[str, Any] = node.state
        path: List[MCTSNode] = [node]
        while (
            not state["is_terminal"]
            and node.is_fully_expanded()
            and node.children
        ):
            node = node.best_child(self.exploration_constant)
            assert node.state is not None
            state = node.state
            path.append(node)
        return node, state, path

    def _expand(
        self,
        node: MCTSNode,
        state: Dict[str, Any],
    ) -> Tuple[MCTSNode, Dict[str, Any]]:
        """
        Select one untried action uniformly at random, call step_from_state()
        to advance the state, create the child node, and return
        (child_node, resulting_state).
        """
        action = node.untried_actions.pop(
            random.randrange(len(node.untried_actions))
        )
        next_state = self._env.step_from_state(state, action)
        child = MCTSNode(
            parent=node,
            action_taken=action,
            current_player=next_state["current_player"],
            legal_actions=list(next_state["legal_actions"]),
            untried_actions=list(next_state["legal_actions"]),
            state=next_state,
        )
        node.children[action] = child
        return child, next_state

    def _simulate(self, state: Dict[str, Any]) -> float:
        """
        Rollout: delegate the entire heuristic rollout to the Go engine via
        a single rollout_from_state() call. Returns +1.0 if perspective_player
        wins, -1.0 if the opponent wins, 0.0 if the depth limit was reached.
        """
        if state["is_terminal"]:
            winner = state["winner"]
        else:
            result = self._env.rollout_from_state(state, self.rollout_policy)
            winner = result.get("winner", -1)

        if winner == self.perspective_player:
            return 1.0
        if winner != -1:
            return -1.0
        return 0.0

    def _backpropagate(
        self, path: List[MCTSNode], value: float
    ) -> None:
        """
        Negamax backpropagation. The simulation value is from
        perspective_player's POV (+1.0 = perspective_player wins).
        Each node stores returns from its own acting player's POV, so
        the value is negated for opponent nodes.
        """
        for node in reversed(path):
            node.visit_count += 1
            if node.current_player == self.perspective_player:
                node.total_value += value
            else:
                node.total_value -= value
