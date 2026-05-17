"""
Per-game CSV benchmark runner.

Generates a CSV with one row per game, enabling paired analysis (Wilson CI,
McNemar's test) across matchups that share the same per-game seed.

Columns: tournament_label, p0_name, p1_name, master_seed, game_idx,
         game_seed, winner_seat

Usage:
    python bench_csv.py --seed 42 --out bench_per_game.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from typing import Any, Callable, List, Optional, Tuple

from truco_env.env import TrucoEnv
from truco_env.seeding import derive_game_seed, seed_all
from truco_env.wrappers import TrucoVectorObservation

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.reinforce_agent import ReinforceAgent
from agents.cfr_agent import CFRAgent
from agents.hmm_agent import HMMAgent
from agents.hmm_cfr_agent import HMMCFRAgent
from agents.always_fold_agent import AlwaysFoldAgent
from agents.always_raise_agent import AlwaysRaiseAgent
from agents.deterministic_agent import DeterministicAgent


CFR_MODEL = "models/cfr_v9_scorehand_6M.json.gz"
REINFORCE_MODEL = "models/reinforce.pth"


def make_agent(name: str, seat: int) -> Any:
    """Factory mirrors play.py but returns fresh instances per call."""
    key = name.lower()
    if key == "random":
        return RandomAgent()
    if key == "heuristic":
        return HeuristicAgent()
    if key == "always_fold":
        return AlwaysFoldAgent()
    if key == "always_raise":
        return AlwaysRaiseAgent()
    if key == "deterministic":
        return DeterministicAgent()
    if key == "hmm":
        return HMMAgent(perspective=seat)
    if key == "hmm_cfr":
        # Use the HMMCFRAgent default cfr_model_path (v8_fullbucket_2M).
        # This matches the agent configuration that produced the thesis
        # numbers; overriding to v9 here would change HMM+CFR results.
        return HMMCFRAgent(perspective=seat)
    if key == "cfr":
        agent = CFRAgent()
        agent.load(CFR_MODEL)
        return agent
    if key == "reinforce":
        agent = ReinforceAgent()
        agent.load(REINFORCE_MODEL)
        return agent
    raise ValueError(f"Unknown agent name: {name}")


def get_action(agent: Any, state_vector: Any, raw_state: dict, info: dict) -> int:
    if isinstance(agent, ReinforceAgent):
        return agent.act(state_vector, info)
    return agent.act(raw_state, info)


def play_one(
    env: TrucoVectorObservation,
    agent_p0: Any,
    agent_p1: Any,
    seed: int,
) -> int:
    seed_all(env, seed)
    state_vector, info = env.reset()
    for agent in (agent_p0, agent_p1):
        if hasattr(agent, "reset"):
            agent.reset()

    terminated = False
    truncated = False
    while not (terminated or truncated):
        raw_state = env.raw_env.current_state
        current_player: int = raw_state["current_player"]
        if current_player == 0:
            action = get_action(agent_p0, state_vector, raw_state, info)
        else:
            action = get_action(agent_p1, state_vector, raw_state, info)
        state_vector, _, terminated, truncated, info = env.step(action)

    return env.raw_env.current_state["winner"]


def run_tournament(
    env: TrucoVectorObservation,
    label: str,
    p0_name: str,
    p1_name: str,
    num_games: int,
    master_seed: int,
    writer: csv.writer,
) -> Tuple[int, int, float]:
    """
    Run num_games and log each result to writer. No seat swap — caller
    is responsible for ordering. Returns (p0_wins, p1_wins, elapsed_s).
    """
    agent_p0 = make_agent(p0_name, 0)
    agent_p1 = make_agent(p1_name, 1)

    p0_wins = 0
    p1_wins = 0
    start = time.time()

    for game_idx in range(num_games):
        game_seed = derive_game_seed(master_seed, game_idx)
        winner = play_one(env, agent_p0, agent_p1, seed=game_seed)
        if winner == 0:
            p0_wins += 1
        elif winner == 1:
            p1_wins += 1
        writer.writerow(
            [label, p0_name, p1_name, master_seed, game_idx, game_seed, winner]
        )

    elapsed = time.time() - start
    rate = p0_wins / num_games * 100
    print(
        f"  [{label}] {p0_name} vs {p1_name} | "
        f"{p0_wins}/{num_games} ({rate:.1f}%) | {elapsed:.1f}s"
    )
    return p0_wins, p1_wins, elapsed


# ---------------------------------------------------------------------------
# Tournament plan
# ---------------------------------------------------------------------------
# Each tuple: (label, p0_name, p1_name, num_games)

TOURNAMENTS: List[Tuple[str, str, str, int]] = [
    # ---- Final-benchmark column: vs Heuristic (5000) ----
    ("final_vs_heur",   "hmm_cfr",   "heuristic",  5000),
    ("final_vs_heur",   "hmm",       "heuristic",  5000),
    ("final_vs_heur",   "cfr",       "heuristic",  5000),

    # ---- Final-benchmark column: vs Random (5000) ----
    ("final_vs_rand",   "hmm_cfr",   "random",     5000),
    ("final_vs_rand",   "hmm",       "random",     5000),
    ("final_vs_rand",   "cfr",       "random",     5000),
    ("final_vs_rand",   "heuristic", "random",     5000),

    # ---- Archetype columns: vs AlwaysFold (1000) ----
    ("arch_vs_fold",    "hmm_cfr",   "always_fold", 1000),
    ("arch_vs_fold",    "hmm",       "always_fold", 1000),
    ("arch_vs_fold",    "cfr",       "always_fold", 1000),
    ("arch_vs_fold",    "heuristic", "always_fold", 1000),

    # ---- Archetype columns: vs AlwaysRaise (1000) ----
    ("arch_vs_raise",   "hmm_cfr",   "always_raise", 1000),
    ("arch_vs_raise",   "hmm",       "always_raise", 1000),
    ("arch_vs_raise",   "cfr",       "always_raise", 1000),
    ("arch_vs_raise",   "heuristic", "always_raise", 1000),

    # ---- Third-party Deterministic baseline (5000) ----
    ("deterministic",   "random",    "deterministic", 5000),
    ("deterministic",   "heuristic", "deterministic", 5000),
    ("deterministic",   "reinforce", "deterministic", 5000),
    ("deterministic",   "cfr",       "deterministic", 5000),
    ("deterministic",   "hmm",       "deterministic", 5000),
    ("deterministic",   "hmm_cfr",   "deterministic", 5000),

    # ---- Head-to-head matrix, non-MCTS pairings (5000) ----
    # Each unordered pair (i,j) runs both directions so the seat-swap analysis
    # can recover the symmetric average. Skip self-play; bench_matrix_results
    # already shows all self-play diagonal entries within noise of 50%.
    ("h2h",             "heuristic", "reinforce",  5000),
    ("h2h",             "reinforce", "heuristic",  5000),
    ("h2h",             "heuristic", "cfr",        5000),
    ("h2h",             "cfr",       "heuristic",  5000),
    ("h2h",             "heuristic", "hmm_cfr",    5000),
    ("h2h",             "hmm_cfr",   "heuristic",  5000),
    ("h2h",             "reinforce", "cfr",        5000),
    ("h2h",             "cfr",       "reinforce",  5000),
    ("h2h",             "reinforce", "hmm_cfr",    5000),
    ("h2h",             "hmm_cfr",   "reinforce",  5000),
    ("h2h",             "cfr",       "hmm_cfr",    5000),
    ("h2h",             "hmm_cfr",   "cfr",        5000),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-game CSV benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Master seed.")
    parser.add_argument(
        "--out", type=str, default="bench_per_game.csv",
        help="CSV output path."
    )
    args = parser.parse_args()

    base_env = TrucoEnv()
    env = TrucoVectorObservation(base_env)

    overall_start = time.time()
    print(f"Writing per-game results to {args.out}")
    print(f"Master seed: {args.seed}")
    print(f"Total tournaments: {len(TOURNAMENTS)}")
    print()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tournament_label", "p0_name", "p1_name",
            "master_seed", "game_idx", "game_seed", "winner_seat",
        ])

        for i, (label, p0, p1, n) in enumerate(TOURNAMENTS):
            print(f"[{i+1}/{len(TOURNAMENTS)}] {label}: {p0} vs {p1} ({n} games)")
            run_tournament(env, label, p0, p1, n, args.seed, writer)
            f.flush()  # make CSV inspectable while running

    elapsed = time.time() - overall_start
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"CSV written to: {args.out}")


if __name__ == "__main__":
    main()
