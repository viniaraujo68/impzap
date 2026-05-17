"""
Partial bench re-run: only HMM+CFR tournaments, with HMM+CFR using its
default CFR backing (v8_fullbucket_2M), matching the configuration that
produced the existing thesis numbers. Output rows are merged into the
main CSV by replacing the HMM+CFR rows.
"""

from __future__ import annotations

import argparse
import csv
import time
from typing import Any, List, Tuple

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


REINFORCE_MODEL = "models/reinforce.pth"


def make_agent(name: str, seat: int) -> Any:
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
        # NOTE: default cfr_model_path = v8_fullbucket_2M; matches thesis text.
        return HMMCFRAgent(perspective=seat)
    if key == "cfr":
        agent = CFRAgent()
        agent.load("models/cfr_v9_scorehand_6M.json.gz")
        return agent
    if key == "reinforce":
        agent = ReinforceAgent()
        agent.load(REINFORCE_MODEL)
        return agent
    raise ValueError(f"Unknown agent name: {name}")


def get_action(agent: Any, state_vector, raw_state, info):
    if isinstance(agent, ReinforceAgent):
        return agent.act(state_vector, info)
    return agent.act(raw_state, info)


def play_one(env, agent_p0, agent_p1, seed):
    seed_all(env, seed)
    state_vector, info = env.reset()
    for agent in (agent_p0, agent_p1):
        if hasattr(agent, "reset"):
            agent.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        raw_state = env.raw_env.current_state
        current_player = raw_state["current_player"]
        if current_player == 0:
            action = get_action(agent_p0, state_vector, raw_state, info)
        else:
            action = get_action(agent_p1, state_vector, raw_state, info)
        state_vector, _, terminated, truncated, info = env.step(action)
    return env.raw_env.current_state["winner"]


# Only HMM+CFR-touching tournaments.
TOURNAMENTS: List[Tuple[str, str, str, int]] = [
    ("final_vs_heur", "hmm_cfr", "heuristic",    5000),
    ("final_vs_rand", "hmm_cfr", "random",       5000),
    ("arch_vs_fold",  "hmm_cfr", "always_fold",  1000),
    ("arch_vs_raise", "hmm_cfr", "always_raise", 1000),
    ("deterministic", "hmm_cfr", "deterministic", 5000),
    ("h2h",           "heuristic", "hmm_cfr",    5000),
    ("h2h",           "hmm_cfr",   "heuristic",  5000),
    ("h2h",           "reinforce", "hmm_cfr",    5000),
    ("h2h",           "hmm_cfr",   "reinforce",  5000),
    ("h2h",           "cfr",       "hmm_cfr",    5000),
    ("h2h",           "hmm_cfr",   "cfr",        5000),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="bench_per_game_hmmcfr_v8.csv")
    args = parser.parse_args()

    base_env = TrucoEnv()
    env = TrucoVectorObservation(base_env)

    overall_start = time.time()
    print(f"Re-running HMM+CFR tournaments with v8 default backing")
    print(f"Output: {args.out}")
    print(f"Seed: {args.seed} | {len(TOURNAMENTS)} tournaments\n")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tournament_label", "p0_name", "p1_name",
            "master_seed", "game_idx", "game_seed", "winner_seat",
        ])
        for i, (label, p0, p1, n) in enumerate(TOURNAMENTS):
            agent_p0 = make_agent(p0, 0)
            agent_p1 = make_agent(p1, 1)
            p0_wins = 0
            start = time.time()
            for game_idx in range(n):
                game_seed = derive_game_seed(args.seed, game_idx)
                winner = play_one(env, agent_p0, agent_p1, seed=game_seed)
                if winner == 0:
                    p0_wins += 1
                writer.writerow([label, p0, p1, args.seed, game_idx, game_seed, winner])
            elapsed = time.time() - start
            print(f"[{i+1}/{len(TOURNAMENTS)}] {label}: {p0} vs {p1} "
                  f"({p0_wins}/{n} = {p0_wins/n*100:.2f}%) {elapsed:.1f}s")
            f.flush()

    print(f"\nTotal: {(time.time()-overall_start)/60:.1f} min")


if __name__ == "__main__":
    main()
