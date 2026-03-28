"""
REINFORCE training script for Truco Paulista.

Usage:
    python train.py                    # 50k episodes (default)
    python train.py --episodes 100000
    python train.py --output models/my_model.pth
"""

import argparse
import copy
import os
import random
import time
from collections import deque
from typing import Any, Deque, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.reinforce_agent import ReinforceAgent
from truco_env.env import TrucoEnv
from truco_env.wrappers import TrucoVectorObservation

DEFAULT_EPISODES: int = 50_000
DEFAULT_MODEL_PATH: str = "models/reinforce.pth"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """
    Compute reward-to-go G_t for a sequence of per-step rewards.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Parameters
    ----------
    rewards : List[float]
        Rewards at each step where the agent acted (same order as log_probs).
    gamma : float
        Discount factor.

    Returns
    -------
    List[float]
        Discounted reward-to-go for each step, same length as rewards.
    """
    returns: List[float] = []
    running: float = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


def run_evaluation(
    env: TrucoVectorObservation,
    agent: ReinforceAgent,
    opponent: Any,
    num_games: int,
) -> float:
    """
    Evaluate agent win rate against a fixed opponent without gradient updates.

    Parameters
    ----------
    env : TrucoVectorObservation
        The wrapped environment.
    agent : ReinforceAgent
        Agent being evaluated (policy used in no_grad mode).
    opponent : Any
        Opponent agent with .act(state, info) -> int interface.
    num_games : int
        Number of games to play.

    Returns
    -------
    float
        Win rate as a percentage (0.0 to 100.0).
    """
    wins = 0
    raw = env.raw_env
    for _ in range(num_games):
        state_vector, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            current_player = info["current_player"]
            if current_player == 0:
                with torch.no_grad():
                    action = agent.act(state_vector, info)
                # Discard the log_prob stored during evaluation
                agent.saved_log_probs.clear()
            else:
                action = opponent.act(raw.current_state, info)
            state_vector, _, terminated, truncated, info = env.step(action)
        if raw.current_state.get("winner") == 0:
            wins += 1
    return (wins / num_games) * 100.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    episodes: List[int],
    train_win_rates: List[float],
    eval_vs_random: List[float],
    eval_vs_heuristic: List[float],
    avg_returns: List[float],
) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    ax1.plot(episodes, train_win_rates, color="blue", label="Train Win Rate")
    ax1.axhline(y=50.0, color="red", linestyle="--", label="Baseline (50%)")
    ax1.set_title("Training Win Rate (Rolling 500-Episode Window)")
    ax1.set_ylabel("Win Rate (%)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.7)

    ax2.plot(episodes, eval_vs_random, color="green", label="vs Random")
    ax2.plot(episodes, eval_vs_heuristic, color="orange", label="vs Heuristic")
    ax2.axhline(y=50.0, color="red", linestyle="--", label="Baseline (50%)")
    ax2.set_title("Evaluation Win Rate (100 Games Each)")
    ax2.set_ylabel("Win Rate (%)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.7)

    ax3.plot(episodes, avg_returns, color="purple", label="Avg Episode Return")
    ax3.set_title("Average Episode Return (Rolling 500-Episode Window)")
    ax3.set_xlabel("Episodes")
    ax3.set_ylabel("Return")
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=300)
    print("Plot saved to training_results.png")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(num_episodes: int = DEFAULT_EPISODES, output: str = DEFAULT_MODEL_PATH) -> None:
    base_env = TrucoEnv()
    env = TrucoVectorObservation(base_env)

    agent_p0 = ReinforceAgent(lr=1e-3, gamma=0.99, ema_alpha=0.05)
    heuristic_opp = HeuristicAgent()
    random_opp = RandomAgent()

    # Opponent starts as heuristic; at 65% win rate, switches to 50/50 mix
    # of heuristic and frozen self-play snapshot to prevent catastrophic forgetting.
    current_opponent: Any = heuristic_opp
    frozen_snapshot: Optional[ReinforceAgent] = None
    using_self_play: bool = False
    SELF_PLAY_MIX: float = 0.5  # fraction of self-play episodes once curriculum activates

    # Scale intervals with num_episodes: ~20 log points, window = 2x eval interval
    EVAL_INTERVAL: int = max(500, num_episodes // 20)
    WINDOW: int = EVAL_INTERVAL * 2
    win_window: Deque[int] = deque(maxlen=WINDOW)

    # Tracking for plots
    tracked_episodes: List[int] = []
    tracked_train_win_rates: List[float] = []
    tracked_eval_vs_random: List[float] = []
    tracked_eval_vs_heuristic: List[float] = []
    tracked_avg_returns: List[float] = []
    return_window: Deque[float] = deque(maxlen=WINDOW)

    print("=" * 60)
    print(f"STARTING REINFORCE TRAINING: {num_episodes} EPISODES")
    print("=" * 60)
    start_time = time.time()

    for episode in range(num_episodes):
        state_vector, info = env.reset()
        terminated = False
        truncated = False
        # Decide opponent for this episode (only relevant once self-play is active)
        use_self_play_this_ep: bool = random.random() < SELF_PLAY_MIX

        # Per-episode buffers
        # all_rewards: reward_p0 at every step (including opponent's turns)
        # p0_step_indices: which positions in all_rewards correspond to p0 actions
        all_rewards: List[float] = []
        p0_step_indices: List[int] = []
        episode_return: float = 0.0

        while not (terminated or truncated):
            current_player = info["current_player"]

            if current_player == 0:
                action = agent_p0.act(state_vector, info)
                p0_step_indices.append(len(all_rewards))
            else:
                opp_state = base_env.current_state
                if using_self_play and frozen_snapshot is not None and use_self_play_this_ep:
                    with torch.no_grad():
                        action = frozen_snapshot.act(state_vector, info)
                    frozen_snapshot.saved_log_probs.clear()
                else:
                    action = current_opponent.act(opp_state, info)

            state_vector, _, terminated, truncated, info = env.step(action)

            reward_p0: float = float(info.get("reward_p0", 0.0))
            all_rewards.append(reward_p0)

        episode_return = sum(all_rewards)

        # End of game — compute full-sequence returns, extract p0's action steps
        if p0_step_indices:
            all_returns = compute_returns(all_rewards, agent_p0.gamma)
            p0_returns = [all_returns[i] for i in p0_step_indices]
            agent_p0.update_policy(p0_returns)

        # Track outcome
        winner = base_env.current_state.get("winner", -1)
        win_window.append(1 if winner == 0 else 0)
        return_window.append(episode_return)

        # Self-play curriculum: switch at 65% rolling win rate
        if not using_self_play and len(win_window) >= EVAL_INTERVAL:
            rolling_wr = sum(win_window) / WINDOW * 100.0
            if rolling_wr >= 65.0:
                using_self_play = True
                frozen_snapshot = copy.deepcopy(agent_p0)
                frozen_snapshot.policy.eval()
                print(
                    f"  [Episode {episode + 1}] Self-play curriculum activated "
                    f"(rolling win rate: {rolling_wr:.1f}%)"
                )

        # Update frozen snapshot every eval interval once self-play is active
        if using_self_play and (episode + 1) % EVAL_INTERVAL == 0:
            frozen_snapshot = copy.deepcopy(agent_p0)
            frozen_snapshot.policy.eval()

        # Periodic evaluation and logging
        if (episode + 1) % EVAL_INTERVAL == 0:
            rolling_wr = (sum(win_window) / len(win_window)) * 100.0 if win_window else 0.0
            avg_ret = float(np.mean(return_window)) if return_window else 0.0

            eval_rand = run_evaluation(env, agent_p0, random_opp, 300)
            eval_heur = run_evaluation(env, agent_p0, heuristic_opp, 300)

            tracked_episodes.append(episode + 1)
            tracked_train_win_rates.append(rolling_wr)
            tracked_eval_vs_random.append(eval_rand)
            tracked_eval_vs_heuristic.append(eval_heur)
            tracked_avg_returns.append(avg_ret)

            opp_label = "mixed" if using_self_play else "heuristic"
            print(
                f"Episode {episode + 1:>6}/{num_episodes} | "
                f"Train WR: {rolling_wr:5.1f}% ({opp_label}) | "
                f"Eval vs Random: {eval_rand:5.1f}% | "
                f"Eval vs Heuristic: {eval_heur:5.1f}% | "
                f"Avg Return: {avg_ret:+.3f}"
            )

    end_time = time.time()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print(f"Total time: {end_time - start_time:.2f}s")
    print("=" * 60)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    agent_p0.save(output)
    print(f"Model saved to: {output}")

    plot_results(
        tracked_episodes,
        tracked_train_win_rates,
        tracked_eval_vs_random,
        tracked_eval_vs_heuristic,
        tracked_avg_returns,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REINFORCE agent for Truco Paulista")
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help=f"Number of training episodes (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Output model path (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()
    train(num_episodes=args.episodes, output=args.output)
