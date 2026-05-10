"""
Head-to-head benchmark matrix between the main agents.

Each pair plays num_games total with seats swapped at the halfway point,
so the win rate from agent A's perspective is unbiased by who deals first.

The five agents in the matrix are: Heuristic, REINFORCE, MCTS, CFR (v9@6M)
and HMM+CFR. The matrix is printed as both a Markdown table and a LaTeX
tabular block, and the raw counts are saved to bench_matrix_results.txt
so the run can be reused later without re-executing.
"""

import time
from typing import Any, Callable, Dict, List, Tuple

from truco_env.env import TrucoEnv
from truco_env.wrappers import TrucoVectorObservation
from agents.heuristic_agent import HeuristicAgent
from agents.reinforce_agent import ReinforceAgent
from agents.mcts_agent import MCTSAgent
from agents.cfr_agent import CFRAgent
from agents.hmm_cfr_agent import HMMCFRAgent


CFR_MODEL = "models/cfr_v9_scorehand_6M.json.gz"
REINFORCE_MODEL = "models/reinforce.pth"

GAMES_FAST = 5000
GAMES_MCTS = 1000


def get_action(agent: Any, state_vector: Any, raw_state: Dict[str, Any], info: Dict[str, Any]) -> int:
    if isinstance(agent, ReinforceAgent):
        return agent.act(state_vector, info)
    return agent.act(raw_state, info)


def play_one(env: TrucoVectorObservation, agent_p0: Any, agent_p1: Any) -> int:
    """Return the index of the winning seat (0 or 1)."""
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


def head_to_head(
    env: TrucoVectorObservation,
    factory_a: Callable[[int], Any],
    factory_b: Callable[[int], Any],
    num_games: int,
    label_a: str,
    label_b: str,
) -> Tuple[int, int, float]:
    """Run num_games with seats swapped halfway. Returns (a_wins, b_wins, elapsed)."""
    half = num_games // 2
    a_wins = 0
    b_wins = 0
    start = time.time()

    # First half: A as P0, B as P1
    a0 = factory_a(0)
    b1 = factory_b(1)
    for _ in range(half):
        winner = play_one(env, a0, b1)
        if winner == 0:
            a_wins += 1
        elif winner == 1:
            b_wins += 1

    # Second half: B as P0, A as P1
    b0 = factory_b(0)
    a1 = factory_a(1)
    for _ in range(num_games - half):
        winner = play_one(env, b0, a1)
        if winner == 0:
            b_wins += 1
        elif winner == 1:
            a_wins += 1

    elapsed = time.time() - start
    rate_a = a_wins / num_games * 100
    print(
        f"  {label_a} vs {label_b}: {a_wins}/{num_games} ({rate_a:.1f}%) "
        f"vs {b_wins} ties={num_games - a_wins - b_wins} | {elapsed:.1f}s"
    )
    return a_wins, b_wins, elapsed


def main() -> None:
    base_env = TrucoEnv()
    env = TrucoVectorObservation(base_env)

    # Each factory takes the seat (0 or 1) and returns a fresh agent for that seat.
    # Stateless agents ignore the seat argument.
    def make_heuristic(_seat: int) -> Any:
        return HeuristicAgent()

    def make_reinforce(_seat: int) -> Any:
        ag = ReinforceAgent()
        ag.load(REINFORCE_MODEL)
        return ag

    def make_mcts(seat: int) -> Any:
        return MCTSAgent(
            env=base_env,
            n_simulations=500,
            n_determinizations=10,
            perspective_player=seat,
        )

    def make_cfr(_seat: int) -> Any:
        ag = CFRAgent()
        ag.load(CFR_MODEL)
        return ag

    def make_hmm_cfr(seat: int) -> Any:
        return HMMCFRAgent(perspective=seat)

    agents: List[Tuple[str, Callable[[int], Any]]] = [
        ("Heuristic", make_heuristic),
        ("REINFORCE", make_reinforce),
        ("MCTS", make_mcts),
        ("CFR", make_cfr),
        ("HMM+CFR", make_hmm_cfr),
    ]

    n = len(agents)
    win_rate: List[List[float]] = [[0.0] * n for _ in range(n)]
    sample_size: List[List[int]] = [[0] * n for _ in range(n)]

    print("\n=== Head-to-Head Matrix ===\n")
    overall_start = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            label_a, factory_a = agents[i]
            label_b, factory_b = agents[j]
            num_games = GAMES_MCTS if "MCTS" in (label_a, label_b) else GAMES_FAST
            a_wins, b_wins, _ = head_to_head(
                env, factory_a, factory_b, num_games, label_a, label_b
            )
            rate_a = a_wins / num_games * 100
            rate_b = b_wins / num_games * 100
            win_rate[i][j] = rate_a
            win_rate[j][i] = rate_b
            sample_size[i][j] = num_games
            sample_size[j][i] = num_games

    total_elapsed = time.time() - overall_start
    print(f"\nTotal benchmark time: {total_elapsed/60:.1f} min")

    # ---- Markdown matrix ----
    print("\n### Markdown matrix (row vs column win rate)\n")
    header = "| | " + " | ".join(name for name, _ in agents) + " |"
    sep = "|---|" + "|".join([":---:"] * n) + "|"
    print(header)
    print(sep)
    for i, (name_i, _) in enumerate(agents):
        cells = []
        for j in range(n):
            if i == j:
                cells.append("—")
            else:
                cells.append(f"{win_rate[i][j]:.1f}%")
        print(f"| **{name_i}** | " + " | ".join(cells) + " |")

    # ---- LaTeX tabular ----
    print("\n### LaTeX tabular\n")
    col_spec = "l" + "c" * n
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")
    print("& " + " & ".join(name for name, _ in agents) + r" \\")
    print(r"\midrule")
    for i, (name_i, _) in enumerate(agents):
        cells = []
        for j in range(n):
            if i == j:
                cells.append("---")
            else:
                cells.append(f"{win_rate[i][j]:.1f}\\%")
        print(name_i + " & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ---- Raw counts (for reuse) ----
    with open("bench_matrix_results.txt", "w") as f:
        f.write("# Head-to-head benchmark results\n")
        f.write(f"# Generated in {total_elapsed/60:.1f} minutes\n")
        f.write(f"# CFR model: {CFR_MODEL}\n")
        f.write(f"# REINFORCE model: {REINFORCE_MODEL}\n\n")
        f.write("agent_a, agent_b, num_games, a_win_rate_pct\n")
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                f.write(
                    f"{agents[i][0]}, {agents[j][0]}, {sample_size[i][j]}, "
                    f"{win_rate[i][j]:.2f}\n"
                )
    print("\nRaw counts written to bench_matrix_results.txt")


if __name__ == "__main__":
    main()
