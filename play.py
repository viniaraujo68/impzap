import time
from typing import Any, Dict, Tuple

from truco_env.env import TrucoEnv
from truco_env.wrappers import TrucoVectorObservation
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.reinforce_agent import ReinforceAgent
from agents.mcts_agent import MCTSAgent
from agents.cfr_agent import CFRAgent
from agents.hmm_agent import HMMAgent
from agents.hmm_cfr_agent import HMMCFRAgent
from agents.always_fold_agent import AlwaysFoldAgent
from agents.always_raise_agent import AlwaysRaiseAgent


def translate_action(action: int, raw_state: Dict[str, Any]) -> str:
    """Return a human-readable description of an action in the given state."""
    is_mao_de_onze: bool = raw_state.get("waiting_for_mao_de_onze", False)

    if action == 3:
        return "RAISED"
    if action == 4:
        return "ACCEPTED MAO DE ONZE" if is_mao_de_onze else "ACCEPTED RAISE"
    if action == 5:
        return "REFUSED MAO DE ONZE" if is_mao_de_onze else "FOLDED"

    hand: list = raw_state.get("hand", [])
    if 0 <= action <= 2:
        card = hand[action] if action < len(hand) else "?"
        return f"PLAYED {card} (face-up)"
    if 6 <= action <= 8:
        idx = action - 6
        card = hand[idx] if idx < len(hand) else "?"
        return f"PLAYED {card} (face-down)"

    return f"UNKNOWN ACTION {action}"


def get_agent_name(agent: Any) -> str:
    return getattr(agent, "name", agent.__class__.__name__)


def get_action_for_agent(
    agent: Any,
    state_vector: Any,
    raw_state: Dict[str, Any],
    info: Dict[str, Any],
) -> int:
    if isinstance(agent, ReinforceAgent):
        return agent.act(state_vector, info)
    return agent.act(raw_state, info)


def play_verbose_match(
    env: TrucoVectorObservation,
    agent_p0: Any,
    agent_p1: Any,
) -> None:
    """Play one game and print every action to stdout."""
    state_vector, info = env.reset()
    terminated = False
    truncated = False
    turn = 1
    hand_number = 1

    name_p0 = get_agent_name(agent_p0)
    name_p1 = get_agent_name(agent_p1)

    print("=" * 60)
    print(f"MATCH: {name_p0} (P0) vs {name_p1} (P1)")
    print("=" * 60)

    while not (terminated or truncated):
        raw_state = env.raw_env.current_state
        current_player: int = raw_state["current_player"]

        print(
            f"\n[Hand {hand_number} | Turn {turn}] "
            f"Score: P0 [{raw_state['score'][0]}] x [{raw_state['score'][1]}] P1"
        )
        print(f"Vira: {raw_state['vira']} | Current Bet: {raw_state['current_bet_value']}")
        print(f"Table: {raw_state['table_cards']}")
        print(f"Player {current_player}'s Hand: {raw_state['hand']}")

        if info.get("waiting_for_mao_de_onze"):
            print(f"MAO DE ONZE: Player {current_player} must decide.")

        if current_player == 0:
            action = get_action_for_agent(agent_p0, state_vector, raw_state, info)
            actor = f"P0 ({name_p0})"
        else:
            action = get_action_for_agent(agent_p1, state_vector, raw_state, info)
            actor = f"P1 ({name_p1})"

        print(f"  -> {actor}: {translate_action(action, raw_state)}")

        next_state_vector, reward, terminated, truncated, next_info = env.step(action)

        if next_info["reward_p0"] != 0:
            next_raw = env.raw_env.current_state
            print("-" * 60)
            print(
                f"HAND ENDED. Score: P0 [{next_raw['score'][0]}] x [{next_raw['score'][1]}] P1"
            )
            print("-" * 60)
            hand_number += 1

        state_vector = next_state_vector
        info = next_info
        turn += 1

    final = env.raw_env.current_state
    print("=" * 60)
    print(f"MATCH ENDED. WINNER: PLAYER {final['winner']}")
    print("=" * 60)


def simulate_tournament(
    env: TrucoVectorObservation,
    agent_p0: Any,
    agent_p1: Any,
    num_games: int = 1000,
) -> Tuple[int, int]:
    """
    Run num_games silent matches and print a summary.

    Returns
    -------
    Tuple[int, int]
        (wins_p0, wins_p1)
    """
    wins_p0 = 0
    wins_p1 = 0

    name_p0 = get_agent_name(agent_p0)
    name_p1 = get_agent_name(agent_p1)

    print(f"\nSTARTING TOURNAMENT: {num_games} games | {name_p0} (P0) vs {name_p1} (P1)")
    start_time = time.time()

    for _ in range(num_games):
        state_vector, info = env.reset()
        terminated = False
        truncated = False

        for agent in (agent_p0, agent_p1):
            if hasattr(agent, "reset"):
                agent.reset()

        while not (terminated or truncated):
            raw_state = env.raw_env.current_state
            current_player: int = raw_state["current_player"]

            if current_player == 0:
                action = get_action_for_agent(agent_p0, state_vector, raw_state, info)
            else:
                action = get_action_for_agent(agent_p1, state_vector, raw_state, info)

            state_vector, _, terminated, truncated, info = env.step(action)

        final = env.raw_env.current_state
        if final["winner"] == 0:
            wins_p0 += 1
        elif final["winner"] == 1:
            wins_p1 += 1

    elapsed = time.time() - start_time
    rate_p0 = wins_p0 / num_games * 100
    rate_p1 = wins_p1 / num_games * 100

    print("=" * 50)
    print("TOURNAMENT RESULTS")
    print("=" * 50)
    print(f"Time: {elapsed:.2f}s")
    print(f"P0 ({name_p0}): {wins_p0} wins ({rate_p0:.1f}%)")
    print(f"P1 ({name_p1}): {wins_p1} wins ({rate_p1:.1f}%)")

    return wins_p0, wins_p1


def main() -> None:
    base_env = TrucoEnv()
    env = TrucoVectorObservation(base_env)

    cfr = CFRAgent()
    cfr.load("models/cfr_v8_fullbucket_2M.json.gz")

    random_agent = RandomAgent()
    heuristic = HeuristicAgent()

    mcts = MCTSAgent(
        env=base_env,
        n_simulations=500,
        n_determinizations=10,
        perspective_player=1,
    )

    reinforce = ReinforceAgent()
    reinforce.load("models/reinforce.pth")

    hmm_p0 = HMMAgent(perspective=0)
    hmm_cfr_p0 = HMMCFRAgent(perspective=0)
    always_fold = AlwaysFoldAgent()
    always_raise = AlwaysRaiseAgent()

    print("\n=== Archetype Baselines ===\n")
    simulate_tournament(env, heuristic, always_fold, num_games=1000)
    simulate_tournament(env, heuristic, always_raise, num_games=1000)
    simulate_tournament(env, cfr, always_fold, num_games=1000)
    simulate_tournament(env, cfr, always_raise, num_games=1000)
    simulate_tournament(env, hmm_p0, always_fold, num_games=1000)
    simulate_tournament(env, hmm_p0, always_raise, num_games=1000)
    simulate_tournament(env, hmm_cfr_p0, always_fold, num_games=1000)
    simulate_tournament(env, hmm_cfr_p0, always_raise, num_games=1000)

    print("\n=== Full Benchmark ===\n")
    simulate_tournament(env, hmm_p0, random_agent, num_games=1000)
    simulate_tournament(env, hmm_p0, heuristic, num_games=1000)
    simulate_tournament(env, hmm_p0, reinforce, num_games=1000)
    simulate_tournament(env, hmm_p0, cfr, num_games=1000)
    simulate_tournament(env, cfr, random_agent, num_games=1000)
    simulate_tournament(env, cfr, heuristic, num_games=1000)
    simulate_tournament(env, cfr, reinforce, num_games=1000)
    simulate_tournament(env, cfr, mcts, num_games=500)
    simulate_tournament(env, hmm_cfr_p0, random_agent, num_games=1000)
    simulate_tournament(env, hmm_cfr_p0, heuristic, num_games=1000)
    simulate_tournament(env, hmm_cfr_p0, cfr, num_games=1000)
    simulate_tournament(env, hmm_cfr_p0, mcts, num_games=500)
    # simulate_tournament(env, heuristic, mcts, num_games=100)
    # simulate_tournament(env, reinforce, mcts, num_games=100)
    # simulate_tournament(env, reinforce, heuristic, num_games=1000)
    # simulate_tournament(env, reinforce, random_agent, num_games=1000)


if __name__ == "__main__":
    main()
