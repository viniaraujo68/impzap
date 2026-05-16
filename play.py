import argparse
import time
from typing import Any, Callable, Dict, Optional, Tuple

from truco_env.env import TrucoEnv
from truco_env.seeding import derive_game_seed, seed_all
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
from agents.deterministic_agent import DeterministicAgent


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
    seed: Optional[int] = None,
) -> None:
    """
    Play one game and print every action to stdout.

    If `seed` is given, every RNG source (Go engine, random, numpy, torch)
    is seeded before reset so the game is reproducible.
    """
    if seed is not None:
        seed_all(env, seed)
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
    seed: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Run num_games silent matches and print a summary.

    If `seed` is given, each game N is seeded with derive_game_seed(seed, N)
    so deals (and agent stochasticity) are reproducible across runs and
    aligned across opponents: game N has the same starting hand regardless
    of which agents play it.

    Returns
    -------
    Tuple[int, int]
        (wins_p0, wins_p1)
    """
    wins_p0 = 0
    wins_p1 = 0

    name_p0 = get_agent_name(agent_p0)
    name_p1 = get_agent_name(agent_p1)

    seed_note = f" | seed={seed}" if seed is not None else ""
    print(
        f"\nSTARTING TOURNAMENT: {num_games} games | "
        f"{name_p0} (P0) vs {name_p1} (P1){seed_note}"
    )
    start_time = time.time()

    for game_idx in range(num_games):
        if seed is not None:
            seed_all(env, derive_game_seed(seed, game_idx))
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


AGENT_FACTORIES: Dict[str, Callable[[TrucoEnv, int], Any]] = {
    "random": lambda _base, _seat: RandomAgent(),
    "heuristic": lambda _base, _seat: HeuristicAgent(),
    "always_fold": lambda _base, _seat: AlwaysFoldAgent(),
    "always_raise": lambda _base, _seat: AlwaysRaiseAgent(),
    "deterministic": lambda _base, _seat: DeterministicAgent(),
    "hmm": lambda _base, seat: HMMAgent(perspective=seat),
    "hmm_cfr": lambda _base, seat: HMMCFRAgent(perspective=seat),
    "cfr": lambda _base, _seat: _build_cfr(),
    "reinforce": lambda _base, _seat: _build_reinforce(),
    "mcts": lambda base, seat: MCTSAgent(
        env=base, n_simulations=500, n_determinizations=10,
        perspective_player=seat,
    ),
}


def _build_cfr() -> CFRAgent:
    agent = CFRAgent()
    agent.load("models/cfr_v9_scorehand_6M.json.gz")
    return agent


def _build_reinforce() -> ReinforceAgent:
    agent = ReinforceAgent()
    agent.load("models/reinforce.pth")
    return agent


def build_agent(name: str, base_env: TrucoEnv, seat: int) -> Any:
    """Resolve an agent name (case-insensitive) to a fresh instance."""
    key = name.lower()
    if key not in AGENT_FACTORIES:
        raise ValueError(
            f"Unknown agent '{name}'. Valid: {sorted(AGENT_FACTORIES)}"
        )
    return AGENT_FACTORIES[key](base_env, seat)


def run_default_benchmark(env: TrucoVectorObservation, seed: Optional[int]) -> None:
    """Run the full hardcoded benchmark suite (P0/P1 agent pairings)."""
    base_env = env.raw_env

    cfr = _build_cfr()
    random_agent = RandomAgent()
    heuristic = HeuristicAgent()
    mcts = MCTSAgent(
        env=base_env, n_simulations=500, n_determinizations=10,
        perspective_player=1,
    )
    _ = mcts  # kept for parity with prior main(); benchmarks below don't use it
    reinforce = _build_reinforce()
    hmm_p0 = HMMAgent(perspective=0)
    hmm_cfr_p0 = HMMCFRAgent(perspective=0)
    always_fold = AlwaysFoldAgent()
    always_raise = AlwaysRaiseAgent()
    deterministic = DeterministicAgent()

    print("\n=== Archetype Baselines ===\n")
    simulate_tournament(env, heuristic, always_fold, num_games=1000, seed=seed)
    simulate_tournament(env, heuristic, always_raise, num_games=1000, seed=seed)
    simulate_tournament(env, cfr, always_fold, num_games=1000, seed=seed)
    simulate_tournament(env, cfr, always_raise, num_games=1000, seed=seed)
    simulate_tournament(env, hmm_p0, always_fold, num_games=1000, seed=seed)
    simulate_tournament(env, hmm_p0, always_raise, num_games=1000, seed=seed)
    simulate_tournament(env, hmm_cfr_p0, always_fold, num_games=1000, seed=seed)
    simulate_tournament(env, hmm_cfr_p0, always_raise, num_games=1000, seed=seed)

    print("\n=== Full Benchmark ===\n")
    simulate_tournament(env, hmm_p0, random_agent, num_games=5000, seed=seed)
    simulate_tournament(env, hmm_p0, heuristic, num_games=5000, seed=seed)
    simulate_tournament(env, cfr, random_agent, num_games=5000, seed=seed)
    simulate_tournament(env, cfr, heuristic, num_games=5000, seed=seed)
    simulate_tournament(env, hmm_cfr_p0, random_agent, num_games=5000, seed=seed)
    simulate_tournament(env, hmm_cfr_p0, heuristic, num_games=5000, seed=seed)

    print("\n=== Deterministic Strategist (Filevich, 2023) Third-Party Baseline ===\n")
    simulate_tournament(env, random_agent, deterministic, num_games=5000, seed=seed)
    simulate_tournament(env, heuristic, deterministic, num_games=5000, seed=seed)
    simulate_tournament(env, reinforce, deterministic, num_games=5000, seed=seed)
    simulate_tournament(env, cfr, deterministic, num_games=5000, seed=seed)
    simulate_tournament(env, hmm_p0, deterministic, num_games=5000, seed=seed)
    simulate_tournament(env, hmm_cfr_p0, deterministic, num_games=5000, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Truco Paulista match/tournament driver.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Master seed. Each game N uses derive_game_seed(seed, N), so the "
        "same N has the same deal across runs/opponents.",
    )
    parser.add_argument(
        "--p0", type=str, default=None,
        help="P0 agent name. If given (with --p1), runs a single matchup "
        "instead of the default benchmark suite.",
    )
    parser.add_argument(
        "--p1", type=str, default=None,
        help="P1 agent name.",
    )
    parser.add_argument(
        "--games", type=int, default=1000,
        help="Number of games when running a single --p0/--p1 matchup "
        "(default: 1000).",
    )
    parser.add_argument(
        "--replay", type=int, default=None, metavar="GAME_INDEX",
        help="Verbosely replay a single seeded game by index. Requires "
        "--seed, --p0, --p1.",
    )
    args = parser.parse_args()

    base_env = TrucoEnv()
    env = TrucoVectorObservation(base_env)

    if args.replay is not None:
        if args.seed is None or args.p0 is None or args.p1 is None:
            parser.error("--replay requires --seed, --p0, and --p1.")
        agent_p0 = build_agent(args.p0, base_env, 0)
        agent_p1 = build_agent(args.p1, base_env, 1)
        game_seed = derive_game_seed(args.seed, args.replay)
        print(
            f"Replaying game #{args.replay} | master_seed={args.seed} "
            f"| game_seed={game_seed}"
        )
        play_verbose_match(env, agent_p0, agent_p1, seed=game_seed)
        return

    if args.p0 is not None and args.p1 is not None:
        agent_p0 = build_agent(args.p0, base_env, 0)
        agent_p1 = build_agent(args.p1, base_env, 1)
        simulate_tournament(
            env, agent_p0, agent_p1, num_games=args.games, seed=args.seed,
        )
        return

    if args.p0 is not None or args.p1 is not None:
        parser.error("--p0 and --p1 must be given together.")

    run_default_benchmark(env, args.seed)


if __name__ == "__main__":
    main()
