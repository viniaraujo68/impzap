"""
Standalone CFR training script for Truco Paulista.

Usage:
    python train_cfr.py                   # 50k iterations (default)
    python train_cfr.py --iterations 100000
    python train_cfr.py --resume           # resume from existing model
"""

import argparse
import os

from agents.cfr_agent import CFRAgent
from truco_env.env import TrucoEnv

DEFAULT_ITERATIONS: int = 50_000
DEFAULT_MODEL_PATH: str = "models/cfr.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFR agent for Truco Paulista")
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help=f"Number of CFR iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Output model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from existing model",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    env = TrucoEnv()
    agent = CFRAgent(env=env)

    if args.resume and os.path.exists(args.output):
        agent.load(args.output)
        print(f"Resuming from {agent._iterations} iterations")

    agent.train(num_iterations=args.iterations)
    agent.save(args.output)


if __name__ == "__main__":
    main()
