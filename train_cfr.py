"""
Standalone CFR training script for Truco Paulista.
Uses Go-native CFR traversal for high performance.

Usage:
    python train_cfr.py --output models/cfr_v3_5buck_1M.json.gz --iterations 1000000
    python train_cfr.py --output models/cfr_v3_5buck_2M.json.gz --iterations 1000000 --resume models/cfr_v3_5buck_1M.json.gz
"""

import argparse
import os

from truco_env.env import TrucoEnv

DEFAULT_ITERATIONS: int = 1_000_000
DEFAULT_MODEL_PATH: str = "models/cfr_v3_5buck_1M.json.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFR agent for Truco Paulista (Go-native)")
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help=f"Number of CFR iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Output model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to existing model to resume training from",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    env = TrucoEnv()

    result = env.cfr_train(args.iterations, resume_path=args.resume)
    print(f"Training result: {result}")

    save_result = env.cfr_save(args.output)
    print(f"Save result: {save_result}")


if __name__ == "__main__":
    main()
