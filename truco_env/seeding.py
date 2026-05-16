"""
Centralized seeding for reproducible Truco Paulista games.

Two-level scheme:
- A *master* seed seeds the whole tournament reproducibly.
- Per-game seeds are derived deterministically from (master_seed, game_index),
  so game N is bit-identical across runs regardless of which agents play it.
  This is what enables direct agent-vs-agent comparison on the same deal.

Seeded RNG sources:
- Go engine (deck shuffles, MCTS rollout policy, CFR sampling) — via
  TrucoEnv.seed_engine.
- Python stdlib `random` — used by HeuristicAgent, HMMAgent, CFRAgent,
  HMMCFRAgent, RandomAgent, MCTSAgent.
- NumPy `np.random` — used in observation construction and HMM matrices.
- PyTorch `torch.manual_seed` — used by REINFORCE's Categorical sampling.
"""

import hashlib
import random as _random
from typing import Any

import numpy as _np

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def derive_game_seed(master_seed: int, game_index: int) -> int:
    """
    Deterministically derive a per-game seed from a master seed and game index.

    Uses BLAKE2s as a fast, well-mixed hash so consecutive game indices give
    uncorrelated seeds. Returns a non-negative 63-bit int that fits in a
    Go int64 and a Python int.
    """
    h = hashlib.blake2s(
        f"{int(master_seed)}:{int(game_index)}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "big", signed=False) & 0x7FFFFFFFFFFFFFFF


def seed_all(env: Any, seed: int) -> None:
    """
    Seed every RNG source: Go engine, `random`, NumPy, and PyTorch (if
    installed). Accepts either a base `TrucoEnv` or a `TrucoVectorObservation`
    wrapper (auto-unwrapped via `.raw_env`).
    """
    seed = int(seed)
    raw_env = getattr(env, "raw_env", env)
    raw_env.seed_engine(seed)
    _random.seed(seed)
    _np.random.seed(seed & 0xFFFFFFFF)
    if _HAS_TORCH:
        _torch.manual_seed(seed)
