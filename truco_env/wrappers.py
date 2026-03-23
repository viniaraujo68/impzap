from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agents.card_utils import RANKS, SUITS
from truco_env.env import TrucoEnv

# Observation vector layout (total: 164 dimensions):
#   [  0.. 39] One-hot encoding of own hand cards       (40 cards)
#   [ 40.. 79] One-hot encoding of table cards          (40 cards)
#   [ 80..119] One-hot encoding of vira card            (40 cards)
#   [120..159] One-hot encoding of played cards history (40 cards)
#   [160]      P0 score normalised to [0, 1]            (score / 12)
#   [161]      P1 score normalised to [0, 1]
#   [162]      Current bet value normalised to [0, 1]   (bet / 12)
#   [163]      Mao-de-onze flag                         (0 or 1)
_OBS_SIZE: int = 164


class TrucoVectorObservation(gym.ObservationWrapper):
    """
    Observation wrapper that converts the raw dict from TrucoEnv into a
    fixed-length float32 numpy vector suitable for neural network input.
    """

    def __init__(self, env: TrucoEnv) -> None:
        super().__init__(env)
        self._raw_env: TrucoEnv = env
        self._card_to_idx: Dict[str, int] = {
            f"{r}_{s}": i
            for i, (r, s) in enumerate(
                (r, s) for r in RANKS for s in SUITS
            )
        }
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(_OBS_SIZE,), dtype=np.float32
        )

    @property
    def raw_env(self) -> TrucoEnv:
        """Return the underlying TrucoEnv with full type information."""
        return self._raw_env

    def observation(self, obs: Dict[str, Any]) -> np.ndarray:
        vec = np.zeros(_OBS_SIZE, dtype=np.float32)

        for card in obs.get("hand", []):
            if card in self._card_to_idx:
                vec[self._card_to_idx[card]] = 1.0

        for card in obs.get("table_cards", []):
            if card in self._card_to_idx:
                vec[40 + self._card_to_idx[card]] = 1.0

        vira = obs.get("vira", "")
        if vira in self._card_to_idx:
            vec[80 + self._card_to_idx[vira]] = 1.0

        for card in obs.get("played_cards", []):
            if card in self._card_to_idx:
                vec[120 + self._card_to_idx[card]] = 1.0

        score = obs.get("score", [0, 0])
        vec[160] = min(score[0] / 12.0, 1.0)
        vec[161] = min(score[1] / 12.0, 1.0)

        vec[162] = min(obs.get("current_bet_value", 1) / 12.0, 1.0)
        vec[163] = 1.0 if obs.get("waiting_for_mao_de_onze", False) else 0.0

        return vec
