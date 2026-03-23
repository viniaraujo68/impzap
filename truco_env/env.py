import os
import ctypes
import json
from typing import Any, Dict, Optional, Tuple  # Optional kept for reset/step signatures

import gymnasium as gym
from gymnasium import spaces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(BASE_DIR, "trucolib.so")


class TrucoEnv(gym.Env):
    """
    Gymnasium wrapper around the Go Truco engine (trucolib.so).

    Observation space: raw dict returned by the Go engine's JSON serializer.
    Action space: Discrete(9), mapping to the 9 possible game actions.

    Actions
    -------
    0-2 : Play card at hand index face-up.
    3   : Request/raise truco bet.
    4   : Accept current bet or mao-de-onze challenge.
    5   : Fold current bet or refuse mao-de-onze challenge.
    6-8 : Play card at hand index (index - 6) face-down.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, lib_path: str = LIB_PATH) -> None:
        super().__init__()

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Compiled engine not found at: {lib_path}. "
                "Run 'make build' at the project root first."
            )

        self._lib = ctypes.CDLL(lib_path)
        self._lib.InitGame.restype = ctypes.c_void_p
        self._lib.Step.argtypes = [ctypes.c_int]
        self._lib.Step.restype = ctypes.c_void_p
        self._lib.StepFromState.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.StepFromState.restype = ctypes.c_void_p
        self._lib.RolloutFromState.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.RolloutFromState.restype = ctypes.c_void_p
        self._lib.FreeString.argtypes = [ctypes.c_void_p]

        self.action_space: spaces.Discrete = spaces.Discrete(9)
        self.observation_space: spaces.Dict = spaces.Dict({})
        self.current_state: Dict[str, Any] = {}

    def _parse_and_free(self, ptr: int) -> Dict[str, Any]:
        if not ptr:
            raise RuntimeError("The game engine returned a null pointer.")
        try:
            json_str = ctypes.string_at(ptr).decode("utf-8")
            state = json.loads(json_str)
            if "error" in state:
                raise ValueError(f"Logical error in Go engine: {state['error']}")
            return state
        finally:
            self._lib.FreeString(ptr)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "legal_actions": self.current_state.get("legal_actions", []),
            "current_player": self.current_state.get("current_player", -1),
            "waiting_for_mao_de_onze": self.current_state.get(
                "waiting_for_mao_de_onze", False
            ),
            "score_p0": self.current_state.get("score", [0, 0])[0],
            "score_p1": self.current_state.get("score", [0, 0])[1],
            "reward_p0": self.current_state.get("reward", [0.0, 0.0])[0],
            "reward_p1": self.current_state.get("reward", [0.0, 0.0])[1],
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        ptr = self._lib.InitGame()
        self.current_state = self._parse_and_free(ptr)
        return self.current_state, self._get_info()

    def rollout_from_state(
        self, state: Dict[str, Any], policy_id: int = 1
    ) -> Dict[str, Any]:
        """
        Run a complete rollout from the given full GameState dict to a
        terminal state entirely inside the Go engine.

        Parameters
        ----------
        state : Dict[str, Any]
            Full GameState dict (as produced by step_from_state or
            _determinize). Must use internal field names.
        policy_id : int
            Rollout policy. 0 = random, 1 = heuristic (default).

        Returns
        -------
        Dict[str, Any]
            ``{"winner": int, "score": [int, int]}`` where winner is
            0, 1, or -1 (depth limit reached without a terminal state).
        """
        state_bytes = json.dumps(state).encode("utf-8")
        ptr = self._lib.RolloutFromState(state_bytes, policy_id)
        return self._parse_and_free(ptr)

    def step_from_state(self, state: Dict[str, Any], action: int) -> Dict[str, Any]:
        """
        Stateless transition: apply action to the given full GameState dict and
        return the resulting full GameState dict. Does not touch self.current_state.

        Parameters
        ----------
        state : Dict[str, Any]
            Full GameState as returned by a previous step_from_state call or
            built from a determinization. Must use the internal field names
            (e.g. ``hands``, ``round_history``) rather than View field names.
        action : int
            Action index (0-8) to apply.

        Returns
        -------
        Dict[str, Any]
            The new full GameState after the transition.
        """
        state_bytes = json.dumps(state).encode("utf-8")
        ptr = self._lib.StepFromState(state_bytes, action)
        return self._parse_and_free(ptr)

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        ptr = self._lib.Step(int(action))
        self.current_state = self._parse_and_free(ptr)

        info = self._get_info()
        terminated = bool(self.current_state.get("is_terminal", False))
        reward = info["reward_p0"]

        return self.current_state, reward, terminated, False, info
