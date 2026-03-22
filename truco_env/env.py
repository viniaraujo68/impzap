import os
import ctypes
import json
import gymnasium as gym
from gymnasium import spaces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(BASE_DIR, 'trucolib.so')

class TrucoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, lib_path=LIB_PATH):
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
        self._lib.FreeString.argtypes = [ctypes.c_void_p]

        self.action_space = spaces.Discrete(9)

        self.observation_space = spaces.Dict({}) 

        self.current_state = None

    def _parse_and_free(self, ptr):
        if not ptr:
            raise RuntimeError("The game engine returned a null pointer.")
        try:
            json_str = ctypes.string_at(ptr).decode('utf-8')
            state = json.loads(json_str)
            if 'error' in state:
                raise ValueError(f"Logical error in Go engine: {state['error']}")
            return state
        finally:
            self._lib.FreeString(ptr)

    def _get_info(self):
        return {
            'legal_actions': self.current_state.get('legal_actions', []),
            'current_player': self.current_state.get('current_player', -1),
            'waiting_for_mao_de_onze': self.current_state.get('waiting_for_mao_de_onze', False),
            'score_p0': self.current_state.get('score', [0, 0])[0],
            'score_p1': self.current_state.get('score', [0, 0])[1],
            'reward_p0': self.current_state.get('reward', [0.0, 0.0])[0], 
            'reward_p1': self.current_state.get('reward', [0.0, 0.0])[1]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        ptr = self._lib.InitGame()
        self.current_state = self._parse_and_free(ptr)
        
        return self.current_state, self._get_info()

    def step(self, action: int):
        ptr = self._lib.Step(int(action))
        self.current_state = self._parse_and_free(ptr)
        
        info = self._get_info()
        terminated = bool(self.current_state.get('is_terminal', False))
        truncated = False 
        
        reward = info['reward_p0'] 
        
        return self.current_state, reward, terminated, truncated, info