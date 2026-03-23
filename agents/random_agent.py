import random
from typing import Any, Dict, List


class RandomAgent:
    """Agent that selects uniformly at random from the legal actions."""

    def act(self, state: Dict[str, Any], info: Dict[str, Any]) -> int:
        """
        Return a random legal action.

        Parameters
        ----------
        state : Dict[str, Any]
            Observable state dict from TrucoEnv (unused).
        info : Dict[str, Any]
            Info dict from TrucoEnv. Must contain 'legal_actions'.

        Returns
        -------
        int
            A randomly chosen legal action.
        """
        legal_actions: List[int] = info["legal_actions"]
        return random.choice(legal_actions)
