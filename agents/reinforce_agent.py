from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Two-hidden-layer MLP that maps a state vector to action logits."""

    def __init__(
        self,
        input_size: int = 164,
        hidden_size: int = 128,
        output_size: int = 9,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReinforceAgent:
    """
    Policy gradient (REINFORCE) agent for Truco Paulista.

    Collects the full game trajectory, then performs a single policy update
    at game end using reward-to-go returns with a cross-episode EMA baseline.
    Gradient norms are clipped to max_norm=1.0 to stabilise training.
    """

    def __init__(
        self,
        input_size: int = 164,
        hidden_size: int = 128,
        output_size: int = 9,
        lr: float = 1e-3,
        gamma: float = 0.99,
        ema_alpha: float = 0.05,
    ) -> None:
        self.name: str = "REINFORCE"
        self.policy: PolicyNetwork = PolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )
        self.gamma: float = gamma
        self.ema_alpha: float = ema_alpha
        self._ema_baseline: float = 0.0
        self.saved_log_probs: List[torch.Tensor] = []

    def act(self, state_vector: np.ndarray, info: Dict[str, Any]) -> int:
        """
        Sample an action from the masked policy distribution.

        Legal-action masking is applied by setting logits for illegal actions
        to -inf before the softmax, ensuring zero probability for those actions.

        Parameters
        ----------
        state_vector : np.ndarray
            Observation vector produced by TrucoVectorObservation.
        info : Dict[str, Any]
            Info dict from TrucoEnv. Must contain 'legal_actions'.

        Returns
        -------
        int
            Sampled legal action.
        """
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)
        logits = self.policy(state_tensor)

        legal_actions: List[int] = info["legal_actions"]
        mask = torch.full((1, 9), float("-inf"))
        for a in legal_actions:
            mask[0, a] = 0.0

        probs = F.softmax(logits + mask, dim=1)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update_policy(self, returns: List[float]) -> None:
        """
        Perform one REINFORCE gradient step using the full-game trajectory.

        The EMA baseline is updated using the initial return G_0 (total game
        return), then subtracted from each G_t to form advantages. No std
        normalisation is applied to avoid distorting single-episode return
        geometry. Clears saved_log_probs after the update.

        Parameters
        ----------
        returns : List[float]
            Reward-to-go returns G_t for each step where the agent acted,
            pre-computed by the training loop with discount factor gamma.
        """
        if not self.saved_log_probs or not returns:
            self.saved_log_probs.clear()
            return

        assert len(self.saved_log_probs) == len(returns), (
            f"log_probs length {len(self.saved_log_probs)} != returns length {len(returns)}"
        )

        # Update EMA baseline with G_0 (total discounted return for this game)
        g0: float = returns[0]
        self._ema_baseline = (
            self.ema_alpha * g0 + (1.0 - self.ema_alpha) * self._ema_baseline
        )

        advantages = [g - self._ema_baseline for g in returns]

        policy_loss = torch.cat(
            [-lp * adv for lp, adv in zip(self.saved_log_probs, advantages)]
        ).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.saved_log_probs.clear()

    def save(self, filepath: str) -> None:
        """Save the policy network weights to filepath."""
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """Load policy network weights from filepath."""
        self.policy.load_state_dict(
            torch.load(filepath, weights_only=True)
        )
        self.policy.eval()
