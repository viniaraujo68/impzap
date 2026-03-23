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

    Maintains a PolicyNetwork and accumulates log-probabilities and rewards
    within each game. Policy update is triggered by the training loop after
    each hand reward is received.
    """

    def __init__(
        self,
        input_size: int = 164,
        hidden_size: int = 128,
        output_size: int = 9,
        lr: float = 1e-3,
    ) -> None:
        self.name: str = "REINFORCE"
        self.policy: PolicyNetwork = PolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )
        self.saved_log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

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

    def store_reward(self, reward: float) -> None:
        """Append a scalar reward to the episode buffer."""
        self.rewards.append(reward)

    def update_policy(self) -> None:
        """
        Perform one REINFORCE gradient step using the buffered log-probs and
        rewards. Clears both buffers after the update.
        """
        if not self.saved_log_probs:
            return

        final_reward: float = sum(self.rewards)
        policy_loss = torch.cat(
            [-lp * final_reward for lp in self.saved_log_probs]
        ).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.saved_log_probs.clear()
        self.rewards.clear()

    def save(self, filepath: str) -> None:
        """Save the policy network weights to filepath."""
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """Load policy network weights from filepath."""
        self.policy.load_state_dict(
            torch.load(filepath, weights_only=True)
        )
        self.policy.eval()
