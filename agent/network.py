"""
Dueling Deep Q-Network for UAV path planning.

Architecture:
    Input (15-dim state)
      → Shared FC layers (256→128, ReLU)
      → ├── Value stream → 1 (state value V(s))
      → └── Advantage stream → N_ACTIONS (A(s,a))
      → Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from env.config import INPUT_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, N_ACTIONS


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network with shared feature extraction layers."""

    def __init__(self):
        super().__init__()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
        )

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN_DIM_2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage stream: estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(HIDDEN_DIM_2, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: (batch_size, INPUT_DIM) observation tensor

        Returns:
            Q-values: (batch_size, N_ACTIONS)
        """
        features = self.shared(state)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, N_ACTIONS)
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
