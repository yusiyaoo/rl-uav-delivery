"""
Double DQN Agent with Dueling Network and Prioritized Experience Replay.

Key features:
    1. Double Q-learning — decouples action selection from evaluation
    2. Dueling architecture — separate value and advantage streams
    3. Prioritized Experience Replay — trains on high-TD-error transitions
    4. Soft target network updates — stable convergence
"""

import numpy as np
import torch
import torch.nn.functional as F
from env.config import *
from agent.network import DuelingQNetwork
from agent.replay_buffer import PrioritizedReplayBuffer


class DoubleDQNAgent:
    """Double DQN agent for UAV path planning."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.online_net = DuelingQNetwork().to(self.device)
        self.target_net = DuelingQNetwork().to(self.device)
        self._hard_update(self.target_net, self.online_net)  # sync at init

        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LR)

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer()

        # Exploration
        self.epsilon = EPS_START

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: (15,) observation array
            evaluate: if True, always greedy (no exploration)

        Returns:
            action index (0..6)
        """
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def learn(self):
        """
        Perform one learning step.

        Returns:
            loss (float): the TD loss for logging, or 0.0 if buffer too small
        """
        if len(self.buffer) < MIN_BUFFER_SIZE:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(BATCH_SIZE)

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).unsqueeze(1).to(self.device)
        weights = torch.from_numpy(weights).unsqueeze(1).to(self.device)

        # --- Double DQN update ---
        # Current Q-values
        q_current = self.online_net(states).gather(1, actions)

        # Target: r + γ * Q_target(s', argmax_a Q_online(s', a))
        with torch.no_grad():
            # Action selection by online network
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # Action evaluation by target network
            q_next = self.target_net(next_states).gather(1, next_actions)
            q_target = rewards + GAMMA * q_next * (1.0 - dones)

        # TD errors
        td_errors = (q_target - q_current).detach().cpu().numpy()

        # Weighted MSE loss (PER importance sampling)
        loss = (weights * F.mse_loss(q_current, q_target, reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors.flatten())

        # Soft update target network
        self._soft_update()

        return float(loss.item())

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def anneal_beta(self):
        """Anneal importance-sampling correction."""
        self.buffer.anneal_beta()

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', EPS_END)

    def _soft_update(self):
        """θ_target = τ·θ_online + (1-τ)·θ_target"""
        for target_param, online_param in zip(self.target_net.parameters(),
                                               self.online_net.parameters()):
            target_param.data.copy_(
                TAU * online_param.data + (1.0 - TAU) * target_param.data
            )

    @staticmethod
    def _hard_update(target_net, online_net):
        """θ_target = θ_online"""
        target_net.load_state_dict(online_net.state_dict())
