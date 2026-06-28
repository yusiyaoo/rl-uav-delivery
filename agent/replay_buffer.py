"""
Prioritized Experience Replay Buffer.

Implements proportional prioritization (Schaul et al., 2016).
TD-error magnitude |δ| determines sampling probability:
    P(i) ∝ (|δ_i| + ε)^α
With importance-sampling weights to correct the bias:
    w_i = (N · P(i))^{-β}  (normalized by max weight)
"""

import numpy as np
from env.config import *


class SumTree:
    """
    Sum-tree data structure for efficient weighted sampling.

    Stores priorities in a complete binary tree where leaves hold
    transition priorities and internal nodes hold their sum.
    Allows O(log N) update and O(log N) sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree with 2*capacity nodes; root is at index 1
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def total(self) -> float:
        return self.tree[1]

    def add(self, priority: float, data):
        """Add data with given priority."""
        idx = self.write_idx + self.capacity  # leaf index
        self.data[self.write_idx] = data
        self._update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, tree_idx: int, priority: float):
        """Update priority at tree_idx and propagate up."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] += change

    def update_priority(self, tree_idx: int, priority: float):
        """Update priority for a specific transition.

        Args:
            tree_idx: leaf index in the sum-tree (already >= capacity), as returned by get()
            priority: new priority value to set
        """
        self._update(tree_idx, priority)

    def get(self, value: float):
        """
        Sample a transition given a cumulative probability value.

        Args:
            value: a scalar in [0, total_priority)

        Returns:
            (tree_idx, priority, data)
        """
        idx = 1  # root
        while idx < self.capacity:  # not a leaf
            left = 2 * idx
            if value < self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay with proportional prioritization.
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY,
                 alpha: float = PER_ALPHA,
                 beta: float = PER_BETA_START,
                 beta_anneal: float = PER_BETA_ANNEAL,
                 epsilon: float = PER_EPSILON):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = beta_anneal
        self.beta_end = PER_BETA_END
        self.epsilon = epsilon
        self.max_priority = 1.0  # initial max priority for new experiences

    def add(self, state, action, reward, next_state, done):
        """Store a transition with maximum priority."""
        data = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, data)

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones,
                       importance_weights, tree_indices)
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(value)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Importance-sampling weights
        probs = np.array(priorities) / self.tree.total()
        n = self.tree.size
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()  # normalize so max weight is 1

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices, td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self.epsilon) ** self.alpha
            self.tree.update_priority(idx, priority)
            if priority > self.max_priority:
                self.max_priority = priority

    def anneal_beta(self):
        """Anneal beta toward 1.0."""
        self.beta = min(self.beta_end, self.beta * self.beta_anneal)
        # Prevent beta from staying below beta_end due to floating point
        if self.beta > self.beta_end - 0.01:
            self.beta = self.beta_end

    def __len__(self) -> int:
        return self.tree.size
