"""Experience replay buffer for DQN.

Provides a fixed-size circular buffer with pre-allocated torch tensors for
 efficient uniform random sampling.
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    """Fixed-size circular replay buffer with pre-allocated tensors.

    Transitions are stored as ``(state, action, reward, next_state, done)``
    tuples.  When the buffer reaches *capacity* the oldest entries are
    overwritten in FIFO order.
    """

    def __init__(self, capacity: int, state_dim: int, device: torch.device) -> None:
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            state_dim: Dimensionality of state vectors.
            device: Torch device for tensor storage.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        self._size = 0
        self._pos = 0

        self.states = torch.zeros(
            capacity, state_dim, dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros(
            capacity, state_dim, dtype=torch.float32, device=device
        )
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Store a transition in the buffer.

        Args:
            state: Current state tensor of shape ``(state_dim,)``.
            action: Action index in ``[0, NUM_ACTIONS)``.
            reward: Scalar reward.
            next_state: Next state tensor of shape ``(state_dim,)``.
            done: Whether the episode terminated.
        """
        idx = self._pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample a random minibatch of transitions without replacement.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of ``(states, actions, rewards, next_states, dones)``
            tensors, all on :attr:`device`.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        batch_size = min(batch_size, self._size)
        indices = torch.randperm(self._size, device=self.device)[:batch_size]

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self._size
