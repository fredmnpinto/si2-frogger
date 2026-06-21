"""DQN trainer with epsilon-greedy exploration and target network updates.

This module provides:
    - :class:`DQNConfig`: Dataclass holding all DQN hyperparameters.
    - :class:`DQNTrainer`: Manages the policy network, target network, replay
      buffer, and optimisation loop.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn_network import DQNNetwork, NUM_ACTIONS, STATE_DIM
from training.replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    """Hyperparameter configuration for DQN training.

    All fields have sensible defaults based on the original DQN paper and
    the Frogger environment characteristics.
    """

    learning_rate: float = 1e-3
    """Adam learning rate."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    epsilon_start: float = 1.0
    """Initial epsilon for epsilon-greedy exploration."""

    epsilon_end: float = 0.01
    """Final epsilon after decay."""

    epsilon_decay_steps: int = 10000
    """Number of steps over which epsilon linearly decays."""

    buffer_size: int = 50000
    """Maximum capacity of the experience replay buffer."""

    batch_size: int = 32
    """Number of transitions sampled per training update."""

    target_update_freq: int = 1000
    """Frequency (in training steps) for hard target network updates."""

    tau: float = 1.0
    """Polyak averaging coefficient.

    * ``tau == 1.0`` → hard copy every :attr:`target_update_freq` steps.
    * ``0 < tau < 1.0`` → soft update at every training step.
    """

    update_frequency: int = 4
    """Perform a training update only once every *N* calls to :meth:`train_step`."""

    hidden_size: int = 128
    """Number of neurons in each hidden layer."""

    gradient_clip: float = 1.0
    """Maximum L2 norm for gradient clipping."""

    device: str = "auto"
    """Torch device string.  ``"auto"`` selects CUDA when available."""


class DQNTrainer:
    """DQN trainer managing policy network, target network, replay buffer, and optimisation.

    Typical usage inside a training loop::

        trainer = DQNTrainer(config)
        for episode in range(num_episodes):
            state = env.reset()
            state_tensor = encoder.encode(state)
            done = False
            while not done:
                epsilon = trainer.get_epsilon(step)
                action = trainer.select_action(state_tensor, epsilon)
                next_state, reward, done, _ = env.step(action)
                next_state_tensor = encoder.encode(next_state)
                trainer.replay_buffer.push(
                    state_tensor, action, reward, next_state_tensor, done
                )
                loss = trainer.train_step()
                state_tensor = next_state_tensor
                step += 1
    """

    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        """Initialize the DQN trainer.

        Args:
            config: Hyperparameter configuration.  Uses :class:`DQNConfig`
                defaults when ``None``.
        """
        self.config = config if config is not None else DQNConfig()

        # Device selection with automatic CUDA fallback
        if self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)

        # Networks
        self.policy_net = DQNNetwork(
            hidden_dim=self.config.hidden_size
        ).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        # Optimiser
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.config.learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            state_dim=STATE_DIM,
            device=self.device,
        )

        # Training state
        self.step_count = 0
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    def get_epsilon(self, step: int) -> float:
        """Compute epsilon for the given step using linear decay.

        Epsilon decays linearly from :attr:`~DQNConfig.epsilon_start` to
        :attr:`~DQNConfig.epsilon_end` over
        :attr:`~DQNConfig.epsilon_decay_steps` steps.

        Args:
            step: Current training step (environment step count).

        Returns:
            Epsilon value in ``[epsilon_end, epsilon_start]``.
        """
        if self.config.epsilon_decay_steps <= 0:
            return self.config.epsilon_end
        if step >= self.config.epsilon_decay_steps:
            return self.config.epsilon_end
        slope = (
            self.config.epsilon_end - self.config.epsilon_start
        ) / self.config.epsilon_decay_steps
        return self.config.epsilon_start + slope * step

    def select_action(self, state: torch.Tensor, epsilon: float) -> int:
        """Select an action using the epsilon-greedy policy.

        The *policy network* is used for action selection; the target network
        is never used for this purpose.

        Args:
            state: Encoded state tensor of shape ``(STATE_DIM,)``.
            epsilon: Exploration probability in ``[0, 1]``.

        Returns:
            Selected action index in ``[0, NUM_ACTIONS)``.
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, NUM_ACTIONS, (1,)).item()

        with torch.no_grad():
            q_values = self.policy_net(state.unsqueeze(0).to(self.device))
            return int(q_values.argmax(dim=1).item())

    def update_target_network(self) -> None:
        """Update target network weights.

        * If :attr:`~DQNConfig.tau` ``< 1.0`` a soft Polyak update is applied
          every time this method is called.
        * If :attr:`~DQNConfig.tau` ``== 1.0`` a hard copy is performed only
          when :attr:`step_count` is a multiple of
          :attr:`~DQNConfig.target_update_freq`.
        """
        if self.config.tau < 1.0:
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * policy_param.data
                    + (1.0 - self.config.tau) * target_param.data
                )
        elif self.step_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_step(self) -> Optional[float]:
        """Perform one DQN training step.

        Samples a batch from the replay buffer, computes the temporal-difference
        loss, back-propagates, and clips gradients.

        Returns:
            The loss value as a Python ``float``, or ``None`` if no training
            occurred (buffer too small or not on an update step).

        Raises:
            RuntimeError: If the computed loss is NaN or Inf.
        """
        self.step_count += 1

        # Only train every ``update_frequency`` calls
        if self.step_count % self.config.update_frequency != 0:
            return None

        if len(self.replay_buffer) < self.config.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.config.batch_size)
        )

        # Compute predicted Q-values for the taken actions
        q_pred = (
            self.policy_net(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )

        # Compute target Q-values using the target network
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
            q_target = rewards + self.config.gamma * q_next * (1.0 - dones)

        # Loss
        loss = self.loss_fn(q_pred, q_target)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss is not finite: {loss.item()}"
            )

        # Back-propagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()

        # Update target network
        self.update_target_network()

        return float(loss.item())

    def save_checkpoint(self, path: str, episode: int, epsilon: float, best_score: float) -> None:
        """Save the trainer state to a checkpoint file.

        Args:
            path: File path to save the checkpoint.
            episode: Current training episode number.
            epsilon: Current epsilon value for exploration.
            best_score: Best evaluation score achieved so far.
        """
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "episode": episode,
            "epsilon": epsilon,
            "best_score": best_score,
            "config": dataclasses.asdict(self.config),
        }
        checkpoint["config"]["device"] = str(self.device)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> dict:
        """Load trainer state from a checkpoint file.

        Args:
            path: File path to the checkpoint.

        Returns:
            Dictionary containing metadata (episode, epsilon, best_score).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint is corrupted or incompatible.
        """
        if not path or not isinstance(path, str):
            raise ValueError("Checkpoint path must be a non-empty string.")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint.get("step_count", 0)

        return {
            "episode": checkpoint.get("episode", 0),
            "epsilon": checkpoint.get("epsilon", self.config.epsilon_start),
            "best_score": checkpoint.get("best_score", float("-inf")),
        }
