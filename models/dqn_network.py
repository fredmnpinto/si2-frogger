"""DQN network and state encoder for the Frogger DQN agent.

This module provides:
    - :class:`StateEncoder`: Converts raw Frogger game state dictionaries into
      fixed-size 22-dimensional feature vectors.
    - :class:`DQNNetwork`: A simple MLP that maps state vectors to Q-values
      for the four discrete actions (NORTH, SOUTH, EAST, WEST).
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

STATE_DIM = 22
"""Dimensionality of the encoded state vector."""

NUM_ACTIONS = 4
"""Number of discrete actions (NORTH, SOUTH, EAST, WEST)."""

HIDDEN_SIZE = 128
"""Default hidden layer size for the DQN network."""

MAX_SPEED = 1.8
"""Maximum obstacle speed magnitude in the game (used for normalization)."""

TRAFFIC_LANES = (1, 2, 3, 5, 6, 7)
"""Y-coordinates of the six traffic lanes."""


class StateEncoder:
    """Encodes raw Frogger game state into a fixed-size feature vector.

    The output vector has :data:`STATE_DIM` (22) elements:

    +---------+------------------------------------------+
    | Indices | Description                              |
    +=========+==========================================+
    | 0-1     | Normalised frog position                 |
    | 2       | Normalised lives                         |
    | 3       | Checkpoint flag (0 or 1)                 |
    | 4-6     | Lane 1: distance, speed, width           |
    | 7-9     | Lane 2: distance, speed, width           |
    | 10-12   | Lane 3: distance, speed, width           |
    | 13-15   | Lane 5: distance, speed, width           |
    | 16-18   | Lane 6: distance, speed, width           |
    | 19-21   | Lane 7: distance, speed, width           |
    +---------+------------------------------------------+
    """

    def __init__(self, width: int = 11, height: int = 9) -> None:
        """Initialize the encoder.

        Args:
            width: Grid width (number of columns).
            height: Grid height (number of rows).
        """
        self.width = width
        self.height = height

    def encode(self, state: Dict[str, Any]) -> torch.Tensor:
        """Encode a raw game state dictionary into a 22-D tensor.

        Args:
            state: Raw state from :meth:`server.logic.Frogger.get_state` or
                :meth:`env.frogger_env.FroggerEnv.step`.

        Returns:
            Float tensor of shape ``(STATE_DIM,)`` with values roughly in
            ``[0, 1]``.
        """
        features = torch.zeros(STATE_DIM, dtype=torch.float32)

        # Frog position (2 features)
        features[0] = float(state["frog_x"]) / self.width
        features[1] = float(state["frog_y"]) / (self.height - 1)

        # Lives (1 feature)
        features[2] = float(state["lives"]) / 3.0

        # Checkpoint flag (1 feature) — infer from score if not present
        checkpoint = state.get("current_lap_checkpoint")
        if checkpoint is None:
            score = state.get("score", 0)
            checkpoint = 50 if (score % 100) >= 50 else 0
        features[3] = float(checkpoint) / 50.0

        # Per-lane features (6 lanes × 3 = 18 features)
        obstacles = state.get("obstacles", [])
        frog_x = float(state["frog_x"])

        for lane_idx, lane in enumerate(TRAFFIC_LANES):
            base = 4 + lane_idx * 3
            lane_obs = [obs for obs in obstacles if obs.get("y") == lane]

            if not lane_obs:
                features[base] = 1.0
                features[base + 1] = 0.0
                features[base + 2] = 0.0
                continue

            min_dist = float("inf")
            nearest_obs = None

            for obs in lane_obs:
                dist = self._distance_to_obstacle(
                    frog_x, float(obs["x"]), float(obs["width"])
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_obs = obs

            features[base] = min_dist / self.width
            features[base + 1] = float(nearest_obs["speed"]) / MAX_SPEED
            features[base + 2] = float(nearest_obs["width"]) / self.width

        return features

    @staticmethod
    def _distance_to_obstacle(
        frog_x: float, obs_x: float, obs_w: float, width: float = 11.0
    ) -> float:
        """Compute minimum wrap-around distance from frog BODY to obstacle.

        The lane is treated as a circle of circumference *width*.  If the frog
        overlaps the obstacle the distance is ``0``.

        Args:
            frog_x: Frog x-coordinate.
            obs_x: Obstacle left edge.
            obs_w: Obstacle width.
            width: Grid width.

        Returns:
            Minimum distance to the obstacle (``0`` if overlapping).
        """
        # Frog body: [frog_x + 0.1, frog_x + 0.9]
        frog_left = frog_x + 0.1
        frog_right = frog_x + 0.9
        frog_center = (frog_left + frog_right) / 2  # = frog_x + 0.5
        frog_half_width = 0.4

        # Compute distance from frog center to obstacle
        dx = (frog_center - obs_x) % width

        # Check collision (accounting for frog body width)
        if dx <= obs_w + frog_half_width:
            return 0.0

        # Distance from frog's nearest edge to obstacle's nearest edge
        return min(
            dx - obs_w - frog_half_width,
            width - dx - frog_half_width
        )


class DQNNetwork(nn.Module):
    """Simple MLP DQN network mapping state vectors to Q-values.

    Architecture (default):
        ``Input[batch, 22] -> Linear(22, 128) + ReLU
                              -> Linear(128, 128) + ReLU
                              -> Linear(128, 4)``

    No activation is applied to the output layer.
    """

    def __init__(
        self, input_dim: int = STATE_DIM, hidden_dim: int = HIDDEN_SIZE
    ) -> None:
        """Initialize the DQN network.

        Args:
            input_dim: Dimensionality of the state vector.
            hidden_dim: Size of the two hidden layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State tensor of shape ``(batch, input_dim)`` or
                ``(input_dim,)`` for a single state.

        Returns:
            Q-values of shape ``(batch, NUM_ACTIONS)`` or
            ``(NUM_ACTIONS,)``.
        """
        return self.net(x)
