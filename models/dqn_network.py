"""DQN network and state encoder for the Frogger DQN agent.

This module provides:
    - :class:`StateEncoder`: Converts raw Frogger game state dictionaries into
      fixed-size 30-dimensional feature vectors.
    - :class:`DQNNetwork`: A simple MLP that maps state vectors to Q-values
      for the five discrete actions (NORTH, SOUTH, EAST, WEST, STAY).
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

STATE_DIM = 32
"""Dimensionality of the encoded state vector."""

NUM_ACTIONS = 5
"""Number of discrete actions (NORTH, SOUTH, EAST, WEST, STAY)."""

HIDDEN_SIZE = 64
"""Default hidden layer size for the DQN network."""

MAX_SPEED = 1.8
"""Maximum obstacle speed magnitude in the game (used for normalization)."""

TRAFFIC_LANES = (1, 2, 3, 5, 6, 7)
"""Y-coordinates of the six traffic lanes."""

# Directional sensor offsets (dx, dy) for 8 directions.
SENSOR_DIRECTIONS = [
    (0, 1),    # N
    (0, -1),   # S
    (1, 0),    # E
    (-1, 0),   # W
    (1, 1),    # NE
    (-1, 1),   # NW
    (1, -1),   # SE
    (-1, -1),  # SW
]
"""Eight directional offsets used for obstacle sensors."""

SENSOR_MAX_RANGE = 3
"""Maximum range (in grid cells) for each directional sensor."""


class StateEncoder:
    """Encodes raw Frogger game state into a fixed-size feature vector.

    The output vector has :data:`STATE_DIM` (32) elements:

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
    | 22-29   | Directional sensors (8 directions)       |
    | 30      | Danger current (1.0 if collision)        |
    | 31      | Safe north (1.0 if clear above)          |
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
        """Encode a raw game state dictionary into a 32-D tensor.

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

        # Directional sensors (8 features)
        frog_y = int(state["frog_y"])
        for direction_idx, (dx, dy) in enumerate(SENSOR_DIRECTIONS):
            sensor_dist = self._sensor_distance(
                frog_x, frog_y, dx, dy, obstacles
            )
            features[22 + direction_idx] = sensor_dist

        # Danger flags (2 features)
        # danger_current: is there a car in the frog's current cell?
        features[30] = 1.0 if self._is_collision(frog_x, frog_y, obstacles) else 0.0

        # safe_north: is the cell above clear?
        if frog_y >= self.height - 1:
            features[31] = 1.0  # Goal is always safe
        else:
            features[31] = 0.0 if self._is_collision(frog_x, frog_y + 1, obstacles) else 1.0

        return features

    def _is_collision(
        self,
        frog_x: float,
        frog_y: int,
        obstacles: List[Dict[str, Any]],
    ) -> bool:
        """Check if the frog at (frog_x, frog_y) collides with any obstacle."""
        for obs in obstacles:
            if obs.get("y") != frog_y:
                continue
            if self._distance_to_obstacle(
                frog_x, float(obs["x"]), float(obs["width"]), self.width
            ) == 0.0:
                return True
        return False

    def _sensor_distance(
        self,
        frog_x: float,
        frog_y: int,
        dx: int,
        dy: int,
        obstacles: List[Dict[str, Any]],
    ) -> float:
        """Cast a ray in direction ``(dx, dy)`` and return nearest obstacle distance.

        The returned value is normalised to ``[0, 1]`` where ``1.0`` means no
        obstacle was found within :data:`SENSOR_MAX_RANGE` cells.

        Args:
            frog_x: Frog x-coordinate.
            frog_y: Frog y-coordinate.
            dx: X direction offset (-1, 0, or 1).
            dy: Y direction offset (-1, 0, or 1).
            obstacles: List of obstacle dictionaries.

        Returns:
            Normalised distance to the nearest obstacle in the given direction.
        """
        for dist in range(1, SENSOR_MAX_RANGE + 1):
            check_x = (frog_x + dx * dist) % self.width
            check_y = frog_y + dy * dist

            for obs in obstacles:
                if obs.get("y") != check_y:
                    continue
                obs_left = float(obs["x"])
                obs_width = float(obs["width"])
                obs_right = obs_left + obs_width

                # Direct overlap
                if obs_left <= check_x < obs_right:
                    return dist / SENSOR_MAX_RANGE

                # Wrap-around right
                if obs_right > self.width:
                    if 0 <= check_x < (obs_right - self.width):
                        return dist / SENSOR_MAX_RANGE

        return 1.0

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
        ``Input[batch, 32] -> Linear(32, 64) + ReLU
                              -> Linear(64, 64) + ReLU
                              -> Linear(64, 5)``

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
