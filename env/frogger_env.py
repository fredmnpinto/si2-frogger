"""Gym-like environment wrapper for the Frogger game logic."""

from __future__ import annotations

import random
from typing import Any, Dict, Union

import numpy as np
import torch

from server.logic import Frogger


class Discrete:
    """Minimal discrete action space helper."""

    def __init__(self, n: int) -> None:
        """Initialize a discrete space with ``n`` possible values.

        Args:
            n: Number of discrete actions (must be positive).
        """
        if n <= 0:
            raise ValueError("Discrete space must have a positive size.")
        self.n = n

    def sample(self) -> int:
        """Return a random integer in ``[0, n)``."""
        return random.randrange(self.n)

    def contains(self, x: Any) -> bool:
        """Check whether ``x`` is a valid action index.

        Args:
            x: Value to validate.

        Returns:
            ``True`` if ``x`` is an integer in ``[0, n)``.
        """
        return isinstance(x, int) and 0 <= x < self.n


class FroggerEnv:
    """Gym-like wrapper around :class:`server.logic.Frogger`.

    The wrapper exposes ``reset()`` and ``step(action)`` and handles the
    frame-based movement cooldown internally so that callers do not need to
    worry about it.
    """

    VALID_ACTIONS = ("NORTH", "SOUTH", "EAST", "WEST", "STAY")
    ACTION_TO_INDEX = {"NORTH": 0, "SOUTH": 1, "EAST": 2, "WEST": 3, "STAY": 4}
    INDEX_TO_ACTION = {0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST", 4: "STAY"}

    def __init__(
        self,
        seed: int | None = None,
        reward_forward: float = 10.0,
        reward_checkpoint: float = 50.0,
        reward_lap: float = 100.0,
        reward_death: float = -50.0,
        reward_backward: float = -2.0,
        reward_time: float = +1.0,
        reward_stay: float = -0.5,
    ) -> None:
        """Create a new ``FroggerEnv``.

        Args:
            seed: Optional random seed for reproducibility.
            reward_forward: Reward for reaching a new lane.
            reward_checkpoint: Reward for reaching the middle checkpoint.
            reward_lap: Reward for completing a lap.
            reward_death: Penalty for dying.
            reward_backward: Penalty for moving backward.
            reward_time: Small survival bonus every step to reward staying alive.
            reward_stay: Penalty for staying still to discourage idle behaviour.
        """
        self.game = Frogger(width=11, height=9, fps=30)
        self._dt = 1.0 / self.game.fps
        self._done = True
        self._episode_length = 0
        self._prev_frog_y = 0
        self._prev_frog_x = 0.0
        self._max_y_reached = 0
        self._laps_completed = 0
        self._steps_at_lap_start = 0

        self.reward_forward = reward_forward
        self.reward_checkpoint = reward_checkpoint
        self.reward_lap = reward_lap
        self.reward_death = reward_death
        self.reward_backward = reward_backward
        self.reward_time = reward_time
        self.reward_stay = reward_stay

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int) -> None:
        """Seed the random number generators.

        Args:
            seed: Integer seed used for ``random``, ``numpy``, and ``torch``.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def action_space(self) -> Discrete:
        """Return the action space (5 discrete actions)."""
        return Discrete(5)

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Return a descriptor of the observation space."""
        return {
            "width": self.game.width,
            "height": self.game.height,
            "frog_x": "float",
            "frog_y": "int",
            "lives": "int",
            "score": "int",
            "high_score": "int",
            "game_over": "bool",
            "win": "bool",
            "obstacles": "list[dict]",
        }

    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return the initial state.

        Returns:
            Initial game state dictionary.
        """
        self.game.reset_game()
        self._done = False
        self._episode_length = 0
        self._prev_frog_y = self.game.frog_y
        self._prev_frog_x = self.game.frog_x
        self._max_y_reached = 0
        self._laps_completed = 0
        self._steps_at_lap_start = 0
        return self.game.get_state()

    def _get_info(self) -> Dict[str, Any]:
        """Build the info dictionary returned by :meth:`step`."""
        return {
            "episode_length": self._episode_length,
            "lives": self.game.lives,
            "score": self.game.score,
            "laps": self.game.laps,
            "laps_completed": self._laps_completed,
            "high_score": self.game.high_score,
            "game_over": self.game.game_over,
            "win": self.game.win,
            "max_y_reached": self._max_y_reached,
        }

    def step(self, action: Union[int, str]) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one action in the environment.

        The method internally advances the game clock until the movement
        cooldown is satisfied, executes the action, and performs one final
        game tick.

        Args:
            action: Either an action index (0-3) or a direction string
                ("NORTH", "SOUTH", "EAST", "WEST").

        Returns:
            A tuple of ``(next_state, reward, done, info)``.

        Raises:
            RuntimeError: If :meth:`step` is called after the episode is done.
            ValueError: If the action is not a valid action index or string.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        # Normalize action --------------------------------------------------
        if isinstance(action, int):
            if action not in self.INDEX_TO_ACTION:
                raise ValueError(f"Invalid action: {action}")
            action_str = self.INDEX_TO_ACTION[action]
        elif isinstance(action, str):
            if action not in self.VALID_ACTIONS:
                raise ValueError(f"Invalid action: {action}")
            action_str = action
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        # Snapshot pre-step state -------------------------------------------
        prev_lives = self.game.lives
        prev_laps = self.game.laps
        prev_checkpoint = self.game.current_lap_checkpoint
        prev_max_y = self.game.max_y_reached_in_checkpoint
        self._prev_frog_y = self.game.frog_y
        self._prev_frog_x = self.game.frog_x
        self._episode_length += 1

        # Advance ticks until cooldown satisfied ----------------------------
        while self.game.frames_since_last_move < self.game.move_cooldown_frames:
            self.game.update(self._dt)
            if self.game.game_over or self.game.win:
                self._done = True
                return (
                    self.game.get_state(),
                    self.reward_death,
                    self._done,
                    self._get_info(),
                )

        # Execute action ----------------------------------------------------
        self.game.move_frog(action_str)

        # STAY does not reset cooldown in the underlying logic; mimic it here
        if action_str == "STAY":
            self.game.frames_since_last_move = 0

        # Track max y reached during the episode
        if self.game.frog_y > self._max_y_reached:
            self._max_y_reached = self.game.frog_y

        # One final update --------------------------------------------------
        self.game.update(self._dt)

        # Death dominates all other signals.
        if self.game.lives < prev_lives:
            self._done = True
            return (
                self.game.get_state(),
                self.reward_death,
                self._done,
                self._get_info(),
            )

        # Check if lap completed (frog reached y=8 and reset to y=0)
        if self.game.frog_y == 0 and self._prev_frog_y > 0 and self.game.laps > prev_laps:
            self._laps_completed += 1

        # Compute reward ----------------------------------------------------
        reward = self._compute_reward(
            prev_lives, prev_laps, prev_checkpoint, prev_max_y, action_str
        )

        # Check done --------------------------------------------------------
        self._done = self.game.game_over or self.game.win

        return self.game.get_state(), reward, self._done, self._get_info()

    def _compute_reward(
        self,
        prev_lives: int,
        prev_laps: int,
        prev_checkpoint: int,
        prev_max_y: int,
        action: str,
    ) -> float:
        """Compute the reward for the current step.

        Args:
            prev_lives: Lives before the step.
            prev_laps: Laps before the step.
            prev_checkpoint: Checkpoint value before the step.
            prev_max_y: ``max_y_reached_in_checkpoint`` before the step.
            action: The action string executed this step.

        Returns:
            Scalar reward for this transition.
        """
        reward = self.reward_time  # Small survival bonus every step

        # Forward progress into a new lane.
        progress = self.game.max_y_reached_in_checkpoint - prev_max_y
        if progress > 0:
            reward += self.reward_forward + progress

        # Backward movement (only if no lap was completed, because a lap
        # resets the frog to the start lane).
        if self.game.frog_y < self._prev_frog_y and self.game.laps == prev_laps:
            reward += self.reward_backward

        # Middle checkpoint reached.
        if self.game.current_lap_checkpoint == 50 and prev_checkpoint == 0:
            reward += self.reward_checkpoint

        # Lap completed.
        if self.game.laps > prev_laps:
            reward += self.reward_lap

        # Penalty for staying still.
        if action == "STAY":
            reward += self.reward_stay

        return reward
