"""Training logger for console and CSV file output.

This module provides:
    - :class:`TrainingLogger`: Logs training metrics to both the console and
      a CSV file.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from typing import Optional, TextIO

from tqdm import tqdm


class TrainingLogger:
    """Logger that writes training metrics to console and CSV file.

    The CSV format is::

        episode,total_reward,epsilon,loss,episode_length,high_score

    Console output prints a formatted line per episode.
    """

    def __init__(
        self,
        log_dir: str,
        log_file: str = "training.csv",
        console: Optional[TextIO] = None,
        log_to_file: bool = True,
    ) -> None:
        """Initialize the logger.

        Creates the log directory if it does not exist and opens the CSV
        file for writing.

        Args:
            log_dir: Directory where the CSV log will be stored.
            log_file: Name of the CSV log file.
            console: Output stream for console logging (defaults to ``sys.stdout``).
            log_to_file: If ``True``, write metrics to a CSV file. If ``False``,
                only console logging is performed.
        """
        self.log_dir = log_dir
        self.log_file = log_file
        self.console = console if console is not None else sys.stdout
        self._high_score = float("-inf")
        self._log_to_file = log_to_file

        os.makedirs(self.log_dir, exist_ok=True)
        self._csv_path = os.path.join(self.log_dir, self.log_file)
        if self._log_to_file:
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._csv_file)
            self._writer.writerow(
                ["episode", "total_reward", "epsilon", "loss", "episode_length", "high_score"]
            )
            self._csv_file.flush()
        else:
            self._csv_file = None
            self._writer = None

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        epsilon: float,
        loss: Optional[float],
        episode_length: int,
    ) -> None:
        """Log metrics for a completed episode to CSV only.

        Args:
            episode: Episode number (1-indexed).
            total_reward: Cumulative reward for the episode.
            epsilon: Current epsilon value.
            loss: Average loss for the episode, or ``None`` if no training
                occurred.
            episode_length: Number of steps in the episode.
        """
        if total_reward > self._high_score:
            self._high_score = total_reward

        # CSV log only (no console output)
        if self._log_to_file and self._writer is not None:
            self._writer.writerow(
                [
                    episode,
                    f"{total_reward:.4f}",
                    f"{epsilon:.4f}",
                    f"{loss:.4f}" if loss is not None and self._is_finite(loss) else "",
                    episode_length,
                    f"{self._high_score:.4f}",
                ]
            )
            if self._csv_file is not None:
                self._csv_file.flush()

    def log_summary(self, episodes: int, best_score: float, final_epsilon: float) -> None:
        """Print an end-of-training summary to the console.

        Args:
            episodes: Total number of episodes trained.
            best_score: Best score achieved during training.
            final_epsilon: Final epsilon value.
        """
        summary = (
            f"\nTraining complete! Episodes: {episodes} | "
            f"Best score: {best_score:.2f} | Final epsilon: {final_epsilon:.4f}\n"
        )
        tqdm.write(summary, file=self.console)

    def _is_finite(self, value: float) -> bool:
        """Check if a float value is finite (not NaN or Inf).

        Args:
            value: Float value to check.

        Returns:
            ``True`` if the value is finite, ``False`` otherwise.
        """
        return math.isfinite(value)

    def close(self) -> None:
        """Close the CSV file and release resources."""
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()

    def __enter__(self) -> "TrainingLogger":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, ensuring the file is closed."""
        self.close()
