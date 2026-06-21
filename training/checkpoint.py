"""Checkpoint manager for saving and loading training state.

This module provides:
    - :class:`CheckpointManager`: Handles periodic checkpoints, best-model
      tracking, emergency checkpoints, and crash recovery.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from training.dqn_trainer import DQNTrainer


DEFAULT_CHECKPOINT_PREFIX: str = "checkpoint_ep"
BEST_MODEL_FILENAME: str = "best_model.pt"
EMERGENCY_CHECKPOINT_FILENAME: str = "emergency_checkpoint.pt"
CRASH_RECOVERY_FILENAME: str = "crash_recovery.pt"


class CheckpointManager:
    """Manages checkpoint saving, loading, and best-model tracking.

    Checkpoints are saved to the configured directory with descriptive
    filenames. The best-performing model (by evaluation score) is saved
    separately and never overwritten by subsequent checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_freq: int,
        trainer: DQNTrainer,
    ) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory where checkpoints are stored.
            checkpoint_freq: Save a checkpoint every N episodes.
            trainer: The :class:`DQNTrainer` instance to checkpoint.
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.trainer = trainer
        self._best_score = float("-inf")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def should_save(self, episode: int) -> bool:
        """Determine whether a periodic checkpoint should be saved.

        Args:
            episode: Current episode number (1-indexed).

        Returns:
            ``True`` if a checkpoint should be saved this episode.
        """
        return episode > 0 and episode % self.checkpoint_freq == 0

    def save(
        self,
        episode: int,
        epsilon: float,
        best_score: float,
        filename: Optional[str] = None,
    ) -> str:
        """Save a checkpoint.

        Args:
            episode: Current episode number.
            epsilon: Current epsilon value.
            best_score: Best evaluation score achieved so far.
            filename: Optional custom filename. If ``None``, a default name
                based on the episode is used.

        Returns:
            Path to the saved checkpoint file.

        Raises:
            ValueError: If the filename contains path traversal characters.
        """
        if filename is None:
            filename = f"{DEFAULT_CHECKPOINT_PREFIX}{episode:04d}.pt"
        # Sanitize filename to prevent path traversal
        if ".." in filename or os.path.isabs(filename):
            raise ValueError(
                f"Invalid checkpoint filename: {filename!r}. "
                "Filenames must not contain '..' or absolute paths."
            )
        basename = os.path.basename(filename)
        path = os.path.join(self.checkpoint_dir, basename)
        self.trainer.save_checkpoint(path, episode, epsilon, best_score)
        return path

    def save_best(self, episode: int, epsilon: float, score: float) -> Optional[str]:
        """Save the best model if the given score improves upon the previous best.

        The best model is saved to ``checkpoints/best_model.pt`` and is
        never overwritten by a worse score.

        Args:
            episode: Current episode number.
            epsilon: Current epsilon value.
            score: Evaluation score for this episode.

        Returns:
            Path to the saved best model, or ``None`` if the score did not
            improve.
        """
        if score > self._best_score:
            self._best_score = score
            path = os.path.join(self.checkpoint_dir, BEST_MODEL_FILENAME)
            self.trainer.save_checkpoint(path, episode, epsilon, self._best_score)
            return path
        return None

    def save_emergency(self, episode: int, epsilon: float, best_score: float) -> str:
        """Save an emergency checkpoint (e.g. on SIGINT).

        Args:
            episode: Current episode number.
            epsilon: Current epsilon value.
            best_score: Best evaluation score achieved so far.

        Returns:
            Path to the saved emergency checkpoint.
        """
        path = os.path.join(self.checkpoint_dir, EMERGENCY_CHECKPOINT_FILENAME)
        return self.save(episode, epsilon, best_score, filename=EMERGENCY_CHECKPOINT_FILENAME)

    def save_crash_recovery(self, episode: int, epsilon: float, best_score: float) -> str:
        """Save a crash recovery checkpoint (e.g. on NaN/Inf loss).

        Args:
            episode: Current episode number.
            epsilon: Current epsilon value.
            best_score: Best evaluation score achieved so far.

        Returns:
            Path to the saved crash recovery checkpoint.
        """
        path = os.path.join(self.checkpoint_dir, CRASH_RECOVERY_FILENAME)
        return self.save(episode, epsilon, best_score, filename=CRASH_RECOVERY_FILENAME)

    def load(self, path: str) -> dict:
        """Load a checkpoint from the given path.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Dictionary containing metadata (episode, epsilon, best_score).
        """
        meta = self.trainer.load_checkpoint(path)
        if meta["best_score"] > self._best_score:
            self._best_score = meta["best_score"]
        return meta

    def list_checkpoints(self) -> list[str]:
        """List all periodic checkpoint files in the checkpoint directory.

        Returns:
            Sorted list of checkpoint file paths.
        """
        files = [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith(DEFAULT_CHECKPOINT_PREFIX) and f.endswith(".pt")
        ]
        return sorted(files)

    def get_best_score(self) -> float:
        """Return the best score tracked so far.

        Returns:
            Best evaluation score.
        """
        return self._best_score

    def set_best_score(self, score: float) -> None:
        """Set the best score tracked so far.

        Args:
            score: New best evaluation score.
        """
        self._best_score = score

    def get_latest_checkpoint(self) -> Optional[str]:
        """Return the path to the most recent periodic checkpoint.

        Returns:
            Path to the latest checkpoint, or ``None`` if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None
