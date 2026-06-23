"""Training orchestrator that runs the full DQN training loop.

This module provides:
    - :class:`TrainingOrchestrator`: Manages the episode loop, evaluation,
      checkpointing, and signal handling.
"""

from __future__ import annotations

import math
import signal
import sys
import time
from typing import Any, Dict, Optional

import torch

from rich.progress import ProgressColumn
from rich.text import Text

from env.frogger_env import FroggerEnv
from models.dqn_network import StateEncoder
from training.checkpoint import CheckpointManager
from training.config import TrainingConfig
from training.dqn_trainer import DQNTrainer
from training.high_score_tracker import HighScoreTracker
from training.logger import TrainingLogger


class SmoothedTimeRemainingColumn(ProgressColumn):
    """Renders estimated time remaining with exponential smoothing.

    Uses a smoothed speed estimate to reduce jitter when task
    completion rate varies (e.g. episodes of very different lengths).
    """

    def __init__(self, smoothing_factor: float = 0.1, **kwargs):
        """Initialize with smoothing factor.

        Args:
            smoothing_factor: Weight for new observations
                (0.1 = 10% new, 90% history).
        """
        self.smoothing_factor = smoothing_factor
        self._smoothed_speed: Optional[float] = None
        self._last_update_time: float = 0.0
        self._last_rendered: str = ""
        super().__init__(**kwargs)

    def render(self, task) -> Text:
        """Render smoothed time remaining."""
        if task.total is None or task.speed is None:
            return Text("--:--", style="progress.remaining")

        # Update smoothed speed at most once per second to reduce flicker,
        # but always initialize on first call.
        now = time.time()
        if self._smoothed_speed is None or now - self._last_update_time >= 1.0:
            self._last_update_time = now
            current_speed = task.speed
            if self._smoothed_speed is None:
                self._smoothed_speed = current_speed
            else:
                self._smoothed_speed = (
                    self.smoothing_factor * current_speed
                    + (1 - self.smoothing_factor) * self._smoothed_speed
                )

        # Calculate ETA from smoothed speed
        remaining = task.total - task.completed
        if remaining <= 0 or self._smoothed_speed is None or self._smoothed_speed <= 0:
            return Text("00:00", style="progress.remaining")

        eta_seconds = remaining / self._smoothed_speed

        # Format as MM:SS or HH:MM:SS
        if eta_seconds < 3600:
            minutes, seconds = divmod(int(eta_seconds), 60)
            text = f"{minutes:02d}:{seconds:02d}"
        else:
            hours, remainder = divmod(int(eta_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            text = f"{hours:d}:{minutes:02d}:{seconds:02d}"

        self._last_rendered = text
        return Text(text, style="progress.remaining")


class TrainingOrchestrator:
    """Orchestrates the full DQN training loop.

    The orchestrator wires together the environment, state encoder, DQN
    trainer, logger, and checkpoint manager. It runs episodes, performs
    evaluation, saves checkpoints, and handles graceful interruption.
    """

    def __init__(
        self,
        config: TrainingConfig,
        env: Optional[FroggerEnv] = None,
        trainer: Optional[DQNTrainer] = None,
        logger: Optional[TrainingLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        """Initialize the training orchestrator.

        Args:
            config: Training configuration.
            env: Optional pre-created environment.
            trainer: Optional pre-created DQN trainer.
            logger: Optional pre-created training logger.
            checkpoint_manager: Optional pre-created checkpoint manager.
        """
        self.config = config
        self.env = env if env is not None else FroggerEnv()
        self.encoder = StateEncoder()
        self.trainer = trainer if trainer is not None else DQNTrainer(config.dqn)
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager

        self._episode = 0
        self._best_score = float("-inf")
        self._best_laps = 0
        self._best_steps = 0
        self._best_steps_per_lap = float("inf")  # Lower is better
        self._interrupted = False
        self._emergency_saved = False
        self.high_score_tracker = HighScoreTracker(max_entries=3)

        # Banner state for Rich Live display
        self._banner_message: Optional[str] = None
        self._banner_time: float = 0.0
        self._banner_duration: float = 8.0  # Show banner for 8 seconds

        # Seed if requested
        if config.seed is not None:
            self.env.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        # Resume from checkpoint if requested
        if config.resume is not None:
            self._resume_from_checkpoint(config.resume)

        # Register SIGINT handler
        signal.signal(signal.SIGINT, self._sigint_handler)

    def _sigint_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGINT by saving an emergency checkpoint and exiting."""
        self._interrupted = True
        if not self._emergency_saved and self.checkpoint_manager is not None:
            self.checkpoint_manager.save_emergency(
                self._episode,
                self._get_current_epsilon(),
                self._best_score,
            )
            self._emergency_saved = True
            if self.logger is not None:
                print(
                    "\nEmergency checkpoint saved. Exiting gracefully.\n",
                    file=self.logger.console,
                )
        sys.exit(0)

    def _resume_from_checkpoint(self, path: str) -> None:
        """Resume training state from a checkpoint file.

        Args:
            path: Path to the checkpoint file.
        """
        meta = self.trainer.load_checkpoint(path)
        self._episode = meta["episode"]
        self._best_score = meta["best_score"]
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.set_best_score(self._best_score)

    def _get_current_epsilon(self) -> float:
        """Compute epsilon for the current training step.

        Returns:
            Current epsilon value.
        """
        return self.trainer.get_epsilon(self.trainer.step_count)

    def _run_episode(self) -> tuple[float, int, Optional[float], int, int, float]:
        """Run a single training episode.

        Returns:
            Tuple of ``(total_reward, episode_length, average_loss, max_y,
            laps_completed, steps_per_lap)``.
        """
        state = self.env.reset()
        state_tensor = self.encoder.encode(state).to(self.trainer.device)
        done = False
        total_reward = 0.0
        episode_losses: list[float] = []
        steps = 0
        max_steps_per_lap = self.config.max_steps_per_lap
        max_total_steps = self.config.max_total_steps
        max_y = 0
        laps_completed = 0
        steps_at_lap_start = 0
        truncated = False

        while not done and steps < max_total_steps:
            epsilon = self._get_current_epsilon()
            action = self.trainer.select_action(state_tensor, epsilon)
            next_state, reward, done, info = self.env.step(action)
            if info.get("max_y_reached", 0) > max_y:
                max_y = info["max_y_reached"]

            # Track lap completion and reset lap timer
            if info.get("laps_completed", 0) > laps_completed:
                laps_completed = info["laps_completed"]
                steps_at_lap_start = steps

            # Check steps-per-lap limit
            steps_since_lap = steps - steps_at_lap_start
            if steps_since_lap >= max_steps_per_lap:
                done = True
                truncated = True
                break

            next_state_tensor = self.encoder.encode(next_state).to(self.trainer.device)

            self.trainer.replay_buffer.push(
                state_tensor, action, reward, next_state_tensor, done
            )

            loss = self.trainer.train_step()
            if loss is not None:
                episode_losses.append(loss)
                if not math.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss detected at episode {self._episode}, "
                        f"step {steps}: {loss}"
                    )

            total_reward += reward
            state_tensor = next_state_tensor
            steps += 1

        # If truncated by total step limit, mark as done
        if steps >= max_total_steps and not done:
            done = True
            truncated = True

        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else None
        steps_per_lap = steps / laps_completed if laps_completed > 0 else 0.0
        return total_reward, steps, avg_loss, max_y, laps_completed, steps_per_lap

    def _evaluate(self) -> float:
        """Run an evaluation over multiple episodes with epsilon=0.

        Returns:
            Mean total reward over the evaluation episodes.
        """
        scores: list[float] = []
        max_steps_per_lap = self.config.max_steps_per_lap
        max_total_steps = self.config.max_total_steps

        for _ in range(self.config.eval_episodes):
            state = self.env.reset()
            state_tensor = self.encoder.encode(state).to(self.trainer.device)
            done = False
            total_reward = 0.0
            steps = 0
            laps_completed = 0
            steps_at_lap_start = 0

            while not done and steps < max_total_steps:
                action = self.trainer.select_action(state_tensor, epsilon=0.0)
                next_state, reward, done, info = self.env.step(action)

                # Track lap completion and reset lap timer
                if info.get("laps_completed", 0) > laps_completed:
                    laps_completed = info["laps_completed"]
                    steps_at_lap_start = steps

                # Check steps-per-lap limit
                steps_since_lap = steps - steps_at_lap_start
                if steps_since_lap >= max_steps_per_lap:
                    done = True
                    break

                state_tensor = self.encoder.encode(next_state).to(self.trainer.device)
                total_reward += reward
                steps += 1

            scores.append(total_reward)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        return mean_score

    def run(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary containing training summary statistics.
        """
        if self.logger is None:
            self.logger = TrainingLogger(
                self.config.log_dir,
                self.config.log_file,
            )
        if self.checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(
                self.config.checkpoint_dir,
                self.config.checkpoint_freq,
                self.trainer,
            )

        # Local imports so tests can easily mock Rich components
        from rich.progress import (
            Progress,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
            TimeElapsedColumn,
            MofNCompleteColumn,
        )
        from rich.live import Live
        from rich.console import Group

        progress = Progress(
            TextColumn("[bold blue]Training"),
            MofNCompleteColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("•"),
            SmoothedTimeRemainingColumn(smoothing_factor=0.05),
            TextColumn("• EPS: {task.fields[epsilon]:.3f}"),
            TextColumn("• Best: {task.fields[best_score]:.1f}"),
            TextColumn("• Reward: {task.fields[reward]:.1f}"),
            TextColumn("• Laps: {task.fields[laps]}"),
            TextColumn("• MaxY: {task.fields[max_y]}"),
        )

        task_id = progress.add_task(
            "training",
            total=self.config.episodes,
            completed=self._episode,
            epsilon=self._get_current_epsilon(),
            best_score=self._best_score,
            reward=0.0,
            laps=0,
            max_y=0,
        )

        # Create a group that combines progress + optional banner + high score table
        def render_display():
            from rich.panel import Panel

            elements = [progress]

            if self._banner_message is not None:
                banner_panel = Panel(
                    self._banner_message,
                    title="[bold yellow]🎉 Achievement",
                    border_style="yellow",
                    padding=(0, 1),
                )
                elements.append(banner_panel)

            table = self.high_score_tracker.get_table()
            elements.append(table)

            return Group(*elements)

        try:
            with Live(render_display(), refresh_per_second=4) as live:
                for episode in range(self._episode + 1, self.config.episodes + 1):
                    self._episode = episode
                    total_reward, length, loss, max_y, laps, steps_per_lap = self._run_episode()

                    # Check if new best
                    if total_reward > self._best_score:
                        prev_best_score = self._best_score
                        prev_best_laps = self._best_laps
                        prev_best_steps = self._best_steps
                        prev_best_steps_per_lap = self._best_steps_per_lap

                        self._best_score = total_reward
                        self._best_laps = laps
                        self._best_steps = length
                        self._best_steps_per_lap = steps_per_lap

                        # Log the new best! (only after episode 100 to reduce early noise)
                        if self.logger is not None and episode >= 100:
                            banner = self.logger.log_new_best(
                                episode=episode,
                                score=total_reward,
                                laps=laps,
                                total_steps=length,
                                steps_per_lap=steps_per_lap,
                                prev_best_score=prev_best_score,
                                prev_best_laps=prev_best_laps,
                                prev_best_steps=prev_best_steps,
                                prev_best_steps_per_lap=prev_best_steps_per_lap,
                            )
                            self._banner_message = banner
                            self._banner_time = time.time()

                        # Show in-place high score tracker (only after ep 100)
                        if episode >= 100:
                            self.high_score_tracker.add_score(
                                episode=episode,
                                score=total_reward,
                                laps=laps,
                                total_steps=length,
                                steps_per_lap=steps_per_lap,
                            )

                    # Clear banner after duration
                    if self._banner_message is not None and time.time() - self._banner_time >= self._banner_duration:
                        self._banner_message = None

                    # Still log to CSV for plotting
                    self.logger.log_episode(
                        episode, total_reward, self._get_current_epsilon(), loss, length, max_y,
                        laps, steps_per_lap,
                    )

                    # Update progress bar
                    progress.update(
                        task_id,
                        advance=1,
                        epsilon=self._get_current_epsilon(),
                        best_score=self._best_score,
                        reward=total_reward,
                        laps=laps,
                        max_y=max_y,
                    )

                    # Refresh live display (triggers re-render with updated table)
                    live.update(render_display())

                    # Periodic checkpoint
                    if self.checkpoint_manager.should_save(episode):
                        self.checkpoint_manager.save(
                            episode,
                            self._get_current_epsilon(),
                            self._best_score,
                        )

                    # Periodic evaluation and best-model tracking (silent)
                    if episode % self.config.eval_freq == 0:
                        eval_score = self._evaluate()
                        self.checkpoint_manager.save_best(
                            episode,
                            self._get_current_epsilon(),
                            eval_score,
                        )
                        if eval_score > self._best_score:
                            self._best_score = eval_score

        except RuntimeError as exc:
            if "Non-finite loss" in str(exc):
                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.save_crash_recovery(
                        self._episode,
                        self._get_current_epsilon(),
                        self._best_score,
                    )
                if self.logger is not None:
                    print(
                        f"\nTraining aborted due to non-finite loss: {exc}\n"
                        f"Crash recovery checkpoint saved.\n",
                        file=self.logger.console,
                    )
            raise

        finally:
            if self.logger is not None:
                self.logger.close()

        if self.logger is not None:
            self.logger.log_summary(
                self._episode,
                self._best_score,
                self._get_current_epsilon(),
            )

        return {
            "episodes": self._episode,
            "best_score": self._best_score,
            "final_epsilon": self._get_current_epsilon(),
        }

    def save_emergency_checkpoint(self) -> None:
        """Public method to save an emergency checkpoint.

        This is called by the SIGINT handler and can also be invoked
        externally.
        """
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save_emergency(
                self._episode,
                self._get_current_epsilon(),
                self._best_score,
            )
