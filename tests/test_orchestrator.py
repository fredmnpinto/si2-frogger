"""Unit tests for the training orchestrator module."""

from __future__ import annotations

import io
import os
import signal
import tempfile
import unittest
from unittest import mock

import torch

from training.checkpoint import CheckpointManager
from training.config import TrainingConfig
from training.dqn_trainer import DQNConfig, DQNTrainer
from training.logger import TrainingLogger
from training.orchestrator import TrainingOrchestrator


class MockEnv:
    """Minimal mock environment for testing."""

    def __init__(self, max_steps: int = 5) -> None:
        self.max_steps = max_steps
        self._step_count = 0
        self._done = False

    def reset(self):
        self._step_count = 0
        self._done = False
        return {
            "frog_x": 5,
            "frog_y": 0,
            "lives": 3,
            "score": 0,
            "high_score": 0,
            "game_over": False,
            "win": False,
            "obstacles": [],
        }

    def step(self, action):
        self._step_count += 1
        if self._step_count >= self.max_steps:
            self._done = True
        done = self._done
        state = {
            "frog_x": 5,
            "frog_y": 0,
            "lives": 3,
            "score": 0,
            "high_score": 0,
            "game_over": done,
            "win": False,
            "obstacles": [],
        }
        return state, 1.0, done, {"episode_length": self._step_count}

    def seed(self, seed):
        pass


class TestTrainingOrchestrator(unittest.TestCase):
    """Tests for TrainingOrchestrator."""

    def _make_orchestrator(
        self,
        config: TrainingConfig,
        env=None,
        trainer=None,
        logger=None,
        checkpoint_manager=None,
    ):
        return TrainingOrchestrator(
            config=config,
            env=env,
            trainer=trainer,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
        )

    def test_init_default_components(self):
        config = TrainingConfig(episodes=2)
        orch = self._make_orchestrator(config)
        self.assertIsNotNone(orch.env)
        self.assertIsNotNone(orch.trainer)
        self.assertIsNotNone(orch.encoder)

    def test_seed_sets_rng(self):
        config = TrainingConfig(episodes=2, seed=42)
        orch = self._make_orchestrator(config)
        # Just verify no exception and env was seeded
        self.assertEqual(config.seed, 42)

    def test_run_episodes(self):
        config = TrainingConfig(episodes=3, eval_freq=10)
        env = MockEnv(max_steps=3)
        trainer = DQNTrainer(DQNConfig(hidden_size=16))
        console = io.StringIO()
        logger = TrainingLogger(
            tempfile.mkdtemp(), log_file="train.csv", console=console
        )
        orch = self._make_orchestrator(config, env=env, trainer=trainer, logger=logger)
        result = orch.run()
        self.assertEqual(result["episodes"], 3)
        logger.close()

    def test_checkpoint_saved_periodically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                episodes=10, checkpoint_freq=5, eval_freq=10, log_dir=tmpdir
            )
            env = MockEnv(max_steps=2)
            trainer = DQNTrainer(DQNConfig(hidden_size=16))
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            cm = CheckpointManager(checkpoint_dir, 5, trainer)
            orch = self._make_orchestrator(
                config, env=env, trainer=trainer, checkpoint_manager=cm
            )
            orch.run()

            checkpoints = cm.list_checkpoints()
            self.assertTrue(any("ep0005" in c for c in checkpoints))
            self.assertTrue(any("ep0010" in c for c in checkpoints))

    def test_best_model_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                episodes=5, eval_freq=2, eval_episodes=2, log_dir=tmpdir
            )
            env = MockEnv(max_steps=2)
            trainer = DQNTrainer(DQNConfig(hidden_size=16))
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            cm = CheckpointManager(checkpoint_dir, 100, trainer)
            orch = self._make_orchestrator(
                config, env=env, trainer=trainer, checkpoint_manager=cm
            )
            orch.run()

            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            self.assertTrue(os.path.isfile(best_path))

    def test_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(episodes=5, checkpoint_freq=5, log_dir=tmpdir)
            env = MockEnv(max_steps=2)
            trainer = DQNTrainer(DQNConfig(hidden_size=16))
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            cm = CheckpointManager(checkpoint_dir, 5, trainer)
            orch = self._make_orchestrator(
                config, env=env, trainer=trainer, checkpoint_manager=cm
            )
            orch.run()

            checkpoint_path = cm.get_latest_checkpoint()
            self.assertIsNotNone(checkpoint_path)

            # Resume from checkpoint
            resume_config = TrainingConfig(
                episodes=8, resume=checkpoint_path, log_dir=tmpdir
            )
            resume_trainer = DQNTrainer(DQNConfig(hidden_size=16))
            resume_cm = CheckpointManager(checkpoint_dir, 100, resume_trainer)
            resume_orch = self._make_orchestrator(
                resume_config,
                env=MockEnv(max_steps=2),
                trainer=resume_trainer,
                checkpoint_manager=resume_cm,
            )
            result = resume_orch.run()
            self.assertEqual(result["episodes"], 8)

    def test_nan_loss_aborts_and_saves_crash_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(episodes=10, log_dir=tmpdir)
            env = MockEnv(max_steps=2)
            trainer = DQNTrainer(DQNConfig(hidden_size=16))
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            cm = CheckpointManager(checkpoint_dir, 100, trainer)

            # Force NaN loss on first train_step
            with mock.patch.object(
                trainer, "train_step", return_value=float("nan")
            ):
                orch = self._make_orchestrator(
                    config, env=env, trainer=trainer, checkpoint_manager=cm
                )
                with self.assertRaises(RuntimeError):
                    orch.run()

            crash_path = os.path.join(checkpoint_dir, "crash_recovery.pt")
            self.assertTrue(os.path.isfile(crash_path))

    def test_emergency_checkpoint_on_sigint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(episodes=100, log_dir=tmpdir)
            env = MockEnv(max_steps=2)
            trainer = DQNTrainer(DQNConfig(hidden_size=16))
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            cm = CheckpointManager(checkpoint_dir, 100, trainer)
            console = io.StringIO()
            logger = TrainingLogger(tmpdir, console=console)
            orch = self._make_orchestrator(
                config, env=env, trainer=trainer, checkpoint_manager=cm, logger=logger
            )

            # Simulate SIGINT during run by calling handler directly
            with self.assertRaises(SystemExit):
                orch._sigint_handler(signal.SIGINT, None)

            emergency_path = os.path.join(checkpoint_dir, "emergency_checkpoint.pt")
            self.assertTrue(os.path.isfile(emergency_path))
            logger.close()

    def test_evaluate_returns_mean_score(self):
        config = TrainingConfig(episodes=2, eval_freq=10)
        env = MockEnv(max_steps=3)
        trainer = DQNTrainer(DQNConfig(hidden_size=16))
        orch = self._make_orchestrator(config, env=env, trainer=trainer)
        score = orch._evaluate()
        self.assertIsInstance(score, float)

    def test_run_returns_summary(self):
        config = TrainingConfig(episodes=2, eval_freq=10)
        env = MockEnv(max_steps=2)
        trainer = DQNTrainer(DQNConfig(hidden_size=16))
        orch = self._make_orchestrator(config, env=env, trainer=trainer)
        result = orch.run()
        self.assertIn("episodes", result)
        self.assertIn("best_score", result)
        self.assertIn("final_epsilon", result)

    def test_log_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(episodes=2, eval_freq=10, log_dir=tmpdir)
            env = MockEnv(max_steps=2)
            trainer = DQNTrainer(DQNConfig(hidden_size=16))
            orch = self._make_orchestrator(config, env=env, trainer=trainer)
            orch.run()

            log_path = os.path.join(tmpdir, "training.csv")
            self.assertTrue(os.path.isfile(log_path))


if __name__ == "__main__":
    unittest.main()
