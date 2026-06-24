"""Unit tests for the training configuration module."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from training.config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_EPISODES,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_FREQ,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_FILE,
    DEFAULT_MAX_STEPS_PER_LAP,
    DEFAULT_MAX_TOTAL_STEPS,
    DEFAULT_PLOT_DIR,
    TrainingConfig,
    parse_args,
)
from training.dqn_trainer import DQNConfig


class TestTrainingConfigDefaults(unittest.TestCase):
    """Tests for default configuration values."""

    def test_default_episodes(self):
        config = TrainingConfig()
        self.assertEqual(config.episodes, DEFAULT_EPISODES)

    def test_default_checkpoint_freq(self):
        config = TrainingConfig()
        self.assertEqual(config.checkpoint_freq, DEFAULT_CHECKPOINT_FREQ)

    def test_default_eval_freq(self):
        config = TrainingConfig()
        self.assertEqual(config.eval_freq, DEFAULT_EVAL_FREQ)

    def test_default_eval_episodes(self):
        config = TrainingConfig()
        self.assertEqual(config.eval_episodes, DEFAULT_EVAL_EPISODES)

    def test_default_directories(self):
        config = TrainingConfig()
        self.assertEqual(config.log_dir, DEFAULT_LOG_DIR)
        self.assertEqual(config.checkpoint_dir, DEFAULT_CHECKPOINT_DIR)
        self.assertEqual(config.plot_dir, DEFAULT_PLOT_DIR)

    def test_default_log_file(self):
        config = TrainingConfig()
        self.assertEqual(config.log_file, DEFAULT_LOG_FILE)

    def test_default_seed_is_none(self):
        config = TrainingConfig()
        self.assertIsNone(config.seed)

    def test_default_resume_is_none(self):
        config = TrainingConfig()
        self.assertIsNone(config.resume)

    def test_default_max_steps_per_lap(self):
        config = TrainingConfig()
        self.assertEqual(config.max_steps_per_lap, DEFAULT_MAX_STEPS_PER_LAP)

    def test_default_max_total_steps(self):
        config = TrainingConfig()
        self.assertEqual(config.max_total_steps, DEFAULT_MAX_TOTAL_STEPS)

    def test_default_dqn_config(self):
        config = TrainingConfig()
        self.assertIsInstance(config.dqn, DQNConfig)
        self.assertEqual(config.dqn.learning_rate, 5e-4)


class TestTrainingConfigSerialization(unittest.TestCase):
    """Tests for config serialization and deserialization."""

    def test_to_dict(self):
        config = TrainingConfig(episodes=500, seed=42)
        d = config.to_dict()
        self.assertEqual(d["episodes"], 500)
        self.assertEqual(d["seed"], 42)
        self.assertIn("dqn_learning_rate", d)

    def test_from_dict(self):
        data = {
            "episodes": 2000,
            "seed": 7,
            "dqn_learning_rate": 5e-4,
            "dqn_hidden_size": 64,
        }
        config = TrainingConfig.from_dict(data)
        self.assertEqual(config.episodes, 2000)
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.dqn.learning_rate, 5e-4)
        self.assertEqual(config.dqn.hidden_size, 64)

    def test_from_dict_ignores_invalid_keys(self):
        data = {"episodes": 100, "invalid_key": "should_be_ignored"}
        config = TrainingConfig.from_dict(data)
        self.assertEqual(config.episodes, 100)

    def test_round_trip(self):
        original = TrainingConfig(episodes=750, seed=99)
        d = original.to_dict()
        restored = TrainingConfig.from_dict(d)
        self.assertEqual(restored.episodes, 750)
        self.assertEqual(restored.seed, 99)
        self.assertEqual(restored.dqn.learning_rate, original.dqn.learning_rate)

    def test_from_json(self):
        config = TrainingConfig(episodes=300, seed=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f)
            loaded = TrainingConfig.from_json(path)
            self.assertEqual(loaded.episodes, 300)
            self.assertEqual(loaded.seed, 1)

    def test_from_json_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            TrainingConfig.from_json("/nonexistent/config.json")

    def test_from_json_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("not json")
            with self.assertRaises(json.JSONDecodeError):
                TrainingConfig.from_json(path)


class TestParseArgs(unittest.TestCase):
    """Tests for CLI argument parsing."""

    def test_default_args(self):
        config = parse_args([])
        self.assertEqual(config.episodes, DEFAULT_EPISODES)
        self.assertIsNone(config.seed)

    def test_override_episodes(self):
        config = parse_args(["--episodes", "2000"])
        self.assertEqual(config.episodes, 2000)

    def test_override_seed(self):
        config = parse_args(["--seed", "42"])
        self.assertEqual(config.seed, 42)

    def test_override_resume(self):
        config = parse_args(["--resume", "checkpoints/checkpoint.pt"])
        self.assertEqual(config.resume, "checkpoints/checkpoint.pt")

    def test_override_dqn_params(self):
        config = parse_args([
            "--learning-rate", "0.005",
            "--hidden-size", "64",
            "--batch-size", "64",
        ])
        self.assertEqual(config.dqn.learning_rate, 0.005)
        self.assertEqual(config.dqn.hidden_size, 64)
        self.assertEqual(config.dqn.batch_size, 64)

    def test_config_file_override(self):
        data = {"episodes": 500, "seed": 10, "dqn_learning_rate": 0.001}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            config = parse_args(["--config", path])
            self.assertEqual(config.episodes, 500)
            self.assertEqual(config.seed, 10)
            self.assertEqual(config.dqn.learning_rate, 0.001)

    def test_cli_overrides_config_file(self):
        data = {"episodes": 500, "seed": 10}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            config = parse_args(["--config", path, "--episodes", "2000"])
            self.assertEqual(config.episodes, 2000)
            self.assertEqual(config.seed, 10)

    def test_override_max_steps_per_lap(self):
        config = parse_args(["--max-steps-per-lap", "150"])
        self.assertEqual(config.max_steps_per_lap, 150)

    def test_override_max_total_steps(self):
        config = parse_args(["--max-total-steps", "3000"])
        self.assertEqual(config.max_total_steps, 3000)

    def test_all_dqn_overrides(self):
        config = parse_args([
            "--learning-rate", "0.005",
            "--gamma", "0.95",
            "--epsilon-start", "1.0",
            "--epsilon-end", "0.05",
            "--epsilon-decay-steps", "5000",
            "--buffer-size", "25000",
            "--batch-size", "64",
            "--target-update-freq", "500",
            "--tau", "0.5",
            "--update-frequency", "2",
            "--hidden-size", "64",
            "--gradient-clip", "5.0",
            "--device", "cpu",
        ])
        self.assertEqual(config.dqn.learning_rate, 0.005)
        self.assertEqual(config.dqn.gamma, 0.95)
        self.assertEqual(config.dqn.epsilon_start, 1.0)
        self.assertEqual(config.dqn.epsilon_end, 0.05)
        self.assertEqual(config.dqn.epsilon_decay_steps, 5000)
        self.assertEqual(config.dqn.buffer_size, 25000)
        self.assertEqual(config.dqn.batch_size, 64)
        self.assertEqual(config.dqn.target_update_freq, 500)
        self.assertEqual(config.dqn.tau, 0.5)
        self.assertEqual(config.dqn.update_frequency, 2)
        self.assertEqual(config.dqn.hidden_size, 64)
        self.assertEqual(config.dqn.gradient_clip, 5.0)
        self.assertEqual(config.dqn.device, "cpu")


if __name__ == "__main__":
    unittest.main()
