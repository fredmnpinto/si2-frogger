"""Unit tests for the checkpoint manager module."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from training.checkpoint import (
    BEST_MODEL_FILENAME,
    CRASH_RECOVERY_FILENAME,
    EMERGENCY_CHECKPOINT_FILENAME,
    CheckpointManager,
)
from training.dqn_trainer import DQNConfig, DQNTrainer


class TestCheckpointManager(unittest.TestCase):
    """Tests for CheckpointManager."""

    def setUp(self):
        self.config = DQNConfig(hidden_size=32)
        self.trainer = DQNTrainer(self.config)

    def test_init_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            CheckpointManager(checkpoint_dir, 100, self.trainer)
            self.assertTrue(os.path.isdir(checkpoint_dir))

    def test_should_save_every_n_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            self.assertFalse(mgr.should_save(1))
            self.assertFalse(mgr.should_save(99))
            self.assertTrue(mgr.should_save(100))
            self.assertTrue(mgr.should_save(200))

    def test_should_save_zero_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            self.assertFalse(mgr.should_save(0))

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            path = mgr.save(100, epsilon=0.5, best_score=50.0)
            self.assertTrue(os.path.isfile(path))
            self.assertIn("checkpoint_ep0100.pt", path)

    def test_save_custom_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            path = mgr.save(1, epsilon=1.0, best_score=0.0, filename="custom.pt")
            self.assertTrue(os.path.isfile(path))
            self.assertIn("custom.pt", path)

    def test_save_best_improves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            path = mgr.save_best(50, epsilon=0.5, score=100.0)
            self.assertIsNotNone(path)
            self.assertIn(BEST_MODEL_FILENAME, path)

    def test_save_best_no_improvement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            mgr.save_best(50, epsilon=0.5, score=100.0)
            path = mgr.save_best(60, epsilon=0.4, score=90.0)
            self.assertIsNone(path)

    def test_save_best_never_overwrites_worse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            path1 = mgr.save_best(50, epsilon=0.5, score=100.0)
            path2 = mgr.save_best(60, epsilon=0.4, score=90.0)
            self.assertIsNotNone(path1)
            self.assertIsNone(path2)
            self.assertTrue(os.path.isfile(path1))

    def test_save_emergency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            path = mgr.save_emergency(123, epsilon=0.3, best_score=50.0)
            self.assertTrue(os.path.isfile(path))
            self.assertIn(EMERGENCY_CHECKPOINT_FILENAME, path)

    def test_save_crash_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            path = mgr.save_crash_recovery(456, epsilon=0.2, best_score=75.0)
            self.assertTrue(os.path.isfile(path))
            self.assertIn(CRASH_RECOVERY_FILENAME, path)

    def test_load_restores_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            mgr.save(100, epsilon=0.5, best_score=50.0)

            # Modify trainer state
            self.trainer.step_count = 999

            meta = mgr.load(os.path.join(tmpdir, "checkpoint_ep0100.pt"))
            self.assertEqual(meta["episode"], 100)
            self.assertAlmostEqual(meta["epsilon"], 0.5)
            self.assertAlmostEqual(meta["best_score"], 50.0)

    def test_load_updates_best_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            path = mgr.save(100, epsilon=0.5, best_score=200.0)
            mgr.load(path)
            self.assertEqual(mgr.get_best_score(), 200.0)

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            mgr.save(100, epsilon=0.5, best_score=50.0)
            mgr.save(200, epsilon=0.3, best_score=100.0)
            mgr.save_best(150, epsilon=0.4, score=150.0)

            checkpoints = mgr.list_checkpoints()
            self.assertEqual(len(checkpoints), 2)
            self.assertIn("checkpoint_ep0100.pt", checkpoints[0])
            self.assertIn("checkpoint_ep0200.pt", checkpoints[1])

    def test_get_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            self.assertIsNone(mgr.get_latest_checkpoint())
            mgr.save(100, epsilon=0.5, best_score=50.0)
            mgr.save(200, epsilon=0.3, best_score=100.0)
            latest = mgr.get_latest_checkpoint()
            self.assertIn("checkpoint_ep0200.pt", latest)

    def test_save_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            with self.assertRaises(ValueError):
                mgr.save(1, epsilon=1.0, best_score=0.0, filename="../escape.pt")
            with self.assertRaises(ValueError):
                mgr.save(1, epsilon=1.0, best_score=0.0, filename="/absolute/path.pt")

    def test_save_uses_basename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            # basename extraction should prevent writing outside checkpoint_dir
            path = mgr.save(1, epsilon=1.0, best_score=0.0, filename="subdir/model.pt")
            self.assertTrue(os.path.isfile(path))
            self.assertIn(tmpdir, path)
            self.assertIn("model.pt", path)

    def test_get_best_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            self.assertEqual(mgr.get_best_score(), float("-inf"))
            mgr.save_best(1, epsilon=1.0, score=100.0)
            self.assertEqual(mgr.get_best_score(), 100.0)

    def test_set_best_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, 100, self.trainer)
            mgr.set_best_score(50.0)
            self.assertEqual(mgr.get_best_score(), 50.0)


if __name__ == "__main__":
    unittest.main()
