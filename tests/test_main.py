"""Integration tests for the training CLI entry point."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from unittest import mock

from training.__main__ import main


class TestTrainingMain(unittest.TestCase):
    """Integration tests for the training CLI."""

    def test_main_runs_successfully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")
            plot_dir = os.path.join(tmpdir, "plots")

            with mock.patch("sys.argv", [
                "training",
                "--episodes", "2",
                "--seed", "42",
                "--log-dir", log_dir,
                "--checkpoint-dir", checkpoint_dir,
                "--plot-dir", plot_dir,
                "--checkpoint-freq", "1",
                "--eval-freq", "10",
            ]):
                exit_code = main()
                self.assertEqual(exit_code, 0)

            self.assertTrue(os.path.isfile(os.path.join(log_dir, "training.csv")))

    def test_main_with_config_file(self):
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"episodes": 2, "seed": 7}, f)

            log_dir = os.path.join(tmpdir, "logs")
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")

            with mock.patch("sys.argv", [
                "training",
                "--config", config_path,
                "--log-dir", log_dir,
                "--checkpoint-dir", checkpoint_dir,
                "--eval-freq", "10",
            ]):
                exit_code = main()
                self.assertEqual(exit_code, 0)

            self.assertTrue(os.path.isfile(os.path.join(log_dir, "training.csv")))

    def test_main_invalid_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "bad.json")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("not json")

            with mock.patch("sys.argv", [
                "training",
                "--config", config_path,
            ]):
                exit_code = main()
                self.assertEqual(exit_code, 1)

    def test_main_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            checkpoint_dir = os.path.join(tmpdir, "checkpoints")

            with mock.patch("sys.argv", [
                "training",
                "--episodes", "1",
                "--log-dir", log_dir,
                "--checkpoint-dir", checkpoint_dir,
                "--eval-freq", "10",
            ]):
                main()

            self.assertTrue(os.path.isdir(log_dir))
            self.assertTrue(os.path.isdir(checkpoint_dir))


if __name__ == "__main__":
    unittest.main()
