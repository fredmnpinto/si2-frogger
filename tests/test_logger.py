"""Unit tests for the training logger module."""

from __future__ import annotations

import csv
import io
import os
import tempfile
import unittest

from training.logger import TrainingLogger


class TestTrainingLogger(unittest.TestCase):
    """Tests for TrainingLogger."""

    def test_creates_log_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "nested", "logs")
            logger = TrainingLogger(log_dir)
            self.assertTrue(os.path.isdir(log_dir))
            logger.close()

    def test_creates_csv_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir, "train.csv")
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "train.csv")))
            logger.close()

    def test_csv_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir)
            logger.close()
            with open(os.path.join(tmpdir, "training.csv"), "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(
                    header,
                    ["episode", "total_reward", "epsilon", "loss", "episode_length", "high_score"],
                )

    def test_log_episode_writes_csv_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            console = io.StringIO()
            logger = TrainingLogger(tmpdir, console=console)
            logger.log_episode(1, 10.5, 1.0, 0.5234, 20)
            logger.close()

            with open(os.path.join(tmpdir, "training.csv"), "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                row = next(reader)
                self.assertEqual(row[0], "1")
                self.assertEqual(row[1], "10.5000")
                self.assertEqual(row[2], "1.0000")
                self.assertEqual(row[3], "0.5234")
                self.assertEqual(row[4], "20")
                self.assertEqual(row[5], "10.5000")

    def test_log_episode_no_console_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            console = io.StringIO()
            logger = TrainingLogger(tmpdir, console=console)
            logger.log_episode(1, 10.5, 1.0, 0.5234, 20)
            logger.close()

            output = console.getvalue()
            self.assertEqual(output, "")

    def test_high_score_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            console = io.StringIO()
            logger = TrainingLogger(tmpdir, console=console)
            logger.log_episode(1, 10.0, 1.0, 0.5, 10)
            logger.log_episode(2, 15.0, 0.9, 0.4, 12)
            logger.log_episode(3, 12.0, 0.8, 0.3, 11)
            logger.close()

            with open(os.path.join(tmpdir, "training.csv"), "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                rows = list(reader)
                self.assertEqual(rows[0][5], "10.0000")
                self.assertEqual(rows[1][5], "15.0000")
                self.assertEqual(rows[2][5], "15.0000")

    def test_none_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            console = io.StringIO()
            logger = TrainingLogger(tmpdir, console=console)
            logger.log_episode(1, 5.0, 1.0, None, 10)
            logger.close()

            with open(os.path.join(tmpdir, "training.csv"), "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)
                row = next(reader)
                self.assertEqual(row[3], "")

            output = console.getvalue()
            self.assertEqual(output, "")

    def test_nan_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            console = io.StringIO()
            logger = TrainingLogger(tmpdir, console=console)
            logger.log_episode(1, 5.0, 1.0, float("nan"), 10)
            logger.close()

            with open(os.path.join(tmpdir, "training.csv"), "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)
                row = next(reader)
                self.assertEqual(row[3], "")

    def test_log_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            console = io.StringIO()
            logger = TrainingLogger(tmpdir, console=console)
            logger.log_summary(episodes=100, best_score=42.5, final_epsilon=0.01)
            logger.close()

            output = console.getvalue()
            self.assertIn("Training complete!", output)
            self.assertIn("100", output)
            self.assertIn("42.50", output)
            self.assertIn("0.0100", output)

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with TrainingLogger(tmpdir) as logger:
                logger.log_episode(1, 1.0, 1.0, 0.1, 5)
            # After context exit, file should be closed
            self.assertTrue(logger._csv_file.closed)

    def test_log_to_file_false_no_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            console = io.StringIO()
            logger = TrainingLogger(tmpdir, console=console, log_to_file=False)
            logger.log_episode(1, 10.5, 1.0, 0.5234, 20)
            logger.close()

            # CSV file should not exist
            self.assertFalse(os.path.isfile(os.path.join(tmpdir, "training.csv")))
            # Console output should also be empty (no per-episode console logging)
            output = console.getvalue()
            self.assertEqual(output, "")

    def test_log_to_file_false_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            console = io.StringIO()
            with TrainingLogger(tmpdir, console=console, log_to_file=False) as logger:
                logger.log_episode(1, 1.0, 1.0, 0.1, 5)
            self.assertIsNone(logger._csv_file)

    def test_multiple_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            console = io.StringIO()
            with TrainingLogger(tmpdir, console=console) as logger:
                for i in range(1, 6):
                    logger.log_episode(i, float(i), 1.0, 0.1, i * 10)

            with open(os.path.join(tmpdir, "training.csv"), "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                rows = list(reader)
                self.assertEqual(len(rows), 5)
                self.assertEqual(rows[-1][0], "5")


if __name__ == "__main__":
    unittest.main()
