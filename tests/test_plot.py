"""Unit tests for the visualization plot module."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest

import numpy as np

from visualization.plot import (
    _moving_average,
    _parse_csv,
    generate_plots,
    main,
    plot_epsilon,
    plot_loss,
    plot_rewards,
    plot_score_distribution,
    plot_steps_per_lap,
)


class TestParseCsv(unittest.TestCase):
    """Tests for CSV parsing."""

    def test_parse_valid_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward", "epsilon", "loss", "episode_length", "high_score", "max_y", "laps_completed", "steps_per_lap"])
                writer.writerow(["1", "10.5", "1.0", "0.5", "20", "10.5", "3", "0", "0.00"])
                writer.writerow(["2", "15.0", "0.9", "", "25", "15.0", "4", "1", "25.00"])

            data = _parse_csv(path)
            self.assertEqual(data["episode"], [1.0, 2.0])
            self.assertEqual(data["total_reward"], [10.5, 15.0])
            self.assertTrue(np.isnan(data["loss"][1]))

    def test_parse_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            _parse_csv("/nonexistent/log.csv")


class TestMovingAverage(unittest.TestCase):
    """Tests for moving average computation."""

    def test_empty_list(self):
        self.assertEqual(_moving_average([], 3), [])

    def test_constant_values(self):
        result = _moving_average([5.0, 5.0, 5.0], 2)
        self.assertEqual(result, [5.0, 5.0, 5.0])

    def test_simple_sequence(self):
        result = _moving_average([1.0, 2.0, 3.0, 4.0], 2)
        expected = [1.0, 1.5, 2.5, 3.5]
        self.assertEqual(result, expected)

    def test_window_larger_than_data(self):
        result = _moving_average([1.0, 2.0], 5)
        expected = [1.0, 1.5]
        self.assertEqual(result, expected)


class TestPlotFunctions(unittest.TestCase):
    """Tests for individual plot functions."""

    def test_plot_rewards_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rewards")
            plot_rewards([1, 2, 3], [10, 20, 15], 2, path)
            self.assertTrue(os.path.isfile(path + ".png"))
            self.assertGreater(os.path.getsize(path + ".png"), 0)

    def test_plot_loss_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "loss")
            plot_loss([1, 2, 3], [0.5, 0.4, 0.3], path)
            self.assertTrue(os.path.isfile(path + ".png"))

    def test_plot_loss_no_valid_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "loss")
            plot_loss([1, 2], [float("nan"), float("nan")], path)
            self.assertTrue(os.path.isfile(path + ".png"))

    def test_plot_epsilon_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "epsilon")
            plot_epsilon([1, 2, 3], [1.0, 0.5, 0.1], path)
            self.assertTrue(os.path.isfile(path + ".png"))

    def test_plot_steps_per_lap_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "steps_per_lap")
            plot_steps_per_lap([1, 2, 3], [50.0, 45.0, 40.0], 2, path)
            self.assertTrue(os.path.isfile(path + ".png"))
            self.assertGreater(os.path.getsize(path + ".png"), 0)

    def test_plot_steps_per_lap_no_valid_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "steps_per_lap")
            plot_steps_per_lap([1, 2], [0.0, 0.0], 2, path)
            self.assertTrue(os.path.isfile(path + ".png"))

    def test_plot_score_distribution_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dist")
            plot_score_distribution([10, 20, 15, 25, 30], path)
            self.assertTrue(os.path.isfile(path + ".png"))

    def test_plot_score_distribution_no_valid_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dist")
            plot_score_distribution([float("nan")], path)
            self.assertTrue(os.path.isfile(path + ".png"))


class TestGeneratePlots(unittest.TestCase):
    """Tests for the full plot generation pipeline."""

    def test_generate_all_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "training.csv")
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward", "epsilon", "loss", "episode_length", "high_score", "max_y", "laps_completed", "steps_per_lap"])
                for i in range(1, 11):
                    writer.writerow([i, i * 10, 1.0 - i * 0.1, 1.0 / i, i * 5, i * 10, i, i // 2, float(i * 5)])

            output_dir = os.path.join(tmpdir, "plots")
            paths = generate_plots(log_path, output_dir, window=3)

            self.assertEqual(len(paths), 5)
            self.assertTrue(any("rewards.png" in p for p in paths))
            self.assertTrue(any("loss.png" in p for p in paths))
            self.assertTrue(any("epsilon.png" in p for p in paths))
            self.assertTrue(any("score_distribution.png" in p for p in paths))
            self.assertTrue(any("steps_per_lap.png" in p for p in paths))

    def test_generate_plots_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "training.csv")
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward"])
                writer.writerow([1, 10])

            output_dir = os.path.join(tmpdir, "plots")
            paths = generate_plots(log_path, output_dir)
            self.assertEqual(len(paths), 2)  # rewards and distribution only

    def test_generate_plots_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "training.csv")
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward"])
                writer.writerow([1, 10])

            output_dir = os.path.join(tmpdir, "nested", "plots")
            generate_plots(log_path, output_dir)
            self.assertTrue(os.path.isdir(output_dir))


class TestPlotMain(unittest.TestCase):
    """Tests for the plot CLI entry point."""

    def test_main_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "training.csv")
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward", "epsilon", "loss", "episode_length", "high_score", "max_y", "laps_completed", "steps_per_lap"])
                for i in range(1, 6):
                    writer.writerow([i, i * 10, 1.0 - i * 0.1, 1.0 / i, i * 5, i * 10, i, i // 2, float(i * 5)])

            output_dir = os.path.join(tmpdir, "plots")
            exit_code = main(["--log", log_path, "--output", output_dir])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "rewards.png")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "steps_per_lap.png")))

    def test_main_svg_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "training.csv")
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward", "epsilon", "loss", "episode_length", "high_score", "max_y", "laps_completed", "steps_per_lap"])
                for i in range(1, 6):
                    writer.writerow([i, i * 10, 1.0 - i * 0.1, 1.0 / i, i * 5, i * 10, i, i // 2, float(i * 5)])

            output_dir = os.path.join(tmpdir, "plots")
            exit_code = main(["--log", log_path, "--output", output_dir, "--format", "svg"])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "rewards.svg")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "loss.svg")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "steps_per_lap.svg")))

    def test_main_missing_file(self):
        exit_code = main(["--log", "/nonexistent.csv", "--output", "/tmp/plots"])
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
