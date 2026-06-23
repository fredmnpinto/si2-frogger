"""Unit tests for the high score tracker module."""

from __future__ import annotations

import io
import unittest
from unittest import mock

from training.high_score_tracker import HighScoreTracker


class TestHighScoreTracker(unittest.TestCase):
    """Tests for HighScoreTracker."""

    def test_init_default_max_entries(self):
        tracker = HighScoreTracker()
        self.assertEqual(tracker.max_entries, 3)
        self.assertEqual(tracker.scores, [])
        self.assertEqual(tracker._lines_printed, 0)

    def test_init_custom_max_entries(self):
        tracker = HighScoreTracker(max_entries=5)
        self.assertEqual(tracker.max_entries, 5)
        self.assertEqual(tracker.scores, [])
        self.assertEqual(tracker._lines_printed, 0)

    def test_add_score_stores_data(self):
        tracker = HighScoreTracker(max_entries=3)
        tracker.add_score(
            episode=10,
            score=100.0,
            laps=2,
            total_steps=120,
            steps_per_lap=60.0,
        )
        self.assertEqual(len(tracker.scores), 1)
        self.assertEqual(tracker.scores[0]["episode"], 10)
        self.assertEqual(tracker.scores[0]["score"], 100.0)
        self.assertEqual(tracker.scores[0]["laps"], 2)
        self.assertEqual(tracker.scores[0]["total_steps"], 120)
        self.assertEqual(tracker.scores[0]["steps_per_lap"], 60.0)

    def test_add_score_updates_lines_printed(self):
        tracker = HighScoreTracker(max_entries=3)
        with mock.patch("builtins.print"):
            tracker.add_score(
                episode=1,
                score=10.0,
                laps=1,
                total_steps=50,
                steps_per_lap=50.0,
            )
        # Header + 1 score + 2 empty slots + footer = 5 lines
        self.assertEqual(tracker._lines_printed, 5)

    def test_max_entries_limit(self):
        tracker = HighScoreTracker(max_entries=3)
        with mock.patch("builtins.print"):
            for i in range(5):
                tracker.add_score(
                    episode=i,
                    score=float(i * 10),
                    laps=i,
                    total_steps=i * 10,
                    steps_per_lap=10.0,
                )
        self.assertEqual(len(tracker.scores), 3)
        # Should keep the last 3 entries (episodes 2, 3, 4)
        self.assertEqual(tracker.scores[0]["episode"], 2)
        self.assertEqual(tracker.scores[1]["episode"], 3)
        self.assertEqual(tracker.scores[2]["episode"], 4)

    def test_clear_resets_lines_printed(self):
        tracker = HighScoreTracker(max_entries=3)
        with mock.patch("builtins.print"):
            tracker.add_score(
                episode=1,
                score=10.0,
                laps=1,
                total_steps=50,
                steps_per_lap=50.0,
            )
            self.assertGreater(tracker._lines_printed, 0)
            tracker.clear()
        self.assertEqual(tracker._lines_printed, 0)

    def test_clear_with_no_lines_printed(self):
        tracker = HighScoreTracker(max_entries=3)
        with mock.patch("builtins.print") as mock_print:
            tracker.clear()
            # Should not print anything when no lines have been printed
            mock_print.assert_not_called()
        self.assertEqual(tracker._lines_printed, 0)

    def test_redraw_output_contains_header(self):
        tracker = HighScoreTracker(max_entries=3)
        with mock.patch("builtins.print") as mock_print:
            tracker.add_score(
                episode=1,
                score=10.0,
                laps=1,
                total_steps=50,
                steps_per_lap=50.0,
            )
            calls = [call.args[0] for call in mock_print.call_args_list]
            header = calls[0]
            self.assertIn("Recent High Scores", header)

    def test_redraw_output_contains_score_data(self):
        tracker = HighScoreTracker(max_entries=3)
        with mock.patch("builtins.print") as mock_print:
            tracker.add_score(
                episode=42,
                score=123.4,
                laps=3,
                total_steps=100,
                steps_per_lap=33.3,
            )
            calls = [call.args[0] for call in mock_print.call_args_list]
            score_line = calls[1]
            self.assertIn("Ep   42", score_line)
            self.assertIn("123.4", score_line)
            self.assertIn("3", score_line)
            self.assertIn("33.3", score_line)

    def test_redraw_output_contains_footer(self):
        tracker = HighScoreTracker(max_entries=3)
        with mock.patch("builtins.print") as mock_print:
            tracker.add_score(
                episode=1,
                score=10.0,
                laps=1,
                total_steps=50,
                steps_per_lap=50.0,
            )
            calls = [call.args[0] for call in mock_print.call_args_list]
            footer = calls[-1]
            self.assertIn("└", footer)

    def test_newest_entry_highlighted(self):
        tracker = HighScoreTracker(max_entries=3)
        with mock.patch("builtins.print") as mock_print:
            tracker.add_score(
                episode=1,
                score=10.0,
                laps=1,
                total_steps=50,
                steps_per_lap=50.0,
            )
            calls = [call.args[0] for call in mock_print.call_args_list]
            score_line = calls[1]
            self.assertIn("▶", score_line)

    def test_multiple_scores_newest_first(self):
        tracker = HighScoreTracker(max_entries=3)
        with mock.patch("builtins.print") as mock_print:
            tracker.add_score(
                episode=1,
                score=10.0,
                laps=1,
                total_steps=50,
                steps_per_lap=50.0,
            )
            tracker.add_score(
                episode=2,
                score=20.0,
                laps=2,
                total_steps=60,
                steps_per_lap=30.0,
            )
            calls = [call.args[0] for call in mock_print.call_args_list]
            # Find all score lines (containing "Ep" and "Score:")
            score_lines = [
                c for c in calls
                if "Ep" in c and "Score:" in c
            ]
            # The last two score lines should be from the final redraw (newest first)
            self.assertGreaterEqual(len(score_lines), 2)
            # In the final redraw, episode 2 should come before episode 1
            final_score_lines = score_lines[-2:]
            self.assertIn("Ep    2", final_score_lines[0])
            self.assertIn("Ep    1", final_score_lines[1])

    def test_empty_slots_filled(self):
        tracker = HighScoreTracker(max_entries=3)
        with mock.patch("builtins.print") as mock_print:
            tracker.add_score(
                episode=1,
                score=10.0,
                laps=1,
                total_steps=50,
                steps_per_lap=50.0,
            )
            calls = [call.args[0] for call in mock_print.call_args_list]
            # With 1 score and max_entries=3, there should be 2 empty slot lines
            empty_lines = [
                c for c in calls
                if c.startswith("│") and "Ep" not in c and "Score:" not in c and "Recent" not in c and "└" not in c
            ]
            self.assertEqual(len(empty_lines), 2)

    def test_redraw_uses_ansi_escape_codes(self):
        tracker = HighScoreTracker(max_entries=3)
        tracker._lines_printed = 5
        with mock.patch("builtins.print") as mock_print:
            tracker.add_score(
                episode=1,
                score=10.0,
                laps=1,
                total_steps=50,
                steps_per_lap=50.0,
            )
            # Check that ANSI escape codes are used for clearing
            ansi_calls = [
                call for call in mock_print.call_args_list
                if isinstance(call.args[0], str) and "\033[" in call.args[0]
            ]
            self.assertGreater(len(ansi_calls), 0)

    def test_clear_uses_ansi_escape_codes(self):
        tracker = HighScoreTracker(max_entries=3)
        tracker._lines_printed = 5
        with mock.patch("builtins.print") as mock_print:
            tracker.clear()
            ansi_calls = [
                call for call in mock_print.call_args_list
                if isinstance(call.args[0], str) and "\033[" in call.args[0]
            ]
            self.assertGreater(len(ansi_calls), 0)


if __name__ == "__main__":
    unittest.main()
