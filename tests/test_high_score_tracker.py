"""Unit tests for the high score tracker module."""

from __future__ import annotations

import unittest

from rich.table import Table

from training.high_score_tracker import HighScoreTracker


class TestHighScoreTracker(unittest.TestCase):
    """Tests for HighScoreTracker."""

    def test_init_default_max_entries(self):
        tracker = HighScoreTracker()
        self.assertEqual(tracker.max_entries, 3)
        self.assertEqual(tracker.scores, [])

    def test_init_custom_max_entries(self):
        tracker = HighScoreTracker(max_entries=5)
        self.assertEqual(tracker.max_entries, 5)
        self.assertEqual(tracker.scores, [])

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

    def test_max_entries_limit(self):
        tracker = HighScoreTracker(max_entries=3)
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

    def test_clear_resets_scores(self):
        tracker = HighScoreTracker(max_entries=3)
        tracker.add_score(
            episode=1,
            score=10.0,
            laps=1,
            total_steps=50,
            steps_per_lap=50.0,
        )
        self.assertEqual(len(tracker.scores), 1)
        tracker.clear()
        self.assertEqual(tracker.scores, [])

    def test_get_table_returns_rich_table(self):
        tracker = HighScoreTracker(max_entries=3)
        tracker.add_score(
            episode=1,
            score=10.0,
            laps=1,
            total_steps=50,
            steps_per_lap=50.0,
        )
        table = tracker.get_table()
        self.assertIsInstance(table, Table)
        self.assertEqual(table.title, "Recent High Scores")

    def test_get_table_columns(self):
        tracker = HighScoreTracker(max_entries=3)
        table = tracker.get_table()
        column_names = [col.header for col in table.columns]
        self.assertEqual(column_names, ["Rank", "Episode", "Score", "Laps", "Steps/Lap"])

    def test_get_table_empty(self):
        tracker = HighScoreTracker(max_entries=3)
        table = tracker.get_table()
        self.assertEqual(len(table.rows), 0)

    def test_get_table_newest_first_ordering(self):
        tracker = HighScoreTracker(max_entries=3)
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
        table = tracker.get_table()
        self.assertEqual(len(table.rows), 2)
        # Newest first: episode 2 should be row 0, episode 1 should be row 1
        # Cells are stored per column: table.columns[col_index]._cells[row_index]
        row0_episode = table.columns[1]._cells[0]
        row1_episode = table.columns[1]._cells[1]
        self.assertEqual(str(row0_episode), "2")
        self.assertEqual(str(row1_episode), "1")

    def test_get_table_highlights_newest(self):
        tracker = HighScoreTracker(max_entries=3)
        tracker.add_score(
            episode=1,
            score=10.0,
            laps=1,
            total_steps=50,
            steps_per_lap=50.0,
        )
        table = tracker.get_table()
        # The newest (and only) entry should have bold green style
        self.assertEqual(table.rows[0].style, "bold green")

    def test_get_table_rank_indicator(self):
        tracker = HighScoreTracker(max_entries=3)
        tracker.add_score(
            episode=1,
            score=10.0,
            laps=1,
            total_steps=50,
            steps_per_lap=50.0,
        )
        table = tracker.get_table()
        rank_cell = table.columns[0]._cells[0]
        self.assertIn("▶", str(rank_cell))

    def test_get_table_multiple_ranks(self):
        tracker = HighScoreTracker(max_entries=3)
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
        table = tracker.get_table()
        # First row (newest) should have ▶ indicator
        rank0 = str(table.columns[0]._cells[0])
        self.assertIn("▶", rank0)
        # Second row should not have ▶ indicator
        rank1 = str(table.columns[0]._cells[1])
        self.assertNotIn("▶", rank1)

    def test_get_table_formatting(self):
        tracker = HighScoreTracker(max_entries=3)
        tracker.add_score(
            episode=42,
            score=123.4,
            laps=3,
            total_steps=100,
            steps_per_lap=33.3,
        )
        table = tracker.get_table()
        # Cells are stored per column: table.columns[col_index]._cells[row_index]
        self.assertEqual(str(table.columns[1]._cells[0]), "42")  # Episode
        self.assertEqual(str(table.columns[2]._cells[0]), "123.4")  # Score
        self.assertEqual(str(table.columns[3]._cells[0]), "3")  # Laps
        self.assertEqual(str(table.columns[4]._cells[0]), "33.3")  # Steps/Lap


if __name__ == "__main__":
    unittest.main()
