"""High score tracker using Rich tables for in-place display."""

from typing import Dict, List

from rich.table import Table


class HighScoreTracker:
    """Tracks last N high scores and renders them as a Rich table.

    The table is designed to be used within a Rich Live display for
    in-place updates without scrolling.
    """

    def __init__(self, max_entries: int = 3) -> None:
        """Initialize the tracker.

        Args:
            max_entries: Number of high scores to display (default 3).
        """
        self.max_entries = max_entries
        self.scores: List[Dict] = []

    def add_score(
        self,
        episode: int,
        score: float,
        laps: int,
        total_steps: int,
        steps_per_lap: float,
    ) -> None:
        """Add a new high score.

        Args:
            episode: Episode number.
            score: Total reward score.
            laps: Number of laps completed.
            total_steps: Total steps in episode.
            steps_per_lap: Average steps per lap.
        """
        self.scores.append({
            "episode": episode,
            "score": score,
            "laps": laps,
            "total_steps": total_steps,
            "steps_per_lap": steps_per_lap,
        })
        # Keep only last N entries
        self.scores = self.scores[-self.max_entries:]

    def get_table(self) -> Table:
        """Generate a Rich table of the current high scores.

        Returns:
            Rich Table with the last N high scores (newest first).
        """
        table = Table(
            title="Recent High Scores",
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 1),
        )
        table.add_column("Rank", style="bold", width=4)
        table.add_column("Episode", justify="right", width=6)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Laps", justify="right", width=4)
        table.add_column("Steps/Lap", justify="right", width=9)

        for i, score in enumerate(reversed(self.scores)):
            rank = i + 1
            ep = score["episode"]
            sc = score["score"]
            lp = score["laps"]
            spl = score["steps_per_lap"]

            # Highlight the newest entry
            rank_str = f"▶#{rank}" if i == 0 else f"  #{rank}"
            style = "bold green" if i == 0 else ""

            table.add_row(
                rank_str,
                str(ep),
                f"{sc:.1f}",
                str(lp),
                f"{spl:.1f}",
                style=style,
            )

        return table

    def clear(self) -> None:
        """Clear tracked scores."""
        self.scores.clear()
