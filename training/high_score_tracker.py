"""In-place high score tracker using ANSI escape codes."""

from typing import Dict, List


class HighScoreTracker:
    """Tracks last N high scores and displays them in-place in the terminal.

    Uses ANSI escape codes to overwrite previous output, creating a fixed
    window that updates without scrolling.
    """

    def __init__(self, max_entries: int = 3) -> None:
        """Initialize the tracker.

        Args:
            max_entries: Number of high scores to display (default 3).
        """
        self.max_entries = max_entries
        self.scores: List[Dict] = []
        self._lines_printed = 0

    def add_score(
        self,
        episode: int,
        score: float,
        laps: int,
        total_steps: int,
        steps_per_lap: float,
    ) -> None:
        """Add a new high score and redraw the display.

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
        self._redraw()

    def _redraw(self) -> None:
        """Clear previous lines and redraw the high score table."""
        # Clear previous output
        if self._lines_printed > 0:
            # Move cursor up N lines
            print(f"\033[{self._lines_printed}F", end="", flush=True)
            # Clear each line
            for _ in range(self._lines_printed):
                print("\033[K", end="")
                if _ < self._lines_printed - 1:
                    print("\033[1B", end="")
            # Move back up
            print(f"\033[{self._lines_printed}F", end="", flush=True)

        # Print header
        print("┌─ Recent High Scores ─────────────────────────────────────┐")
        self._lines_printed = 1

        # Print scores (newest first)
        for i, score in enumerate(reversed(self.scores)):
            rank = i + 1
            ep = score["episode"]
            sc = score["score"]
            lp = score["laps"]
            spl = score["steps_per_lap"]

            # Highlight the newest entry
            prefix = "▶" if i == 0 else " "

            print(
                f"│ {prefix}#{rank} Ep {ep:4d} │ Score: {sc:7.1f} │ "
                f"Laps: {lp} │ Steps/Lap: {spl:5.1f} │"
            )
            self._lines_printed += 1

        # Fill empty slots
        for _ in range(self.max_entries - len(self.scores)):
            print("│                                                          │")
            self._lines_printed += 1

        # Print footer
        print("└──────────────────────────────────────────────────────────┘")
        self._lines_printed += 1

    def clear(self) -> None:
        """Clear the display."""
        if self._lines_printed > 0:
            print(f"\033[{self._lines_printed}F", end="", flush=True)
            for _ in range(self._lines_printed):
                print("\033[K", end="")
                if _ < self._lines_printed - 1:
                    print("\033[1B", end="")
            print(f"\033[{self._lines_printed}F", end="", flush=True)
            self._lines_printed = 0
