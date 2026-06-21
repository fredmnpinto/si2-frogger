"""CLI entry point for training the DQN agent.

Usage::

    python -m training                           # Default: 1000 episodes
    python -m training --episodes 2000 --seed 42 # Override
    python -m training --config config.json      # Config file
    python -m training --resume checkpoint.pt    # Resume
"""

from __future__ import annotations

import json
import sys

from training.config import parse_args
from training.orchestrator import TrainingOrchestrator


def main() -> int:
    """Run the training CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        config = parse_args()
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    orchestrator = TrainingOrchestrator(config)

    try:
        result = orchestrator.run()
        print(f"\nTraining complete. Best score: {result['best_score']:.2f}")
        return 0
    except Exception as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
