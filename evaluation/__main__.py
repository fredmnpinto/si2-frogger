"""CLI entry point for evaluating a trained DQN agent.

Usage::

    python -m evaluation --model checkpoints/best.pt
    python -m evaluation --model checkpoints/best.pt --episodes 100 --seed 42
    python -m evaluation --model checkpoints/best.pt --output results/
"""

from __future__ import annotations

import sys

from evaluation.config import parse_args
from evaluation.evaluator import Evaluator


def main() -> int:
    """Run the evaluation CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        config = parse_args()
    except SystemExit as exc:
        # argparse raises SystemExit on --help or invalid args
        return exc.code if isinstance(exc.code, int) else 1

    evaluator = Evaluator(device=config.device)

    try:
        dqn_result = evaluator.evaluate_dqn(
            checkpoint_path=config.model_path,
            n_episodes=config.n_episodes,
            seed=config.seed,
            max_steps_per_lap=config.max_steps_per_lap,
            max_total_steps=config.max_total_steps,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error loading checkpoint: {exc}", file=sys.stderr)
        return 1

    try:
        random_result = evaluator.evaluate_random(
            n_episodes=config.n_episodes,
            seed=config.seed,
            max_steps_per_lap=config.max_steps_per_lap,
            max_total_steps=config.max_total_steps,
        )
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.", file=sys.stderr)
        return 1

    comparison = evaluator.compare(dqn_result, random_result)

    try:
        evaluator.save_results(comparison, output_dir=config.output_dir)
    except OSError as exc:
        print(f"Error saving results: {exc}", file=sys.stderr)
        return 1

    evaluator.print_summary(comparison)
    return 0


if __name__ == "__main__":
    sys.exit(main())
