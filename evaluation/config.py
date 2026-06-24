"""Evaluation configuration and CLI argument parsing.

This module provides:
    - :class:`EvaluationConfig`: Dataclass holding all evaluation parameters.
    - :func:`parse_args`: Parses command-line arguments and returns an
      :class:`EvaluationConfig` instance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional


DEFAULT_N_EPISODES: int = 100
DEFAULT_OUTPUT_DIR: str = "results"
DEFAULT_DEVICE: str = "auto"
DEFAULT_MAX_STEPS_PER_LAP: int = 200
DEFAULT_MAX_TOTAL_STEPS: int = 2000


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run.

    Attributes:
        n_episodes: Number of episodes to evaluate each agent.
        seed: Optional random seed for reproducibility.
        model_path: Path to the trained DQN checkpoint file.
        output_dir: Directory where JSON/CSV results are saved.
        device: Torch device string ("auto", "cpu", or "cuda").
        max_steps_per_lap: Maximum steps allowed without completing a lap.
        max_total_steps: Absolute maximum steps per episode.
    """

    n_episodes: int = DEFAULT_N_EPISODES
    seed: Optional[int] = None
    model_path: str = ""
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = DEFAULT_DEVICE
    max_steps_per_lap: int = DEFAULT_MAX_STEPS_PER_LAP
    max_total_steps: int = DEFAULT_MAX_TOTAL_STEPS


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluation CLI.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Evaluate a trained DQN agent against a random baseline.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_N_EPISODES,
        help=f"Number of evaluation episodes per agent (default: {DEFAULT_N_EPISODES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for evaluation results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f'Torch device to use: "auto", "cpu", or "cuda" (default: {DEFAULT_DEVICE}).',
    )
    parser.add_argument(
        "--max-steps-per-lap",
        type=int,
        default=DEFAULT_MAX_STEPS_PER_LAP,
        help=f"Maximum steps allowed without completing a lap (default: {DEFAULT_MAX_STEPS_PER_LAP}).",
    )
    parser.add_argument(
        "--max-total-steps",
        type=int,
        default=DEFAULT_MAX_TOTAL_STEPS,
        help=f"Absolute maximum steps per episode (default: {DEFAULT_MAX_TOTAL_STEPS}).",
    )

    return parser


def parse_args(args: Optional[List[str]] = None) -> EvaluationConfig:
    """Parse command-line arguments into an :class:`EvaluationConfig`.

    Args:
        args: Optional list of argument strings (defaults to ``sys.argv``).

    Returns:
        Parsed :class:`EvaluationConfig`.
    """
    parser = _build_parser()
    namespace = parser.parse_args(args)

    return EvaluationConfig(
        n_episodes=namespace.episodes,
        seed=namespace.seed,
        model_path=namespace.model,
        output_dir=namespace.output,
        device=namespace.device,
        max_steps_per_lap=namespace.max_steps_per_lap,
        max_total_steps=namespace.max_total_steps,
    )
