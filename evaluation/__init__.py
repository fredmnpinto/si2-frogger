"""Evaluation package for benchmarking trained DQN agents.

This package provides standalone evaluation capabilities that assess trained
DQN agents against a random baseline without requiring a running game server
or WebSocket connection.

Exports:
    - :class:`Evaluator`: Core evaluation orchestrator.
    - :class:`RandomAgent`: Offline synchronous random action selector.
    - :class:`EvaluationConfig`: Configuration dataclass for evaluation runs.
    - :class:`EvaluationResult`: Aggregated statistics from an evaluation run.
    - :class:`EvaluationComparison`: Side-by-side comparison of DQN vs random.
    - :class:`PerEpisodeResult`: Per-episode metrics.
"""

from __future__ import annotations

from evaluation.config import EvaluationConfig
from evaluation.evaluator import (
    EvaluationComparison,
    EvaluationResult,
    Evaluator,
    PerEpisodeResult,
)
from evaluation.random_agent import RandomAgent

__all__ = [
    "Evaluator",
    "RandomAgent",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationComparison",
    "PerEpisodeResult",
]
