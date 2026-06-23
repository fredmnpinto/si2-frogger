"""Training configuration and CLI argument parsing.

This module provides:
    - :class:`TrainingConfig`: Dataclass holding all training hyperparameters
      and orchestration settings.
    - :func:`parse_args`: Parses command-line arguments and returns a
      :class:`TrainingConfig` instance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from training.dqn_trainer import DQNConfig


DEFAULT_EPISODES: int = 1000
DEFAULT_CHECKPOINT_FREQ: int = 100
DEFAULT_EVAL_FREQ: int = 100
DEFAULT_EVAL_EPISODES: int = 10
DEFAULT_LOG_DIR: str = "logs"
DEFAULT_CHECKPOINT_DIR: str = "checkpoints"
DEFAULT_PLOT_DIR: str = "plots"
DEFAULT_LOG_FILE: str = "training.csv"
DEFAULT_MAX_STEPS_PER_LAP: int = 200
DEFAULT_MAX_TOTAL_STEPS: int = 2000


@dataclass
class TrainingConfig:
    """Configuration for the training orchestration.

    Attributes:
        episodes: Total number of training episodes.
        checkpoint_freq: Save a checkpoint every N episodes.
        eval_freq: Run evaluation every N episodes.
        eval_episodes: Number of episodes per evaluation run.
        log_dir: Directory for log files.
        checkpoint_dir: Directory for checkpoint files.
        plot_dir: Directory for generated plots.
        log_file: Name of the CSV log file.
        seed: Random seed for reproducibility. ``None`` means unseeded.
        resume: Path to a checkpoint file to resume from.
        config_file: Path to a JSON configuration file.
        max_steps_per_lap: Max steps allowed without completing a lap.
        max_total_steps: Absolute max steps per episode.
        dqn: DQN hyperparameter configuration.
    """

    episodes: int = DEFAULT_EPISODES
    checkpoint_freq: int = DEFAULT_CHECKPOINT_FREQ
    eval_freq: int = DEFAULT_EVAL_FREQ
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    log_dir: str = DEFAULT_LOG_DIR
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    plot_dir: str = DEFAULT_PLOT_DIR
    log_file: str = DEFAULT_LOG_FILE
    seed: Optional[int] = None
    resume: Optional[str] = None
    config_file: Optional[str] = None
    max_steps_per_lap: int = DEFAULT_MAX_STEPS_PER_LAP
    max_total_steps: int = DEFAULT_MAX_TOTAL_STEPS
    dqn: DQNConfig = field(default_factory=DQNConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        d = asdict(self)
        # Flatten nested DQNConfig for JSON compatibility
        dqn_dict = d.pop("dqn")
        d.update({f"dqn_{k}": v for k, v in dqn_dict.items()})
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Deserialize a configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            A new :class:`TrainingConfig` instance.
        """
        # Extract DQN-specific keys
        dqn_keys = {k[4:]: v for k, v in data.items() if k.startswith("dqn_")}
        top_level = {k: v for k, v in data.items() if not k.startswith("dqn_")}

        dqn_config = DQNConfig(**dqn_keys) if dqn_keys else DQNConfig()
        top_level["dqn"] = dqn_config

        # Filter to only valid fields
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in top_level.items() if k in valid_keys}

        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """Load a configuration from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A new :class:`TrainingConfig` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m training",
        description="Train a DQN agent for the Frogger game.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=f"Number of training episodes (default: {DEFAULT_EPISODES}).",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help=f"Checkpoint every N episodes (default: {DEFAULT_CHECKPOINT_FREQ}).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help=f"Evaluate every N episodes (default: {DEFAULT_EVAL_FREQ}).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help=f"Episodes per evaluation (default: {DEFAULT_EVAL_EPISODES}).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=f"Directory for logs (default: {DEFAULT_LOG_DIR}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=f"Directory for checkpoints (default: {DEFAULT_CHECKPOINT_DIR}).",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help=f"Directory for plots (default: {DEFAULT_PLOT_DIR}).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=f"CSV log filename (default: {DEFAULT_LOG_FILE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        dest="config_file",
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--max-steps-per-lap",
        type=int,
        default=None,
        help=f"Maximum steps allowed without completing a lap (default: {DEFAULT_MAX_STEPS_PER_LAP}).",
    )
    parser.add_argument(
        "--max-total-steps",
        type=int,
        default=None,
        help=f"Absolute maximum steps per episode (default: {DEFAULT_MAX_TOTAL_STEPS}).",
    )

    # DQN hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor for future rewards.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=None,
        help="Initial epsilon for exploration.",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=None,
        help="Final epsilon after decay.",
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=None,
        help="Number of steps over which epsilon decays.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Replay buffer capacity.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Minibatch size for training updates.",
    )
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=None,
        help="Steps between target network updates.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Polyak averaging coefficient (1.0 = hard copy).",
    )
    parser.add_argument(
        "--update-frequency",
        type=int,
        default=None,
        help="Train every N environment steps.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Number of neurons in hidden layers.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=None,
        help="Maximum L2 norm for gradient clipping.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device ("auto", "cpu", "cuda").',
    )

    return parser


def parse_args(args: Optional[List[str]] = None) -> TrainingConfig:
    """Parse command-line arguments into a :class:`TrainingConfig`.

    If ``--config`` is provided, the JSON file is loaded first and CLI
    arguments override its values.

    Args:
        args: Optional list of argument strings (defaults to ``sys.argv``).

    Returns:
        Parsed :class:`TrainingConfig`.
    """
    parser = _build_parser()
    namespace = parser.parse_args(args)

    # Start with defaults
    config = TrainingConfig()

    # Load config file if provided
    if namespace.config_file is not None:
        config = TrainingConfig.from_json(namespace.config_file)

    # Override with CLI arguments only when explicitly provided
    if namespace.episodes is not None:
        config.episodes = namespace.episodes
    if namespace.checkpoint_freq is not None:
        config.checkpoint_freq = namespace.checkpoint_freq
    if namespace.eval_freq is not None:
        config.eval_freq = namespace.eval_freq
    if namespace.eval_episodes is not None:
        config.eval_episodes = namespace.eval_episodes
    if namespace.log_dir is not None:
        config.log_dir = namespace.log_dir
    if namespace.checkpoint_dir is not None:
        config.checkpoint_dir = namespace.checkpoint_dir
    if namespace.plot_dir is not None:
        config.plot_dir = namespace.plot_dir
    if namespace.log_file is not None:
        config.log_file = namespace.log_file
    if namespace.seed is not None:
        config.seed = namespace.seed
    if namespace.resume is not None:
        config.resume = namespace.resume
    if namespace.config_file is not None:
        config.config_file = namespace.config_file
    if namespace.max_steps_per_lap is not None:
        config.max_steps_per_lap = namespace.max_steps_per_lap
    if namespace.max_total_steps is not None:
        config.max_total_steps = namespace.max_total_steps

    # Override DQN config
    dqn_overrides = {
        "learning_rate": namespace.learning_rate,
        "gamma": namespace.gamma,
        "epsilon_start": namespace.epsilon_start,
        "epsilon_end": namespace.epsilon_end,
        "epsilon_decay_steps": namespace.epsilon_decay_steps,
        "buffer_size": namespace.buffer_size,
        "batch_size": namespace.batch_size,
        "target_update_freq": namespace.target_update_freq,
        "tau": namespace.tau,
        "update_frequency": namespace.update_frequency,
        "hidden_size": namespace.hidden_size,
        "gradient_clip": namespace.gradient_clip,
        "device": namespace.device,
    }
    for key, value in dqn_overrides.items():
        if value is not None:
            setattr(config.dqn, key, value)

    return config
