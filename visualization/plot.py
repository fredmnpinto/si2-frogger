"""Plot generation script for training logs.

Generates five plots from a training CSV log:
    1. Episode rewards over time (smoothed moving average)
    2. Loss curve over training steps
    3. Epsilon decay over episodes
    4. Evaluation score distribution (histogram/boxplot)
    5. Steps per lap over time (smoothed moving average)

Usage::

    python -m visualization.plot --log logs/training.csv --output plots/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_WINDOW_SIZE: int = 50


def _parse_csv(path: str) -> dict[str, List[float]]:
    """Parse a training CSV file into columnar data.

    Args:
        path: Path to the CSV log file.

    Returns:
        Dictionary mapping column names to lists of float values.
        Non-numeric entries are converted to ``float('nan')``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    data: dict[str, List[float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for col in header:
            data[col] = []
        for row in reader:
            for col, val in zip(header, row):
                try:
                    data[col].append(float(val))
                except ValueError:
                    data[col].append(float("nan"))
    return data


def _moving_average(values: List[float], window: int) -> List[float]:
    """Compute a simple moving average.

    Args:
        values: Input sequence.
        window: Window size.

    Returns:
        Smoothed sequence of the same length.  The first ``window-1``
        entries are computed with a shrinking window.
    """
    if not values:
        return []
    arr = np.array(values, dtype=float)
    result = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = np.nanmean(arr[start : i + 1])
    return result.tolist()


def plot_rewards(episodes: List[float], rewards: List[float], window: int, output_path: str, format: str = "png") -> None:
    """Plot episode rewards with a smoothed moving average.

    Args:
        episodes: Episode numbers.
        rewards: Raw episode rewards.
        window: Moving average window size.
        output_path: Path to save the plot file (without extension).
        format: Output format ("png" or "svg").
    """
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, label="Raw reward")
    smoothed = _moving_average(rewards, window)
    plt.plot(episodes, smoothed, label=f"Moving average (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards Over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_path}.{format}", dpi=150, format=format)
    plt.close()


def plot_loss(episodes: List[float], losses: List[float], output_path: str, format: str = "png") -> None:
    """Plot the loss curve over training steps.

    Only finite loss values are plotted.

    Args:
        episodes: Episode numbers.
        losses: Loss values (may contain NaN).
        output_path: Path to save the plot file (without extension).
        format: Output format ("png" or "svg").
    """
    valid = [(e, l) for e, l in zip(episodes, losses) if np.isfinite(l)]
    if not valid:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No valid loss data", ha="center", va="center")
        plt.axis("off")
        plt.savefig(f"{output_path}.{format}", dpi=150, format=format)
        plt.close()
        return

    ep, ls = zip(*valid)
    plt.figure(figsize=(10, 6))
    plt.plot(ep, ls)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss Curve Over Training")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_path}.{format}", dpi=150, format=format)
    plt.close()


def plot_epsilon(episodes: List[float], epsilons: List[float], output_path: str, format: str = "png") -> None:
    """Plot epsilon decay over episodes.

    Args:
        episodes: Episode numbers.
        epsilons: Epsilon values.
        output_path: Path to save the plot file (without extension).
        format: Output format ("png" or "svg").
    """
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Episodes")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_path}.{format}", dpi=150, format=format)
    plt.close()


def plot_score_distribution(rewards: List[float], output_path: str, format: str = "png") -> None:
    """Plot evaluation score distribution as a histogram and boxplot.

    Args:
        rewards: Episode reward values.
        output_path: Path to save the plot file (without extension).
        format: Output format ("png" or "svg").
    """
    valid = [r for r in rewards if np.isfinite(r)]
    if not valid:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No valid reward data", ha="center", va="center")
        plt.axis("off")
        plt.savefig(f"{output_path}.{format}", dpi=150, format=format)
        plt.close()
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(valid, bins=min(20, len(valid)), edgecolor="black")
    axes[0].set_xlabel("Episode Reward")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Reward Distribution")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Boxplot
    axes[1].boxplot(valid, orientation="vertical")
    axes[1].set_ylabel("Episode Reward")
    axes[1].set_title("Reward Boxplot")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_path}.{format}", dpi=150, format=format)
    plt.close()


def plot_steps_per_lap(episodes: List[float], steps_per_lap: List[float], window: int, output_path: str, format: str = "png") -> None:
    """Plot steps per lap over time with smoothed moving average.

    Only episodes with laps_completed > 0 are plotted.

    Args:
        episodes: Episode numbers.
        steps_per_lap: Steps per lap values.
        window: Moving average window size.
        output_path: Path to save the plot file (without extension).
        format: Output format ("png" or "svg").
    """
    valid = [(e, s) for e, s in zip(episodes, steps_per_lap) if s > 0]
    if not valid:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No lap completion data", ha="center", va="center")
        plt.axis("off")
        plt.savefig(f"{output_path}.{format}", dpi=150, format=format)
        plt.close()
        return

    ep, sp = zip(*valid)
    plt.figure(figsize=(10, 6))
    plt.plot(ep, sp, alpha=0.3, label="Raw steps/lap")
    smoothed = _moving_average(list(sp), window)
    plt.plot(ep, smoothed, label=f"Moving average (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Steps per Lap")
    plt.title("Steps per Lap Over Time (Lower is Better)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_path}.{format}", dpi=150, format=format)
    plt.close()


def generate_plots(log_path: str, output_dir: str, window: int = DEFAULT_WINDOW_SIZE, format: str = "png") -> List[str]:
    """Generate all five plots from a training log.

    Args:
        log_path: Path to the CSV training log.
        output_dir: Directory where plot files will be saved.
        window: Moving average window size for the reward plot.
        format: Output format ("png" or "svg").

    Returns:
        List of paths to the generated plot files.

    Raises:
        FileNotFoundError: If the log file does not exist.
    """
    data = _parse_csv(log_path)
    os.makedirs(output_dir, exist_ok=True)

    episodes = data.get("episode", [])
    rewards = data.get("total_reward", [])
    losses = data.get("loss", [])
    epsilons = data.get("epsilon", [])
    steps_per_lap = data.get("steps_per_lap", [])

    files: List[str] = []

    if episodes and rewards:
        path = os.path.join(output_dir, "rewards")
        plot_rewards(episodes, rewards, window, path, format=format)
        files.append(f"{path}.{format}")

    if episodes and losses:
        path = os.path.join(output_dir, "loss")
        plot_loss(episodes, losses, path, format=format)
        files.append(f"{path}.{format}")

    if episodes and epsilons:
        path = os.path.join(output_dir, "epsilon")
        plot_epsilon(episodes, epsilons, path, format=format)
        files.append(f"{path}.{format}")

    if rewards:
        path = os.path.join(output_dir, "score_distribution")
        plot_score_distribution(rewards, path, format=format)
        files.append(f"{path}.{format}")

    if episodes and steps_per_lap:
        path = os.path.join(output_dir, "steps_per_lap")
        plot_steps_per_lap(episodes, steps_per_lap, window, path, format=format)
        files.append(f"{path}.{format}")

    return files


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for plot generation.

    Args:
        args: Optional list of argument strings.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot",
        description="Generate training visualisation plots from a CSV log.",
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to the training CSV log file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Moving average window size (default: {DEFAULT_WINDOW_SIZE}).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "svg"],
        help="Output plot format (default: png).",
    )
    parsed = parser.parse_args(args)

    try:
        paths = generate_plots(parsed.log, parsed.output, parsed.window, format=parsed.format)
        for p in paths:
            print(f"Saved: {p}")
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
