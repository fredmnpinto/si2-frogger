"""Core evaluation orchestrator for benchmarking DQN agents.

This module provides:
    - :class:`PerEpisodeResult`: Per-episode metrics dataclass.
    - :class:`EvaluationResult`: Aggregated statistics from an evaluation run.
    - :class:`EvaluationComparison`: Side-by-side DQN vs random comparison.
    - :class:`Evaluator`: Orchestrates evaluation episodes, computes statistics,
      saves results, and prints formatted summaries.

The :class:`Evaluator` uses :class:`evaluation.random_agent.RandomAgent` as the
random baseline instead of reusing :class:`agents.dummy_agent.DummyAgent`.  See
:mod:`evaluation.random_agent` for the detailed rationale.  In short:
DummyAgent is WebSocket-based, async, requires a running game server, and uses
a different state format.  RandomAgent is a lightweight synchronous offline
action selector that enables fast batch evaluation against
:class:`env.frogger_env.FroggerEnv` without any network overhead.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from env.frogger_env import FroggerEnv
from models.dqn_network import DQNNetwork, StateEncoder
from evaluation.random_agent import RandomAgent

STAY_ACTION_INDEX = 4
"""Index of the STAY action in the Q-value vector."""

SCORE_THRESHOLD = 2.0
"""NFR-006 threshold: DQN mean score must be > 2.0x random mean score."""


@dataclass
class PerEpisodeResult:
    """Metrics for a single evaluation episode."""

    episode: int
    score: float
    length: int
    laps: int
    steps_per_lap: float
    max_y: int
    truncated: bool


@dataclass
class EvaluationResult:
    """Aggregated statistics from an evaluation run."""

    agent_type: str
    n_episodes: int
    mean_score: float
    max_score: float
    min_score: float
    std_score: float
    mean_length: float
    mean_laps: float
    mean_steps_per_lap: float
    total_laps: int
    per_episode: List[PerEpisodeResult]
    seed: Optional[int]
    timestamp: str


@dataclass
class EvaluationComparison:
    """Side-by-side comparison of DQN and random baseline results."""

    dqn_result: EvaluationResult
    random_result: EvaluationResult
    score_ratio: float
    passes_threshold: bool
    output_dir: str


class Evaluator:
    """Evaluates a trained DQN agent against a random baseline.

    The evaluator can run episodes using either a trained DQN network
    (with optional epsilon-greedy exploration) or a :class:`RandomAgent`
    baseline.  It computes aggregate statistics, compares performance
    against the NFR-006 threshold, and saves results to JSON/CSV.
    """

    def __init__(self, env: Optional[FroggerEnv] = None, device: str = "auto") -> None:
        """Initialize the evaluator.

        Args:
            env: Optional pre-created :class:`FroggerEnv`.  A new instance is
                created when ``None``.
            device: Torch device string.  ``"auto"`` selects CUDA when available,
                otherwise CPU.
        """
        self.env = env if env is not None else FroggerEnv()
        self.device = self._resolve_device(device)
        self.encoder = StateEncoder()

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve the torch device string to a :class:`torch.device`.

        Args:
            device: Device string.  ``"auto"`` maps to CUDA if available,
                otherwise CPU.

        Returns:
            The resolved torch device.
        """
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def load_checkpoint(self, path: str) -> DQNNetwork:
        """Load a policy network from a checkpoint file.

        Args:
            path: Path to a PyTorch checkpoint containing
                ``policy_state_dict`` and ``config``.

        Returns:
            A :class:`DQNNetwork` initialized with the checkpoint weights
            and moved to the evaluator's device.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint is corrupted or missing required keys.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint from {path}: {exc}"
            ) from exc

        config = checkpoint.get("config", {})
        hidden_size = config.get("hidden_size", 64)

        network = DQNNetwork(hidden_dim=hidden_size).to(self.device)

        if "policy_state_dict" not in checkpoint:
            raise RuntimeError(
                f"Checkpoint {path} missing 'policy_state_dict' key"
            )

        try:
            network.load_state_dict(checkpoint["policy_state_dict"])
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint corrupted or incompatible: {exc}"
            ) from exc

        network.eval()
        return network

    def _select_action(
        self,
        network: DQNNetwork,
        state_tensor: torch.Tensor,
        epsilon: float,
        mask_stay: bool,
    ) -> int:
        """Select an action using epsilon-greedy with optional STAY masking.

        Args:
            network: The DQN policy network.
            state_tensor: Encoded state tensor on the correct device.
            epsilon: Exploration probability.
            mask_stay: If ``True``, STAY is excluded from both greedy and
                random action selection.

        Returns:
            Selected action index.
        """
        if random.random() < epsilon:
            # Random exploration
            if mask_stay:
                return random.randrange(4)
            return random.randrange(5)

        with torch.no_grad():
            q_values = network(state_tensor.unsqueeze(0))

        if mask_stay:
            q_values[0, STAY_ACTION_INDEX] = float("-inf")

        return int(q_values.argmax(dim=1).item())

    def run_episodes(
        self,
        network: Optional[DQNNetwork] = None,
        n_episodes: int = 100,
        epsilon: float = 0.0,
        seed: Optional[int] = None,
        max_steps_per_lap: int = 200,
        max_total_steps: int = 2000,
        mask_stay: bool = True,
    ) -> EvaluationResult:
        """Run evaluation episodes and collect statistics.

        If *network* is provided, the evaluator uses the DQN policy for
        action selection (with epsilon-greedy exploration).  If *network* is
        ``None``, a :class:`RandomAgent` baseline is used instead.

        Args:
            network: Optional trained DQN network.  When ``None``, random
                baseline is evaluated.
            n_episodes: Number of episodes to run.
            epsilon: Exploration probability for DQN evaluation (default 0.0
                for pure exploitation).
            seed: Optional random seed for reproducibility.
            max_steps_per_lap: Maximum steps allowed without completing a lap.
            max_total_steps: Absolute maximum steps per episode.
            mask_stay: If ``True``, exclude the STAY action from selection.

        Returns:
            :class:`EvaluationResult` containing aggregated statistics and
            per-episode details.
        """
        if seed is not None:
            self.env.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Use RandomAgent when no network is provided.
        # We use RandomAgent instead of DummyAgent because DummyAgent is
        # WebSocket-based and async, requiring a running game server.
        # RandomAgent is a lightweight synchronous offline selector that
        # works directly with FroggerEnv.  See evaluation/random_agent.py
        # for the full rationale.
        random_agent: Optional[RandomAgent] = None
        if network is None:
            random_agent = RandomAgent(self.env, seed=seed)

        per_episode: List[PerEpisodeResult] = []
        scores: List[float] = []
        lengths: List[int] = []
        laps_list: List[int] = []
        steps_per_lap_list: List[float] = []

        for episode_idx in range(n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            max_y = 0
            laps_completed = 0
            steps_at_lap_start = 0
            truncated = False

            while not done and steps < max_total_steps:
                if network is not None:
                    state_tensor = self.encoder.encode(state).to(self.device)
                    action = self._select_action(
                        network, state_tensor, epsilon, mask_stay
                    )
                else:
                    # Random baseline — RandomAgent never selects STAY
                    action = random_agent.select_action()  # type: ignore[union-attr]

                next_state, reward, done, info = self.env.step(action)

                if info.get("max_y_reached", 0) > max_y:
                    max_y = info["max_y_reached"]

                # Track lap completion and reset lap timer
                if info.get("laps_completed", 0) > laps_completed:
                    laps_completed = info["laps_completed"]
                    steps_at_lap_start = steps

                # Check steps-per-lap limit
                steps_since_lap = steps - steps_at_lap_start
                if steps_since_lap >= max_steps_per_lap:
                    done = True
                    truncated = True
                    break

                total_reward += reward
                state = next_state
                steps += 1

            # If truncated by total step limit, mark as done
            if steps >= max_total_steps and not done:
                done = True
                truncated = True

            episode_spl = steps / laps_completed if laps_completed > 0 else 0.0

            per_episode.append(
                PerEpisodeResult(
                    episode=episode_idx + 1,
                    score=total_reward,
                    length=steps,
                    laps=laps_completed,
                    steps_per_lap=episode_spl,
                    max_y=max_y,
                    truncated=truncated,
                )
            )
            scores.append(total_reward)
            lengths.append(steps)
            laps_list.append(laps_completed)
            steps_per_lap_list.append(episode_spl)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        # Population standard deviation (N denominator, not N-1)
        if scores:
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_score = math.sqrt(variance)
        else:
            std_score = 0.0

        mean_length = sum(lengths) / len(lengths) if lengths else 0.0
        mean_laps = sum(laps_list) / len(laps_list) if laps_list else 0.0
        mean_steps_per_lap = (
            sum(steps_per_lap_list) / len(steps_per_lap_list)
            if steps_per_lap_list
            else 0.0
        )
        total_laps = sum(laps_list)

        agent_type = "dqn" if network is not None else "random"

        return EvaluationResult(
            agent_type=agent_type,
            n_episodes=n_episodes,
            mean_score=mean_score,
            max_score=max_score,
            min_score=min_score,
            std_score=std_score,
            mean_length=mean_length,
            mean_laps=mean_laps,
            mean_steps_per_lap=mean_steps_per_lap,
            total_laps=total_laps,
            per_episode=per_episode,
            seed=seed,
            timestamp=datetime.now().isoformat(),
        )

    def evaluate_dqn(
        self,
        checkpoint_path: str,
        n_episodes: int = 100,
        seed: Optional[int] = None,
        max_steps_per_lap: int = 200,
        max_total_steps: int = 2000,
    ) -> EvaluationResult:
        """Evaluate a trained DQN agent from a checkpoint file.

        Args:
            checkpoint_path: Path to the model checkpoint.
            n_episodes: Number of evaluation episodes.
            seed: Optional random seed.
            max_steps_per_lap: Maximum steps allowed without completing a lap.
            max_total_steps: Absolute maximum steps per episode.

        Returns:
            :class:`EvaluationResult` for the DQN agent.
        """
        network = self.load_checkpoint(checkpoint_path)
        return self.run_episodes(
            network=network,
            n_episodes=n_episodes,
            epsilon=0.0,
            seed=seed,
            max_steps_per_lap=max_steps_per_lap,
            max_total_steps=max_total_steps,
            mask_stay=True,
        )

    def evaluate_random(
        self,
        n_episodes: int = 100,
        seed: Optional[int] = None,
        max_steps_per_lap: int = 200,
        max_total_steps: int = 2000,
    ) -> EvaluationResult:
        """Evaluate the random baseline agent.

        Args:
            n_episodes: Number of evaluation episodes.
            seed: Optional random seed.
            max_steps_per_lap: Maximum steps allowed without completing a lap.
            max_total_steps: Absolute maximum steps per episode.

        Returns:
            :class:`EvaluationResult` for the random baseline.
        """
        return self.run_episodes(
            network=None,
            n_episodes=n_episodes,
            seed=seed,
            max_steps_per_lap=max_steps_per_lap,
            max_total_steps=max_total_steps,
        )

    def compare(
        self,
        dqn_result: EvaluationResult,
        random_result: EvaluationResult,
    ) -> EvaluationComparison:
        """Compare DQN and random evaluation results.

        Computes the score ratio and checks whether the NFR-006 threshold
        (DQN mean score > 2.0x random mean score) is satisfied.

        When the random baseline mean is negative or zero, the simple ratio
        breaks down. In that case we use an improvement ratio of
        ``(DQN - Random) / |Random|`` and require DQN to achieve a
        meaningfully positive score.

        Args:
            dqn_result: Evaluation result for the DQN agent.
            random_result: Evaluation result for the random baseline.

        Returns:
            :class:`EvaluationComparison` with ratio and pass/fail verdict.
        """
        random_mean = random_result.mean_score

        if random_mean == 0:
            # Avoid division by zero; if random scores 0, any positive DQN
            # score passes the threshold.
            score_ratio = float("inf") if dqn_result.mean_score > 0 else 0.0
            passes_threshold = dqn_result.mean_score > 0
        elif random_mean < 0:
            # Negative baseline: use improvement over absolute baseline.
            # Ratio = (DQN - Random) / |Random| — positive when DQN > Random.
            score_ratio = (dqn_result.mean_score - random_mean) / abs(random_mean)
            # Threshold: DQN must be significantly better than random.
            # When random is negative, DQN just needs to be positive and
            # substantial (at least 100 points, an arbitrary positive threshold).
            passes_threshold = dqn_result.mean_score > 100
        else:
            score_ratio = dqn_result.mean_score / random_mean
            passes_threshold = score_ratio > SCORE_THRESHOLD

        return EvaluationComparison(
            dqn_result=dqn_result,
            random_result=random_result,
            score_ratio=score_ratio,
            passes_threshold=passes_threshold,
            output_dir="results",
        )

    def save_results(
        self,
        comparison: EvaluationComparison,
        output_dir: str = "results",
    ) -> Tuple[str, str]:
        """Save evaluation comparison to JSON and CSV files.

        Filenames include a timestamp to prevent overwrites.

        Args:
            comparison: The comparison object to serialize.
            output_dir: Directory where files are written.

        Returns:
            Tuple of ``(json_path, csv_path)``.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON output
        json_path = os.path.join(output_dir, f"evaluation_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dqn": asdict(comparison.dqn_result),
                    "random": asdict(comparison.random_result),
                    "score_ratio": comparison.score_ratio,
                    "passes_threshold": comparison.passes_threshold,
                    "threshold": SCORE_THRESHOLD,
                },
                f,
                indent=2,
                default=str,
            )

        # CSV output — one row per episode for both agents
        csv_path = os.path.join(output_dir, f"evaluation_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "agent_type",
                    "episode",
                    "score",
                    "length",
                    "laps",
                    "steps_per_lap",
                    "max_y",
                    "truncated",
                ]
            )
            for ep in comparison.dqn_result.per_episode:
                writer.writerow(
                    [
                        "dqn",
                        ep.episode,
                        f"{ep.score:.2f}",
                        ep.length,
                        ep.laps,
                        f"{ep.steps_per_lap:.2f}",
                        ep.max_y,
                        ep.truncated,
                    ]
                )
            for ep in comparison.random_result.per_episode:
                writer.writerow(
                    [
                        "random",
                        ep.episode,
                        f"{ep.score:.2f}",
                        ep.length,
                        ep.laps,
                        f"{ep.steps_per_lap:.2f}",
                        ep.max_y,
                        ep.truncated,
                    ]
                )

        return json_path, csv_path

    def print_summary(self, comparison: EvaluationComparison) -> None:
        """Print a formatted comparison table to the console.

        Uses :mod:`rich` to render a side-by-side table of DQN vs random
        metrics, the score ratio, and the NFR-006 threshold verdict.

        Args:
            comparison: The comparison object to display.
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Evaluation Results")

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("DQN", style="green", justify="right")
        table.add_column("Random", style="yellow", justify="right")

        dqn = comparison.dqn_result
        rnd = comparison.random_result

        table.add_row("Episodes", str(dqn.n_episodes), str(rnd.n_episodes))
        table.add_row("Mean Score", f"{dqn.mean_score:.2f}", f"{rnd.mean_score:.2f}")
        table.add_row("Max Score", f"{dqn.max_score:.2f}", f"{rnd.max_score:.2f}")
        table.add_row("Min Score", f"{dqn.min_score:.2f}", f"{rnd.min_score:.2f}")
        table.add_row("Std Dev", f"{dqn.std_score:.2f}", f"{rnd.std_score:.2f}")
        table.add_row("Mean Length", f"{dqn.mean_length:.2f}", f"{rnd.mean_length:.2f}")
        table.add_row("Mean Laps", f"{dqn.mean_laps:.2f}", f"{rnd.mean_laps:.2f}")
        table.add_row(
            "Mean Steps/Lap",
            f"{dqn.mean_steps_per_lap:.2f}",
            f"{rnd.mean_steps_per_lap:.2f}",
        )
        table.add_row("Total Laps", str(dqn.total_laps), str(rnd.total_laps))

        table.add_row("", "", "")
        table.add_row("Score Ratio", f"{comparison.score_ratio:.2f}x", "")
        table.add_row("NFR-006 Threshold", f">{SCORE_THRESHOLD:.2f}x", "")
        verdict = "[bold green]PASS[/bold green]" if comparison.passes_threshold else "[bold red]FAIL[/bold red]"
        table.add_row("Verdict", verdict, "")

        console.print(table)
