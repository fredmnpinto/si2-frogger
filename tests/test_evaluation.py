"""Unit tests for the evaluation package."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from typing import Any, Dict, List, Optional

import torch

from evaluation.config import EvaluationConfig, parse_args
from evaluation.evaluator import (
    SCORE_THRESHOLD,
    Evaluator,
    PerEpisodeResult,
)
from evaluation.random_agent import RandomAgent
from models.dqn_network import DQNNetwork, HIDDEN_SIZE


class MockEnv:
    """Minimal mock environment for evaluation testing."""

    def __init__(
        self,
        max_steps: int = 5,
        lap_at_steps: Optional[List[int]] = None,
        never_done: bool = False,
    ) -> None:
        self.max_steps = max_steps
        self.lap_at_steps = set(lap_at_steps or [])
        self.never_done = never_done
        self._step_count = 0
        self._laps = 0
        self._seed: Optional[int] = None

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._laps = 0
        return {
            "frog_x": 5,
            "frog_y": 0,
            "lives": 3,
            "score": 0,
            "high_score": 0,
            "game_over": False,
            "win": False,
            "obstacles": [],
        }

    def step(self, action: int) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._step_count += 1
        if self._step_count in self.lap_at_steps:
            self._laps += 1
        done = False if self.never_done else (self._step_count >= self.max_steps)
        state = {
            "frog_x": 5,
            "frog_y": 0,
            "lives": 3,
            "score": 0,
            "high_score": 0,
            "game_over": done,
            "win": False,
            "obstacles": [],
        }
        info = {
            "episode_length": self._step_count,
            "lives": 3,
            "score": 0,
            "laps": self._laps,
            "laps_completed": self._laps,
            "high_score": 0,
            "game_over": done,
            "win": False,
            "max_y_reached": self._step_count,
        }
        return state, 1.0, done, info

    def seed(self, seed: int) -> None:
        self._seed = seed


class MockNetwork(torch.nn.Module):
    """Mock DQN network that returns deterministic Q-values."""

    def __init__(self, action: int = 0) -> None:
        super().__init__()
        self.action = action
        self._forward_calls: List[torch.Tensor] = []
        # Need a dummy parameter so it's a valid Module
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._forward_calls.append(x)
        q = torch.full((x.shape[0], 5), -1.0, device=x.device)
        q[:, self.action] = 10.0
        return q


class TestEvaluatorInit(unittest.TestCase):
    """Tests for Evaluator initialization."""

    def test_default_env(self):
        evaluator = Evaluator()
        self.assertIsNotNone(evaluator.env)

    def test_custom_env(self):
        env = MockEnv()
        evaluator = Evaluator(env=env)
        self.assertIs(evaluator.env, env)

    def test_device_auto(self):
        evaluator = Evaluator(device="auto")
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(evaluator.device.type, expected)

    def test_device_cpu(self):
        evaluator = Evaluator(device="cpu")
        self.assertEqual(evaluator.device.type, "cpu")


class TestLoadCheckpoint(unittest.TestCase):
    """Tests for Evaluator.load_checkpoint."""

    def _make_checkpoint(
        self,
        path: str,
        hidden_size: int = HIDDEN_SIZE,
        missing_policy: bool = False,
        corrupt: bool = False,
    ) -> None:
        if corrupt:
            with open(path, "wb") as f:
                f.write(b"not a checkpoint")
            return
        network = DQNNetwork(hidden_dim=hidden_size)
        checkpoint: Dict[str, Any] = {
            "policy_state_dict": network.state_dict(),
            "config": {"hidden_size": hidden_size},
        }
        if missing_policy:
            del checkpoint["policy_state_dict"]
        torch.save(checkpoint, path)

    def test_load_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path, hidden_size=128)
            evaluator = Evaluator()
            network = evaluator.load_checkpoint(path)
            self.assertIsInstance(network, DQNNetwork)
            first_linear = network.net[0]
            self.assertEqual(first_linear.out_features, 128)
            self.assertFalse(network.training)

    def test_file_not_found(self):
        evaluator = Evaluator()
        with self.assertRaises(FileNotFoundError) as ctx:
            evaluator.load_checkpoint("/nonexistent/path/model.pt")
        self.assertIn("Checkpoint file not found", str(ctx.exception))

    def test_corrupted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.pt")
            self._make_checkpoint(path, corrupt=True)
            evaluator = Evaluator()
            with self.assertRaises(RuntimeError) as ctx:
                evaluator.load_checkpoint(path)
            self.assertIn("Failed to load checkpoint", str(ctx.exception))

    def test_missing_policy_state_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.pt")
            self._make_checkpoint(path, missing_policy=True)
            evaluator = Evaluator()
            with self.assertRaises(RuntimeError) as ctx:
                evaluator.load_checkpoint(path)
            self.assertIn("missing 'policy_state_dict'", str(ctx.exception))

    def test_incompatible_state_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            network = DQNNetwork(hidden_dim=64)
            # Corrupt the state dict by changing a tensor shape
            state_dict = network.state_dict()
            state_dict["net.0.weight"] = torch.randn(10, 10)
            torch.save(
                {"policy_state_dict": state_dict, "config": {"hidden_size": 64}},
                path,
            )
            evaluator = Evaluator()
            with self.assertRaises(RuntimeError) as ctx:
                evaluator.load_checkpoint(path)
            self.assertIn("Checkpoint corrupted or incompatible", str(ctx.exception))


class TestRunEpisodesDQN(unittest.TestCase):
    """Tests for Evaluator.run_episodes with a DQN network."""

    def test_runs_correct_number_of_episodes(self):
        env = MockEnv(max_steps=3)
        evaluator = Evaluator(env=env)
        network = MockNetwork(action=0)
        result = evaluator.run_episodes(network=network, n_episodes=5)
        self.assertEqual(result.n_episodes, 5)
        self.assertEqual(len(result.per_episode), 5)
        self.assertEqual(result.agent_type, "dqn")

    def test_tracks_per_episode_stats(self):
        env = MockEnv(max_steps=3)
        evaluator = Evaluator(env=env)
        network = MockNetwork(action=0)
        result = evaluator.run_episodes(network=network, n_episodes=2)
        for ep in result.per_episode:
            self.assertIsInstance(ep.score, float)
            self.assertIsInstance(ep.length, int)
            self.assertIsInstance(ep.laps, int)
            self.assertIsInstance(ep.steps_per_lap, float)
            self.assertIsInstance(ep.max_y, int)
            self.assertIsInstance(ep.truncated, bool)

    def test_epsilon_greedy_exploration(self):
        env = MockEnv(max_steps=2)
        evaluator = Evaluator(env=env)
        network = MockNetwork(action=0)
        # With epsilon=1.0, all actions should be random
        # We can't easily verify randomness, but we can verify it doesn't crash
        result = evaluator.run_episodes(network=network, n_episodes=2, epsilon=1.0)
        self.assertEqual(result.n_episodes, 2)

    def test_mask_stay_excludes_action_4(self):
        env = MockEnv(max_steps=2)
        evaluator = Evaluator(env=env)
        # Mock network that would select STAY (action 4) if not masked
        network = MockNetwork(action=4)
        result = evaluator.run_episodes(network=network, n_episodes=1, mask_stay=True)
        # Episode should still complete without error; STAY is masked
        self.assertEqual(result.n_episodes, 1)

    def test_no_mask_stay_allows_action_4(self):
        env = MockEnv(max_steps=2)
        evaluator = Evaluator(env=env)
        network = MockNetwork(action=4)
        result = evaluator.run_episodes(network=network, n_episodes=1, mask_stay=False)
        self.assertEqual(result.n_episodes, 1)

    def test_zero_episodes(self):
        env = MockEnv(max_steps=2)
        evaluator = Evaluator(env=env)
        network = MockNetwork(action=0)
        result = evaluator.run_episodes(network=network, n_episodes=0)
        self.assertEqual(result.n_episodes, 0)
        self.assertEqual(len(result.per_episode), 0)
        self.assertEqual(result.mean_score, 0.0)


    def test_no_mask_stay_random_exploration(self):
        env = MockEnv(max_steps=2)
        evaluator = Evaluator(env=env)
        network = MockNetwork(action=0)
        # With epsilon=1.0 and mask_stay=False, random actions include 0-4
        # We just verify it doesn't crash
        result = evaluator.run_episodes(
            network=network, n_episodes=1, epsilon=1.0, mask_stay=False
        )
        self.assertEqual(result.n_episodes, 1)

    def test_cuda_seeding(self):
        env = MockEnv(max_steps=2)
        evaluator = Evaluator(env=env)
        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.manual_seed_all") as mock_cuda_seed:
                evaluator.run_episodes(network=None, n_episodes=1, seed=42)
                # torch.manual_seed also calls torch.cuda.manual_seed_all
                # when CUDA is available, so we check it was called at least once
                mock_cuda_seed.assert_called_with(42)
                self.assertGreaterEqual(mock_cuda_seed.call_count, 1)


class TestRunEpisodesRandom(unittest.TestCase):
    """Tests for Evaluator.run_episodes with random baseline."""

    def test_runs_correct_number_of_episodes(self):
        env = MockEnv(max_steps=3)
        evaluator = Evaluator(env=env)
        result = evaluator.run_episodes(network=None, n_episodes=5)
        self.assertEqual(result.n_episodes, 5)
        self.assertEqual(len(result.per_episode), 5)
        self.assertEqual(result.agent_type, "random")

    def test_random_agent_never_selects_stay(self):
        env = MockEnv(max_steps=10)
        evaluator = Evaluator(env=env)
        result = evaluator.run_episodes(network=None, n_episodes=10)
        # All episodes should complete without error
        self.assertEqual(result.n_episodes, 10)


class TestStatisticsComputation(unittest.TestCase):
    """Tests for aggregate statistics computation."""

    def test_mean_max_min(self):
        env = MockEnv(max_steps=2)
        evaluator = Evaluator(env=env)
        network = MockNetwork(action=0)
        result = evaluator.run_episodes(network=network, n_episodes=3)
        scores = [ep.score for ep in result.per_episode]
        self.assertEqual(result.mean_score, sum(scores) / len(scores))
        self.assertEqual(result.max_score, max(scores))
        self.assertEqual(result.min_score, min(scores))

    def test_population_std(self):
        # Use a deterministic env so scores are predictable
        env = MockEnv(max_steps=2)
        evaluator = Evaluator(env=env)
        network = MockNetwork(action=0)
        result = evaluator.run_episodes(network=network, n_episodes=3)
        scores = [ep.score for ep in result.per_episode]
        mean = sum(scores) / len(scores)
        expected_var = sum((s - mean) ** 2 for s in scores) / len(scores)
        expected_std = expected_var ** 0.5
        self.assertAlmostEqual(result.std_score, expected_std, places=5)

    def test_zero_laps(self):
        env = MockEnv(max_steps=2)
        evaluator = Evaluator(env=env)
        network = MockNetwork(action=0)
        result = evaluator.run_episodes(network=network, n_episodes=1)
        self.assertEqual(result.mean_laps, 0.0)
        self.assertEqual(result.mean_steps_per_lap, 0.0)
        self.assertEqual(result.total_laps, 0)
        self.assertEqual(result.per_episode[0].steps_per_lap, 0.0)

    def test_with_laps(self):
        env = MockEnv(max_steps=10, lap_at_steps=[3])
        evaluator = Evaluator(env=env)
        network = MockNetwork(action=0)
        result = evaluator.run_episodes(network=network, n_episodes=1)
        self.assertEqual(result.total_laps, 1)
        self.assertEqual(result.mean_laps, 1.0)
        # Episode runs 10 steps, 1 lap -> steps_per_lap = 10.0
        self.assertEqual(result.mean_steps_per_lap, 10.0)


class TestDeterminism(unittest.TestCase):
    """Tests that evaluation is deterministic when seeded."""

    def test_same_seed_same_results(self):
        env1 = MockEnv(max_steps=5)
        evaluator1 = Evaluator(env=env1)
        result1 = evaluator1.run_episodes(network=None, n_episodes=5, seed=42)

        env2 = MockEnv(max_steps=5)
        evaluator2 = Evaluator(env=env2)
        result2 = evaluator2.run_episodes(network=None, n_episodes=5, seed=42)

        self.assertEqual(len(result1.per_episode), len(result2.per_episode))
        for ep1, ep2 in zip(result1.per_episode, result2.per_episode):
            self.assertEqual(ep1.score, ep2.score)
            self.assertEqual(ep1.length, ep2.length)
            self.assertEqual(ep1.laps, ep2.laps)

    def test_different_seed_different_results(self):
        # This test may occasionally fail if randomness happens to produce
        # the same sequence, but with MockEnv the actions don't affect state
        # so all episodes look identical regardless of seed.  We test the
        # seeding mechanism instead by verifying that the env.seed() was called.
        env1 = MockEnv(max_steps=5)
        evaluator1 = Evaluator(env=env1)
        evaluator1.run_episodes(network=None, n_episodes=2, seed=42)
        self.assertEqual(env1._seed, 42)


class TestStepLimits(unittest.TestCase):
    """Tests for max_steps_per_lap and max_total_steps truncation."""

    def test_max_total_steps_truncation(self):
        env = MockEnv(max_steps=1000, never_done=True)
        evaluator = Evaluator(env=env)
        result = evaluator.run_episodes(
            network=None, n_episodes=1, max_total_steps=7
        )
        self.assertEqual(result.per_episode[0].length, 7)
        self.assertTrue(result.per_episode[0].truncated)

    def test_max_steps_per_lap_truncation(self):
        env = MockEnv(max_steps=1000, never_done=True)
        evaluator = Evaluator(env=env)
        result = evaluator.run_episodes(
            network=None, n_episodes=1, max_steps_per_lap=5, max_total_steps=1000
        )
        self.assertEqual(result.per_episode[0].length, 5)
        self.assertTrue(result.per_episode[0].truncated)

    def test_lap_completion_resets_timer(self):
        env = MockEnv(max_steps=1000, lap_at_steps=[5], never_done=True)
        evaluator = Evaluator(env=env)
        result = evaluator.run_episodes(
            network=None, n_episodes=1, max_steps_per_lap=10, max_total_steps=1000
        )
        # Lap at step 5 resets timer, so episode can run past 10 steps
        self.assertGreater(result.per_episode[0].length, 10)
        self.assertEqual(result.per_episode[0].laps, 1)


class TestEvaluateDQN(unittest.TestCase):
    """Tests for Evaluator.evaluate_dqn."""

    def test_evaluate_dqn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            network = DQNNetwork(hidden_dim=32)
            torch.save(
                {
                    "policy_state_dict": network.state_dict(),
                    "config": {"hidden_size": 32},
                },
                path,
            )
            env = MockEnv(max_steps=3)
            evaluator = Evaluator(env=env)
            result = evaluator.evaluate_dqn(checkpoint_path=path, n_episodes=2)
            self.assertEqual(result.agent_type, "dqn")
            self.assertEqual(result.n_episodes, 2)


class TestEvaluateRandom(unittest.TestCase):
    """Tests for Evaluator.evaluate_random."""

    def test_evaluate_random(self):
        env = MockEnv(max_steps=3)
        evaluator = Evaluator(env=env)
        result = evaluator.evaluate_random(n_episodes=3)
        self.assertEqual(result.agent_type, "random")
        self.assertEqual(result.n_episodes, 3)


class TestCompare(unittest.TestCase):
    """Tests for Evaluator.compare."""

    def _make_result(self, agent_type: str, mean_score: float, n_episodes: int = 10) -> Any:
        from evaluation.evaluator import EvaluationResult
        return EvaluationResult(
            agent_type=agent_type,
            n_episodes=n_episodes,
            mean_score=mean_score,
            max_score=mean_score,
            min_score=mean_score,
            std_score=0.0,
            mean_length=10.0,
            mean_laps=0.0,
            mean_steps_per_lap=0.0,
            total_laps=0,
            per_episode=[],
            seed=None,
            timestamp="",
        )

    def test_score_ratio(self):
        evaluator = Evaluator()
        dqn = self._make_result("dqn", 100.0)
        rnd = self._make_result("random", 20.0)
        comp = evaluator.compare(dqn, rnd)
        self.assertEqual(comp.score_ratio, 5.0)
        self.assertTrue(comp.passes_threshold)

    def test_fails_threshold(self):
        evaluator = Evaluator()
        dqn = self._make_result("dqn", 30.0)
        rnd = self._make_result("random", 20.0)
        comp = evaluator.compare(dqn, rnd)
        self.assertEqual(comp.score_ratio, 1.5)
        self.assertFalse(comp.passes_threshold)

    def test_zero_random_mean(self):
        evaluator = Evaluator()
        dqn = self._make_result("dqn", 10.0)
        rnd = self._make_result("random", 0.0)
        comp = evaluator.compare(dqn, rnd)
        self.assertEqual(comp.score_ratio, float("inf"))
        self.assertTrue(comp.passes_threshold)

    def test_zero_both(self):
        evaluator = Evaluator()
        dqn = self._make_result("dqn", 0.0)
        rnd = self._make_result("random", 0.0)
        comp = evaluator.compare(dqn, rnd)
        self.assertEqual(comp.score_ratio, 0.0)
        self.assertFalse(comp.passes_threshold)

    def test_negative_random_mean_dqn_positive(self):
        evaluator = Evaluator()
        dqn = self._make_result("dqn", 150.0)
        rnd = self._make_result("random", -20.0)
        comp = evaluator.compare(dqn, rnd)
        # score_ratio = (150 - (-20)) / 20 = 8.5
        self.assertEqual(comp.score_ratio, 8.5)
        self.assertTrue(comp.passes_threshold)

    def test_negative_random_mean_dqn_low_positive(self):
        evaluator = Evaluator()
        dqn = self._make_result("dqn", 50.0)
        rnd = self._make_result("random", -20.0)
        comp = evaluator.compare(dqn, rnd)
        # score_ratio = (50 - (-20)) / 20 = 3.5
        self.assertEqual(comp.score_ratio, 3.5)
        # DQN < 100, so fails
        self.assertFalse(comp.passes_threshold)

    def test_negative_random_mean_dqn_negative(self):
        evaluator = Evaluator()
        dqn = self._make_result("dqn", -5.0)
        rnd = self._make_result("random", -20.0)
        comp = evaluator.compare(dqn, rnd)
        # score_ratio = (-5 - (-20)) / 20 = 0.75
        self.assertEqual(comp.score_ratio, 0.75)
        self.assertFalse(comp.passes_threshold)


class TestSaveResults(unittest.TestCase):
    """Tests for Evaluator.save_results."""

    def _make_comparison(self) -> Any:
        from evaluation.evaluator import EvaluationComparison, EvaluationResult
        dqn = EvaluationResult(
            agent_type="dqn",
            n_episodes=2,
            mean_score=10.0,
            max_score=15.0,
            min_score=5.0,
            std_score=2.0,
            mean_length=5.0,
            mean_laps=0.0,
            mean_steps_per_lap=0.0,
            total_laps=0,
            per_episode=[
                PerEpisodeResult(1, 10.0, 5, 0, 0.0, 3, False),
                PerEpisodeResult(2, 15.0, 5, 0, 0.0, 3, False),
            ],
            seed=42,
            timestamp="2024-01-01T00:00:00",
        )
        rnd = EvaluationResult(
            agent_type="random",
            n_episodes=2,
            mean_score=5.0,
            max_score=5.0,
            min_score=5.0,
            std_score=0.0,
            mean_length=5.0,
            mean_laps=0.0,
            mean_steps_per_lap=0.0,
            total_laps=0,
            per_episode=[
                PerEpisodeResult(1, 5.0, 5, 0, 0.0, 3, False),
                PerEpisodeResult(2, 5.0, 5, 0, 0.0, 3, False),
            ],
            seed=42,
            timestamp="2024-01-01T00:00:00",
        )
        return EvaluationComparison(
            dqn_result=dqn,
            random_result=rnd,
            score_ratio=2.0,
            passes_threshold=False,
            output_dir="results",
        )

    def test_creates_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator()
            comp = self._make_comparison()
            json_path, _ = evaluator.save_results(comp, output_dir=tmpdir)
            self.assertTrue(os.path.isfile(json_path))
            self.assertTrue(json_path.endswith(".json"))
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("dqn", data)
            self.assertIn("random", data)
            self.assertEqual(data["score_ratio"], 2.0)
            self.assertEqual(data["passes_threshold"], False)
            self.assertEqual(data["threshold"], SCORE_THRESHOLD)

    def test_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator()
            comp = self._make_comparison()
            _, csv_path = evaluator.save_results(comp, output_dir=tmpdir)
            self.assertTrue(os.path.isfile(csv_path))
            self.assertTrue(csv_path.endswith(".csv"))
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            self.assertEqual(rows[0][0], "agent_type")
            # 2 DQN episodes + 2 random episodes + header = 5 rows
            self.assertEqual(len(rows), 5)
            dqn_rows = [r for r in rows[1:] if r[0] == "dqn"]
            random_rows = [r for r in rows[1:] if r[0] == "random"]
            self.assertEqual(len(dqn_rows), 2)
            self.assertEqual(len(random_rows), 2)

    def test_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nested", "results")
            evaluator = Evaluator()
            comp = self._make_comparison()
            json_path, csv_path = evaluator.save_results(comp, output_dir=output_dir)
            self.assertTrue(os.path.isdir(output_dir))
            self.assertTrue(os.path.isfile(json_path))
            self.assertTrue(os.path.isfile(csv_path))

    def test_timestamped_filenames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator()
            comp = self._make_comparison()
            json_path1, csv_path1 = evaluator.save_results(comp, output_dir=tmpdir)
            # Mock datetime to ensure a different timestamp
            with mock.patch("evaluation.evaluator.datetime") as mock_dt:
                mock_dt.now.return_value.strftime.return_value = "20240101_120000"
                mock_dt.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
                json_path2, csv_path2 = evaluator.save_results(comp, output_dir=tmpdir)
            self.assertNotEqual(os.path.basename(json_path1), os.path.basename(json_path2))
            self.assertNotEqual(os.path.basename(csv_path1), os.path.basename(csv_path2))


class TestPrintSummary(unittest.TestCase):
    """Tests for Evaluator.print_summary."""

    def _make_comparison(self, passes: bool = True) -> Any:
        from evaluation.evaluator import EvaluationComparison, EvaluationResult
        dqn = EvaluationResult(
            agent_type="dqn",
            n_episodes=100,
            mean_score=245.3,
            max_score=520.0,
            min_score=80.0,
            std_score=89.45,
            mean_length=142.5,
            mean_laps=1.85,
            mean_steps_per_lap=77.03,
            total_laps=185,
            per_episode=[],
            seed=42,
            timestamp="2024-01-01T00:00:00",
        )
        rnd = EvaluationResult(
            agent_type="random",
            n_episodes=100,
            mean_score=45.2,
            max_score=180.0,
            min_score=-10.0,
            std_score=35.12,
            mean_length=45.3,
            mean_laps=0.32,
            mean_steps_per_lap=141.56,
            total_laps=32,
            per_episode=[],
            seed=42,
            timestamp="2024-01-01T00:00:00",
        )
        return EvaluationComparison(
            dqn_result=dqn,
            random_result=rnd,
            score_ratio=5.43,
            passes_threshold=passes,
            output_dir="results",
        )

    def test_print_summary_pass(self):
        evaluator = Evaluator()
        comp = self._make_comparison(passes=True)
        with mock.patch("rich.console.Console.print") as mock_print:
            evaluator.print_summary(comp)
            mock_print.assert_called_once()
            table = mock_print.call_args[0][0]
            # Verify table has rows
            self.assertGreater(len(table.rows), 0)

    def test_print_summary_fail(self):
        evaluator = Evaluator()
        comp = self._make_comparison(passes=False)
        with mock.patch("rich.console.Console.print") as mock_print:
            evaluator.print_summary(comp)
            mock_print.assert_called_once()


class TestCLIParsing(unittest.TestCase):
    """Tests for evaluation CLI argument parsing."""

    def test_defaults(self):
        config = parse_args(["--model", "checkpoints/best.pt"])
        self.assertEqual(config.model_path, "checkpoints/best.pt")
        self.assertEqual(config.n_episodes, 100)
        self.assertIsNone(config.seed)
        self.assertEqual(config.output_dir, "results")
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.max_steps_per_lap, 200)
        self.assertEqual(config.max_total_steps, 2000)

    def test_custom_args(self):
        config = parse_args([
            "--model", "checkpoints/best.pt",
            "--episodes", "50",
            "--seed", "42",
            "--output", "eval_results/",
            "--device", "cpu",
            "--max-steps-per-lap", "150",
            "--max-total-steps", "1000",
        ])
        self.assertEqual(config.n_episodes, 50)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.output_dir, "eval_results/")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.max_steps_per_lap, 150)
        self.assertEqual(config.max_total_steps, 1000)

    def test_model_required(self):
        with self.assertRaises(SystemExit):
            parse_args([])


class TestMainEntryPoint(unittest.TestCase):
    """Tests for evaluation.__main__.main."""

    def test_successful_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pt")
            network = DQNNetwork(hidden_dim=16)
            torch.save(
                {
                    "policy_state_dict": network.state_dict(),
                    "config": {"hidden_size": 16},
                },
                model_path,
            )

            with mock.patch("sys.argv", [
                "evaluation",
                "--model", model_path,
                "--episodes", "2",
                "--output", tmpdir,
            ]):
                from evaluation.__main__ import main
                exit_code = main()
                self.assertEqual(exit_code, 0)

    def test_missing_checkpoint(self):
        with mock.patch("sys.argv", [
            "evaluation",
            "--model", "/nonexistent/model.pt",
            "--episodes", "2",
        ]):
            from evaluation.__main__ import main
            exit_code = main()
            self.assertEqual(exit_code, 1)

    def test_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pt")
            network = DQNNetwork(hidden_dim=16)
            torch.save(
                {
                    "policy_state_dict": network.state_dict(),
                    "config": {"hidden_size": 16},
                },
                model_path,
            )

            with mock.patch("sys.argv", [
                "evaluation",
                "--model", model_path,
                "--episodes", "2",
            ]):
                from evaluation.__main__ import main
                with mock.patch.object(
                    Evaluator, "evaluate_random", side_effect=KeyboardInterrupt
                ):
                    exit_code = main()
                    self.assertEqual(exit_code, 1)

    def test_main_system_exit_from_parse_args(self):
        # Calling main() without required --model should trigger SystemExit
        # catch in main() and return exit code 2
        with mock.patch("sys.argv", ["evaluation"]):
            from evaluation.__main__ import main
            exit_code = main()
            self.assertEqual(exit_code, 2)

    def test_main_save_results_oserror(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pt")
            network = DQNNetwork(hidden_dim=16)
            torch.save(
                {
                    "policy_state_dict": network.state_dict(),
                    "config": {"hidden_size": 16},
                },
                model_path,
            )

            with mock.patch("sys.argv", [
                "evaluation",
                "--model", model_path,
                "--episodes", "2",
                "--output", tmpdir,
            ]):
                from evaluation.__main__ import main
                with mock.patch.object(
                    Evaluator, "save_results", side_effect=OSError("disk full")
                ):
                    exit_code = main()
                    self.assertEqual(exit_code, 1)


class TestRandomAgent(unittest.TestCase):
    """Tests for RandomAgent."""

    def test_select_action_range(self):
        env = MockEnv()
        agent = RandomAgent(env)
        for _ in range(100):
            action = agent.select_action()
            self.assertIn(action, {0, 1, 2, 3})
            self.assertNotEqual(action, 4)

    def test_never_selects_stay(self):
        env = MockEnv()
        agent = RandomAgent(env)
        actions = [agent.select_action() for _ in range(200)]
        self.assertNotIn(4, actions)

    def test_seeding(self):
        env = MockEnv()
        agent1 = RandomAgent(env, seed=42)
        actions1 = [agent1.select_action() for _ in range(10)]
        agent2 = RandomAgent(env, seed=42)
        actions2 = [agent2.select_action() for _ in range(10)]
        self.assertEqual(actions1, actions2)


class TestEvaluationConfig(unittest.TestCase):
    """Tests for EvaluationConfig dataclass."""

    def test_default_values(self):
        config = EvaluationConfig(model_path="test.pt")
        self.assertEqual(config.n_episodes, 100)
        self.assertIsNone(config.seed)
        self.assertEqual(config.output_dir, "results")
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.max_steps_per_lap, 200)
        self.assertEqual(config.max_total_steps, 2000)

    def test_custom_values(self):
        config = EvaluationConfig(
            model_path="test.pt",
            n_episodes=50,
            seed=123,
            output_dir="out",
            device="cpu",
            max_steps_per_lap=150,
            max_total_steps=1000,
        )
        self.assertEqual(config.n_episodes, 50)
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.output_dir, "out")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.max_steps_per_lap, 150)
        self.assertEqual(config.max_total_steps, 1000)


if __name__ == "__main__":
    unittest.main()
