"""Unit tests for the DQN inference agent."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from unittest import mock
from typing import Any, Dict

import torch

from agents.dqn_agent import DQNAgent, FALLBACK_ACTION, STAY_ACTION_INDEX, main
from env.frogger_env import FroggerEnv
from models.dqn_network import DQNNetwork, HIDDEN_SIZE, STATE_DIM


def _run_coro(coro):
    """Run a coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


async def _fake_run(self):
    """No-op async replacement for DQNAgent.run."""
    pass


class TestDQNAgentInit(unittest.TestCase):
    """Tests for :class:`DQNAgent` initialization and checkpoint loading."""

    def _make_checkpoint(
        self,
        path: str,
        hidden_size: int = HIDDEN_SIZE,
        corrupt: bool = False,
    ) -> None:
        """Create a realistic checkpoint file at *path*."""
        if corrupt:
            with open(path, "wb") as f:
                f.write(b"this is not a valid checkpoint")
            return

        network = DQNNetwork(hidden_dim=hidden_size)
        checkpoint = {
            "policy_state_dict": network.state_dict(),
            "config": {"hidden_size": hidden_size},
        }
        torch.save(checkpoint, path)

    def _make_state(
        self,
        frog_x: float = 5.0,
        frog_y: int = 0,
        lives: int = 3,
        game_over: bool = False,
    ) -> Dict[str, Any]:
        """Build a minimal WebSocket state dictionary."""
        return {
            "width": 11,
            "height": 9,
            "frog_x": frog_x,
            "frog_y": frog_y,
            "lives": lives,
            "score": 0,
            "high_score": 0,
            "game_over": game_over,
            "win": False,
            "obstacles": [],
        }

    def test_successful_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path, hidden_size=128)
            agent = DQNAgent(model_path=path)
            self.assertIsInstance(agent.network, DQNNetwork)
            # Verify hidden_size was reconstructed correctly
            # First linear layer: input -> hidden
            first_linear = agent.network.net[0]
            self.assertEqual(first_linear.out_features, 128)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            DQNAgent(model_path="/nonexistent/path/model.pt")
        self.assertIn("Checkpoint file not found", str(ctx.exception))

    def test_corrupted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.pt")
            self._make_checkpoint(path, corrupt=True)
            with self.assertRaises(RuntimeError) as ctx:
                DQNAgent(model_path=path)
            self.assertIn("Failed to load checkpoint", str(ctx.exception))

    def test_device_override_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path, device="cpu")
            self.assertEqual(agent.device.type, "cpu")

    def test_device_auto(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path, device="auto")
            expected = "cuda" if torch.cuda.is_available() else "cpu"
            self.assertEqual(agent.device.type, expected)

    def test_network_eval_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path)
            self.assertFalse(agent.network.training)


class TestDQNAgentDeliberate(unittest.TestCase):
    """Tests for :meth:`DQNAgent.deliberate`."""

    def _make_checkpoint(self, path: str) -> None:
        network = DQNNetwork()
        checkpoint = {
            "policy_state_dict": network.state_dict(),
            "config": {"hidden_size": HIDDEN_SIZE},
        }
        torch.save(checkpoint, path)

    def _make_state(
        self,
        frog_x: float = 5.0,
        frog_y: int = 0,
        lives: int = 3,
        game_over: bool = False,
    ) -> Dict[str, Any]:
        return {
            "width": 11,
            "height": 9,
            "frog_x": frog_x,
            "frog_y": frog_y,
            "lives": lives,
            "score": 0,
            "high_score": 0,
            "game_over": game_over,
            "win": False,
            "obstacles": [],
        }

    def test_current_state_none_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path)
            agent.current_state = None
            result = asyncio_run(agent.deliberate())
            self.assertIsNone(result)

    def test_game_over_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path)
            agent.current_state = self._make_state(game_over=True)
            result = asyncio_run(agent.deliberate())
            self.assertIsNone(result)

    def test_state_encoding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path)
            agent.current_state = self._make_state(frog_x=3.0, frog_y=2)

            with mock.patch.object(agent.network, "forward") as mock_forward:
                mock_forward.return_value = torch.tensor(
                    [1.0, 0.5, 0.2, 0.1, 0.0]
                )
                asyncio_run(agent.deliberate())

            # Verify forward was called with a tensor of correct shape
            call_args = mock_forward.call_args
            self.assertIsNotNone(call_args)
            input_tensor = call_args[0][0]
            self.assertEqual(input_tensor.shape, (STATE_DIM,))

    def test_greedy_action_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path)
            agent.current_state = self._make_state()

            # Q-values: NORTH=0.2, SOUTH=0.5, EAST=1.0, WEST=0.1, STAY=2.0
            # After masking STAY, EAST should be selected
            with mock.patch.object(agent.network, "forward") as mock_forward:
                mock_forward.return_value = torch.tensor(
                    [0.2, 0.5, 1.0, 0.1, 2.0]
                )
                action = asyncio_run(agent.deliberate())

            self.assertEqual(action, "EAST")

    def test_stay_action_masking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path)
            agent.current_state = self._make_state()

            # Q-values: STAY is highest, but NORTH is second highest
            with mock.patch.object(agent.network, "forward") as mock_forward:
                mock_forward.return_value = torch.tensor(
                    [0.5, 0.3, 0.2, 0.1, 10.0]
                )
                action = asyncio_run(agent.deliberate())

            self.assertEqual(action, "NORTH")
            # Verify STAY was not selected
            self.assertNotEqual(action, "STAY")

    def test_nan_inf_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path)
            agent.current_state = self._make_state()

            with mock.patch.object(agent.network, "forward") as mock_forward:
                mock_forward.return_value = torch.tensor(
                    [float("nan"), 1.0, 1.0, 1.0, 1.0]
                )
                action = asyncio_run(agent.deliberate())

            self.assertEqual(action, FALLBACK_ACTION)

    def test_inf_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path)
            agent.current_state = self._make_state()

            with mock.patch.object(agent.network, "forward") as mock_forward:
                mock_forward.return_value = torch.tensor(
                    [1.0, float("inf"), 1.0, 1.0, 1.0]
                )
                action = asyncio_run(agent.deliberate())

            self.assertEqual(action, FALLBACK_ACTION)

    def test_action_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            self._make_checkpoint(path)
            agent = DQNAgent(model_path=path)
            agent.current_state = self._make_state()

            for idx, expected_action in FroggerEnv.INDEX_TO_ACTION.items():
                if idx == STAY_ACTION_INDEX:
                    continue  # STAY is masked

                q_values = torch.full((5,), -1.0)
                q_values[idx] = 10.0

                with mock.patch.object(agent.network, "forward") as mock_forward:
                    mock_forward.return_value = q_values.clone()
                    action = asyncio_run(agent.deliberate())

                self.assertEqual(action, expected_action)


class TestDQNAgentCLI(unittest.TestCase):
    """Tests for the DQN agent CLI entry point."""

    def test_model_required(self):
        with mock.patch("sys.argv", ["agents.dqn_agent"]):
            with self.assertRaises(SystemExit):
                main()

    def test_default_server_uri(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            network = DQNNetwork()
            torch.save(
                {
                    "policy_state_dict": network.state_dict(),
                    "config": {"hidden_size": HIDDEN_SIZE},
                },
                path,
            )

            with mock.patch("sys.argv", [
                "agents.dqn_agent",
                "--model", path,
            ]):
                with mock.patch.object(DQNAgent, "run", _fake_run):
                    with mock.patch("agents.dqn_agent.asyncio.run", side_effect=_run_coro):
                        main()

    def test_default_device(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            network = DQNNetwork()
            torch.save(
                {
                    "policy_state_dict": network.state_dict(),
                    "config": {"hidden_size": HIDDEN_SIZE},
                },
                path,
            )

            with mock.patch("sys.argv", [
                "agents.dqn_agent",
                "--model", path,
            ]):
                with mock.patch.object(DQNAgent, "run", _fake_run):
                    with mock.patch("agents.dqn_agent.asyncio.run", side_effect=_run_coro):
                        main()

    def test_custom_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            network = DQNNetwork()
            torch.save(
                {
                    "policy_state_dict": network.state_dict(),
                    "config": {"hidden_size": HIDDEN_SIZE},
                },
                path,
            )

            with mock.patch("sys.argv", [
                "agents.dqn_agent",
                "--model", path,
                "--server-uri", "ws://example.com:9999/ws",
                "--device", "cpu",
            ]):
                with mock.patch.object(DQNAgent, "run", _fake_run):
                    with mock.patch("agents.dqn_agent.asyncio.run", side_effect=_run_coro):
                        main()

    def test_main_invocation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            network = DQNNetwork()
            torch.save(
                {
                    "policy_state_dict": network.state_dict(),
                    "config": {"hidden_size": HIDDEN_SIZE},
                },
                path,
            )

            with mock.patch("sys.argv", [
                "agents.dqn_agent",
                "--model", path,
            ]):
                with mock.patch.object(DQNAgent, "run", _fake_run):
                    with mock.patch("agents.dqn_agent.asyncio.run", side_effect=_run_coro):
                        main()


def asyncio_run(coro):
    """Helper to run an async coroutine synchronously for testing."""
    return _run_coro(coro)


if __name__ == "__main__":
    unittest.main()
