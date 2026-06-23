"""Unit tests for the DQN network and state encoder."""

from __future__ import annotations

import time
import unittest
from typing import Any, Dict

import torch

from models.dqn_network import (
    DQNNetwork,
    HIDDEN_SIZE,
    NUM_ACTIONS,
    STATE_DIM,
    StateEncoder,
)


class TestStateEncoder(unittest.TestCase):
    """Tests for :class:`StateEncoder`."""

    def _make_state(
        self,
        frog_x: float = 5.0,
        frog_y: int = 0,
        lives: int = 3,
        score: int = 0,
        obstacles: list[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Build a minimal state dictionary."""
        return {
            "width": 11,
            "height": 9,
            "frog_x": frog_x,
            "frog_y": frog_y,
            "lives": lives,
            "score": score,
            "high_score": 0,
            "game_over": False,
            "win": False,
            "obstacles": obstacles if obstacles is not None else [],
        }

    def test_output_shape_and_dtype(self):
        encoder = StateEncoder()
        state = self._make_state()
        tensor = encoder.encode(state)
        self.assertEqual(tensor.shape, (STATE_DIM,))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_normalization_ranges(self):
        encoder = StateEncoder(width=11, height=9)
        state = self._make_state(frog_x=5.0, frog_y=4, lives=2, score=50)
        tensor = encoder.encode(state)

        # frog_x_norm in [0, 1]
        self.assertGreaterEqual(tensor[0].item(), 0.0)
        self.assertLessEqual(tensor[0].item(), 1.0)

        # frog_y_norm in [0, 1]
        self.assertGreaterEqual(tensor[1].item(), 0.0)
        self.assertLessEqual(tensor[1].item(), 1.0)

        # lives_norm in [0, 1]
        self.assertGreaterEqual(tensor[2].item(), 0.0)
        self.assertLessEqual(tensor[2].item(), 1.0)

        # checkpoint flag in {0, 1}
        self.assertIn(tensor[3].item(), {0.0, 1.0})

        # Per-lane distances in [0, 1]
        for lane_idx in range(6):
            base = 4 + lane_idx * 3
            self.assertGreaterEqual(tensor[base].item(), 0.0)
            self.assertLessEqual(tensor[base].item(), 1.0)

    def test_empty_obstacles(self):
        encoder = StateEncoder()
        state = self._make_state(obstacles=[])
        tensor = encoder.encode(state)

        for lane_idx in range(6):
            base = 4 + lane_idx * 3
            self.assertEqual(tensor[base].item(), 1.0)      # safe distance
            self.assertEqual(tensor[base + 1].item(), 0.0)  # speed
            self.assertEqual(tensor[base + 2].item(), 0.0)  # width

    def test_checkpoint_inference_from_score(self):
        encoder = StateEncoder()
        state_no_cp = self._make_state(score=0)
        self.assertEqual(encoder.encode(state_no_cp)[3].item(), 0.0)

        state_cp = self._make_state(score=50)
        self.assertEqual(encoder.encode(state_cp)[3].item(), 1.0)

        state_lap = self._make_state(score=100)
        self.assertEqual(encoder.encode(state_lap)[3].item(), 0.0)

    def test_checkpoint_from_dict_field(self):
        encoder = StateEncoder()
        state = self._make_state(score=0)
        state["current_lap_checkpoint"] = 50
        self.assertEqual(encoder.encode(state)[3].item(), 1.0)

    def test_nearest_obstacle_selection(self):
        encoder = StateEncoder(width=11)
        state = self._make_state(
            frog_x=5.0,
            obstacles=[
                {"x": 0.0, "y": 1, "width": 1.0, "speed": 1.0, "type": "car"},
                {"x": 8.0, "y": 1, "width": 1.0, "speed": 0.5, "type": "car"},
            ],
        )
        tensor = encoder.encode(state)

        # Lane 1 features start at index 4
        # Frog body center at 5.5, half-width 0.4
        # Distance to obs at x=8 (width=1) on circle of 11:
        # dx = (5.5 - 8.0) % 11 = 8.5; dx > width + half_width (1.4), so
        # dist = min(8.5 - 1.0 - 0.4, 11 - 8.5 - 0.4) = min(7.1, 2.1) = 2.1
        # Normalized: 2.1/11 ≈ 0.1909
        self.assertAlmostEqual(tensor[4].item(), 2.1 / 11.0, places=5)
        # Speed of nearest obstacle (the one at x=8) is 0.5
        self.assertAlmostEqual(tensor[5].item(), 0.5 / 1.8, places=5)

    def test_wrap_around_distance(self):
        encoder = StateEncoder(width=10)
        # Frog at 1, obstacle at 8 with width 2 (covers [8, 10] which wraps to [8, 0])
        # Frog body center at 1.5, half-width 0.4
        # dx = (1.5 - 8.0) % 10 = 3.5; dx > 2 + 0.4 = 2.4, so
        # dist = min(3.5 - 2.0 - 0.4, 10 - 3.5 - 0.4) = min(1.1, 6.1) = 1.1
        dist = encoder._distance_to_obstacle(1.0, 8.0, 2.0, width=10.0)
        self.assertEqual(dist, 1.1)

        # Frog at 9, obstacle at 8 with width 2 (covers [8, 10])
        # Frog body center at 9.5, half-width 0.4
        # dx = (9.5 - 8.0) % 10 = 1.5; dx <= 2 + 0.4 = 2.4, so dist = 0 (overlap)
        dist = encoder._distance_to_obstacle(9.0, 8.0, 2.0, width=10.0)
        self.assertEqual(dist, 0.0)


class TestDQNNetwork(unittest.TestCase):
    """Tests for :class:`DQNNetwork`."""

    def test_forward_single_state(self):
        net = DQNNetwork()
        state = torch.randn(STATE_DIM)
        q_values = net(state)
        self.assertEqual(q_values.shape, (NUM_ACTIONS,))

    def test_forward_batch(self):
        net = DQNNetwork()
        batch = torch.randn(8, STATE_DIM)
        q_values = net(batch)
        self.assertEqual(q_values.shape, (8, NUM_ACTIONS))

    def test_parameter_count(self):
        net = DQNNetwork()
        total = sum(p.numel() for p in net.parameters())
        # Expected: ~20K parameters (allow generous tolerance)
        self.assertGreater(total, 15000)
        self.assertLess(total, 25000)

    def test_cpu_forward(self):
        net = DQNNetwork()
        state = torch.randn(STATE_DIM)
        q_values = net(state)
        self.assertTrue(torch.isfinite(q_values).all())

    def test_cuda_forward_if_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        net = DQNNetwork().to("cuda")
        state = torch.randn(STATE_DIM, device="cuda")
        q_values = net(state)
        self.assertEqual(q_values.device.type, "cuda")
        self.assertTrue(torch.isfinite(q_values).all())

    def test_forward_timing_cpu(self):
        net = DQNNetwork()
        net.eval()
        state = torch.randn(STATE_DIM)

        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                net(state)

        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                net(state)
                end = time.perf_counter()
                times.append(end - start)

        avg_time_ms = (sum(times) / len(times)) * 1000
        self.assertLess(avg_time_ms, 10.0, f"Average forward pass too slow: {avg_time_ms:.3f}ms")

    def test_no_output_activation(self):
        net = DQNNetwork()
        # Set all weights to a positive constant so ReLU is a no-op
        with torch.no_grad():
            for p in net.parameters():
                p.fill_(0.5)
        state = torch.ones(STATE_DIM)
        q_values = net(state)
        # With no output activation the value should be large and positive
        self.assertTrue(q_values.min().item() > 10.0)


if __name__ == "__main__":
    unittest.main()
