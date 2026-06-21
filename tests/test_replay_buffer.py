"""Unit tests for the experience replay buffer."""

from __future__ import annotations

import unittest

import torch

from training.replay_buffer import ReplayBuffer

STATE_DIM = 22


class TestReplayBuffer(unittest.TestCase):
    """Tests for :class:`ReplayBuffer`."""

    def test_capacity_and_len(self):
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity=10, state_dim=STATE_DIM, device=device)
        self.assertEqual(len(buffer), 0)

        for i in range(5):
            buffer.push(
                state=torch.ones(STATE_DIM) * i,
                action=i,
                reward=float(i),
                next_state=torch.ones(STATE_DIM) * (i + 1),
                done=False,
            )
        self.assertEqual(len(buffer), 5)

    def test_fifo_overwrite(self):
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity=3, state_dim=STATE_DIM, device=device)

        for i in range(5):
            buffer.push(
                state=torch.ones(STATE_DIM) * i,
                action=i,
                reward=float(i),
                next_state=torch.ones(STATE_DIM) * (i + 1),
                done=(i == 4),
            )

        self.assertEqual(len(buffer), 3)
        # The oldest entry (i=0) should have been overwritten.
        # After 5 pushes into capacity-3 buffer:
        #   i=3 overwrote index 0, i=4 overwrote index 1, i=2 remains at index 2.
        self.assertTrue(torch.equal(buffer.states[0], torch.ones(STATE_DIM) * 3))
        self.assertTrue(torch.equal(buffer.states[1], torch.ones(STATE_DIM) * 4))
        self.assertTrue(torch.equal(buffer.states[2], torch.ones(STATE_DIM) * 2))

    def test_sample_shapes(self):
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity=100, state_dim=STATE_DIM, device=device)

        for i in range(50):
            buffer.push(
                state=torch.randn(STATE_DIM),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(STATE_DIM),
                done=False,
            )

        batch_size = 16
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        self.assertEqual(states.shape, (batch_size, STATE_DIM))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(next_states.shape, (batch_size, STATE_DIM))
        self.assertEqual(dones.shape, (batch_size,))

        self.assertEqual(states.dtype, torch.float32)
        self.assertEqual(actions.dtype, torch.long)
        self.assertEqual(rewards.dtype, torch.float32)
        self.assertEqual(dones.dtype, torch.float32)

    def test_sample_less_than_batch_size(self):
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity=100, state_dim=STATE_DIM, device=device)

        for i in range(5):
            buffer.push(
                state=torch.randn(STATE_DIM),
                action=0,
                reward=1.0,
                next_state=torch.randn(STATE_DIM),
                done=False,
            )

        batch_size = 32
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        self.assertEqual(states.shape[0], 5)

    def test_sample_empty_raises(self):
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity=10, state_dim=STATE_DIM, device=device)
        with self.assertRaises(ValueError):
            buffer.sample(1)

    def test_sample_without_replacement(self):
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity=10, state_dim=STATE_DIM, device=device)

        for i in range(5):
            buffer.push(
                state=torch.ones(STATE_DIM) * i,
                action=i,
                reward=float(i),
                next_state=torch.ones(STATE_DIM) * i,
                done=False,
            )

        # Sample the full buffer multiple times; with true without-replacement
        # sampling every element should appear exactly once in each sample.
        for _ in range(10):
            states, actions, rewards, next_states, dones = buffer.sample(5)
            self.assertEqual(len(torch.unique(rewards)), 5)
            self.assertTrue(torch.equal(torch.sort(rewards)[0], torch.arange(5, dtype=torch.float32)))

    def test_device_consistency(self):
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity=10, state_dim=STATE_DIM, device=device)
        buffer.push(
            state=torch.randn(STATE_DIM),
            action=0,
            reward=1.0,
            next_state=torch.randn(STATE_DIM),
            done=False,
        )
        states, _, _, _, _ = buffer.sample(1)
        self.assertEqual(states.device, device)

    def test_done_storage(self):
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity=10, state_dim=STATE_DIM, device=device)
        buffer.push(
            state=torch.randn(STATE_DIM),
            action=0,
            reward=1.0,
            next_state=torch.randn(STATE_DIM),
            done=True,
        )
        _, _, _, _, dones = buffer.sample(1)
        self.assertEqual(dones[0].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
