"""Unit tests for the DQN trainer."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import torch

from models.dqn_network import DQNNetwork, STATE_DIM
from training.dqn_trainer import DQNConfig, DQNTrainer


class TestDQNTrainerInitialization(unittest.TestCase):
    """Tests for trainer initialisation."""

    def test_default_config(self):
        trainer = DQNTrainer()
        self.assertIsInstance(trainer.config, DQNConfig)
        self.assertEqual(trainer.config.learning_rate, 5e-4)
        self.assertEqual(trainer.config.gamma, 0.99)

    def test_custom_config(self):
        config = DQNConfig(learning_rate=5e-4, batch_size=64)
        trainer = DQNTrainer(config)
        self.assertEqual(trainer.config.learning_rate, 5e-4)
        self.assertEqual(trainer.config.batch_size, 64)

    def test_networks_created(self):
        trainer = DQNTrainer()
        self.assertIsInstance(trainer.policy_net, DQNNetwork)
        self.assertIsInstance(trainer.target_net, DQNNetwork)

    def test_target_net_eval_mode(self):
        trainer = DQNTrainer()
        self.assertFalse(trainer.target_net.training)

    def test_optimizer_created(self):
        trainer = DQNTrainer()
        self.assertIsNotNone(trainer.optimizer)
        self.assertEqual(len(list(trainer.optimizer.param_groups)), 1)

    def test_buffer_created(self):
        trainer = DQNTrainer()
        self.assertEqual(len(trainer.replay_buffer), 0)
        self.assertEqual(trainer.replay_buffer.capacity, trainer.config.buffer_size)

    def test_device_auto_fallback(self):
        trainer = DQNTrainer(DQNConfig(device="auto"))
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(trainer.device.type, expected)

    def test_explicit_device(self):
        trainer = DQNTrainer(DQNConfig(device="cpu"))
        self.assertEqual(trainer.device.type, "cpu")


class TestEpsilonDecay(unittest.TestCase):
    """Tests for the epsilon-greedy decay schedule."""

    def test_start_value(self):
        trainer = DQNTrainer()
        self.assertAlmostEqual(trainer.get_epsilon(0), 1.0)

    def test_end_value(self):
        trainer = DQNTrainer()
        eps = trainer.get_epsilon(trainer.config.epsilon_decay_steps)
        self.assertAlmostEqual(eps, 0.01)

    def test_zero_decay_steps_returns_epsilon_end(self):
        config = DQNConfig(epsilon_decay_steps=0)
        trainer = DQNTrainer(config)
        eps = trainer.get_epsilon(0)
        self.assertAlmostEqual(eps, config.epsilon_end)
        eps = trainer.get_epsilon(100)
        self.assertAlmostEqual(eps, config.epsilon_end)

    def test_past_end_value(self):
        trainer = DQNTrainer()
        eps = trainer.get_epsilon(trainer.config.epsilon_decay_steps + 1000)
        self.assertAlmostEqual(eps, 0.01)

    def test_linear_interpolation(self):
        trainer = DQNTrainer()
        mid = trainer.config.epsilon_decay_steps // 2
        eps = trainer.get_epsilon(mid)
        expected = (1.0 + 0.01) / 2
        self.assertAlmostEqual(eps, expected, places=5)


class TestActionSelection(unittest.TestCase):
    """Tests for epsilon-greedy action selection."""

    def test_greedy_uses_policy_network(self):
        trainer = DQNTrainer()
        state = torch.randn(STATE_DIM)

        # Fix policy network weights for determinism
        with torch.no_grad():
            for p in trainer.policy_net.parameters():
                p.fill_(0.1)

        action = trainer.select_action(state, epsilon=0.0)
        self.assertIn(action, range(5))

    def test_random_action_with_epsilon_one(self):
        trainer = DQNTrainer()
        state = torch.randn(STATE_DIM)
        actions = [trainer.select_action(state, epsilon=1.0) for _ in range(100)]
        self.assertTrue(any(a != actions[0] for a in actions[1:]))

    def test_action_selection_uses_policy_not_target(self):
        trainer = DQNTrainer()
        state = torch.randn(STATE_DIM)

        # Make policy and target outputs differ
        with torch.no_grad():
            for p in trainer.policy_net.parameters():
                p.fill_(1.0)
            for p in trainer.target_net.parameters():
                p.fill_(-1.0)

        action = trainer.select_action(state, epsilon=0.0)
        # Policy net with all-positive weights should give positive Q-values
        # and argmax should be deterministic
        with torch.no_grad():
            q_policy = trainer.policy_net(state.unsqueeze(0))
            expected_action = int(q_policy.argmax(dim=1).item())
        self.assertEqual(action, expected_action)


class TestTargetNetworkUpdates(unittest.TestCase):
    """Tests for target network update mechanisms."""

    def test_hard_update_copies_weights(self):
        config = DQNConfig(tau=1.0, target_update_freq=2)
        trainer = DQNTrainer(config)

        # Modify policy weights
        with torch.no_grad():
            for p in trainer.policy_net.parameters():
                p.add_(1.0)

        # Before update, target should differ
        diffs_before = [
            torch.sum((tp - pp).abs()).item()
            for tp, pp in zip(trainer.target_net.parameters(), trainer.policy_net.parameters())
        ]
        self.assertTrue(any(d > 0.1 for d in diffs_before))

        # Trigger hard update at step_count == target_update_freq
        trainer.step_count = config.target_update_freq
        trainer.update_target_network()

        # After update, target should match policy
        for tp, pp in zip(trainer.target_net.parameters(), trainer.policy_net.parameters()):
            self.assertTrue(torch.allclose(tp, pp))

    def test_hard_update_no_copy_off_frequency(self):
        config = DQNConfig(tau=1.0, target_update_freq=1000)
        trainer = DQNTrainer(config)

        with torch.no_grad():
            for p in trainer.policy_net.parameters():
                p.add_(1.0)

        trainer.step_count = 1
        trainer.update_target_network()

        diffs_after = [
            torch.sum((tp - pp).abs()).item()
            for tp, pp in zip(trainer.target_net.parameters(), trainer.policy_net.parameters())
        ]
        self.assertTrue(any(d > 0.1 for d in diffs_after))

    def test_soft_update_averages_weights(self):
        config = DQNConfig(tau=0.1)
        trainer = DQNTrainer(config)

        # Set policy and target to known values
        with torch.no_grad():
            for p in trainer.policy_net.parameters():
                p.fill_(1.0)
            for p in trainer.target_net.parameters():
                p.fill_(0.0)

        trainer.update_target_network()

        for tp in trainer.target_net.parameters():
            self.assertAlmostEqual(tp.mean().item(), 0.1, places=5)


class TestTrainingStep(unittest.TestCase):
    """Tests for the DQN training step."""

    def _fill_buffer(self, trainer: DQNTrainer, n: int = 100) -> None:
        """Populate the replay buffer with random transitions."""
        for _ in range(n):
            trainer.replay_buffer.push(
                state=torch.randn(STATE_DIM),
                action=0,
                reward=1.0,
                next_state=torch.randn(STATE_DIM),
                done=False,
            )

    def test_train_step_returns_none_when_buffer_small(self):
        trainer = DQNTrainer()
        # Force past the update_frequency check so we hit the buffer-size check
        trainer.step_count = trainer.config.update_frequency - 1
        result = trainer.train_step()
        self.assertIsNone(result)

    def test_train_step_returns_finite_loss(self):
        trainer = DQNTrainer()
        self._fill_buffer(trainer, n=100)
        # Force training by setting step_count so that modulo passes
        trainer.step_count = trainer.config.update_frequency - 1
        loss = trainer.train_step()
        self.assertIsNotNone(loss)
        self.assertTrue(isinstance(loss, float))
        self.assertTrue(loss >= 0.0)

    def test_train_step_skips_on_frequency(self):
        trainer = DQNTrainer(DQNConfig(update_frequency=4))
        self._fill_buffer(trainer, n=100)
        trainer.step_count = 0
        loss = trainer.train_step()  # step_count becomes 1, 1 % 4 != 0
        self.assertIsNone(loss)

    def test_train_step_detects_nan(self):
        trainer = DQNTrainer()
        self._fill_buffer(trainer, n=100)

        # Corrupt buffer rewards to force NaN
        trainer.replay_buffer.rewards.fill_(float("inf"))
        trainer.step_count = trainer.config.update_frequency - 1

        with self.assertRaises(RuntimeError):
            trainer.train_step()

    def test_gradient_clipping_is_applied(self):
        trainer = DQNTrainer()
        self._fill_buffer(trainer, n=100)
        trainer.step_count = trainer.config.update_frequency - 1

        with mock.patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            trainer.train_step()
            mock_clip.assert_called_once()
            args, kwargs = mock_clip.call_args
            self.assertEqual(args[1], trainer.config.gradient_clip)

    def test_policy_network_changes_after_train(self):
        trainer = DQNTrainer()
        self._fill_buffer(trainer, n=100)

        # Snapshot weights before
        before = [p.clone() for p in trainer.policy_net.parameters()]

        trainer.step_count = trainer.config.update_frequency - 1
        trainer.train_step()

        # Weights should have changed
        changed = False
        for b, a in zip(before, trainer.policy_net.parameters()):
            if not torch.allclose(b, a):
                changed = True
                break
        self.assertTrue(changed)

    def test_target_net_used_for_q_target(self):
        trainer = DQNTrainer()
        self._fill_buffer(trainer, n=100)

        # Make target and policy outputs very different
        with torch.no_grad():
            for p in trainer.target_net.parameters():
                p.fill_(10.0)
            for p in trainer.policy_net.parameters():
                p.fill_(0.0)

        trainer.step_count = trainer.config.update_frequency - 1
        loss1 = trainer.train_step()

        # Change target net and train again — loss should change
        with torch.no_grad():
            for p in trainer.target_net.parameters():
                p.fill_(0.0)

        trainer.step_count = trainer.config.update_frequency - 1
        loss2 = trainer.train_step()

        self.assertIsNotNone(loss1)
        self.assertIsNotNone(loss2)
        self.assertNotEqual(loss1, loss2)

    def test_double_dqn_uses_policy_for_selection_target_for_evaluation(self):
        """Verify Double DQN: policy net selects action, target net evaluates it."""
        trainer = DQNTrainer()
        self._fill_buffer(trainer, n=100)

        # Set policy net to always select action 0
        # Set target net to have very different values per action
        with torch.no_grad():
            for p in trainer.policy_net.parameters():
                p.fill_(0.0)
            # Make target net output [100, 0, 0, 0, 0] for all inputs
            for name, p in trainer.target_net.named_parameters():
                if "weight" in name:
                    p.fill_(0.0)
                    # Bias of last layer controls output values
                    if p.shape[0] == 5:
                        p[0, :].fill_(100.0)
                elif "bias" in name:
                    p.fill_(0.0)
                    if p.shape[0] == 5:
                        p[0] = 100.0

        trainer.step_count = trainer.config.update_frequency - 1
        loss1 = trainer.train_step()

        # Now change policy net to always select action 1
        with torch.no_grad():
            for name, p in trainer.policy_net.named_parameters():
                if "weight" in name and p.shape[0] == 5:
                    p.fill_(0.0)
                    p[1, :].fill_(1.0)
                elif "bias" in name and p.shape[0] == 5:
                    p.fill_(0.0)
                    p[1] = 1.0

        trainer.step_count = trainer.config.update_frequency - 1
        loss2 = trainer.train_step()

        self.assertIsNotNone(loss1)
        self.assertIsNotNone(loss2)
        # Loss should differ because policy net now selects a different action
        # which gets evaluated to a different Q-value by the target net
        self.assertNotEqual(loss1, loss2)


class TestCheckpointSaveLoad(unittest.TestCase):
    """Tests for checkpoint save and load functionality."""

    def test_save_checkpoint_creates_file(self):
        trainer = DQNTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            trainer.save_checkpoint(path, episode=10, epsilon=0.5, best_score=100.0)
            self.assertTrue(os.path.isfile(path))

    def test_save_and_load_restores_networks(self):
        config = DQNConfig(hidden_size=64)
        trainer1 = DQNTrainer(config)
        # Set known weights
        with torch.no_grad():
            for p in trainer1.policy_net.parameters():
                p.fill_(0.42)
            for p in trainer1.target_net.parameters():
                p.fill_(0.99)
        trainer1.step_count = 123

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            trainer1.save_checkpoint(path, episode=5, epsilon=0.3, best_score=50.0)

            trainer2 = DQNTrainer(config)
            meta = trainer2.load_checkpoint(path)

        # Networks restored
        for p in trainer2.policy_net.parameters():
            self.assertAlmostEqual(p.mean().item(), 0.42, places=5)
        for p in trainer2.target_net.parameters():
            self.assertAlmostEqual(p.mean().item(), 0.99, places=5)
        self.assertEqual(trainer2.step_count, 123)
        self.assertEqual(meta["episode"], 5)
        self.assertAlmostEqual(meta["epsilon"], 0.3)
        self.assertAlmostEqual(meta["best_score"], 50.0)

    def test_load_missing_file_raises(self):
        trainer = DQNTrainer()
        with self.assertRaises(FileNotFoundError):
            trainer.load_checkpoint("/nonexistent/path/checkpoint.pt")

    def test_load_invalid_path_raises(self):
        trainer = DQNTrainer()
        with self.assertRaises(ValueError):
            trainer.load_checkpoint("")
        with self.assertRaises(ValueError):
            trainer.load_checkpoint(None)  # type: ignore[arg-type]

    def test_checkpoint_contains_config(self):
        trainer = DQNTrainer(DQNConfig(learning_rate=0.005))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            trainer.save_checkpoint(path, episode=1, epsilon=1.0, best_score=0.0)
            checkpoint = torch.load(path, weights_only=True)
            self.assertIn("config", checkpoint)
            self.assertEqual(checkpoint["config"]["learning_rate"], 0.005)


if __name__ == "__main__":
    unittest.main()
