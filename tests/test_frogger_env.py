"""Unit tests for the Frogger environment wrapper."""

from __future__ import annotations

import unittest

from env.frogger_env import Discrete, FroggerEnv
from server.logic import Obstacle


class TestDiscrete(unittest.TestCase):
    """Tests for the minimal Discrete action space helper."""

    def test_init_positive(self):
        d = Discrete(4)
        self.assertEqual(d.n, 4)

    def test_init_zero_raises(self):
        with self.assertRaises(ValueError):
            Discrete(0)

    def test_sample_range(self):
        d = Discrete(4)
        for _ in range(20):
            self.assertIn(d.sample(), range(4))

    def test_contains_valid(self):
        d = Discrete(4)
        self.assertTrue(d.contains(0))
        self.assertTrue(d.contains(3))

    def test_contains_invalid(self):
        d = Discrete(4)
        self.assertFalse(d.contains(-1))
        self.assertFalse(d.contains(4))
        self.assertFalse(d.contains("NORTH"))


class TestFroggerEnv(unittest.TestCase):
    """Tests for the FroggerEnv wrapper."""

    def test_init_default(self):
        env = FroggerEnv()
        self.assertEqual(env.reward_forward, 1.0)
        self.assertEqual(env.reward_checkpoint, 10.0)
        self.assertEqual(env.reward_lap, 20.0)
        self.assertEqual(env.reward_death, -10.0)
        self.assertEqual(env.reward_backward, -1.0)

    def test_init_custom_rewards(self):
        env = FroggerEnv(
            reward_forward=5.0,
            reward_checkpoint=25.0,
            reward_lap=100.0,
            reward_death=-50.0,
            reward_backward=-5.0,
        )
        self.assertEqual(env.reward_forward, 5.0)
        self.assertEqual(env.reward_checkpoint, 25.0)
        self.assertEqual(env.reward_lap, 100.0)
        self.assertEqual(env.reward_death, -50.0)
        self.assertEqual(env.reward_backward, -5.0)

    def test_action_space(self):
        env = FroggerEnv()
        self.assertEqual(env.action_space.n, 4)

    def test_observation_space(self):
        env = FroggerEnv()
        space = env.observation_space
        self.assertIsInstance(space, dict)
        expected_keys = {
            "width",
            "height",
            "frog_x",
            "frog_y",
            "lives",
            "score",
            "high_score",
            "game_over",
            "win",
            "obstacles",
        }
        self.assertEqual(set(space.keys()), expected_keys)

    def test_seed_determinism(self):
        env1 = FroggerEnv(seed=42)
        s1 = env1.reset()
        env2 = FroggerEnv(seed=42)
        s2 = env2.reset()
        self.assertEqual(s1, s2)

    def test_reset_returns_state(self):
        env = FroggerEnv()
        state = env.reset()
        self.assertIsInstance(state, dict)
        expected_keys = {
            "width",
            "height",
            "frog_x",
            "frog_y",
            "lives",
            "score",
            "high_score",
            "game_over",
            "win",
            "obstacles",
        }
        self.assertEqual(set(state.keys()), expected_keys)
        self.assertEqual(state["frog_y"], 0)
        self.assertEqual(state["lives"], 3)

    def test_reset_fresh_game(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        env.step("NORTH")
        env.step("NORTH")
        env.reset()
        self.assertEqual(env.game.frog_y, 0)
        self.assertEqual(env.game.lives, 3)
        self.assertEqual(env.game.score, 0)
        self.assertFalse(env.game.game_over)
        self.assertEqual(env._episode_length, 0)

    def test_step_valid_str_actions(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        for action in FroggerEnv.VALID_ACTIONS:
            env.reset()
            env.game.obstacles = []
            state, reward, done, info = env.step(action)
            self.assertIsInstance(state, dict)

    def test_step_valid_int_actions(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        for i in range(4):
            env.reset()
            env.game.obstacles = []
            state, reward, done, info = env.step(i)
            self.assertIsInstance(state, dict)

    def test_step_invalid_action_str(self):
        env = FroggerEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step("JUMP")

    def test_step_invalid_action_int(self):
        env = FroggerEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(99)

    def test_step_after_done_raises(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        env.game.lives = 1
        env.game.obstacles.append(
            Obstacle(x=5.0, y=1, width=2.5, speed=0.0, type="car", variant="test")
        )
        env.step("NORTH")  # dies, done=True
        with self.assertRaises(RuntimeError):
            env.step("NORTH")

    def test_cooldown_handling(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []

        original_update = env.game.update
        call_count = [0]

        def counting_update(dt: float) -> None:
            call_count[0] += 1
            original_update(dt)

        env.game.update = counting_update

        # First step after reset: no cooldown wait (frames == cooldown)
        env.step("NORTH")
        self.assertEqual(call_count[0], 1)

        # Second step: 4 cooldown ticks + 1 final update = 5 calls
        call_count[0] = 0
        env.step("NORTH")
        self.assertEqual(call_count[0], 5)

    def test_death_during_cooldown(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        env.step("NORTH")  # frog at y=1
        env.game.lives = 1
        env.game.obstacles.append(
            Obstacle(x=5.0, y=1, width=2.5, speed=0.0, type="car", variant="test")
        )
        state, reward, done, info = env.step("NORTH")
        self.assertTrue(done)
        self.assertEqual(reward, -10.0)
        self.assertEqual(info["lives"], 0)

    def test_death_after_action(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        env.game.lives = 1
        env.game.obstacles.append(
            Obstacle(x=5.0, y=1, width=2.5, speed=0.0, type="car", variant="test")
        )
        state, reward, done, info = env.step("NORTH")
        self.assertTrue(done)
        self.assertEqual(reward, -10.0)
        self.assertEqual(info["lives"], 0)

    def test_reward_forward_progress(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        _, reward, _, _ = env.step("NORTH")
        self.assertEqual(reward, 1.0)

    def test_reward_no_double_forward(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        _, r1, _, _ = env.step("NORTH")   # y=0 -> y=1, forward +1
        _, r2, _, _ = env.step("SOUTH")   # y=1 -> y=0, backward -1
        _, r3, _, _ = env.step("NORTH")   # y=0 -> y=1, no extra forward
        self.assertEqual(r1, 1.0)
        self.assertEqual(r2, -1.0)
        self.assertEqual(r3, 0.0)

    def test_reward_backward(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        env.step("NORTH")  # y=0 -> y=1
        _, reward, _, _ = env.step("SOUTH")
        self.assertEqual(reward, -1.0)

    def test_reward_checkpoint(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        for _ in range(3):
            env.step("NORTH")
        # 4th move reaches y=4 (checkpoint)
        _, reward, _, _ = env.step("NORTH")
        self.assertEqual(reward, 10.0)

    def test_reward_lap(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        for _ in range(7):
            env.step("NORTH")
        # 8th move reaches y=8 (lap completion)
        _, reward, _, _ = env.step("NORTH")
        self.assertEqual(reward, 20.0)

    def test_reward_death(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        env.game.lives = 1
        env.game.obstacles.append(
            Obstacle(x=5.0, y=1, width=2.5, speed=0.0, type="car", variant="test")
        )
        _, reward, done, _ = env.step("NORTH")
        self.assertEqual(reward, -10.0)
        self.assertTrue(done)

    def test_info_dict(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        _, _, _, info = env.step("NORTH")
        required_keys = {
            "episode_length",
            "lives",
            "score",
            "laps",
            "high_score",
            "game_over",
            "win",
        }
        self.assertTrue(required_keys.issubset(info.keys()))

    def test_episode_length_tracking(self):
        env = FroggerEnv()
        env.reset()
        env.game.obstacles = []
        _, _, _, info = env.step("NORTH")
        self.assertEqual(info["episode_length"], 1)
        _, _, _, info = env.step("NORTH")
        self.assertEqual(info["episode_length"], 2)

    def test_custom_reward_weights(self):
        env = FroggerEnv(
            reward_forward=5.0,
            reward_death=-50.0,
            reward_checkpoint=25.0,
            reward_lap=100.0,
            reward_backward=-5.0,
        )
        env.reset()
        env.game.obstacles = []
        _, reward, _, _ = env.step("NORTH")
        self.assertEqual(reward, 5.0)

        env.reset()
        env.game.obstacles = []
        env.game.lives = 1
        env.game.obstacles.append(
            Obstacle(x=5.0, y=1, width=2.5, speed=0.0, type="car", variant="test")
        )
        _, reward, done, _ = env.step("NORTH")
        self.assertEqual(reward, -50.0)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
