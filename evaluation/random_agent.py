"""Offline random action baseline for evaluation.

This module provides :class:`RandomAgent`, a simple synchronous action
selector used during batch evaluation. It is intentionally **not** a reuse of
:class:`agents.dummy_agent.DummyAgent` for the following reasons:

1. **Architecture mismatch**: DummyAgent subclasses
   :class:`agents.base_agent.BaseAgent`, which is a WebSocket-based async
   client. It requires a running game server at ``ws://localhost:8765/ws`` and
   communicates via JSON messages. Evaluation, however, needs to run offline
   against :class:`env.frogger_env.FroggerEnv` directly, without any network
   overhead or async event loop.

2. **State format mismatch**: DummyAgent receives raw state dictionaries from
   the WebSocket (with keys like ``frog_x``, ``frog_y``, ``obstacles``, etc.)
   and returns action strings (e.g. ``"NORTH"``). The evaluation pipeline
   works with encoded 30-D tensors and action indices (0-4) that
   :class:`env.frogger_env.FroggerEnv` accepts.

3. **Performance**: Batch evaluation of 100+ episodes must be fast and
   deterministic. A WebSocket round-trip per action would add ~30 ms of
   latency per step, making evaluation orders of magnitude slower.

4. **Simplicity**: RandomAgent is a 10-line synchronous class with no
   external dependencies. DummyAgent is ~25 lines plus the entire BaseAgent
   inheritance chain (~100+ lines of WebSocket logic).

For these reasons, we created a dedicated :class:`RandomAgent` that operates
directly on :class:`env.frogger_env.FroggerEnv` using action indices.
"""

from __future__ import annotations

import random
from typing import Optional

from env.frogger_env import FroggerEnv


class RandomAgent:
    """Synchronous random action selector for :class:`FroggerEnv`.

    This agent selects uniformly from the four directional actions
    (NORTH, SOUTH, EAST, WEST) and **never** selects STAY.  It is designed
    for fast offline batch evaluation without any network or async
    dependencies.

    See the module docstring above for the rationale behind creating a
    separate random agent instead of reusing DummyAgent.
    """

    def __init__(self, env: FroggerEnv, seed: Optional[int] = None) -> None:
        """Initialize the random agent.

        Args:
            env: The Frogger environment instance (used to validate action
                space; the agent does not modify it).
            seed: Optional random seed for reproducible action selection.
        """
        self.env = env
        if seed is not None:
            random.seed(seed)

    def select_action(self) -> int:
        """Select a random action index from {0, 1, 2, 3} (never STAY).

        Returns:
            Action index: 0=NORTH, 1=SOUTH, 2=EAST, 3=WEST.
        """
        # We exclude STAY (index 4) to match the DQN agent's deliberate()
        # behaviour, which masks the STAY action during inference.
        return random.randrange(4)
