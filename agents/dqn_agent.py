"""DQN inference agent that plays Frogger via WebSocket.

This module provides:
    - :class:`DQNAgent`: A :class:`agents.base_agent.BaseAgent` subclass that
      loads trained DQN weights and selects greedy actions in real-time.
    - :func:`main`: CLI entry point for launching the agent.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Optional

import torch

from agents.base_agent import BaseAgent
from env.frogger_env import FroggerEnv
from models.dqn_network import DQNNetwork, StateEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - AGENT - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

STAY_ACTION_INDEX = 4
"""Index of the STAY action in the Q-value vector."""

FALLBACK_ACTION = "NORTH"
"""Safe fallback action when Q-values contain NaN or Inf."""


class DQNAgent(BaseAgent):
    """Inference agent that loads a trained DQN and plays via WebSocket.

    The agent encodes incoming WebSocket state JSON into the same 30-D tensor
    used during training, runs a greedy forward pass through the policy network,
    and returns the action with the highest Q-value (excluding STAY).
    """

    def __init__(
        self,
        model_path: str,
        server_uri: str = "ws://localhost:8765/ws",
        device: str = "auto",
    ) -> None:
        """Initialize the DQN agent.

        Args:
            model_path: Path to a PyTorch checkpoint file containing
                ``policy_state_dict`` and ``config``.
            server_uri: WebSocket URI of the Frogger game server.
            device: Torch device string. ``"auto"`` selects CUDA when available,
                otherwise CPU.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint is corrupted or incompatible.
        """
        super().__init__(server_uri=server_uri)
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.encoder = StateEncoder()
        self.network = self._load_network()
        self.network.eval()

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve the torch device string to a :class:`torch.device`.

        Args:
            device: Device string. ``"auto"`` maps to CUDA if available,
                otherwise CPU.

        Returns:
            The resolved torch device.
        """
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_network(self) -> DQNNetwork:
        """Load the policy network from the checkpoint file.

        Returns:
            A :class:`DQNNetwork` initialized with the checkpoint weights.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint is corrupted or incompatible.
        """
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.model_path}"
            )

        try:
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint from {self.model_path}: {exc}"
            ) from exc

        config = checkpoint.get("config", {})
        hidden_size = config.get("hidden_size", 64)

        network = DQNNetwork(hidden_dim=hidden_size).to(self.device)

        try:
            network.load_state_dict(checkpoint["policy_state_dict"])
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint corrupted or incompatible: {exc}"
            ) from exc

        return network

    async def deliberate(self) -> Optional[str]:
        """Select the next action using the trained DQN.

        The method encodes the current WebSocket state, runs a forward pass
        through the policy network, masks the STAY action, and returns the
        action with the highest Q-value.

        Returns:
            Action string (e.g. ``"NORTH"``) or ``None`` if the game is over
            or no state is available.
        """
        if self.current_state is None:
            return None

        if self.current_state.get("game_over") is True:
            return None

        state_tensor = self.encoder.encode(self.current_state).to(self.device)

        with torch.no_grad():
            q_values = self.network(state_tensor)

        if not torch.isfinite(q_values).all():
            logger.warning(
                "Q-values contain NaN or Inf; falling back to %s",
                FALLBACK_ACTION,
            )
            return FALLBACK_ACTION

        # Mask STAY action by setting its Q-value to -inf
        q_values[STAY_ACTION_INDEX] = float("-inf")

        action_index = int(q_values.argmax(dim=0).item())
        return FroggerEnv.INDEX_TO_ACTION[action_index]


def main() -> None:
    """CLI entry point for the DQN inference agent."""
    parser = argparse.ArgumentParser(
        description="Run the DQN inference agent against the Frogger server."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--server-uri",
        type=str,
        default="ws://localhost:8765/ws",
        help="WebSocket URI of the Frogger game server (default: ws://localhost:8765/ws).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device to use: "auto", "cpu", or "cuda" (default: auto).',
    )

    args = parser.parse_args()

    agent = DQNAgent(
        model_path=args.model,
        server_uri=args.server_uri,
        device=args.device,
    )
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
