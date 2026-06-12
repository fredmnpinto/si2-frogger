__author__ = "Mário Antunes"
__version__ = "1.1.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import asyncio
import random
from typing import Optional
from .base_agent import BaseAgent

class DummyAgent(BaseAgent):
    async def deliberate(self) -> Optional[str]:
        if not self.current_state or self.current_state.get("game_over"):
            return None
        
        # Prefer using valid actions sent by the server
        actions = self.current_state.get("actions") or self.current_state.get("valid_actions")
        if actions:
            return random.choice(actions)
        
        # Fallback to standard directions
        return random.choice(["NORTH", "SOUTH", "EAST", "WEST"])

if __name__ == "__main__":
    agent = DummyAgent()
    asyncio.run(agent.run())
