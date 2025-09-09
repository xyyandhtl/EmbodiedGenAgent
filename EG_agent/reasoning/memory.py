from collections import deque
from typing import Deque, Dict, Any
from environment.actions.planner_level import Action

class Memory:
    """
    Stores (max N) pairs of (observation, action) to provide context to the VLM.
    """
    def __init__(self, size: int = 10) -> None:
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=size)

    def add(self, observation: Dict[str, Any], action: Action) -> None:
        self.buffer.append({"obs": observation, "action": action})

    def summary(self) -> str:
        # Generate a bullet list summary to add to the prompt
        lines = []
        for idx, item in enumerate(self.buffer):
            act = item["action"]
            lines.append(f"{idx+1}. {act.kind} → {act.params}")
        return "\n".join(lines)
