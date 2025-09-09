from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


class NavigationDirection(str, Enum):
    FORWARD = "forward"
    FORWARD_LEFT = "forward_left"
    FORWARD_RIGHT = "forward_right"
    LEFT = "left"
    RIGHT = "right"

class ActionKind(str, Enum):
    NAVIGATION = "navigation"
    INTERACTION = "interaction"

class InteractionType(str, Enum):
    TALK = "talk"
    GESTURE = "gesture"


@dataclass
class Action:
    kind: ActionKind
    params: Dict[str, Any] = field(default_factory=dict)
    # Examples of params:
    #   NAVIGATION: {"direction": NavigationDirection, "angle": float, "distance": float}
    #   INTERACTION: {"interaction_type": InteractionType, "target": str}

    def is_done(self, observation: Dict[str, Any]) -> bool:
        """
        Quick logic to determine if the action is finished, based on the most recent observation
        (e.g., if I'm talking and the door is cleared => finished).
        """
        # ★ Implement according to your sensors or VLM outputs
        return observation.get("status") == "DONE"
    