from __future__ import annotations

import sys
from enum import Enum, auto
from typing import Union
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np

from environment.state.state_tracker import StateTracker, AgentState
from environment.actions.planner_level import Action
from planning.sequence.goal_manager import Goal, GoalManager
from planning.sequence.planner import Planner
from reasoning.perception import Perception, Observation
from reasoning.memory import Memory

try:
    # ConversationManager is optional – import lazily.
    from .conversation import ConversationManager  # type: ignore
except ImportError:
    ConversationManager = None  # type: ignore

__all__ = ["RobotAgent"]


class RobotAgentSystem:
    """Public-facing façade of the whole agent stack.

    Usage
    -----
    >>> agent = RobotAgent(goal_text="Enter office 12")
    >>> while True:
    ...     img = camera.read()
    ...     action = agent.step(img)
    ...     robot.execute(action)
    ...     if agent.finished:
    ...         break
    """

    def __init__(
        self,
        *,
        goal_text: str,
        provider: str = "openai",
        history_size: int = 10,
    ) -> None:
        # 1) Perception engines
        self.perception = Perception(
            goal_text=goal_text,
            provider=provider,
            history_size=history_size,
        )

        # 2) Cognition / memory / planning
        self.goal_manager = GoalManager(Goal(goal_text))
        self.memory = Memory(size=history_size)
        self.planner = Planner()
        self.state_tracker = StateTracker()

        # 3) Optional conversation manager
        self.conversation = ConversationManager(goal_text) if ConversationManager else None

        # 4) Decompose high-level goal into sub-goals
        if hasattr(self.planner, "decompose"):
            subgoals = self.planner.decompose(goal_text)
            for sg in subgoals:
                self.goal_manager.push_subgoal(Goal(sg))

        # 5) Print initial plan
        print("┌ Initial sub-goals plan ─────────────────────────")
        for idx, g in enumerate(self.goal_manager.goal_stack, start=1):
            print(f"│ {idx}. {g.description}")
        print("└──────────────────────────────────────────────────")

    @property
    def finished(self) -> bool:
        """Whether the agent has completed its top-level mission."""
        return self.state_tracker.state == AgentState.FINISHED

    def step(self, img: Union[str, Path, Image.Image, np.ndarray]) -> Action:
        """One control tick:
        1. Perceive with the right mode.
        2. Update goal_manager & state_tracker.
        3. Decide next action via planner.
        4. If INTERACTION, run ConversationManager.
        5. Log to memory.
        """

        # 1) Run perception
        mode = self._current_mode()
        obs: Observation = self.perception.perceive(img, mode=mode)
        print(f'[Perception - {mode}] vlm description: ', obs["description"])

        # 2) Update goals + state
        self.goal_manager.update_from_observation(obs)
        self.goal_manager.pop_finished()
        self.state_tracker.update_last_observation(obs)

        # 3) Plan next action
        action = self.planner.decide(obs)

        # 4) If interaction, trigger conversation turn
        if action.kind.name.lower() == "interaction" and self.conversation:
            utterance = self.conversation.robot_turn()
            action.params["utterance"] = utterance

        # 5) Record into memory
        self.memory.add(obs, action)

        return action

    def _current_mode(self) -> str:
        """Choose 'navigation' vs 'interaction' based on the agent's state."""
        if self.state_tracker.state in {
            AgentState.INTERACTING,
            AgentState.TALKING,
            AgentState.WAITING_REPLY,
        }:
            return "interaction"
        return "navigation"


# TODO: This script should call the various parts or sub-libraries, take the goal as input, as well as an image, 
# create sub-goals for what it needs to do, print the plan with the sub-goals, and return navigation and interaction actions. 
# In navigation actions, it will return movements forward, left, right, with distance. 
# In the case of interaction, it can start a conversation and use TTS and STT to talk to the human, 
# so we need to understand how to return this in the library, as it will later be used in ROS. 
# TODO: After each interaction ends, the environment should be analyzed with an image to verify if the path is clear and switch to navigation actions. 
# TODO: Ensure this library with all its sub-libraries can be used easily in ROS2.

