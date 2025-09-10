from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

@dataclass
class Goal:
    description: str                 # “Enter office X”
    location_hint: Optional[str] = ""  # “coordinates of office X door”
    finished: bool = False

class GoalManager:
    """
    Splits a large goal into micro-goals and updates them.
    """
    def __init__(self, initial_goal: Goal) -> None:
        self.goal_stack: Deque[Goal] = deque([initial_goal])

    @property
    def current(self) -> Goal:
        return self.goal_stack[-1]

    def update_from_observation(self, obs: dict) -> None:
        """
        Analyzes the observation and marks goals as completed or creates new ones.
        """
        if "room_entered" in obs and obs["room_entered"]:
            self.current.finished = True

    def push_subgoal(self, goal: Goal) -> None:
        self.goal_stack.append(goal)

    def pop_finished(self) -> None:
        while self.goal_stack and self.goal_stack[-1].finished:
            self.goal_stack.pop()


