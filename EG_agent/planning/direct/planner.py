from typing import Dict, Any, List
from EG_agent.environment.actions.planner_level_direct import Action, ActionKind, NavigationDirection, InteractionType

class Planner:
    """
    Translate perception + goal → next action and decompose goal into sub-goals.
    """

    def decompose(self, goal_text: str) -> List[str]:
        """
        Divide a high-level goal into a list of sub-goals.
        You can replace this implementation with a call to an LLM
        to dynamically generate intermediate steps.
        """
        lower = goal_text.lower()
        # Static example for goals containing "office"
        if "office" in lower:
            return [
                "Go to the main hallway",
                "Orient towards the office door",
                "Move to the office entrance",
                "Enter the office"
            ]
        # By default, a single step equal to the full goal
        return [goal_text]

    def decide(self, observation: Dict[str, Any]) -> Action:
        """
        Based on the current observation, return the next action.
        - If a blocking person is detected, generate an interaction.
        - Otherwise, generate a forward navigation command.
        """
        status = observation.get("status", "")
        obstacles = observation.get("obstacles", [])

        # 1) If there is a person in the way → spoken interaction
        if "person" in obstacles:
            return Action(
                kind=ActionKind.INTERACTION,
                params={
                    "interaction_type": InteractionType.TALK,
                    "target": "person"
                }
            )

        # 2) If no obstacle → navigate forward
        return Action(
            kind=ActionKind.NAVIGATION,
            params={
                "direction": NavigationDirection.FORWARD,
                "angle": 0.0,
                "distance": 0.5
            }
        )

# TODO: This script should take the goal and split it considering the current observations detected by the VLM. 
# It should call the model, see what happens, observe if there are obstacles, and if it's a human, the first goal should be to approach, interact with the human, convince them, verify the path is clear, and proceed. 
# If any of the sub-goals are not met, the goal or plan should be recalculated.