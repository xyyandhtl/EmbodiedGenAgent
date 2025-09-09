from enum import Enum, auto

class AgentState(Enum):
    IDLE = auto()
    PLANNING = auto()
    MOVING = auto()
    NAVIGATING = auto()     # State for active navigation
    INTERACTING = auto()
    TALKING = auto()        # State for spoken interaction
    WAITING_REPLY = auto()  # Waiting for human response
    FINISHED = auto()       # Finalized state

class StateTracker:
    def __init__(self) -> None:
        self.state = AgentState.NAVIGATING
        self.last_observation = None  # Stores the last observation for internal reference

    def update_last_observation(self, observation: dict) -> None:
        """
        Updates the state based on the last received observation.
        """
        self.last_observation = observation
        # Here you can decide to change the state or keep it
        # Simple example:
        if observation.get("goal_observed") and self.state != AgentState.FINISHED:
            # If interacting and no person is blocking, switch to navigating
            if self.state in {AgentState.INTERACTING, AgentState.TALKING, AgentState.WAITING_REPLY}:
                if "person" not in observation.get("obstacles", []):
                    self.state = AgentState.NAVIGATING
            # If already entered the room, mark as finished
            if observation.get("room_entered"):
                self.state = AgentState.FINISHED

    def update(self, last_action, last_obs) -> None:
        """
        Additional method to update state using the last action and observation,
        can be called externally if preferred.
        """
        self.update_last_observation(last_obs)
