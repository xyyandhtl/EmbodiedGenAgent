from EG_agent.environment.embodied._base.action import EmbodiedAction

class Mark(EmbodiedAction):
    can_be_expanded = True
    num_args = 1
    valid_args = EmbodiedAction.MARKABLE

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls, *arg):
        """
        get_info should return a dict with:
        - pre: a set of strings (preconditions) that must already hold before executing the action.
        - add: a set of strings (effects) that will be added to the agent.condition_set after execution.
        - del_set: a set of strings that will be removed from the agent.condition_set after execution.
        - cost: a numeric cost estimate for planning.
        """
        target = arg[0]
        return {
            "pre": {f"RobotNear({target})"},
            # unify with goal generator: MARK -> IsMarked(X)
            "add": {f"IsMarked({target})"},
            "del_set": set(),
            "cost": 5,
        }

    def change_condition_set(self):
        self.agent_env.condition_set |= (self.info["add"])
