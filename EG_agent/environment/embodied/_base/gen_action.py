from EG_agent.planning.btpg.behavior_tree.base_nodes import Action
from EG_agent.planning.btpg.behavior_tree import Status
# import shared object sets
from EG_agent.prompts.object_sets import *
from EG_agent.system.envs.base_env import BaseAgentEnv

class EmbodiedAction(Action):
    can_be_expanded = True
    num_args = 1

    env: BaseAgentEnv = None  # type: ignore

    # use shared sets from object_sets.py
    LOCATIONS = LOCATIONS
    INSPECTION_POINTS = INSPECTION_POINTS
    INCIDENTS = INCIDENTS
    PERSONS = PERSONS

    # Derived sets for convenience
    NAV_POINTS = NAV_POINTS
    CAPTUREABLE = CAPTUREABLE
    MARKABLE = MARKABLE
    REPORTABLE = REPORTABLE

    # Backward-compat: single roll-up set if needed elsewhere
    AllObject = AllObject

    @property
    def action_class_name(self):
        return self.__class__.__name__

    def change_condition_set(self):
        pass

    def update(self) -> Status:
        # 在这里执行具体的动作逻辑，比如移动、拍照等
  
        cur_action = self.action_class_name.lower()
        print(f"Executing action: {cur_action} on target: {self.args[0].lower()}")
        cur_action_done = False

        if cur_action == "walk":
            cur_goal_place = self.agent.cur_goal_places[self.args[0].lower()]
            cur_cmd_vel = self.agent.cur_agent_states.get("cmd_vel", (0,0,0))
            self.env.run_action("cmd_vel", cur_cmd_vel)
            # If current target has entered camera FOV, consider walk complete
            # if getattr(self.env, "goal_inview", {}).get(self.args[0].lower(), False):
            if self.env.goal_inview[self.args[0].lower()]:
                cur_action_done = True
        elif cur_action == "capture":
            self.env.run_action("enum_command", 0)
            cur_action_done = True
        elif cur_action == "mark":
            self.env.run_action("enum_command", 1)
            cur_action_done = True
        elif cur_action == "report":
            self.env.run_action("enum_command", 2)
            cur_action_done = True
        else:
            raise ValueError(f"Unknown action type: {cur_action}")

        if cur_action_done:
            self.change_condition_set()
        return Status.RUNNING
