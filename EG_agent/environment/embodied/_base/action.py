import math

from EG_agent.planning.btpg.behavior_tree.base_nodes import Action
from EG_agent.planning.btpg.behavior_tree import Status
from EG_agent.prompts.default_objects import *

from EG_agent.system.envs.isaacsim_env import IsaacsimEnv

class EmbodiedAction(Action):
    can_be_expanded = True
    num_args = 1

    agent_env: IsaacsimEnv = None  # type: ignore

    # Backward-compat: single roll-up set if needed elsewhere
    AllObject = AllObject

    @property
    def action_class_name(self):
        return self.__class__.__name__

    def change_condition_set(self):
        raise NotImplementedError

    def update(self) -> Status:
        # 在这里执行具体的动作逻辑，比如移动、拍照等
        cur_action = self.action_class_name.lower()
        print(f"Executing action: {cur_action} on target: {self.args[0]}")
        cur_action_done = False

        if cur_action == "walk":
            cur_cmd_vel = self.agent_env.get_cur_cmd_vel()
            self.agent_env.run_action("cmd_vel", cur_cmd_vel)
            # If current target has entered camera FOV, consider walk complete
            if self.agent_env.goal_inview[self.args[0]]:
                cur_action_done = True
        elif cur_action == "mark":
            self.agent_env.run_action("mark", None)
        elif cur_action == "capture":
            self.agent_env.run_action("enum_command", (0,))
            cur_action_done = True
        elif cur_action == "report":
            self.agent_env.run_action("enum_command", (1,))
            cur_action_done = True
        else:
            raise ValueError(f"Unknown action type: {cur_action}")

        if cur_action_done:
            self.change_condition_set()
        return Status.RUNNING
