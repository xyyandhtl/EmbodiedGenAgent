import math
import py_trees

from EG_agent.planning.btpg.behavior_tree.base_nodes import Action
from EG_agent.planning.btpg.behavior_tree import Status
from EG_agent.prompts.default_objects import *

from EG_agent.system.envs.isaacsim_env import IsaacsimEnv

class EmbodiedAction(Action):
    can_be_expanded = True
    num_args = 1
    agent_env: IsaacsimEnv = None  # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._find_exploration_started = False

    @property
    def action_class_name(self):
        return self.__class__.__name__

    def change_condition_set(self):
        raise NotImplementedError

    def terminate(self, new_status: Status):
        """Reset the internal state of the action when it is stopped or finished."""
        self.logger.debug(f"Terminating with status {new_status}, resetting find exploration state.")
        self._find_exploration_started = False

    def _update_find(self, target_obj: str) -> Status:
        """
        Handles the streamlined, stateful logic for the 'find' action.
        On every tick, it tries to find the object. If not found, it manages the exploration lifecycle.
        """
        vlmap_backend = self.agent_env._vlmap_backend

        # 1. Always check for the object first. This is the success condition.
        position = vlmap_backend.query_object(target_obj)
        if position is not None:
            self.logger.info(f"SUCCESS: Object '{target_obj}' found at {position}.")
            if self._find_exploration_started:
                vlmap_backend.stop_exploration()
            self.agent_env.set_object_places({target_obj: position})
            self.change_condition_set()
            return Status.SUCCESS

        # --- If we are here, the object was NOT found. Now, manage exploration. ---

        # 2. If exploration has finished (but we didn't find the object), it's a failure.
        if self._find_exploration_started and not vlmap_backend.is_exploring:
            self.logger.warning(f"FAILURE: Object '{target_obj}' not found after exploration.")
            return Status.FAILURE

        # 3. If exploration hasn't been started yet, start it now.
        if not self._find_exploration_started:
            self.logger.warning(f"Object '{target_obj}' not found. Starting exploration.")
            vlmap_backend.start_exploration_to_find(target_obj)
            self._find_exploration_started = True

        # 4. In all other cases (exploration is running), the status is RUNNING.
        self.logger.debug(f"RUNNING: Still trying to find '{target_obj}'.")
        return Status.RUNNING

    def update(self) -> Status:
        # 在这里执行具体的动作逻辑，比如移动、拍照等
        # TODO: 当这里代码越来越复杂时,考虑将不同动作的逻辑拆分到各自的子类中
        cur_action = self.action_class_name.lower()
        target_obj = self.args[0]
        if cur_action != "walk":
            self.logger.info(f"Executing action: {cur_action} on target: {target_obj}")
        cur_action_done = False

        if cur_action == "find":
            return self._update_find(target_obj)

        if cur_action == "walk":
            cur_cmd_vel = self.agent_env.get_cur_cmd_vel()
            self.agent_env.run_action("cmd_vel", cur_cmd_vel)
            # If current target has entered camera FOV, consider walk complete
            if self.agent_env.goal_inview.get(target_obj, False):
                cur_action_done = True
        elif cur_action == "mark":
            self.agent_env.run_action("mark", None)
            cur_action_done = True
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
            return Status.SUCCESS
        return Status.RUNNING
