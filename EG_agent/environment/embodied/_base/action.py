from EG_agent.planning.btpg.behavior_tree.base_nodes import Action
from EG_agent.planning.btpg.behavior_tree import Status

from EG_agent.system.envs.isaacsim_env import IsaacsimEnv

class EmbodiedAction(Action):
    can_be_expanded = True
    num_args = 1
    agent_env: IsaacsimEnv = None  # type: ignore

    @property
    def action_class_name(self):
        return self.__class__.__name__

    def change_condition_set(self):
        raise NotImplementedError

    def update(self) -> Status:
        # 在这里执行具体的动作逻辑，比如移动、拍照等
        # TODO: 当这里代码越来越复杂时,考虑将不同动作的逻辑拆分到各自的子类中
        cur_action = self.action_class_name.lower()
        target_obj = self.args[0]
        cur_action_done = False

        if cur_action == "find":
            if self.agent_env._vlmap_backend.object_found(self.args[0]):
                self.logger.info(f"Executing action: {cur_action} on target: {target_obj} done")
                cur_action_done = True
            elif self.agent_env._vlmap_backend.is_exploring(self.args[0]):
                # self.logger.info(f"Executing action: {cur_action} on target: {target_obj} running")
                pass    # TODO: 加上自主探索一次仍失败后再调一次start_find
            else:
                self.logger.info(f"Executing action: {cur_action} on target: {target_obj} begin")
                self.agent_env._vlmap_backend.start_find(self.args[0])
                cur_cmd_vel = self.agent_env.get_cur_cmd_vel()
                self.agent_env.run_action("cmd_vel", cur_cmd_vel)
        elif cur_action == "walk":
            cur_cmd_vel = self.agent_env.get_cur_cmd_vel()
            self.agent_env.run_action("cmd_vel", cur_cmd_vel)
            # If current target has entered camera FOV, consider walk complete
            if self.agent_env.goal_inview.get(target_obj, False):
                cur_action_done = True
        elif cur_action == "mark":
            self.logger.info(f"Executing action: {cur_action} on target: {target_obj}")
            self.agent_env.run_action("mark", None)
            cur_action_done = True
        elif cur_action == "capture":
            self.logger.info(f"Executing action: {cur_action} on target: {target_obj}")
            self.agent_env.run_action("enum_command", (0,))
            cur_action_done = True
        elif cur_action == "report":
            self.logger.info(f"Executing action: {cur_action} on target: {target_obj}")
            self.agent_env.run_action("enum_command", (1,))
            cur_action_done = True
        else:
            raise ValueError(f"Unknown action type: {cur_action}")

        if cur_action_done:
            self.change_condition_set()
            return Status.SUCCESS
        return Status.RUNNING
