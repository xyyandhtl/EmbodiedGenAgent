from typing import Callable, Dict, List, Any, Optional, Tuple

from EG_agent.system.envs.base_env import BaseEnv
from EG_agent.system.module_path import AGENT_ENV_PATH


class IsaacsimEnv(BaseEnv):
    agent_num = 1

    # launch simulator
    # simulator_path = f'{ROOT_PATH}/../simulators/virtualhome/windows/VirtualHome.exe'
    # simulator_path = f'{ROOT_PATH}/../simulators/virtualhome/linux_exec/linux_exec.v2.3.0.x86_64'

    behavior_lib_path = f"{AGENT_ENV_PATH}/embodied"

    def __init__(self):
        if not self.headless:
            self.launch_simulator()
        super().__init__()
        self.action_callbacks_dict = {}

    def register_action_callbacks(self, type: str, fn: Callable):
        self.action_callbacks_dict[type] = fn


    def reset(self):
        raise NotImplementedError

    def task_finished(self):
        raise NotImplementedError

    def launch_simulator(self):
        # todo: maybe set ros2 topic names or register callbacks
        pass

    def load_scenario(self,scenario_id):
        # todo: maybe do nothing
        pass

    def run_action(self, action_type: str, action, verbose=False):
        # todo: convert the bt_action (high-level) to low-level commands and pass them to the simulator
        if action_type in self.action_callbacks_dict:
            self.action_callbacks_dict[action_type](action)
        else:
            raise ValueError(f"Action type {action_type} not registered.")
        if verbose:
            pass

    def close(self):
        # todo: maybe close simulator process
        pass

    def set_navigator

