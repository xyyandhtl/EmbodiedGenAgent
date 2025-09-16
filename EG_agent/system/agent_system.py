import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from enum import Enum
from typing import Union
import numpy as np

from EG_agent.prompts.object_sets import AllObject
from EG_agent.reasoning.logic_goal import LogicGoalGenerator
from EG_agent.planning.bt_planner import BTGenerator, BTAgent


class RobotAgentSystem:
    def __init__(self):
        # 初始化 逻辑Goal 生成器
        self.goal_generator = LogicGoalGenerator()

        # 构建 BT规划器 及其 Agent载体（Agent实现和环境交互）
        bt_generator = BTGenerator(env_name="embodied", 
                                   cur_cond_set=set(), 
                                   key_objects=list(AllObject))
        self.bt_agent = BTAgent(bt_generator)

        # 载入VLM地图导航模块
        self.dual_map = None  # TODO

        # 由外部设置具体环境实例
        self.env = None  

    def set_env(self, env):
        self.env = env

    def update(self, env_ret: dict):
        # todo: update the agent system based on environment feedback
        # self.bt_agent.update(env_ret)
        pass

    @property
    def finished(self) -> bool:
        pass

    @property
    def status(self) -> bool:
        pass

    def feed_observation(self, 
                         pose: np.ndarray,
                         intrinsics: np.ndarray, 
                         image: np.ndarray, 
                         depth: np.ndarray = None):
        pass

    def feed_instruction(self, text: str):
        pass

if __name__ == "__main__":
    agent_system = RobotAgentSystem()
    # agent_system.set_env(isacclab_env)
    # env_loop:
    # ...
