import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from EG_agent.prompts.object_sets import AllObject
from EG_agent.reasoning.logic_goal import LogicGoalGenerator
from EG_agent.planning.bt_planner import BTGenerator
from EG_agent.vlmap.vlmap_nav_ros2 import VLMapNavROS2
from EG_agent.system.agent.agent import Agent
from EG_agent.system.envs.isaacsim_env import IsaacsimEnv


class EGAgentSystem:
    def __init__(self):
        # 初始化 '逻辑Goal生成器'
        self.goal_generator = LogicGoalGenerator()

        # 构建 '行为树规划器'
        self.bt_generator = BTGenerator(env_name="embodied", 
                                        cur_cond_set=set(), 
                                        key_objects=list(AllObject))
        
        # 行为树执行的 'Agent载体’，通过bint_bt动态绑定行为树，被绑定到 '部署环境执行器' 和环境交互
        self.bt_agent = Agent()

        # '部署环境执行器'，定义如何和部署环境交互，和 'Agent载体’ 绑定
        self.env = IsaacsimEnv()
        self.env.place_agent(self.bt_agent)

        # 初始化 'VLM地图导航模块'
        self.vlmap_backend = VLMapNavROS2()

        # todo

    def set_env(self, env):
        self.env = env

    def update(self, env_ret: dict):
        # todo: update the agent system based on environment feedback
        # self.bt_agent.update(env_ret)
        pass

    @property
    def finished(self) -> bool:
        # todo
        return False

    @property
    def status(self) -> bool:
        # todo
        return False

    def feed_observation(self, 
                         pose: np.ndarray,
                         intrinsics: np.ndarray, 
                         image: np.ndarray, 
                         depth: np.ndarray):
        # todo: camrera observation -> vlmap -> update condition 
        #                                  └ -> low-level action generation
        pass

    def feed_instruction(self, text: str):
        # todo: goal_generator->bt_generator->bt_agent.bind_bt
        pass


if __name__ == "__main__":
    agent_system = EGAgentSystem()
    # agent_system.set_env(isacclab_env)
    # env_loop:
    # ...
