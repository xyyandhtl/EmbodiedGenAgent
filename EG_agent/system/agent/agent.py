from typing import Optional, Any

from EG_agent.planning.btpg import BehaviorTree
from EG_agent.planning.bt_planner import BTGenerator
from EG_agent.system.envs.base_env import BaseEnv


# 和行为树生成器分离，每次需要从生成再bind_bt
class Agent(object):
    env: BaseEnv = None # type: ignore
    scene = None
    response_frequency = 1

    def __init__(self):
        self.condition_set = set()
        self.init_statistics()

    def bind_bt(self,bt):
        self.bt = bt
        bt.bind_agent(self)

    def init_statistics(self):
        self.step_num = 1
        self.next_response_time = self.response_frequency
        self.last_tick_output = None

    def step(self):
        if self.env.time > self.next_response_time:
            self.next_response_time += self.response_frequency
            self.step_num += 1

            self.bt.tick()
            bt_output = self.bt.visitor.output_str

            if bt_output != self.last_tick_output:
                if self.env.print_ticks:
                    print(f"==== time:{self.env.time:f}s ======")

                    # print(bt_output)
                    # 分割字符串
                    parts = bt_output.split("Action", 1)
                    # 获取 'Action' 后面的内容
                    if len(parts) > 1:
                        bt_output = parts[1].strip()  # 使用 strip() 方法去除可能的前后空格
                    else:
                        bt_output = ""  # 如果 'Action' 不存在于字符串中，则返回空字符串
                    print("Action ",bt_output)
                    print("\n")

                    self.last_tick_output = bt_output
                return True
            else:
                return False
            

# 整合行为树生成器的Agent，暂不用
class BTAgent(object):
    env: BaseEnv = None # type: ignore
    scene = None
    response_frequency = 1

    def __init__(self, generator: Optional[BTGenerator] = None):
        self.condition_set = set()
        self.init_statistics()
        if generator is not None:
            self.set_generator(generator)

    def bind_bt(self, bt: BehaviorTree):
        """
        Bind a generated BehaviorTree to this agent. Enforce type to BehaviorTree.
        bt must be the object returned by BTGenerator.generate(...)
        """
        if not isinstance(bt, BehaviorTree):
            raise TypeError("bind_bt expects a BehaviorTree instance (result of BTGenerator.generate)")
        self.bt = bt
        bt.bind_agent(self)

    def set_generator(self, generator: BTGenerator):
        """
        Attach a BTGenerator used to generate BTs on demand.
        """
        if not isinstance(generator, BTGenerator):
            raise TypeError("set_generator expects a BTGenerator instance")
        self.generator = generator

    def generate_and_bind(self, goal: Any, btml_name: str = "tree", init_state: Optional[set] = None) -> BehaviorTree:
        """
        Use the attached BTGenerator to create a BehaviorTree and bind it to this agent.
        Returns the bound BehaviorTree.
        """
        if self.generator is None:
            raise RuntimeError("No BTGenerator attached. Call set_generator(...) or pass generator to constructor.")
        bt = self.generator.generate(goal, btml_name=btml_name)
        self.bind_bt(bt)
        if init_state is not None:
            # initialize agent condition set if provided
            self.condition_set = set(init_state)
        return bt

    def init_statistics(self):
        self.step_num = 1
        self.next_response_time = self.response_frequency
        self.last_tick_output = None

    def step(self):
        if self.env.time > self.next_response_time:
            self.next_response_time += self.response_frequency
            self.step_num += 1

            self.bt.tick()
            bt_output = self.bt.visitor.output_str

            if bt_output != self.last_tick_output:
                if self.env.print_ticks:
                    print(f"==== time:{self.env.time:f}s ======")

                    # print(bt_output)
                    # 分割字符串
                    parts = bt_output.split("Action", 1)
                    # 获取 'Action' 后面的内容
                    if len(parts) > 1:
                        bt_output = parts[1].strip()  # 使用 strip() 方法去除可能的前后空格
                    else:
                        bt_output = ""  # 如果 'Action' 不存在于字符串中，则返回空字符串
                    print("Action ",bt_output)
                    print("\n")

                    self.last_tick_output = bt_output
                return True
            else:
                return False
            
    def update(self, env_ret: dict):
        """
        Update the BTAgent's state based on environment feedback.
        env_ret: environment return dictionary containing feedback information.
        """
        pass
