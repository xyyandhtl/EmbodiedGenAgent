import time
from typing import List

from EG_agent.planning.btpg.behavior_tree.behavior_libs import ExecBehaviorLibrary
from EG_agent.planning.btpg import BehaviorTree

class BaseAgentEnv(object):
    # Merged Agent and Env for simplicity
    # If hard to work or clarify the logic, separate them into two classes
    agent_num = 1
    behavior_lib_path = None
    print_ticks = False
    headless = False
    response_frequency = 1  # moved from Agent
    scene = None  # moved from Agent
    cur_action_args = ()

    def __init__(self):
        self.time = 0
        self.start_time = time.time()
        self.env = self  # so code expecting agent.env keeps working
        self.condition_set = set()  # moved from Agent
        self.bt: BehaviorTree = None  # type: ignore

        self.init_statistics()  # moved from Agent

        self.create_behavior_lib()

    # moved from Agent
    def bind_bt(self, bt):
        self.bt = bt
        bt.bind_agent(self)

    # moved from Agent
    def init_statistics(self):
        self.step_num = 1
        self.next_response_time = self.response_frequency
        self.last_tick_output = None

    def step(self):
        self.time = time.time() - self.start_time

        # Integrated Agent.step logic
        if self.bt is not None and self.time > self.next_response_time:
            self.next_response_time += self.response_frequency
            self.step_num += 1

            self.bt.tick()
            bt_output = self.bt.visitor.output_str

            if bt_output != self.last_tick_output:
                if self.print_ticks:
                    print(f"==== time:{self.time:f}s ======")
                    parts = bt_output.split("Action", 1)
                    if len(parts) > 1:
                        bt_output = parts[1].strip()
                    else:
                        bt_output = ""
                    print("Action ", bt_output)
                    print("\n")
                    self.last_tick_output = bt_output

        # self.env_step()
        self.last_step_time = self.time
        return self.task_finished()

    def task_finished(self):
        if {"IsIn(milk,fridge)", "IsClosed(fridge)"} <= self.condition_set:
            return True
        else:
            return False

    def create_behavior_lib(self):
        self.behavior_lib = ExecBehaviorLibrary(self.behavior_lib_path)

    def env_step(self):
        pass

    def reset(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError





