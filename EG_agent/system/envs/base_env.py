import time
import re

from EG_agent.planning.btpg.behavior_tree.behavior_libs import ExecBehaviorLibrary
from EG_agent.planning.btpg import BehaviorTree

# Merged Agent and Env for simplicity
# If hard to work or clarify the logic, separate them into two classes
class BaseAgentEnv:
    behavior_lib_path = None
    # =========================================================
    # base methods
    # =========================================================
    def __init__(self):
        # EnvAgent attributes
        self.print_ticks = True
        self.cur_goal_set = set()  # set of str

        # Behavior Tree attributes
        self.bt: BehaviorTree = None  # type: ignore

        self.response_frequency = 0.1
        self.step_num = 0
        self.next_response_time = 0.0
        self.last_tick_output = None
        self.time = 0.0
        self.start_time = time.time()
        
        self.condition_set = set()

        self.init_statistics()
        self.create_behavior_lib()

        self._paren_pattern = re.compile(r'\((.*?)\)')

    def bind_bt(self, bt: BehaviorTree):
        bt.bind_agent(self)
        self.bt = bt

    def init_statistics(self):
        self.step_num = 1
        self.next_response_time = self.response_frequency
        self.last_tick_output = None

    def extract_targes(self, bt_node_name):
        walk_objects = self._paren_pattern.search(bt_node_name)
        if walk_objects:
            return walk_objects.group(1)
        return None

    # =========================================================
    # methods that low-level env should impl 
    # =========================================================
    def find_path(self, goal_pose):
        # high-level env like VirtualHome has its wrapped function so it is not needed
        raise NotImplementedError
    
    def grab_object(self, object_name: str):
        # high-level env like VirtualHome has its wrapped function so it is not needed
        raise NotImplementedError
    
    def get_target_pos(self, target_name: str):
        raise NotImplementedError

    def step(self):
        if self.bt is None:
            return False
        
        self.time = time.time() - self.start_time
        # Integrated Agent.step logic
        if self.bt is not None and self.time > self.next_response_time:
            self.next_response_time += self.response_frequency
            self.step_num += 1

            self.bt.tick()
            bt_output = self.bt.visitor.output_str

            parts = bt_output.split("Action", 1)
            if len(parts) > 1:
                bt_output = parts[1].strip()
            else:
                bt_output = ""

            if bt_output != self.last_tick_output:
                # Do some work that low-level env is not callable
                if bt_output.startswith("Walk"):
                    walk_objects = self.extract_targes(bt_output)
                    if walk_objects:
                        self.find_path(self.get_target_pos(walk_objects))
                    else:
                        raise ValueError(f"Cannot parse walk object from BT output: {bt_output}")

                if self.print_ticks:
                    print(f"==== time:{self.time:f}s ======")
                    print(f"Action {self.step_num}: {bt_output}")
                    self.last_tick_output = bt_output

        # self.agent_env_step()
        self.last_step_time = self.time
        return self.task_finished()

    # =========================================================
    # control and status methods
    # =========================================================
    def task_finished(self):
        # raise NotImplementedError
        """根据条件集合判定任务完成。"""
        # Delegate scheduling/completion to behavior tree
        return self.cur_goal_set and self.cur_goal_set <= self.condition_set

    def create_behavior_lib(self):
        self.behavior_lib = ExecBehaviorLibrary(self.behavior_lib_path)

    def reset(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError





