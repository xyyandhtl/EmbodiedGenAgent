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
        
        self.step_num = 0
        self._last_tick_time = time.time()
        self._last_plan_time = time.time()

        self.tick_interval = 0.05
        self.path_plan_interval = 5.0
        
        self.last_tick_output: str = ""
        self.tick_updated = False
        
        self.cur_target: str = ""
        self.condition_set = set()

        # self.init_statistics()
        self.create_behavior_lib()

        self._paren_pattern = re.compile(r'\((.*?)\)')

    def bind_bt(self, bt: BehaviorTree):
        bt.bind_agent(self)
        self.bt = bt

    def init_statistics(self):
        self.step_num = 1
        self._last_tick_time = time.time()
        self._last_plan_time = time.time()
        self.last_tick_output = ""

    def extract_targets(self, bt_node_name: str) -> str:
        walk_objects = self._paren_pattern.search(bt_node_name)
        if walk_objects:
            return walk_objects.group(1)
        return ""

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

    def step(self) -> bool:
        now = time.time()
        elapsed = now - self._last_tick_time
        # print(f"time since last tick: {elapsed}")
        if elapsed < self.tick_interval:
            time.sleep(self.tick_interval - elapsed)
            return False

        self._last_tick_time = now  # 更新时间点，保证间隔准确
        self.step_num += 1

        if self.bt is None:
            return False

        self.bt.tick()
        bt_output = self.bt.visitor.output_str
        parts = bt_output.split("Action", 1)
        if len(parts) > 1:
            bt_output = parts[1].strip()
        else:
            bt_output = ""

        # Calculate global path once within every path_plan_interval
        # self.cur_target = self.extract_targets(bt_output)
        # if bt_output.startswith("Walk") and now - self._last_plan_time > self.path_plan_interval:
        #     self._last_plan_time = now
        #     if self.cur_target:
        #         self.find_path(self.get_target_pos(self.cur_target))
        #     else:
        #         raise ValueError(f"Cannot parse walk object from BT output: {bt_output}")

        # when tick node changed
        if bt_output != self.last_tick_output:
            self.last_tick_output = bt_output
            self.tick_updated = True

            # Only calculate global path once
            if bt_output.startswith("Walk"):
                self.cur_target = self.extract_targets(bt_output)
                if self.cur_target:
                    self.find_path(self.get_target_pos(self.cur_target))
                else:
                    raise ValueError(f"Cannot parse walk object from BT output: {bt_output}")

            if self.print_ticks:
                # print(f"==== time:{self.time:f}s ======")
                print(f"Action {self.step_num}: {bt_output}")

        # self.agent_env_step()
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





