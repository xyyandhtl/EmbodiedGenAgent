import time
from typing import Union, List, Optional, Dict, Any

from EG_agent.planning.btpg.algos.llm_client.tools import goal_transfer_str
from EG_agent.planning.btpg.algos.bt_planning.bt_planner_interface import BTPlannerInterface
from EG_agent.planning.btpg.behavior_tree.behavior_libs.ExecBehaviorLibrary import ExecBehaviorLibrary
from EG_agent.planning.btpg import BehaviorTree
from EG_agent.system.path import AGENT_ENV_PATH

class BTGenerator:
    """
    Wrapper around BTPlannerInterface to generate behavior trees from goal strings or goal sets,
    optionally write .btml, and execute the generated BT.

    Constructor kept minimal: only behavior_lib is required; cur_cond_set and key_objects optional.
    Planner is instantiated in the constructor with fixed sensible defaults.
    """
    def __init__(self,
                 env_name: str,
                 cur_cond_set: Optional[set] = None,
                 key_objects: Optional[List[str]] = None):
        self.env_name = env_name
        self.behavior_lib = ExecBehaviorLibrary(f"{AGENT_ENV_PATH}/{env_name}")
        print(f'behavior_lib: {self.behavior_lib}')
        self.cur_cond_set = cur_cond_set or set()
        self.key_objects = key_objects or []
        # fixed defaults (kept minimal and stable)
        self.priority_act_ls: List[str] = []
        self.key_predicates: List[str] = []
        self.selected_algorithm = "hbtp"
        self.mode = "small-objs"
        self.time_limit = 60
        self.heuristic_choice = 0
        self.output_just_best = True
        self.use_priority_act = []

        # instantiate planner here so it's ready for immediate use
        self.planner: BTPlannerInterface = BTPlannerInterface(
            self.behavior_lib,
            cur_cond_set=self.cur_cond_set,
            priority_act_ls=self.priority_act_ls,
            key_predicates=self.key_predicates,
            key_objects=self.key_objects,
            selected_algorithm=self.selected_algorithm,
            mode=self.mode,
            time_limit=self.time_limit,
            heuristic_choice=self.heuristic_choice,
            output_just_best=self.output_just_best,
            use_priority_act=self.use_priority_act
        )

        self.goal_set = None

    def generate(self, goal: Union[str, List[set]], btml_name: str = "tree") -> Any:
        """
        goal: either a goal string (e.g. 'A & B') or a pre-parsed goal_set (list/iterable of condition sets)
        Returns a BehaviorTree instance (and stores ptml/cost/expanded_num in self.last_*).
        """
        start_time = time.time()
        if isinstance(goal, str):
            goal_set = goal_transfer_str(goal)
        else:
            goal_set = goal
        self.goal_set = goal_set

        # planner already created in constructor
        self.planner.process(goal_set)

        ptml_string, cost, expanded_num = self.planner.post_process()
        planning_time_total = time.time() - start_time
        
        path = f"{btml_name}.btml"
        with open(path, "w", encoding="utf-8") as f:
            f.write(ptml_string)
        bt = BehaviorTree(path, self.behavior_lib)

        # store last post-process results for external access if needed
        # self.last_ptml = ptml_string
        # self.last_cost = cost
        # self.last_expanded_num = expanded_num
        error, state, act_num, current_cost, record_act_ls, ticks = self.execute(
            goal_set[0], self.cur_cond_set, verbose=False)
        
        print(f'\x1b[32mGoal:{goal_set[0]}\x1b[0m, \n'
              f'\x1b[31merror:\x1b[0m {error}, \n'
              f'\x1b[33mstate:\x1b[0m {state}, \n'
              f'\x1b[35mact_num:\x1b[0m {act_num}, \n'
              f'\x1b[36mcurrent_cost:\x1b[0m {current_cost}, \n'
              f'\x1b[35mrecord_act_ls:\x1b[0m {record_act_ls}, \n'
              f'\x1b[34mplanning_time_total:\x1b[0m {planning_time_total}, \n'
              f'\x1b[33mexpanded_num:\x1b[0m {expanded_num}, \n'
              f'\x1b[32mticks:\x1b[0m {ticks}')
        
        # If requested, write PTML to the specified file
        bt.draw(file_name=btml_name, png_only=True)
            
        return bt

    def execute(self, goal: set, state: set, verbose: bool = True):
        if not self.planner:
            raise RuntimeError("Planner not initialized. Call generate() first.")
        return self.planner.execute_bt(goal, state, verbose=verbose)
