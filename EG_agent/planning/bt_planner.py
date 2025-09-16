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


class BTAgent(object):
    env = None
    scene = None
    response_frequency = 1

    # new class attribute to hold a BTGenerator (set by deployer)
    generator: Optional[BTGenerator] = None

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