import time
import sys
sys.path.append("./")
sys.path.append("EG_agent/planning")

from EG_agent.planning.btpg.algos.llm_client.tools import goal_transfer_str
from EG_agent.planning.btpg.algos.bt_planning.bt_planner_interface import BTPlannerInterface
from EG_agent.planning.btpg.behavior_tree.behavior_libs.ExecBehaviorLibrary import ExecBehaviorLibrary
from EG_agent.system.path import AGENT_ENV_PATH

max_goal_num=5
diffcult_type= "mix" #"single"  #"mix" "multi"
scene = "VH" # RH RHS RW

headless = False

test_env = "embodied"  # "virtualhome"  # "embodied"

if test_env == "embodied":
    cur_cond_set = set()
    from EG_agent.environment.embodied._base.gen_action import EmbodiedAction
    goal_str = 'RobotNear_doorway & IsCaptured_victim'
    key_objects = ["doorway", "victim"]
    # goal_str = 'IsReported_doorway & IsMarked_obstacle & IsCaptured_victim'
    # key_objects = ["doorway", "obstacle", "victim"]
    behavior_lib = ExecBehaviorLibrary(f"{AGENT_ENV_PATH}/embodied")
elif test_env == "virtualhome":
    cur_cond_set = {'IsSwitchedOff(dishwasher)', 'IsSwitchedOff(tablelamp)', 'IsClose(garbagecan)', 
                    'IsClose(cabinet)', 'IsStanding(self)', 'IsSwitchedOff(toaster)', 'IsClose(dishwasher)', 
                    'IsSwitchedOff(lightswitch)', 'IsRightHandEmpty(self)', 'IsLeftHandEmpty(self)', 
                    'IsSwitchedOff(tv)', 'IsClose(kitchencabinet)', 'IsSwitchedOff(microwave)', 
                    'IsSwitchedOff(faucet)', 'IsClose(stove)', 'IsSwitchedOff(coffeemaker)', 
                    'IsSwitchedOff(computer)', 'IsClose(microwave)', 'IsClose(fridge)', 'IsSwitchedOff(stove)'}
    # goal_str = 'IsSwitchedOn_tv & IsSwitchedOn_computer & IsIn_cutlets_microwave & IsClose_microwave '
    # key_objects = ["tv","computer","cutlets","microwave"]
    goal_str = 'IsSwitchedOn_tv & IsClose_microwave '
    key_objects = ["tv","microwave"]
    behavior_lib = ExecBehaviorLibrary(f"{AGENT_ENV_PATH}/virtualhome")
else:
    raise NotImplementedError

print(f'behavior_lib: {behavior_lib}')

print("goal_str:", goal_str)
algo = BTPlannerInterface(behavior_lib, cur_cond_set=cur_cond_set,
                      priority_act_ls=[], key_predicates=[],
                      key_objects=key_objects,
                      selected_algorithm="hbtp", mode="small-objs",
                      time_limit=60,
                      heuristic_choice=0,output_just_best=True,use_priority_act=[]) #

goal_set = goal_transfer_str(goal_str)

start_time = time.time()
algo.process(goal_set)
end_time = time.time()
planning_time_total = end_time - start_time

time_limit_exceeded = algo.algo.time_limit_exceeded

ptml_string, cost, expanded_num = algo.post_process()
error, state, act_num, current_cost, record_act_ls,ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
      "\x1b[31mERROR\x1b[0m" if error else "",
      "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)


# visualization
file_name = "tree"
file_path = f'./{file_name}.btml'
with open(file_path, 'w') as file:
    file.write(ptml_string)
# read and execute
from EG_agent.planning.btpg import BehaviorTree
bt = BehaviorTree(file_name + ".btml", behavior_lib)
# bt.print()
bt.draw(png_only=True)

# Simulate execution in a simulated scenario.
# goal_str = 'IsIn_milk_fridge & IsClose_fridge'
# goal_str = 'IsOn_bananas_kitchentable'
goal = goal_transfer_str(goal_str)[0]
print(f"goal: {goal}") # {'IsIn(milk,fridge)', 'IsClose(fridge)'}


# if scene in ["VH","RW"]:
#     if not headless:
#         env.agents[0].bind_bt(bt)
#         env.reset()
#         is_finished = False
#         while not is_finished:
#             is_finished = env.step()
#             if goal <= env.agents[0].condition_set:
#                 is_finished=True
#         env.close()
# else:
#     error, state, act_num, current_cost, record_act_ls,ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=True)

# algo.algo.bt.draw()
