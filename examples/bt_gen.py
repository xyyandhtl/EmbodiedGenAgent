from EG_agent.planning.bt_planner import BTGenerator

if __name__ == "__main__":
    test_env = "embodied"  # "virtualhome"  # "embodied"

    # if test_env == "embodied":
    #     cur_cond_set = set()
    #     goal_str = 'RobotNear_Doorway & IsCaptured_Victim'
    #     key_objects = ["StagingArea", "Corridor", "Intersection", "Stairwell",
    #                 "Office", "Warehouse", "ControlRoom", "LoadingBay", 
    #                 "Lobby", "ChargingStation", "Outdoor", "Indoor", "Wall",
    #                 "Doorway", "Window", "ElectricalPanel", "GasMeter", "Equipment",
    #                 "StructuralCrack", "SmokeSource", "WaterLeak", "BlockedExit",
    #                 "Blood", "Fire", "Gas", "Debris",
    #                 "Victim", "Rescuer", "Visitor", "Staff"]
    # elif test_env == "virtualhome":
    #     cur_cond_set = {'IsSwitchedOff(dishwasher)', 'IsSwitchedOff(tablelamp)', 'IsClose(garbagecan)', 
    #                     'IsClose(cabinet)', 'IsStanding(self)', 'IsSwitchedOff(toaster)', 'IsClose(dishwasher)', 
    #                     'IsSwitchedOff(lightswitch)', 'IsRightHandEmpty(self)', 'IsLeftHandEmpty(self)', 
    #                     'IsSwitchedOff(tv)', 'IsClose(kitchencabinet)', 'IsSwitchedOff(microwave)', 
    #                     'IsSwitchedOff(faucet)', 'IsClose(stove)', 'IsSwitchedOff(coffeemaker)', 
    #                     'IsSwitchedOff(computer)', 'IsClose(microwave)', 'IsClose(fridge)', 'IsSwitchedOff(stove)'}
    #     goal_str = 'IsSwitchedOn_tv & IsClose_microwave '
    #     key_objects = ["tv","microwave"]
    # else:
    #     raise NotImplementedError

    # print("goal_str:", goal_str)

    # instantiate wrapper with minimal args
    bt_gen = BTGenerator(
        env_name=test_env,
        cur_cond_set=set(), # set empty
        key_objects=[],     # set empty
    )

    bt_gen.set_key_objects(["TrafficSign", "Trafficbarrier", "Trafficlight"])
    bt_gen.set_goal("RobotNear_TrafficSign & IsCaptured_TrafficSign")
    # bt_gen.set_goal("RobotNear_Trafficbarrier & IsCaptured_Trafficbarrier & RobotNear_Trafficlight & IsMarked_Trafficlight")
    bt = bt_gen.generate(btml_name="tree")

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