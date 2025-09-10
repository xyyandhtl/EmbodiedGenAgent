from btpg.behavior_tree.base_nodes import Action
from btpg.behavior_tree import Status

class EmbodiedAction(Action):
    can_be_expanded = True
    num_args = 1

    # SAR/inspection domain categories
    # Navigation targets (areas/waypoints)
    LOCATIONS = {
        "entry", "staging_area", "corridor", "intersection", "stairwell",
        "room_a", "room_b", "warehouse", "control_room", "loading_bay", "outdoor",
        "lobby", "exit", "charging_station"
    }
    # Points of interest the robot may inspect/capture/mark
    INSPECTION_POINTS = {
        "doorway", "window", "electrical_panel", "gas_meter",
        "structural_crack", "smoke_source", "water_leak", "blocked_exit",
        "anomaly", "equipment"
    }
    # Dynamic entities/incidents
    INCIDENTS = {
        "hazard", "gas_leak", "fire", "obstacle", "debris", "leak"
    }
    PERSONS = {"victim", "rescuer"}

    # Derived sets for convenience
    NAV_POINTS = LOCATIONS | INSPECTION_POINTS
    CAPTUREABLE = INSPECTION_POINTS | INCIDENTS | PERSONS
    MARKABLE = CAPTUREABLE
    REPORTABLE = CAPTUREABLE

    # Backward-compat: single roll-up set if needed elsewhere
    AllObject = NAV_POINTS | CAPTUREABLE

    @property
    def action_class_name(self):
        return self.__class__.__name__

    def change_condition_set(self):
        pass

    def update(self) -> Status:
        # 在这里执行具体的动作逻辑，比如移动、拍照等
  
        # if self.num_args==1:
        #     script = [f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1)']
        # else:
        #     script = [f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1) <{self.args[1].lower()}> (1)']
        # # self.env.run_script(script,verbose=True,camera_mode="PERSON_FROM_BACK") # FIRST_PERSON
        # print("script: ",script)

        self.change_condition_set()
        return Status.RUNNING
