from EG_agent.planning.btpg.behavior_tree.base_nodes import Action
from EG_agent.planning.btpg.behavior_tree import Status

class EmbodiedAction(Action):
    can_be_expanded = True
    num_args = 1

    # SAR/inspection domain categories
    # Navigation targets (areas/waypoints)
    LOCATIONS = {
        "StagingArea", "Corridor", "Intersection", "Stairwell",
        "Office", "Warehouse", "ControlRoom", "LoadingBay", 
        "Lobby", "ChargingStation", "Outdoor", "Indoor", "Wall"
    }
    # Points of interest the robot may inspect/capture/mark
    INSPECTION_POINTS = {
        "Doorway", "Window", "ElectricalPanel", "GasMeter", "Equipment",
        "StructuralCrack", "SmokeSource", "WaterLeak", "BlockedExit",
    }
    # Dynamic entities/incidents
    INCIDENTS = {
        "Blood", "Fire", "Gas", "Debris"
    }
    PERSONS = {
        "Victim", "Rescuer", "Visitor", "Staff"
    }

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
