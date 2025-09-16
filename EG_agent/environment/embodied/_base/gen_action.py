from EG_agent.planning.btpg.behavior_tree.base_nodes import Action
from EG_agent.planning.btpg.behavior_tree import Status
# import shared object sets
from EG_agent.prompts.object_sets import *

class EmbodiedAction(Action):
    can_be_expanded = True
    num_args = 1

    # use shared sets from object_sets.py
    LOCATIONS = LOCATIONS
    INSPECTION_POINTS = INSPECTION_POINTS
    INCIDENTS = INCIDENTS
    PERSONS = PERSONS

    # Derived sets for convenience
    NAV_POINTS = NAV_POINTS
    CAPTUREABLE = CAPTUREABLE
    MARKABLE = MARKABLE
    REPORTABLE = REPORTABLE

    # Backward-compat: single roll-up set if needed elsewhere
    AllObject = AllObject

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
