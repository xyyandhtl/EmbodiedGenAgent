from EG_agent.environment.virtualhome._base.vh_action import VHAction

class Grab(VHAction):
    can_be_expanded = False
    num_args = 1
    valid_args = VHAction.Objects

    def __init__(self, *args):
        super().__init__(*args)
