from EG_agent.environment.base.gen_condition import VHCondition as GenCondition

class IsNear(GenCondition):
    can_be_expanded = True
    num_args = 2

