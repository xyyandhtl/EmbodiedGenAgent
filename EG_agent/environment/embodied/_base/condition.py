from EG_agent.planning.btpg.behavior_tree.base_nodes import Condition
from EG_agent.planning.btpg.behavior_tree import Status

from EG_agent.system.envs.isaacsim_env import IsaacsimEnv

class EmbodiedCondition(Condition):
    can_be_expanded = True
    num_args = 1
    agent_env: IsaacsimEnv = None  # type: ignore

    def update(self) -> Status:
        if self.name in self.agent_env.condition_set:
            return Status.SUCCESS
        else:
            return Status.FAILURE
