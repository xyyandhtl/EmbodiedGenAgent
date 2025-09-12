import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from enum import Enum
from typing import Union
import numpy as np

from EG_agent.system.path import AGENT_SYSTEM_PATH
from EG_agent.reasoning.logic_goal import LogicGoalGenerator
from EG_agent.planning.bt_planning import BTGenerator


class RobotAgentSystem:

    def __init__(
        self,
    ) -> None:
        pass

    @property
    def finished(self) -> bool:
        pass

    @property
    def status(self) -> bool:
        pass

    def step(self, img: Union[str, Path, np.ndarray]):
        pass

