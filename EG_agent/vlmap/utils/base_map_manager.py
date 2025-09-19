from omegaconf import DictConfig

from EG_agent.vlmap.utils.tracker import Tracker
from EG_agent.vlmap.utils.visualizer import ReRunVisualizer

class BaseMapManager:
    def __init__(
        self,
        cfg: DictConfig,
        ) -> None:
        
        # config
        self.cfg = cfg
        
        # init state
        self.is_initialized = False
        
        # tracker init
        self.tracker = Tracker(self.cfg)
        
        # visualizer
        self.visualizer = ReRunVisualizer()
        self.prev_entities = set()
        