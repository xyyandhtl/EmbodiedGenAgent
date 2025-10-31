from dynaconf import Dynaconf

from EG_agent.vlmap.utils.tracker import Tracker
from EG_agent.vlmap.utils.visualizer import ReRunVisualizer
from EG_agent.vlmap.utils.types import Observation, GoalMode, ObjectClasses

class BaseMapManager:
    def __init__(
        self,
        cfg: Dynaconf,
        ) -> None:
        
        # config
        self.cfg = cfg
        
        # init state
        self.is_initialized = False
        
        # tracker init
        self.tracker = Tracker(self.cfg)
        
        self.prev_entities = set()

        # Object classes
        classes_path = cfg.yolo.given_classes_path
        self.obj_classes = ObjectClasses(
            classes_file_path=classes_path,
            bg_classes=self.cfg.yolo.bg_classes,
            skip_bg=self.cfg.yolo.skip_bg)
        
        # visualizer
        if self.cfg.use_rerun:
            self.visualizer = ReRunVisualizer()