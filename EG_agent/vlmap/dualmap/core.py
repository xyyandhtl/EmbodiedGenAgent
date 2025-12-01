import os
import json
import threading
import queue
import time
from pathlib import Path
import logging
import psutil
import re
import yaml
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R
from dynaconf import Dynaconf, LazySettings

from EG_agent.vlmap.dualmap.types import DataInput, GoalMode
from EG_agent.vlmap.dualmap.object_detector import Detector
from EG_agent.vlmap.dualmap.local_map_manager import LocalMapManager
from EG_agent.vlmap.dualmap.global_map_manager import GlobalMapManager
from EG_agent.vlmap.dualmap.visualizer import ReRunVisualizer
from EG_agent.vlmap.dualmap.time_utils import (
    timing_context,
    print_timing_results,
    save_timing_results,
    get_map_memory_usage,
)
from EG_agent.vlmap.dualmap.navigation_helper import (
    remaining_path,
    remove_sharp_turns_3d,
)

def set_thread_priority():
    """Set current thread to higher priority if possible"""
    try:
        # Get current process
        p = psutil.Process()
        # Set to higher priority (on Unix systems)
        p.nice(-10)  # Higher priority (negative values)
    except (psutil.AccessDenied, AttributeError):
        # Access denied or platform doesn't support nice
        pass

# Set up the module-level logger
logger = logging.getLogger(__name__)


class Dualmap:
    def __init__(self, cfg: LazySettings):
        """
        Initialize Dualmap with configuration and essential components.
        """
        self.cfg = cfg
        dualmap_dir = str(Path(__file__).resolve().parent.parent)
        # self.cfg.output_path = f'{dualmap_dir}/{self.cfg.output_path}'
        # self.cfg.logging_config = f'{dualmap_dir}/{self.cfg.logging_config}'
        self.cfg.yolo.given_classes_path = f'{dualmap_dir}/{self.cfg.yolo.given_classes_path}/{self.cfg.dataset_name}.txt'
        self.cfg.yolo.model_path = f'{dualmap_dir}/{self.cfg.yolo.model_path}'
        self.cfg.sam.model_path = f'{dualmap_dir}/{self.cfg.sam.model_path}'
        self.cfg.fastsam.model_path = f'{dualmap_dir}/{self.cfg.fastsam.model_path}'
        self.cfg.clip.model_path = f'{dualmap_dir}/{self.cfg.clip.model_path}'
        self.cfg.preload_path = f'{self.cfg.output_path}/{self.cfg.dataset_name}/{self.cfg.preload_path}'

        self.cfg.map_save_path = f"{self.cfg.output_path}/{self.cfg.dataset_name}/{self.cfg.map_save_path}"
        self.cfg.detection_path = f"{self.cfg.output_path}/{self.cfg.dataset_name}/{self.cfg.detection_path}"

        # print config into console
        self.print_cfg()

        # --- 1. Initialization ---
        self.detector = Detector(cfg)
        self.local_map_manager = LocalMapManager(cfg)
        self.global_map_manager = GlobalMapManager(cfg)

        # Additional initialization for visualization
        if cfg.use_rerun:
            self.visualizer = ReRunVisualizer(cfg)
            self.visualizer.set_use_rerun(cfg.use_rerun)
            self.visualizer.init("refactor_mapping")
            self.visualizer.spawn()

        # Keyframe Selection
        self.keyframe_counter = 0
        self.last_keyframe_time = None
        self.last_keyframe_pose = None
        self.time_threshold = cfg.time_threshold
        self.pose_threshold = cfg.pose_threshold
        self.rotation_threshold = cfg.rotation_threshold

        # pose memory
        self.curr_pose: np.ndarray = None   # Keyframe pose
        self.prev_pose: np.ndarray = None
        self.goal_pose: list = []
        self.wait_count = 0

        # --- 2. Threads & Queues ---
        self.stop_thread = False  # Signal to stop threads

        # Mode for Getting the Goal
        self.goal_mode = GoalMode.NONE
        self.inquiry: str = ""
        self.inquiry_feat = None
        self.inquiry_found = set()
        self.found_obj_name: str = ""

        # Local planning flags & paths
        self.begin_local_planning = False
        self.action_path = []
        self.curr_global_path = []
        self.curr_local_path = []
        self.start_action_path = False

        # debug param: path counter
        self.path_counter = 0

        # Queue provided by DualmapInterface for raw frames
        self.last_layout_kf_idx = -1
        self.last_detector_kf_idx = -1
        self.input_queue = deque(maxlen=1)
        self.detection_results_queue = queue.Queue(maxsize=10)

        self.detector_thread = None
        self.mapping_thread = None
        self.path_planning_thread = None
        self.layout_thread = None
        self.vis_map_thread = None

        self.layout_event = threading.Event()
        self.goal_event = threading.Event()
        self.vis_map_event = threading.Event()
        self.detector_event = threading.Event()
        self.mapper_event = threading.Event()

    # ===============================================
    # Basic Utilities
    # ===============================================
    def print_cfg(self):
        log_file_path = ""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_path = handler.baseFilename

        # Organize configuration items to be printed into a list
        cfg_items = [
            ("Log Path", log_file_path),
            ("Output Dir", self.cfg.output_path),
            ("Map Save Dir", self.cfg.map_save_path),
            ("Class List Path", self.cfg.yolo.given_classes_path),
            ("Use FastSAM for OV?", self.cfg.use_fastsam),
            # ("Running Concrete Map Only?", self.cfg.run_local_mapping_only),
            # ("Save Concrete Map?", self.cfg.save_local_map),
            # ("Save Global Map?", self.cfg.save_global_map),
            # ("Use Preload Global Map?", self.cfg.preload_global_map),
            ("Use Rerun for Visualization?", self.cfg.use_rerun),
            ("Camera Intrinsics", str(self.cfg.intrinsics)),
            ("Cmaera Extrinsics", str(self.cfg.extrinsics)),
            ("Range Sensor", str(self.cfg.range_sensor)),
        ]

        # Define separator line length
        line_length = 60
        print("=" * line_length)
        for key, value in cfg_items:
            print(f"{key:<30} : {value}")
        print("=" * line_length)
        logger.info(f"Config {cfg_items}")

    def save_map(self, map_path=None):
        """
        Save the local and global maps to disk.
        """
        if map_path is not None:
            self.cfg.map_save_path = map_path

        # if self.cfg.save_local_map:
        # self.local_map_manager.save_map()

        # if self.cfg.save_global_map:
        self.global_map_manager.save_map()
        
        # if self.cfg.save_layout:
        self.detector.save_layout()
        layout_pcd = self.detector.get_layout_pointcloud()
        self.global_map_manager.set_layout_info(layout_pcd, force_full_update=True)
        # self.global_map_manager.save_wall_pcd()

    def load_map(self, map_path=None):
        """
        Load the local and global maps from disk.
        """
        if map_path is not None:
            self.cfg.preload_path = map_path

        # check if need to preload the global map
        # if self.cfg.preload_global_map:
        logger.info("[Core][Init] Preloading global map...")
        self.global_map_manager.load_map()

        logger.info("[Core][Init] Preloading layout...")
        self.detector.load_layout()
        # self.global_map_manager.load_wall()
        # TODO: the dualmap save/load logic should be reorgainized
        layout_pcd = self.detector.get_layout_pointcloud()
        self.global_map_manager.set_layout_info(layout_pcd, force_full_update=True)

    def get_keyframe_idx(self):
        return self.keyframe_counter

    def check_keyframe(self, time_stamp, curr_pose):
        """
        Check if the current frame should be selected as a keyframe based on
        time interval, pose difference (translation), and rotation difference.
        """
        is_keyframe = False
        # Translation check
        if self.last_keyframe_pose is not None:
            translation_diff = np.linalg.norm(
                curr_pose[:3, 3] - self.last_keyframe_pose[:3, 3]
            )  # Translation difference
            if translation_diff >= self.pose_threshold:  # 0.1 m
                self.last_keyframe_time = time_stamp
                self.last_keyframe_pose = curr_pose
                logger.debug(
                    "[Core][CheckKeyframe] New keyframe detected by translation"
                )
                is_keyframe = True

            # Rotation check
            curr_rotation = R.from_matrix(curr_pose[:3, :3])
            last_rotation = R.from_matrix(self.last_keyframe_pose[:3, :3])
            rotation_diff = curr_rotation.inv() * last_rotation
            angle_diff = rotation_diff.magnitude() * (180 / np.pi)

            if angle_diff >= self.rotation_threshold:  # > 3 degrees
                self.last_keyframe_time = time_stamp
                self.last_keyframe_pose = curr_pose
                logger.debug("[Core][CheckKeyframe] New keyframe detected by rotation")
                is_keyframe = True

        # Time check
        if (
            self.last_keyframe_time is None
            or abs(time_stamp - self.last_keyframe_time) >= self.time_threshold
        ):  # > 0.5 s
            self.last_keyframe_time = time_stamp
            self.last_keyframe_pose = curr_pose
            logger.debug("[Core][CheckKeyframe] New keyframe detected by time")
            is_keyframe = True

        if is_keyframe:
            self.keyframe_counter += 1
            logger.debug(
                f"[Core][CheckKeyframe] Current frame is keyframe: {self.keyframe_counter}"
            )
            return True
        else:
            # logger.info("Not a new keyframe, abandon")
            return False
    
    # ===============================================
    # Threading Control
    # ===============================================
    def start_threading(self):
        self.detector_thread = threading.Thread(
            target=self.run_detector_thread, daemon=True
        )
        self.detector_thread.start()
        logger.info("[Core] Detector thread started.")

        self.mapping_thread = threading.Thread(
            target=self.run_mapping_thread, daemon=True
        )
        self.mapping_thread.start()
        logger.info("[Core] Mapping thread started.")

        self.path_planning_thread = threading.Thread(
            target=self.run_path_planning_thread, daemon=True
        )
        self.path_planning_thread.start()
        logger.info("[Core] Path planning thread started.")

        self.layout_thread = threading.Thread(
            target=self.run_layout_thread, daemon=True
        )
        self.layout_thread.start()
        logger.info("[Core] Layout thread started.")

        self.vis_map_thread = threading.Thread(
            target=self.run_vis_map_thread, daemon=True
        )
        self.vis_map_thread.start()
        logger.info("[Core] Semantic map thread started.")

    def stop_threading(self):
        self.stop_thread = True
        self.layout_event.set()
        self.goal_event.set()
        self.vis_map_event.set()
        # Join detector thread
        if self.detector_thread and self.detector_thread.is_alive():
            self.detector_thread.join()
        if self.mapping_thread and self.mapping_thread.is_alive():
            self.mapping_thread.join()
        if self.path_planning_thread and self.path_planning_thread.is_alive():
            self.path_planning_thread.join()
        if self.layout_thread and self.layout_thread.is_alive():
            self.layout_thread.join()
        if self.vis_map_thread and self.vis_map_thread.is_alive():
            self.vis_map_thread.join()
        logger.info("[Core] Stopped detector, mapping and path planning thread.")
        logger.info("[Core] Stopped layout thread.")

    def end_process(self):
        """
        The process of ending the sequnce.
        """
        end_frame_id = self.curr_frame_id
        self.stop_threading()

        # end duration
        end_range: int = self.cfg.active_window_size + self.cfg.max_pending_count + 1
        for i in range(end_range):
            # Set timestamp for visualizer
            logger.info("[Core][EndProcess] End Counter: %d", end_frame_id + i + 1)
            if self.cfg.use_rerun:
                self.visualizer.set_time_sequence("frame", end_frame_id + i + 1)

            # local end_process
            # set fake timestamp
            self.local_map_manager.set_curr_idx(end_frame_id + i + 1)
            self.local_map_manager.end_process()
            local_map_obj_num = len(self.local_map_manager.local_map)
            logger.info("[Core][EndProcess] Local Objects num: %d", local_map_obj_num)

            # global process
            global_obs_list = self.local_map_manager.get_global_observations()
            self.local_map_manager.clear_global_observations()
            self.global_map_manager.process_observations(global_obs_list)
            global_map_obj_num = len(self.global_map_manager.global_map)
            logger.info("[Core][EndProcess] Global Objects num: %d", global_map_obj_num)

            if local_map_obj_num == 0:
                logger.warning(
                    "[EndProcess] End Processing End. to: %d", end_frame_id + i + 1
                )
                break

        with timing_context("Merging", self):
            if self.cfg.merge_local_map:
                self.local_map_manager.merge_local_map()
                if self.cfg.use_rerun:
                    self.visualizer.set_time_sequence("frame", end_range + 1)
                logger.info("[Core][EndProcess] Local Map Merged")

        # save the local mapping results
        # self.save_map()

        # Dualmap process timing
        if hasattr(self, "timing_results"):
            print_timing_results("Dualmap", self.timing_results)
            system_time_path = f"{self.cfg.map_save_path}/../system_time.csv"
            save_timing_results(self.timing_results, system_time_path)

        # Detector process timing
        if hasattr(self.detector, "timing_results"):
            print_timing_results("Detector", self.detector.timing_results)
            detector_time_path = f"{self.cfg.map_save_path}/../detector_time.csv"
            save_timing_results(self.detector.timing_results, detector_time_path)

    def run_layout_thread(self):
        """
        Independent thread for layout processing.
        """
        while not self.stop_thread:
            self.layout_event.wait(timeout=2.0)
            self.layout_event.clear()
            if self.stop_thread:
                break           
            if len(self.input_queue) == 0:
                continue
            
            data_input: DataInput = self.input_queue[-1]
            if data_input.idx == self.last_layout_kf_idx:
                continue
            self.last_layout_kf_idx = data_input.idx

            with timing_context("Update layout", self):
                self.detector.run_layout_thread()

        logger.info("[Core] Layout thread exiting.")

    def run_detector_thread(self):
        """
        Independent thread:
          - Pull the latest frame from DualmapInterface.synced_data_queue
          - Update realtime pose and traj
          - Check keyframe
          - Run the full detector pipeline for keyframes
          - Push (curr_obs_list, curr_frame_id) to detection_results_queue
        """
        # set_thread_priority()
        while not self.stop_thread:
            self.detector_event.wait(timeout=0.1)
            self.detector_event.clear()
            if len(self.input_queue) == 0:
                continue

            data_input: DataInput = self.input_queue[-1]
            if data_input.idx == self.last_detector_kf_idx:
                continue
            self.last_detector_kf_idx = data_input.idx

            # Get current frame id
            self.curr_frame_id = data_input.idx
            logger.debug(f"[Core][DetectorThread] Processing frame {self.curr_frame_id}")

            # Get current pose（被判断作为关键帧的 位姿）
            self.curr_pose = data_input.pose

            # Detection process (full pipeline)
            start_time = time.time()
            with timing_context("Observation Generation", self):
                self.detector.set_data_input(data_input)

                with timing_context("Process Detection", self):
                    if self.cfg.run_detection:
                        self.detector.process_detections()
                        with timing_context("Save Detection", self):
                            if self.cfg.save_detection:
                                self.detector.save_detection_results()
                    else:
                        self.detector.load_detection_results()

                with timing_context("Observation Formatting", self):
                    self.detector.calculate_observations()

                curr_obs_list = self.detector.get_curr_observations()

                self.detector.update_state()

                # Push to mapping queue
                try:
                    self.detection_results_queue.put_nowait((curr_obs_list, self.curr_frame_id))
                    # self.mapper_event.set()
                except queue.Full:
                    logger.debug(
                        f"[Core] Mapping queue is full, skipping frame {self.curr_frame_id}."
                    )

            end_time = time.time()
            logger.debug(f"[Core][DetectorThread] Processed frame {self.curr_frame_id} "
                         f"in {end_time - start_time:.2f} seconds")

            # Set time sequence for visualizer
            if self.cfg.use_rerun:
                self.visualizer.set_time_sequence("frame", self.curr_frame_id)
                # Set current camera information
                self.visualizer.set_camera_info(data_input.intrinsics, data_input.pose)
                self.visualizer.set_image(data_input.color)

                elapsed_time = end_time - start_time
                self.detector.visualize_time(elapsed_time)

                # TODO: psutil seems not that correct
                # mem_usage = self.get_total_memory_by_keyword()
                # self.detector.visualize_memory(mem_usage) 

        logger.info("[Core] Detector thread exiting.")

    def run_mapping_thread(self):
        """
        Independent thread: Monitor detection results and process local mapping and global mapping.
        """
        while not self.stop_thread:
            self.mapper_event.wait(timeout=0.3)
            self.mapper_event.clear()
            try:
                # Get detection results from queue
                # logger.info(f"Queue length: {len(self.detection_results_queue)}")
                curr_obs_list, curr_frame_id = self.detection_results_queue.get(
                    timeout=1
                )  # Timeout exception
                logger.debug(
                    f"[Core][MappingThread] Received data for frame {curr_frame_id}, Queue size {self.detection_results_queue.qsize()}"
                )
            except queue.Empty:
                continue

            # Set time sequence for visualizer
            if self.cfg.use_rerun:
                self.visualizer.set_time_sequence("frame", self.curr_frame_id)
                self.detector.visualize_detection()
                
            self.detector.update_data()

            # Local Mapping
            with timing_context("Local Mapping", self):
                self.local_map_manager.set_curr_idx(curr_frame_id)
                self.local_map_manager.process_observations(curr_obs_list)

            # Get global observations
            with timing_context("Global Mapping", self):
                global_obs_list = self.local_map_manager.get_global_observations()
                self.local_map_manager.clear_global_observations()

                # Global Mapping
                self.global_map_manager.process_observations(global_obs_list)

                # if len(global_obs_list) > 0:
                self.global_map_manager.mark_semantic_map_dirty()

        logger.info("[Core] Mapping thread exiting.")

    def run_path_planning_thread(self):
        """
        Independent thread: path planning with low-frequency execution and switching.
        This thread covers the "find" and "walk" action.
        Currently, after goal_mode is set first time, path_plan is always running
        """
        loop_interval: float = self.cfg.path_planning_interval

        while not self.stop_thread:
            logger.debug(f"[PathPlanningThread] Loop begins.")
            self.goal_event.wait(timeout=loop_interval)
            self.goal_event.clear()  # 重置为未触发状态
            if self.stop_thread:
                break
            
            # 每次此低频loop先更新layout_map
            with timing_context("Occupancy Mapping", self):
                if not self.global_map_manager.layout_initialized:
                    layout_pcd = self.detector.get_layout_pointcloud()
                    self.global_map_manager.set_layout_info(layout_pcd)
                else:
                    update_radius = self.cfg.occ_update_radius
                    layout_pcd_partial = self.detector.get_partial_layout_pcd(
                        self.global_map_manager._curr_pose, update_radius)
                    self.global_map_manager.set_layout_info(
                        layout_pcd_partial, update_radius=update_radius)
                self.global_map_manager.process_binary_map()

            # 开始模式判断和路径规划
            if self.goal_pose:
                if self.goal_mode == GoalMode.NONE:
                    self.goal_mode = GoalMode.POSE
            elif self.inquiry == "explore":
                self.goal_pose = self.query_object("explore")
                if self.goal_pose:
                    self.goal_mode = GoalMode.RANDOM
                else:
                    self.goal_mode = GoalMode.NONE
            elif self.inquiry:
                self.goal_pose = self.query_object(self.inquiry)
                if self.goal_pose:
                    self.goal_mode = GoalMode.POSE
                    self.inquiry_found.add(self.inquiry)
                else:
                    # when inquiry not found, try explore
                    self.goal_pose = self.query_object("explore")
                    if self.goal_pose:
                        self.goal_mode = GoalMode.RANDOM
                    else:
                        self.goal_mode = GoalMode.NONE
            else:
                self.goal_mode = GoalMode.NONE
            
            logger.debug(f"[PathPlanningThread] inquiry {self.inquiry}, "
                         f"goal_pose {self.goal_pose}, "
                         f"goal_mode {self.goal_mode}")
            if self.goal_mode != GoalMode.NONE:     # and not self.path_exists()
                logger.info(f"[PathPlanningThread] calculating global_path with "
                            f"goal_mode {self.goal_mode}, "
                            f"goal_pose {self.goal_pose}, "
                            f"inquiry_name {self.inquiry}")
                self.compute_global_path()

        logger.info("[Core] Path_planning thread exiting.")

    def run_vis_map_thread(self):
        """
        Independent thread: semantic map with low-frequency execution and switching.
        """
        while not self.stop_thread:
            self.vis_map_event.wait(timeout=2.0)
            self.vis_map_event.clear()  # 重置为未触发状态
            if self.stop_thread:
                break

            self.global_map_manager.background_map_update()

        logger.info("[Core] Layout thread exiting.")

    # ===============================================
    # Query and Navigation API
    # ===============================================
    def reset_query_and_navigation(self):
        """重置 Find 涉及的导航状态，包括清除索引目标和已索引到的目标位置。"""
        self.inquiry = ""
        self.inquiry_feat = None
        
        self.reset_goal_position()
        self.global_map_manager.mark_semantic_map_dirty()

    def reset_goal_position(self):
        """仅重置目标位置，保留查询内容不变。"""
        self.goal_mode = GoalMode.NONE
        self.goal_pose = []
        
        self.action_path = []
        self.curr_global_path = []
        self.curr_local_path = []

    def path_exists(self):
        """检查当前是否存在有效的全局路径。"""
        return len(self.curr_global_path) > 0

    def convert_inquiry_to_feat(self, inquiry_sentence: str):
        text_query_tokenized = self.detector.clip_tokenizer(inquiry_sentence).to("cuda")
        text_query_ft = self.detector.clip_model.encode_text(text_query_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        return text_query_ft
    
    def query_object(self, query: str):
        """
        根据查询字符串在全局地图中寻找目标位置。
        支持两种查询模式：
        1. "explore"：随机采样一个目标位置用于探索。
        2. "ObjectName" 或 "RobotNear(ObjectName)"：根据物体名称查询其位置。
        """
        if query == "explore":
            logger.info("[Core][Query] set explore mode")
            if self.global_map_manager.path_planner is None:
                planner_ok = self.global_map_manager.update_path_planner()
                if not planner_ok:
                    return []
            position = self.global_map_manager.path_planner.sample_random_world_goal()
            return position
        
        # 1. 从查询（如："desk"/"RobotNear(ControlRoom)"）中提取物体名称
        match = re.search(r'\((.*?)\)', query)
        if match:
            object_name = match.group(1)
        else:
            object_name = query  # 如果格式不匹配，则假定整个查询都是对象名

        logger.info(f"[VLMapNav] [query_object] Received query '{query}', searching for object '{object_name}'.")

        # 2. 检查全局地图是否存在
        if not self.global_map_manager.has_global_map():
            logger.debug("[VLMapNav] [query_object] Global map is empty. Cannot find object.")
            return []

        # 3. 将对象名称转换为 CLIP 特征向量，并传给 global_map_manager 为在 INQUIRY 模式下获取目标点坐标
        self.inquiry_feat = self.convert_inquiry_to_feat(object_name)
        self.global_map_manager.inquiry = self.inquiry_feat

        # 4. 调用 GlobalMapManager 的 find_best_candidate_with_inquiry 来寻找最佳匹配
        #    这将返回 GlobalObject 实例和分数
        best_candidate, best_similarity = self.global_map_manager.find_best_candidate_with_inquiry()
        
        # 5. 处理结果
        if best_candidate is not None:
            # 提取物体边界框的中心点作为其位置
            self.found_obj_name = self.global_map_manager.obj_classes.get_classes_arr()[best_candidate.class_id]
            logger.info(f"[VLMapNav] [query_object] Found best match '{self.found_obj_name}' "
                        f"for query '{object_name}' with score {best_similarity:.4f}")
            if best_similarity > 0.5:
                position = best_candidate.bbox_2d.get_center().tolist()
                return position
        
        logger.warning(f"[VLMapNav] [query_object] No object found for query '{object_name}'")
        return []

    def compute_global_path(self):
        """
        计算全局路径，存入 self.curr_global_path
        """
        if not self.global_map_manager.has_global_map():
            logger.warning("[Core] No global map available for path planning.")
            self.reset_goal_position()
            return

        self.global_map_manager.has_action_path = False

        start = time.time()
        # calculate the path based on current global map
        # Get 3D path point in world coordinate
        # 计算 全局路径
        self.curr_global_path = self.global_map_manager.calculate_global_path(
            goal_mode=self.goal_mode, goal_position=self.goal_pose
        )
        if not self.curr_global_path:
            self.reset_goal_position()
            return

        self.global_map_manager.update_pose_path(nav_path=self.curr_global_path)

        # Clear the local mapping results
        self.curr_local_path = None
        self.begin_local_planning = True

        logger.info(f"[Core] Global path planning time: {time.time() - start:.4f} seconds.")

        if self.cfg.save_all_path:
            # Create a unique file name using the counter
            self.path_counter += 1

            save_dir = f"{self.cfg.output_path}/path/global_path"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # build new file name with counter
            file_name = f"{self.path_counter}.json"
            save_path = os.path.join(save_dir, file_name)

            # Write JSON file
            with open(save_path, "w") as f:
                json.dump(self.curr_global_path, f, indent=4)

            logger.warning(f"[Core] Global path saved to {save_path}")

    def compute_local_path(self):
        """
        计算局部路径，存入 self.curr_local_path
        """
        if not self.begin_local_planning:
            logger.warning("[Core] Local Planning not enabled because no global path available!")
            return None
        
        if not self.local_map_manager.has_local_map():
            logger.warning("[Core] No local map available for path planning.")
            return None

        # set the global inquiry sentence to global map manager
        self.local_map_manager.inquiry = self.inquiry_feat

        # TEMP: Now explicitly pass the goal to local map to establish the workflow
        global_path = self.curr_global_path

        start = global_path[-1]
        x, y, z = start  # Unpack coordinates
        start_pose = np.eye(4)  # Initialize a 4x4 identity matrix

        # Set translation part
        start_pose[0, 3] = x  # X direction translation
        start_pose[1, 3] = y  # Y direction translation
        start_pose[2, 3] = z  # Z direction translation

        # for CLick mode, pass the click goal to local from global (not use now, because the nav_graph is deleted)
        # if self.global_map_manager.nav_graph.snapped_goal:
        #     click_goal = self.global_map_manager.nav_graph.snapped_goal
        #     self.local_map_manager.set_click_goal(click_goal)

        # for Inquiry mode, pass the inquiry goal bbox to local from global
        if self.global_map_manager.global_candidate_bbox is not None:
            goal_bbox = self.global_map_manager.global_candidate_bbox
            goal_score = self.global_map_manager.global_candidate_score
            self.local_map_manager.set_global_bbox(goal_bbox)
            self.local_map_manager.set_global_score(goal_score)

            global_map = self.global_map_manager.global_map
            self.local_map_manager.set_global_map(global_map)

        # calculate the path based on current global map
        self.curr_local_path = self.local_map_manager.calculate_local_path(
            start_pose, goal_mode=GoalMode.POSE, resolution=self.cfg.resolution, goal_position=self.goal_pose
        )

        if self.curr_local_path is not None:
            logger.info("[Core] Local Navigation has finished!")

            self.global_map_manager.global_candidate_score = 0.0
            self.global_map_manager.best_candidate_name = None
            self.global_map_manager.ignore_global_obj_list = []
            self.wait_count = 0

            self.begin_local_planning = False
            self.start_action_path = True

            if self.cfg.save_all_path:
                # Ensure the path directory exists
                save_dir = f"{self.cfg.output_path}/path/local_path"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # build new file name with counter
                file_name = f"{self.path_counter}.json"
                save_path = os.path.join(save_dir, file_name)

                # Write JSON file
                with open(save_path, "w") as f:
                    json.dump(self.curr_local_path, f, indent=4)

                logger.warning(f"[Core] Local path saved to {save_path}")

        self.prev_pose = self.curr_pose

    def compute_action_path(self):
        """
        计算动作路径，存入 self.action_path
        """
        if self.curr_global_path is None:
            logger.info("[Core][ActionPath] No Global Path! Action Path not available!")
            self.action_path = self.global_map_manager.action_path = []
            return

        # Global path is definitely available here
        if self.curr_local_path is None:
            # then just use global path as action path
            logger.info("[Core][ActionPath] No Local Path! Action Path Now Using Global Path!")
            self.action_path = self.global_map_manager.action_path = self.curr_global_path
            return

        # Triggered by local path computing
        if self.start_action_path:
            logger.info("[Core][ActionPath] Start Action Path Calculation!")

            # delete the start point
            self.action_path = self.curr_global_path + self.curr_local_path[1:]

            self.action_path = remaining_path(self.action_path, self.curr_pose)

            # remove sharp returns
            if self.cfg.use_remove_sharp_turns:
                self.action_path = remove_sharp_turns_3d(self.action_path)

            if self.cfg.save_all_path:
                # Ensure the path directory exists
                save_dir = f"{self.cfg.output_path}/path/action_path"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # build new file name with counter
                file_name = f"{self.path_counter}.json"
                save_path = os.path.join(save_dir, file_name)

                # Write JSON file
                with open(save_path, "w") as f:
                    json.dump(self.action_path, f, indent=4)

                logger.warning(f"[Core] Action path saved to {save_path}")

            self.global_map_manager.action_path = self.action_path

            self.global_map_manager.has_action_path = True

            self.start_action_path = False

    def compute_next_waypoint(self, from_path="global"):
        if from_path == "global":
            cur_path = self.curr_global_path
        elif from_path == "local":
            cur_path = self.curr_local_path
        elif from_path == "action":
            cur_path = self.action_path
        if not cur_path:
            logger.debug("[Core] No path available for next waypoint computation.")
            return None
        remain_path = remaining_path(cur_path, self.curr_pose)
        if not remain_path:
            logger.debug("[Core] No remaining path after current pose.")
            return None
        return remain_path[min(1, len(remain_path)-1)]
