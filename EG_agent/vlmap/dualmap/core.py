import os
import json
import threading
import queue
import time
from pathlib import Path
import logging
import psutil
import cv2
import re

import yaml
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R
from dynaconf import Dynaconf, LazySettings

from EG_agent.vlmap.utils.types import DataInput, GoalMode
from EG_agent.vlmap.utils.object_detector import Detector
from EG_agent.vlmap.utils.local_map_manager import LocalMapManager
from EG_agent.vlmap.utils.global_map_manager import GlobalMapManager
from EG_agent.vlmap.utils.visualizer import ReRunVisualizer
from EG_agent.vlmap.utils.time_utils import (
    timing_context,
    print_timing_results,
    save_timing_results,
    get_map_memory_usage,
)
from EG_agent.vlmap.utils.navigation_helper import (
    remaining_path,
    remove_sharp_turns_3d,
)

# Optionally import pdb for debugging purposes
# import pdb

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
        self.cfg.yolo.given_classes_path = f'{dualmap_dir}/{self.cfg.yolo.given_classes_path}'
        self.cfg.yolo.model_path = f'{dualmap_dir}/{self.cfg.yolo.model_path}'
        self.cfg.sam.model_path = f'{dualmap_dir}/{self.cfg.sam.model_path}'
        self.cfg.fastsam.model_path = f'{dualmap_dir}/{self.cfg.fastsam.model_path}'
        self.cfg.config_file_path = f'{dualmap_dir}/{self.cfg.config_file_path}'
        # self.cfg.ros_stream_config_path = f'{dualmap_dir}/{self.cfg.ros_stream_config_path}'
        self.cfg.given_classes_id_color = f'{dualmap_dir}/{self.cfg.given_classes_id_color}'
        self.cfg.preload_path = f'{self.cfg.output_path}/{self.cfg.dataset_name}/{self.cfg.preload_path}'

        self.cfg.map_save_path = f"{self.cfg.output_path}/{self.cfg.dataset_name}/{self.cfg.map_save_path}"
        self.cfg.detection_path = f"{self.cfg.output_path}/{self.cfg.dataset_name}/{self.cfg.detection_path}"

        # print config into console
        self.print_cfg()

        # --- 1. Initialization ---
        self.visualizer = ReRunVisualizer(cfg)  # 用于通过Rerun库进行实时可视化（展示建图过程、机器人位姿、点云、检测框等信息）
        self.detector = Detector(cfg)  # 负责图像中的物体检测（YOLO）、分割（SAM/FastSAM）以及 特征提取（CLIP）
        self.local_map_manager = LocalMapManager(cfg)  # 局部地图管理器（短期记忆）：负责维护机器人当前位置周围一小块区域的、高精度的实时地图，用于精细环境感知与动态避障
        self.global_map_manager = GlobalMapManager(cfg)  # 全局地图管理器（长期记忆）：负责存储和管理长期的、大范围环境地图和物体信息，用于大范围全局路径规划

        # Additional initialization for visualization
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
        self.curr_pose: np.ndarray = None
        self.realtime_pose: np.ndarray = None
        self.prev_pose: np.ndarray = None
        self.goal_pose: np.ndarray = None
        self.wait_count = 0

        # self.load_map()

        # --- 2. Start the file monitoring thread ---
        self.stop_thread = False  # Signal to stop the thread

        # Mode for Getting the Goal
        self.get_goal_mode = GoalMode.POSE
        # The request seq for getting the goal
        self.inquiry = ""
        self.inquiry_feat = None

        # Start local planning
        self.begin_local_planning = False

        # Final path for agent to follow
        self.action_path = []
        self.curr_global_path = []
        self.curr_local_path = []
        self.start_action_path = False

        # debug param: path counter
        self.path_counter = 0

        # History of agent's actual traversed path
        self.traj_path = deque(maxlen=60)

        # Parallel for mapping thread
        if self.cfg.use_parallel:
            self.detection_results_queue = queue.Queue(
                maxsize=10
            )  # Limit queue size to avoid memory leaks
            # Initialize thread
            self.mapping_thread = threading.Thread(
                target=self.run_mapping_thread, daemon=True
            )
            self.mapping_thread.start()

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
            ("Camera Intrinsics", str(self.cfg.intrinsic)),
            # ("Cmaera Extrinsics": str(self.cfg.extrinsics)},
        ]

        # Define separator line length
        line_length = 60
        print("=" * line_length)
        for key, value in cfg_items:
            print(f"{key:<30} : {value}")
        print("=" * line_length)

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
        self.global_map_manager.set_layout_info(layout_pcd)

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

        # if self.cfg.preload_layout:
        logger.info("[Core][Init] Preloading layout...")
        self.detector.load_layout()
        self.global_map_manager.load_wall()
        # TODO: the dualmap save/load logic should be reorgainized
        layout_pcd = self.detector.get_layout_pointcloud()
        self.global_map_manager.set_layout_info(layout_pcd)

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
                logger.info(
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
                logger.info("[Core][CheckKeyframe] New keyframe detected by rotation")
                is_keyframe = True

        # Time check
        if (
            self.last_keyframe_time is None
            or abs(time_stamp - self.last_keyframe_time) >= self.time_threshold
        ):  # > 0.5 s
            self.last_keyframe_time = time_stamp
            self.last_keyframe_pose = curr_pose
            logger.info("[Core][CheckKeyframe] New keyframe detected by time")
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

    def get_total_memory_by_keyword(self, keyword="applications"):
        total_rss = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                cmdline = proc.info.get("cmdline")
                if isinstance(cmdline, list) and keyword in " ".join(cmdline):
                    total_rss += proc.info["memory_info"].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        return total_rss / 1024 / 1024  # MB

    def sequential_process(self, data_input: DataInput):
        """
        Process input data sequentially.
        Note: Sequential processing is not recommended for large datasets.
        It is primarily used for debugging and testing purposes.
        Also, Sequential processing mode does not support navigation.
        """

        # Get current frame id
        self.curr_frame_id = data_input.idx

        # Get current pose
        self.curr_pose = data_input.pose

        # Set time sequence for visualizer
        self.visualizer.set_time_sequence("frame", self.curr_frame_id)

        # Set current camera information & image
        self.visualizer.set_camera_info(data_input.intrinsics, data_input.pose)
        self.visualizer.set_image(data_input.color)

        # Detection process with timing
        with timing_context("Detection Process", self):
            self.detector.set_data_input(data_input)

            if self.cfg.run_detection:
                self.detector.process_detections()
                with timing_context("Save Detection", self):
                    if self.cfg.save_detection:
                        self.detector.save_detection_results()
            else:
                self.detector.load_detection_results()

            # Detection results -> objects observation
            with timing_context("Vis Detection", self):
                self.detector.calculate_observations()
                if self.cfg.use_rerun:
                    self.detector.visualize_detection()

        # Local Mapping with timing
        with timing_context("Local Mapping", self):
            curr_obs_list = self.detector.get_curr_observations()
            self.detector.update_state()
            self.detector.update_data()
            self.local_map_manager.set_curr_idx(self.curr_frame_id)
            self.local_map_manager.process_observations(curr_obs_list)

        # Get global observations
        with timing_context("Global Mapping", self):
            global_obs_list = self.local_map_manager.get_global_observations()
            self.local_map_manager.clear_global_observations()
            self.global_map_manager.process_observations(global_obs_list)


    def parallel_process(self, data_input: DataInput):
        """
        Process input data in parallel. 数据流入的主要入口
            1. 接收一 关键帧 (DataInput，包含RGB、depth、pose等）
            2. 调用 Detector 对图像进行处理，生成物体观测结果。为不阻塞主线程（数据接收线程），将检测和观测结果放入 (`detection_results_queue`) 队列
            3. 更新可视化信息
            4. 更新地图，计算导航路径（全局+局部）
        """
        # Get current frame id
        self.curr_frame_id = data_input.idx

        # Get current pose（被判断作为关键帧的 位姿）
        self.curr_pose = data_input.pose

        # Record the traj_path with realtime pos
        if self.realtime_pose is not None:
            self.traj_path.append(self.realtime_pose[:3, 3].copy())

        # --- 1. Detection process ---
        start_time = time.time()
        with timing_context("Observation Generation", self):
            # (1) 调用 Detector，将 当前帧 设为 curr_data
            # (2) 运行 数据处理线程：检测当前帧是否可作为 全局点云的关键帧，如果是，则计算 当前帧的点云，并 merge 到全局点云中，更新`迁移关键帧数据`
            self.detector.set_data_input(data_input)

            with timing_context("Process Detection", self):
                if self.cfg.run_detection:
                    # (3) 执行检测：
                    #   YOLO + Segmentation + FastSAM，并进行过滤，merge，得到 `filtered_detections`
                    #   Create Object Pointcloud（为每个实例分割生成 点云、颜色 + CLIP（使用 CLIP 提取 图像特征、类别文本特征），得到最终结果 `curr_results`
                    self.detector.process_detections()
                    with timing_context("Save Detection", self):
                        if self.cfg.save_detection:
                            self.detector.save_detection_results()
                else:
                    self.detector.load_detection_results()

            with timing_context("Observation Formatting", self):
                # (4) 将 语义检测结果 ==> 观测对象 `LocalObservation`，并存入 `curr_observations`
                # 每个观测对象包含：
                #   idx:帧ID     class_id:物体类别ID     mask:物体mask     xyxy:物体2D检测框坐标      conf:物体检测置信度
                #   clip_ft:物体CLIP加权特征      pcd:物体3D点云      bbox:物体点云BBOX   distance:物体到相机的距离
                #   is_low_mobility:是否为固定物体   masked_image    cropped_image:物体裁剪图像
                self.detector.calculate_observations()

            # (5) 获取 观测结果 `curr_observations`
            curr_obs_list = self.detector.get_curr_observations()

            self.detector.update_state()  # 检测、观测结果清空，待下一帧处理

            # (6) 将结果放入 `detection_results_queue` 队列，供 建图线程 处理
            try:
                self.detection_results_queue.put(
                    (curr_obs_list, self.curr_frame_id), timeout=1
                )
            except queue.Full:
                logger.warning(
                    f"[Core] Mapping queue is full, skipping frame {self.curr_frame_id}."
                )

        end_time = time.time()

        # Set time sequence for visualizer
        self.visualizer.set_time_sequence("frame", self.curr_frame_id)

        # Set current camera information
        self.visualizer.set_camera_info(data_input.intrinsics, data_input.pose)
        self.visualizer.set_image(data_input.color)

        if self.cfg.use_rerun:
            elapsed_time = end_time - start_time
            self.detector.visualize_time(elapsed_time)

            # TODO: psutil seems not that correct
            # mem_usage = self.get_total_memory_by_keyword()
            # self.detector.visualize_memory(mem_usage)

    def run_mapping_thread(self):
        """
        Independent thread: Monitor detection results and process local mapping and global mapping.
        建图线程（独立的后台线程）：
            1. 不断地从 detection_results_queue 队列中取出观测结果
            2. 将观测结果分别送入 LocalMapManager 和 GlobalMapManager，来更新局部和全局地图
        """
        while not self.stop_thread:
            try:
                # Get detection results from queue
                # logger.info(f"Queue length: {len(self.detection_results_queue)}")
                curr_obs_list, curr_frame_id = self.detection_results_queue.get(
                    timeout=1
                )  # Timeout exception
                logger.info(
                    f"[Core][MappingThread] Received data for frame {curr_frame_id}, Queue size {self.detection_results_queue.qsize()}"
                )

                # Set time stamp
                self.visualizer.set_time_sequence("frame", self.curr_frame_id)

                # Detection Visualization
                if self.cfg.use_rerun:
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

                    self.visualizer.mark_semantic_map_dirty(dirty=True)  # global_map 改变，需更新语义地图图像缓存

                # Get memory usage statistics of local and global maps
                # mem_stats = get_map_memory_usage(self.local_map_manager.local_map,
                #                                 self.global_map_manager.global_map)

                # logger.info(
                #     f"[Core][MappingThread] Memory Usage - Local Map: {mem_stats['local_map_mb']} MB, "
                #     f"Global Map: {mem_stats['global_map_mb']} MB"
                # )

            except queue.Empty:
                continue

    def end_process(self):
        """
        The process of ending the sequnce.
        """
        self.visualizer.shutdown()
        end_frame_id = self.curr_frame_id

        self.stop_threading()

        # end duration
        end_range: int = self.cfg.active_window_size + self.cfg.max_pending_count + 1

        for i in range(end_range):
            # Set timestamp for visualizer
            logger.info("[Core][EndProcess] End Counter: %d", end_frame_id + i + 1)
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
            detector_time_path = self.cfg.map_save_path + "/../detector_time.csv"
            save_timing_results(self.detector.timing_results, detector_time_path)

    def stop_threading(self):
        self.stop_thread = True

        if self.cfg.use_parallel:
            self.mapping_thread.join()

        logger.info("[Core] Stopped monitoring config file and mapping thread.")

    def convert_inquiry_to_feat(self, inquiry_sentence: str):
        text_query_tokenized = self.detector.clip_tokenizer(inquiry_sentence).to("cuda")
        text_query_ft = self.detector.clip_model.encode_text(text_query_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        return text_query_ft

    # ===============================================
    # Public API for navigation and querying
    # ===============================================
    def reset_query_and_navigation(self):
        """重置导航状态，包括清除全局地图查询和重置路径规划器。"""
        self.inquiry_feat = None
        self.action_path = None
        self.curr_global_path = None
        self.curr_local_path = None

    def query_object(self, query: str):
        # 1. 从查询（如："desk"/"RobotNear(ControlRoom)"）中提取物体名称
        match = re.search(r'\((.*?)\)', query)
        if match:
            object_name = match.group(1)
        else:
            object_name = query  # 如果格式不匹配，则假定整个查询都是对象名

        logger.info(f"[VLMapNav] [query_object] Received query '{query}', searching for object '{object_name}'.")

        # 2. 检查全局地图是否存在
        if not self.global_map_manager.has_global_map():
            logger.warning("[VLMapNav] [query_object] Global map is empty. Cannot find object.")
            return None

        # 3. 将对象名称转换为 CLIP 特征向量，并传给 global_map_manager 为在 INQUIRY 模式下获取目标点坐标
        self.inquiry_feat = self.convert_inquiry_to_feat(object_name)
        self.global_map_manager.inquiry = self.inquiry_feat

        # 4. 调用 GlobalMapManager 的 find_best_candidate_with_inquiry 来寻找最佳匹配
        #    这将返回 GlobalObject 实例和分数
        best_candidate, best_similarity = self.global_map_manager.find_best_candidate_with_inquiry()

        # 5. 处理结果
        if best_candidate is not None:
            # 提取物体边界框的中心点作为其位置
            position = best_candidate.bbox_2d.get_center().tolist()
            found_obj_name = self.visualizer.obj_classes.get_classes_arr()[best_candidate.class_id]
            
            logger.info(f"[VLMapNav] [query_object] Found best match '{found_obj_name}' for query '{object_name}' with score {best_similarity:.4f} at position {position}.")
            return position
        else:
            logger.warning(f"[VLMapNav] [query_object] No object found for query '{object_name}' in the global map.")
            return None

    def compute_global_path(self, goal_pose: np.ndarray):
        """
        计算全局路径，存入 self.curr_global_path
        """
        self.goal_pose = goal_pose

        if not self.global_map_manager.has_global_map():
            logger.warning("[Core] No global map available for path planning.")
            return None
        logger.info(f"[Core] calculating global_path to: {goal_pose}")

        self.global_map_manager.has_action_path = False

        # Get Current layout information from detector
        layout_pcd = self.detector.get_layout_pointcloud()
        # (1) 将 点云数据 设置给 GlobalMapManager.layout_map
        # (2) 如果没有墙壁点云，则提取墙壁点云
        self.global_map_manager.set_layout_info(layout_pcd)

        self.visualizer.mark_semantic_map_dirty(dirty=True)  # wall_pcd 改变，需更新语义地图图像缓存

        # calculate the path based on current global map
        # Get 3D path point in world coordinate
        # 计算 全局路径
        self.curr_global_path = self.global_map_manager.calculate_global_path(
            self.curr_pose, goal_mode=self.get_goal_mode, resolution=self.cfg.resolution, goal_position=self.goal_pose
        )

        self.visualizer.mark_traversable_map_dirty(dirty=True)  # nav_graph 改变，需更新语义地图图像缓存

        # Clear the local mapping results
        self.curr_local_path = None
        self.begin_local_planning = True

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

        # logger.info("=====================================")
        # dt, dr = self.local_map_manager.compute_pose_difference(self.curr_pose, self.prev_pose)

        # if dt is not None and dr is not None:
        #     logger.info(f"Translation Difference: {dt}")
        #     logger.info(f"Rotation Difference: {dr}")

        #     if dt < 0.10 and dr < 0.30:
        #         self.wait_count += 1
        #         logger.info(f"Wait Count: {self.wait_count}")
        #     else:
        #         self.wait_count -= 1
        #         if self.wait_count < 0:
        #             self.wait_count = 0

        # if self.wait_count > 10:
        #     logger.info("We cannot find the local path and global path is finished!")
        #     logger.info("Now start a new global path planning!")
        #     self.begin_local_planning = False
        #     self.wait_count = 0
        #     # retrigger the global path planning
        #     self.set_calculate_path(self.cfg.config_file_path)

        # TEMP: Now explicitly pass the goal to local map to establish the workflow
        global_path = self.curr_global_path

        start = global_path[-1]
        x, y, z = start  # Unpack coordinates
        start_pose = np.eye(4)  # Initialize a 4x4 identity matrix

        # Set translation part
        start_pose[0, 3] = x  # X direction translation
        start_pose[1, 3] = y  # Y direction translation
        start_pose[2, 3] = z  # Z direction translation

        # for CLick mode, pass the click goal to local from global
        if self.global_map_manager.nav_graph.snapped_goal:
            click_goal = self.global_map_manager.nav_graph.snapped_goal
            self.local_map_manager.set_click_goal(click_goal)

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
            start_pose, goal_mode=self.get_goal_mode, resolution=self.cfg.resolution, goal_position=self.goal_pose
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
        cur_path = remaining_path(cur_path, self.curr_pose)
        if not cur_path:
            logger.debug("[Core] No remaining path available for next waypoint computation.")
            return None
        return cur_path[min(1, len(cur_path)-1)]

    def get_semantic_map_image(self):
        semantic_map = self.visualizer.get_semantic_map_image(
            self.global_map_manager,
            resolution=self.cfg.resolution,
            curr_pose=self.curr_pose,
            traj_path=self.traj_path,
            nav_path=self.curr_global_path  # TODO: 后续更改为 self.action_path，或者传入 global_path 和 local_path，以不同颜色显示
        )

        if semantic_map is not None:
            save_dir = str(self.cfg.map_save_path)
            save_path = f"{save_dir}/semantic_map.png"
            if os.path.exists(save_dir) and not os.path.exists(save_path):
                cv2.imwrite(save_path, semantic_map)
                print(f"[visualizer] Semantic map saved to {save_path}")

        return semantic_map
    
    def get_traversable_map_image(self):
        return self.visualizer.get_traversable_map_image(
            self.global_map_manager,
            curr_pose=self.curr_pose
        )
