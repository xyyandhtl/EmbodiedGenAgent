# runner_ros_base.py

import logging
import time
import queue  # Add queue module for thread-safe communication
# from collections import deque  # removed

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from EG_agent.vlmap.utils.time_utils import timing_context
from EG_agent.vlmap.utils.types import DataInput
from EG_agent.vlmap.dualmap.core import Dualmap


class DualmapInterface:
    """
    Base class for ROS1 and ROS2 runners.
    Handles shared logic such as intrinsics/extrinsics loading,
    image decompression, pose conversion, and keyframe processing.
    """

    def __init__(self, cfg, dualmap: Dualmap):
        self.cfg = cfg
        self.dualmap = dualmap
        self.logger = logging.getLogger(__name__)

        self.kf_idx = 0
        self.intrinsics = None
        self.extrinsics = None
        # self.synced_data_queue = deque(maxlen=1)  # removed
        self.last_message_time = None

    def load_intrinsics(self, dataset_cfg):
        """Load camera intrinsics from config file."""
        intrinsic_cfg = dataset_cfg.get('intrinsic', None)
        if intrinsic_cfg:
            fx, fy, cx, cy = intrinsic_cfg['fx'], intrinsic_cfg['fy'], intrinsic_cfg['cx'], intrinsic_cfg['cy']
            self.logger.info("[Main] Loaded intrinsics from config.")
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.logger.warning("[Main] No intrinsics provided.")
        return None

    def load_extrinsics(self, dataset_cfg):
        """Load camera extrinsics from config file."""
        extrinsic_cfg = dataset_cfg.get('extrinsics', None)
        if extrinsic_cfg:
            matrix = np.array(extrinsic_cfg)
            if matrix.shape == (4, 4):
                self.logger.info("[Main] Loaded extrinsics from config.")
                return matrix
        self.logger.warning("[Main] No valid extrinsics provided. Using identity matrix.")
        return np.eye(4)

    def create_world_transform(self):
        """Create world coordinate transformation from roll/pitch/yaw."""
        roll = np.radians(self.cfg.world_roll)
        pitch = np.radians(self.cfg.world_pitch)
        yaw = np.radians(self.cfg.world_yaw)

        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

        R_combined = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R_combined
        return T

    def decompress_image(self, msg_data, is_depth=False):
        """Decode compressed image data (RGB or depth)."""
        msg_data = bytes(msg_data)
        if is_depth:
            depth_data = np.frombuffer(msg_data[12:], np.uint8)
            img = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)
        else:
            np_arr = np.frombuffer(msg_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def build_pose_matrix(self, translation, quaternion):
        """Construct 4x4 pose matrix from translation and quaternion."""
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation
        return transformation_matrix

    def push_data(self, rgb_img, depth_img, pose, timestamp):
        """Push synchronized input data into queue for processing."""
        # 用于灵活调整世界坐标系的方向
        # 目前：capture_frame 和 extrinsics 是 单位阵，pose 是相机的世界坐标系(ROS系统，前进轴为Z，上轴为-Y)）
        transformed_pose = self.create_world_transform() @ (pose @ self.extrinsics)

        data_input = DataInput(
            idx=self.kf_idx,
            time_stamp=timestamp,
            color=rgb_img,
            depth=depth_img,
            color_name=str(timestamp),
            intrinsics=self.intrinsics,
            pose=transformed_pose
        )
        self.dualmap.realtime_pose = data_input.pose
        self.dualmap.global_map_manager.update_pose_path(curr_pose=data_input.pose)
        # 根据 时间戳和位姿 判断当前帧是否为 关键帧
        if not self.dualmap.check_keyframe(data_input.time_stamp, data_input.pose):
            return
        
        data_input.idx = self.dualmap.get_keyframe_idx()
        # Push to Dualmap's input queue, waiting for detector thread to process
        self.dualmap.input_queue.append(data_input)

    def run_once(self):
        # 调用 detector 流程处理一帧 input_queue 的 run_once,
        # 已弃用, 把所有 detector 和 mapping 逻辑都放在后台线程
        data_input: DataInput = self.dualmap.input_queue[-1]
        if data_input.idx == self.dualmap.last_keyframe_idx:
            return
        
        self.dualmap.last_keyframe_idx = data_input.idx
        self.logger.info("[Main] ============================================================")
        with timing_context("Time Per Frame", self.dualmap):
            if self.cfg.use_parallel:
                self.dualmap.parallel_process(data_input)
            else:
                self.dualmap.sequential_process(data_input)

        self.logger.info(f"[Main] Processing keyframe {data_input.idx} took {time.time() - data_input.time_stamp:.2f} seconds.")
