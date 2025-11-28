import logging
import time
from dynaconf import Dynaconf
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from EG_agent.system.module_path import AGENT_VLMAP_PATH
from EG_agent.vlmap.dualmap.core import Dualmap
from EG_agent.vlmap.dualmap.logging_helper import setup_logging
from EG_agent.vlmap.dualmap.types import GoalMode, DataInput

class VLMapNav:
    """
    VLMap navigation backend without ROS2 dependencies.
    EGAgentSystem feeds observations; this class maintains Dualmap and exposes navigation APIs.
    """
    def __init__(self, range_sensor: str = "depth"):
        cfg_files = [f"{AGENT_VLMAP_PATH}/config/base_config.yaml",
                     f"{AGENT_VLMAP_PATH}/config/system_config.yaml",
                     f"{AGENT_VLMAP_PATH}/config/mobility_config.yaml",
                     f"{AGENT_VLMAP_PATH}/config/rerun_config.yaml"]
        if range_sensor == "lidar":
            cfg_files.append(f"{AGENT_VLMAP_PATH}/config/lidar_config.yaml")
        elif range_sensor == "depth":
            cfg_files.append(f"{AGENT_VLMAP_PATH}/config/depth_config.yaml")
        else:
            raise ValueError(f"range_sensor must be 'lidar' or 'depth', but got {range_sensor}")
        self.cfg = Dynaconf(settings_files=cfg_files, lowercase_read=True, merge_enabled=False)
        self.cfg.output_path = f'{AGENT_VLMAP_PATH}/{self.cfg.output_path}'
        self.cfg.logging_config = f'{AGENT_VLMAP_PATH}/{self.cfg.logging_config}'
        self.logger = logging.getLogger(__name__)
        setup_logging(output_path=f'{self.cfg.output_path}/{self.cfg.dataset_name}',
                      config_path=str(self.cfg.logging_config))
        self.logger.info("[VLMapNav] initialized")

        # Let the received intrinsics topic decide
        # self.intrinsics = None
        self.intrinsics: np.ndarray = self.load_intrinsics(self.cfg)
        self.dist = self.load_distortion(self.cfg)
        self.extrinsics = self.load_extrinsics(self.cfg, name='extrinsics')
        self.lidar_extrinsics = self.load_extrinsics(self.cfg, name='lidar_extrinsics')

        self.kf_idx = 0
        self.last_message_time = None
        self.realtime_pose: np.ndarray = np.eye(4)

        self.dualmap: Dualmap = None

    def create_backend(self):
        self.dualmap = Dualmap(self.cfg)

    # ===============================================
    # Auxiliary methods
    # ===============================================
    def load_intrinsics(self, dataset_cfg):
        """Load camera intrinsics from config file."""
        intrinsic_cfg = dataset_cfg.intrinsics
        fx, fy, cx, cy = intrinsic_cfg['fx'], intrinsic_cfg['fy'], intrinsic_cfg['cx'], intrinsic_cfg['cy']
        self.logger.info("[Main] Loaded intrinsics from config.")
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def load_distortion(self, dataset_cfg):
        """Load camera distortion parameters from config file."""
        intrinsic_cfg = dataset_cfg.intrinsics
        dist = intrinsic_cfg.get('dist', None)  # Default to None if not specified
        if dist is None:
            self.logger.info("[Main] No distortion parameters found in config, using None")
        else:
            self.logger.info(f"[Main] Loaded distortion parameters from config: {dist}")
        return dist

    def load_extrinsics(self, dataset_cfg, name='lidar_extrinsics'):
        """Load camera extrinsics from config file."""
        # extrinsic_cfg = dataset_cfg.get('extrinsics', None)
        extrinsic_cfg = dataset_cfg.get(name, None)
        if extrinsic_cfg:
            matrix = np.array(extrinsic_cfg)
            if matrix.shape == (4, 4):
                self.logger.info(f"[Main] Loaded [{name}] from config.")
                return matrix
        self.logger.warning(f"[Main] No valid [{name}] provided. Using identity matrix.")
        return np.eye(4)

    def create_world_transform(self):
        """Create world coordinate transformation from roll/pitch/yaw."""
        roll = np.radians(float(self.cfg.world_roll))
        pitch = np.radians(float(self.cfg.world_pitch))
        yaw = np.radians(float(self.cfg.world_yaw))

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

    def push_data(self, rgb_img, depth_img, lidar_points, pose, timestamp):
        """Push synchronized input data into queue for processing."""
        # 用于灵活调整世界坐标系的方向
        # 目前：world_transform 和 extrinsics 是 单位阵，pose 是相机的世界坐标系(ROS系统，前进轴为Z，上轴为-Y)）
        transformed_pose = self.create_world_transform() @ (pose @ self.extrinsics)
        # print("transformed_pose:", transformed_pose)
        # print("lidar_points:", lidar_points.shape)

        if lidar_points is not None:
            lidar_points = (self.lidar_extrinsics @ np.hstack(
                (lidar_points, np.ones((len(lidar_points), 1)))).T).T[:, :3]

        data_input = DataInput(
            idx=self.kf_idx,
            time_stamp=timestamp,
            color=rgb_img,
            depth=depth_img,
            lidar=lidar_points,
            color_name=str(timestamp),
            intrinsics=self.intrinsics,
            dist=self.dist,
            pose=transformed_pose
        )
        self.realtime_pose = data_input.pose
        self.dualmap.global_map_manager.update_pose_path(curr_pose=data_input.pose)
        # 根据 时间戳和位姿 判断当前帧是否为 关键帧
        if not self.dualmap.check_keyframe(data_input.time_stamp, data_input.pose):
            return

        data_input.idx = self.dualmap.get_keyframe_idx()
        # Push to Dualmap's input queue, waiting for detector thread to process
        self.dualmap.input_queue.append(data_input)

    # ===============================================
    # High-level API for navigation and querying
    # ===============================================
    def start_find(self, target_object: str):
        # self.logger.info(f"Starting find '{target_object}'")
        self.dualmap.goal_pose = None
        self.dualmap.inquiry = target_object
        self.dualmap.goal_event.set()   # 即时唤醒触发一次 path_plan

    def object_found(self, target_object: str):
        return target_object in self.dualmap.inquiry_found

    def is_exploring(self, target_object: str):
        return self.dualmap.goal_mode == GoalMode.RANDOM and self.dualmap.inquiry == target_object

    def query_object(self, object: str):
        return self.dualmap.query_object(object)
    
    def get_global_path(self, goal_pose: np.ndarray):
        """ 已弃用,全局路径规划已放至后台线程 """
        self.dualmap.goal_pose = goal_pose.tolist()
        self.dualmap.compute_global_path()

    def get_local_path(self):
        self.dualmap.compute_local_path()

    def get_action_path(self):
        self.dualmap.compute_action_path()

    def get_cmd_vel(self) -> tuple:
        """
        A simple Controller to follow the planned path.
        Generate a velocity command toward the next waypoint.
        """
        # start = time.time()
        goal_pos = self.dualmap.goal_pose
        if not goal_pos:
            self.logger.debug("[VLMapNav] [get_cmd_vel] goal_pose is None!")
            return (0.0, 0.0, 0.0)
        
        next_waypoint = self.dualmap.compute_next_waypoint()

        if next_waypoint is None:
            self.logger.debug("[VLMapNav] [get_cmd_vel] get_next_waypoint failed!")
            return (0.0, 0.0, 0.0)
        
        # === 1. 获取相机位姿 ===
        camera_pose_ros = self.realtime_pose.copy()
        cam_pos = camera_pose_ros[:3, 3]
        cam_rot = camera_pose_ros[:3, :3]

        dist_to_goal = np.linalg.norm(np.array(goal_pos)[:2] - cam_pos[:2])
        if  dist_to_goal < 0.5:
            self.logger.info(f"[VLMapNav] goal_pose arrived")
            self.dualmap.reset_goal_position()
            return (0.0, 0.0, 0.0)

        delta_world = np.array(next_waypoint) - cam_pos
        dist_to_waypoint = np.linalg.norm(delta_world[:2])
        if dist_to_waypoint < 0.1:
            return (0.0, 0.0, 0.0)

        # Controller parameters
        kp_ang = 0.6
        kp_lin = 0.6
        min_lin_vel = 0.6
        max_lin_vel = 2.0
        max_ang_vel = 1.0
        yaw_error_threshold = 0.8

        # === 2. 世界 -> base_link 坐标变换 ===
        delta_body = cam_rot.T @ delta_world
        # 注：ROS系统下，相机forward为 +Z，而机器人forward为 +X
        # 速度指令是给机器人底盘（x-forward），需从 camera系 旋转到 base_link系
        # camera_to_base_link: +Z_cam -> +X_base, +X_cam -> -Y_base, +Y_cam -> -Z_base
        R_cam_to_base = np.array([[0, 0, 1],
                                  [-1, 0, 0],
                                  [0, -1, 0]], dtype=float)
        delta_base = R_cam_to_base @ delta_body

        # === 3. 计算误差 ===
        goal_yaw = np.arctan2(delta_base[1], delta_base[0])
        yaw_error = np.arctan2(np.sin(goal_yaw), np.cos(goal_yaw))

        # === 4. 速度控制 ===
        ang_vel_z = np.clip(kp_ang * yaw_error, -max_ang_vel, max_ang_vel)

        # forward velocity
        lin_vel_x = min_lin_vel + (max_lin_vel - min_lin_vel) * (1 - np.exp(-kp_lin * dist_to_goal))
        yaw_factor = np.clip(np.cos(yaw_error), 0.0, 1.0)
        if np.abs(yaw_error) > yaw_error_threshold:
            yaw_factor *= max(0.3, 1.0 - np.abs(yaw_error) / np.pi)
        lin_vel_x *= yaw_factor
        lin_vel_x = np.clip(lin_vel_x, 0.0, max_lin_vel)

        # === 5. 倒退策略 ===
        # 如果角度误差太大而且目标很近，尝试后退调整姿态
        if np.abs(yaw_error) > 1.2 and dist_to_goal < 0.6:
            lin_vel_x = -0.4  # small backward velocity
            ang_vel_z = np.sign(yaw_error) * 0.5  # turn while backing
            self.logger.debug("[VLMapNav] [get_cmd_vel] Backing off due to large yaw error near wall")

        # === 6. 输出 ===
        # end = time.time()
        # self.logger.debug(
        #     f"[VLMapNav] [get_cmd_vel] Time={end - start:.4f}s | "
        #     f"Pose=({cam_pos[0]:.2f},{cam_pos[1]:.2f}) -> Goal=({next_waypoint[0]:.2f},{next_waypoint[1]:.2f}), "
        #     f"Dist={dist_to_goal:.2f}, YawErr={yaw_error:.2f}, CMD=({lin_vel_x:.2f},0,{ang_vel_z:.2f})"
        # )
        return (float(lin_vel_x), 0.0, float(ang_vel_z))

