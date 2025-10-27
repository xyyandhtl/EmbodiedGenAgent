import logging
import time
from dynaconf import Dynaconf
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from EG_agent.system.module_path import AGENT_VLMAP_PATH
from EG_agent.vlmap.dualmap.core import Dualmap
from EG_agent.vlmap.ros_runner.runner_ros_base import DualmapInterface
from EG_agent.vlmap.utils.logging_helper import setup_logging
from EG_agent.vlmap.utils.types import GoalMode

class VLMapNav(DualmapInterface):
    """
    VLMap navigation backend without ROS2 dependencies.
    EGAgentSystem feeds observations; this class maintains Dualmap and exposes navigation APIs.
    """
    def __init__(self):
        cfg_files = [f"{AGENT_VLMAP_PATH}/config/base_config.yaml",
                     f"{AGENT_VLMAP_PATH}/config/system_config.yaml",
                     f"{AGENT_VLMAP_PATH}/config/support_config/mobility_config.yaml",
                     f"{AGENT_VLMAP_PATH}/config/support_config/demo_config.yaml"]
        self.cfg = Dynaconf(settings_files=cfg_files, lowercase_read=True, merge_enabled=False)
        self.cfg.output_path = f'{AGENT_VLMAP_PATH}/{self.cfg.output_path}'
        self.cfg.logging_config = f'{AGENT_VLMAP_PATH}/{self.cfg.logging_config}'
        self.logger = logging.getLogger(__name__)
        setup_logging(output_path=f'{self.cfg.output_path}/{self.cfg.dataset_name}',
                      config_path=str(self.cfg.logging_config))
        self.logger.info("[VLMapNav] initialized")

        self.dualmap = Dualmap(self.cfg)
        super().__init__(self.cfg, self.dualmap)

        # Let the received intrinsics topic decide
        # self.intrinsics = None
        self.intrinsics = self.load_intrinsics(self.cfg)
        self.extrinsics = self.load_extrinsics(self.cfg)

    # ===============================================
    # High-level API for navigation and querying
    # ===============================================
    def start_find(self, target_object: str):
        # self.logger.info(f"Starting find '{target_object}'")
        self.dualmap.goal_pose = None
        self.dualmap.inquiry = target_object
        self.dualmap.goal_mode = GoalMode.NONE  # 重置以使探索中可再次调用以重设探索点
        self.dualmap.goal_event.set()           # 即时唤醒触发一次 path_plan

    def object_found(self, target_object: str):
        return target_object in self.dualmap.inquiry_found

    def is_exploring(self, target_object: str):
        return self.dualmap.goal_mode == GoalMode.RANDOM and self.dualmap.inquiry == target_object

    def query_object(self, object: str):
        return self.dualmap.query_object(object)
    
    def get_global_path(self, goal_pose: np.ndarray):
        """ 已弃用,全局路径规划已放至后台线程 """
        self.dualmap.goal_pose = goal_pose
        self.dualmap.compute_global_path()

    def get_local_path(self):
        self.dualmap.compute_local_path()

    def get_action_path(self):
        self.dualmap.compute_action_path()

    def get_next_waypoint(self):
        return self.dualmap.compute_next_waypoint()

    def get_cmd_vel(self) -> tuple:
        """
        Generate a single velocity command based on the next waypoint and current pose.
        TODO: if need to put this method to a sub-thread too
        """
        # dualmap.curr_pose 只有在判断为关键帧后运行 self.dualmap.parallel_process() 时才会被更新，所以该值为关键帧位姿
        # 而实时计算速度指令，应用实时位姿
        if self.dualmap.realtime_pose is None:
            self.logger.debug("[VLMapNav] [get_cmd_vel] realtime_pose is None, please ensure start!")
            return (0.0, 0.0, 0.0)

        start = time.time()
        next_waypoint = self.get_next_waypoint()
        if next_waypoint is None:
            self.logger.debug("[VLMapNav] [get_cmd_vel] get_next_waypoint failed!")
            self.dualmap.goal_mode = GoalMode.NONE  # reset goal mode
            return (0.0, 0.0, 0.0)
        
        # Controller constants (tune as needed)
        kp_ang = 0.8
        kp_lin = 0.8
        min_lin_vel = 0.8
        max_lin_vel = 2.0
        max_ang_vel = 1.0
        yaw_error_threshold = 0.8
        goal_reached_threshold = 0.1  # m

        # 1. 计算 目标方向
        camera_pose_ros = self.dualmap.realtime_pose.copy()  # 4x4 pose matrix in ROS frame
        cam_pos = camera_pose_ros[:3, 3]
        cam_rot = camera_pose_ros[:3, :3]

        waypoint_pos = np.array(next_waypoint)
        delta_world = waypoint_pos - cam_pos

        # Distance to goal
        dist_to_goal = np.linalg.norm(delta_world[:2])
        if dist_to_goal <= goal_reached_threshold:
            self.logger.info(f"[VLMapNav] [get_cmd_vel] Waypoint reached (distance = {dist_to_goal: .2f}m).")
            return (0.0, 0.0, 0.0)

        # 2. 将 目标方向 投影到 机器人局部坐标系
        # 将 目标方向 从世界方向 转到 相机坐标系
        delta_body = cam_rot.T @ delta_world
        # 注：ROS系统下，相机forward为 +Z，而机器人forward为 +X
        # 速度指令是给机器人底盘（x-forward），需从 camera系 旋转到 base_link系
        # camera_to_base_link: +Z_cam -> +X_base, +X_cam -> -Y_base, +Y_cam -> -Z_base
        R_cam_to_base = np.array([[0, 0, 1],
                                  [-1, 0, 0],
                                  [0, -1, 0]], dtype=float)
        delta_base = R_cam_to_base @ delta_body

        # 3. 计算偏航误差（基于base_link）
        goal_yaw = np.arctan2(delta_base[1], delta_base[0])
        yaw_error = np.arctan2(np.sin(goal_yaw), np.cos(goal_yaw))

        # Angular velocity
        ang_vel_z = np.clip(kp_ang * yaw_error, -max_ang_vel, max_ang_vel)
        # Linear velocity
        lin_vel_x = min_lin_vel + (max_lin_vel - min_lin_vel) * (1 - np.exp(-kp_lin * dist_to_goal))
        yaw_factor = np.clip(np.cos(yaw_error), 0.0, 1.0)
        if np.abs(yaw_error) > yaw_error_threshold:
            yaw_factor *= max(0.3, 1.0 - np.abs(yaw_error) / np.pi)
        lin_vel_x *= yaw_factor
        lin_vel_x = np.clip(lin_vel_x, 0.0, max_lin_vel)

        end = time.time()
        self.logger.debug(f"[VLMapNav] [get_cmd_vel] Computation time: {end - start: .4f}s")
        self.logger.debug(
            f"[VLMapNav] [get_cmd_vel] Pose = ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}), "
            f"Goal = ({waypoint_pos[0]:.2f}, {waypoint_pos[1]:.2f}), "
            f"Dist = {dist_to_goal:.2f}, yaw_err={yaw_error:.2f}, "
            f"CMD = ({lin_vel_x:.2f}, 0, {ang_vel_z:.2f})"
        )

        return (float(lin_vel_x), 0.0, float(ang_vel_z))

