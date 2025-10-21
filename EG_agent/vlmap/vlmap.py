import logging
import time
from dynaconf import Dynaconf
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from EG_agent.system.module_path import AGENT_VLMAP_PATH
from EG_agent.vlmap.dualmap.core import Dualmap
from EG_agent.vlmap.ros_runner.runner_ros_base import RunnerROSBase
from EG_agent.vlmap.utils.logging_helper import setup_logging
from EG_agent.vlmap.utils.types import GoalMode

class VLMapNav(RunnerROSBase):
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
    def query_object(self, object: str):
        return self.dualmap.query_object(object)
    
    def get_global_path(self, goal_pose: np.ndarray):
        self.dualmap.compute_global_path(goal_pose)

    def get_local_path(self):
        self.dualmap.compute_local_path()

    def get_action_path(self):
        self.dualmap.compute_action_path()

    def get_next_waypoint(self):
        return self.dualmap.compute_next_waypoint()

    def get_cmd_vel(self) -> tuple:
        """
        Generate a single velocity command based on the next waypoint and current pose.

        Args:
            next_waypoint: The target 3D point [x, y, z] in ROS coordinate system.

        Returns:
            - A velocity command tuple (vx, vy, wz) in the robot's base frame.
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
            return (0.0, 0.0, 0.0)
        
        # Controller constants (tune as needed)
        kp = 0.25
        max_lin_vel = 2.0
        max_ang_vel = 1.0
        yaw_error_threshold = 0.8
        goal_reached_threshold = 0.3

        camera_pose_ros = self.dualmap.realtime_pose.copy()  # 4x4 pose matrix in ROS frame
        # if isinstance(camera_pose_ros, np.ndarray) and camera_pose_ros.ndim == 3:
        #     camera_pose_ros = camera_pose_ros[-1, :, :]

        # Rotation from ROS to World frame
        # rot_ros_to_world = R.from_matrix([
        #     [0, 0, 1],
        #     [-1, 0, 0],
        #     [0, -1, 0]
        # ])
        # Camera offset from base in World frame
        # cam_offset_world = np.array([0.0, 0.0, 0.2])

        # Extract camera pose in ROS
        cam_pos_ros = camera_pose_ros[:3, 3]
        cam_rot_ros_matrix = camera_pose_ros[:3, :3]
        cam_rot_ros = R.from_matrix(cam_rot_ros_matrix)

        # Transform camera pose to World
        # cam_pos_world = rot_ros_to_world.apply(cam_pos_ros)
        # cam_rot_world = rot_ros_to_world * cam_rot_ros

        # Compute robot base pose in World
        # robot_pos_world = cam_pos_world - cam_offset_world
        # robot_rot_world = cam_rot_world

        # Transform next waypoint to World
        waypoint_ros = np.array(next_waypoint)
        # waypoint_world = rot_ros_to_world.apply(waypoint_ros)

        # Distance to goal (XY)
        dist_to_goal = np.linalg.norm(waypoint_ros[:2] - cam_pos_ros[:2])
        if dist_to_goal <= goal_reached_threshold:
            self.logger.info(f"[VLMapNav] [get_cmd_vel] Waypoint reached (distance: {dist_to_goal: .2f}m).")
            return (0.0, 0.0, 0.0)

        # Desired heading in World XY plane
        goal_yaw = np.arctan2(
            waypoint_ros[1] - cam_pos_ros[1],
            waypoint_ros[0] - cam_pos_ros[0]
        )

        # Current yaw from rotation matrix (use robot's forward axis projected on XY)
        rot_mat_world = cam_rot_ros.as_matrix()
        forward_world = rot_mat_world[:, 0]  # body +X axis in world frame
        current_yaw = np.arctan2(forward_world[1], forward_world[0])

        # Heading error wrapped to [-pi, pi]
        yaw_error = np.arctan2(np.sin(goal_yaw - current_yaw), np.cos(goal_yaw - current_yaw))

        # Angular velocity
        ang_vel_z = np.clip(kp * yaw_error, -max_ang_vel, max_ang_vel)

        self.logger.info(f"[VLMapNav] [get_cmd_vel] cur_yaw: {current_yaw: .2f}, goal_yaw: {goal_yaw: .2f}")

        # Linear velocity with decoupling during sharp turns
        if np.abs(yaw_error) > yaw_error_threshold:
            lin_vel_x = max_lin_vel * (1.0 - (np.abs(yaw_error) - yaw_error_threshold) / (np.pi - yaw_error_threshold))
            lin_vel_x = float(np.clip(lin_vel_x, 0.0, max_lin_vel))
        else:
            lin_vel_x = max_lin_vel
        
        end = time.time()
        self.logger.debug(f"[VLMapNav] [get_cmd_vel] Computation time: {end - start: .4f}s")
        return (lin_vel_x, 0.0, ang_vel_z)

