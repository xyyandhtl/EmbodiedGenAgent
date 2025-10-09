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
    def get_nav_path(self, goal_query: str, goal_mode: GoalMode = None, timeout_seconds: int = 30) -> list | None:
        # todo: wrap the path planning from dualmap navigation_helper
        """
        Calculates and returns a navigation path based on a goal query.

        This function works by programmatically updating the configuration file that
        the Dualmap core monitors. It sets the navigation goal and triggers the
        path calculation process.

        Args:
            goal_query: The natural language query for the navigation goal (e.g., "a chair").
            goal_mode: The mode for goal specification. Defaults to INQUIRY.

        Returns:
            A list of coordinates representing the navigation path, or None if no path is found.
        """
        if goal_mode is None:
            goal_mode = GoalMode.INQUIRY  # TODO：可能默认为 RANDOM

        self.logger.warning(f"[VLMapNav] [get_nav_path] Received navigation query: '{goal_query}'")
        config_path = self.cfg.config_file_path

        # 1. Update the config file to trigger path planning
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            config_data['inquiry_sentence'] = goal_query
            config_data['get_goal_mode'] = goal_mode.value
            config_data['calculate_path'] = True

            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            self.logger.warning(f"[VLMapNav] [get_nav_path] Updated config file '{config_path}' to trigger path planning.")

        except Exception as e:
            self.logger.warning(f"[VLMapNav] [get_nav_path] Failed to update config file: {e}")
            return None

        # 2. Wait for the path to be calculated by the dualmap thread
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if self.dualmap.global_map_manager.has_action_path:
                self.logger.warning("[VLMapNavROS2] [get_nav_path] Navigation path found.")
                return self.dualmap.action_path  # a list of 3 elements [(x,y,z), ...]
            time.sleep(0.5)  # Check every 0.5 seconds

        self.logger.warning("[VLMapNav] [get_nav_path] Timed out waiting for navigation path.")
        return None

    def get_cmd_vel(self, waypoint: list, kp: float = 0.25, max_lin_vel: float = 2.0, max_ang_vel: float = 1.0,
                    yaw_error_threshold: float = 0.8, goal_reached_threshold: float = 0.3) -> tuple[tuple, bool]:
        """
        Generates a single velocity command in the World frame and checks if the goal is reached.

        Args:
            waypoint: The target 3D waypoint [x, y, z] in ROS coordinate system.
            kp: Proportional gain for the angular velocity controller.
            max_lin_vel: Maximum linear velocity (m/s).
            max_ang_vel: Maximum angular velocity (rad/s).
            yaw_error_threshold: The yaw error (in radians) above which the robot focuses on turning.
            goal_reached_threshold: The distance (in meters) to a waypoint to consider it reached.

        Returns:
            A tuple containing:
            - A velocity command tuple (vx, vy, wz) in the robot's base frame (World system).
            - A boolean `is_reached` flag.
        """
        # --- 1. Check Inputs ---
        camera_pose_ros = np.copy(self.dualmap.curr_pose)  # 4x4 pose matrix in ROS frame
        if waypoint is None or camera_pose_ros is None:
            self.logger.warning("[VLMapNav] [get_cmd_vel] Waypoint or camera_pose is None.")
            return ((0.0, 0.0, 0.0), False)
        if isinstance(camera_pose_ros, np.ndarray) and camera_pose_ros.ndim == 3:
            camera_pose_ros = camera_pose_ros[-1, :, :]

        # --- 2. Define Coordinate Transformations ---
        # Rotation from ROS frame (+Z fwd, -Y up) to World frame (+X fwd, +Z up)
        # World_X = ROS_Z, World_Y = -ROS_X, World_Z = -ROS_Y
        rot_ros_to_world = R.from_matrix([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        # Camera offset from base in World frame (camera is 0.2m above the base) according to the simulation/env/xx_env_cfg
        cam_offset_world = np.array([0.0, 0.0, 0.2])

        # --- 3. Transform Inputs from ROS to World Frame ---
        # a. Extract camera position and rotation from ROS pose matrix
        cam_pos_ros = camera_pose_ros[:3, 3]
        cam_rot_ros_matrix = camera_pose_ros[:3, :3]
        cam_rot_ros = R.from_matrix(cam_rot_ros_matrix)

        # b. Transform camera pose to world frame
        cam_pos_world = rot_ros_to_world.apply(cam_pos_ros)
        cam_rot_world = rot_ros_to_world * cam_rot_ros

        # c. Calculate robot base pose in world frame by applying the offset
        robot_pos_world = cam_pos_world - cam_offset_world
        robot_rot_world = cam_rot_world

        # d. Transform waypoint to world frame
        waypoint_ros = np.array(waypoint)
        waypoint_world = rot_ros_to_world.apply(waypoint_ros)

        # e. Calculate distance and angle to the goal
        dist_to_goal = np.linalg.norm(waypoint_world[:2] - robot_pos_world[:2])  # 3D distance
        if dist_to_goal <= goal_reached_threshold:
            self.logger.info(f"[VLMapNav] [get_cmd_vel] Waypoint reached (distance: {dist_to_goal: .2f}m).")
            return ((0.0, 0.0, 0.0), True)

        # --- 4. Calculate Velocity Command in World Frame (following simulation/mdp/commands.py) ---
        # a. Calculate desired heading in the world XY plane
        goal_yaw = np.arctan2(waypoint_world[1] - robot_pos_world[1], waypoint_world[0] - robot_pos_world[0])  # 2D heading

        # b. Get current robot quaternion in w,x,y,z format for yaw calculation
        robot_quat_xyzw = robot_rot_world.as_quat()
        qw, qx, qy, qz = robot_quat_xyzw[3], robot_quat_xyzw[0], robot_quat_xyzw[1], robot_quat_xyzw[2]
        current_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # c. Calculate heading error, wrapped to the range [-pi, pi]
        yaw_error = goal_yaw - current_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # d. Decide velocity based on distance and heading error
        ang_vel_z = kp * yaw_error
        ang_vel_z = np.clip(ang_vel_z, -max_ang_vel, max_ang_vel)

        # e. Calculate linear velocity (decoupling)
        lin_vel_x = 0.0
        if np.abs(yaw_error) > yaw_error_threshold:
            # Reduce linear velocity significantly during sharp turns
            lin_vel_x = max_lin_vel * (1.0 - (np.abs(yaw_error) - yaw_error_threshold) / (np.pi - yaw_error_threshold))
            lin_vel_x = np.clip(lin_vel_x, 0.0, max_lin_vel)
        else:
            lin_vel_x = max_lin_vel

        # The command is (lin_vel_x, 0.0, ang_vel_z) which corresponds to (vx, vy, wz)
        # in the robot's base frame, aligned with the world frame for this calculation.
        return ((lin_vel_x, 0.0, ang_vel_z), False)

    def query_object(self, query: str):
        import re
        # 1. 从查询（如："desk"/"RobotNear(ControlRoom)"）中提取物体名称
        match = re.search(r'\((.*?)\)', query)
        if match:
            object_name = match.group(1)
        else:
            object_name = query  # 如果格式不匹配，则假定整个查询都是对象名

        self.logger.info(f"[VLMapNav] [query_object] Received query '{query}', searching for object '{object_name}'.")

        # 2. 检查全局地图是否存在
        if not self.dualmap.global_map_manager.has_global_map():
            self.logger.warning("[VLMapNav] [query_object] Global map is empty. Cannot find object.")
            return None

        # 3. 将对象名称转换为 CLIP 特征向量并设置为查询目标
        try:
            query_feat = self.dualmap.convert_inquiry_to_feat(object_name)
            self.dualmap.global_map_manager.inquiry = query_feat  # TODO: 直接更改 global_map_manager 的查询不确定是否会与导航过程中的 parallel_process 中的查询起冲突，如果冲突，则可以将 self.inquiry 删除，直接在函数调用时传入特征向量
        except Exception as e:
            self.logger.error(f"[VLMapNav] [query_object] Failed to convert query to feature vector: {e}")
            return None

        # 4. 调用 GlobalMapManager 的 find_best_candidate_with_inquiry 来寻找最佳匹配
        #    这将返回 GlobalObject 实例和分数
        best_candidate, best_similarity = self.dualmap.global_map_manager.find_best_candidate_with_inquiry()

        # 5. 处理结果
        if best_candidate is not None:
            # 提取物体边界框的中心点作为其位置
            position = best_candidate.bbox_2d.get_center().tolist()
            found_obj_name = self.dualmap.visualizer.obj_classes.get_classes_arr()[best_candidate.class_id]
            
            self.logger.info(f"[VLMapNav] [query_object] Found best match '{found_obj_name}' for query '{object_name}' with score {best_similarity:.4f} at position {position}.")
            return position
        else:
            self.logger.warning(f"[VLMapNav] [query_object] No object found for query '{object_name}' in the global map.")
            return None