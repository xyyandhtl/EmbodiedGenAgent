# runner_ros2.py

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor

from dynaconf import Dynaconf
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage

from scipy.spatial.transform import Rotation as R

from EG_agent.system.module_path import AGENT_VLMAP_PATH
from EG_agent.vlmap.dualmap.core import Dualmap
from EG_agent.vlmap.ros_runner.ros_publisher import ROSPublisher
from EG_agent.vlmap.ros_runner.runner_ros_base import RunnerROSBase
from EG_agent.vlmap.utils.logging_helper import setup_logging
from EG_agent.vlmap.utils.types import GoalMode


class VLMapNavROS2(Node, RunnerROSBase):
    """
    ROS2-specific runner. Uses rclpy and ROS2 message_filters for synchronization,
    subscription, and publishing.
    """
    def __init__(self):
        Node.__init__(self, 'runner_ros')
        cfg_files = [f"{AGENT_VLMAP_PATH}/config/base_config.yaml", 
                     f"{AGENT_VLMAP_PATH}/config/system_config.yaml", 
                     f"{AGENT_VLMAP_PATH}/config/support_config/mobility_config.yaml",
                     f"{AGENT_VLMAP_PATH}/config/support_config/demo_config.yaml", 
                     f"{AGENT_VLMAP_PATH}/config/runner_ros.yaml",]
        self.cfg = Dynaconf(settings_files=cfg_files, 
                            lowercase_read=True, 
                            merge_enabled=False,)
        self.cfg.output_path = f'{AGENT_VLMAP_PATH}/{self.cfg.output_path}'
        self.cfg.logging_config = f'{AGENT_VLMAP_PATH}/{self.cfg.logging_config}'
        self.logger = logging.getLogger(__name__)
        setup_logging(output_path=f'{self.cfg.output_path}/{self.cfg.dataset_name}', 
                      config_path=str(self.cfg.logging_config))
        self.logger.info("[VLMapNavROS2 RUNNER]")
        # self.logger.info(self.cfg.as_dict())

        self.dualmap = Dualmap(self.cfg)
        RunnerROSBase.__init__(self, self.cfg, self.dualmap)

        self.bridge = CvBridge()
        # Let the received intrinsics topic decide
        # self.intrinsics = None
        self.intrinsics = self.load_intrinsics(self.cfg)
        self.extrinsics = self.load_extrinsics(self.cfg)

        # Topic Subscribers
        if self.cfg.use_compressed_topic:
            self.logger.warning("[Main] Using compressed topics.")
            self.rgb_sub = Subscriber(self, CompressedImage, self.cfg.ros_topics.rgb)
            self.depth_sub = Subscriber(self, CompressedImage, self.cfg.ros_topics.depth)
        else:
            self.logger.warning("[Main] Using uncompressed topics.")
            self.rgb_sub = Subscriber(self, Image, self.cfg.ros_topics.rgb)
            self.depth_sub = Subscriber(self, Image, self.cfg.ros_topics.depth)

        self.odom_sub = Subscriber(self, Odometry, self.cfg.ros_topics.odom)

        # Sync messages：使用同步器 ApproximateTimeSynchronizer 来确保上述topic的数据在时间上是大致对齐
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            queue_size=10,
            slop=self.cfg.sync_threshold
        )
        self.sync.registerCallback(self.synced_callback)

        # CameraInfo fallback, currently directly read from cfg
        # self.create_subscription(CameraInfo, self.cfg.ros_topics.camera_info, self.camera_info_callback, 10)

        # Publisher (发布地图、路径等信息) and timer (以固定的频率 (ros_rate) 调用 run 方法)
        self.publisher = ROSPublisher(self, self.cfg)
        self.publish_executor = ThreadPoolExecutor(max_workers=2)

        timer_period = 1.0 / self.cfg.ros_rate
        self.timer = self.create_timer(timer_period, self.run)

    def synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """Callback for synced RGB-D-Odom input.
        当同步器收到一组对齐的消息后，触发该回调函数：
            1. 负责解析 ROS 消息（例如，将 sensor_msgs/Image 转为 OpenCV 格式，从 nav_msgs/Odometry 提取 位姿），并将这些数据打包
            2. 调用 push_data 方法，将打包好的数据送入 Dualmap 核心进行处理
        """
        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9

        if self.cfg.use_compressed_topic:
            rgb_img = self.decompress_image(rgb_msg.data, is_depth=False)
            depth_img = self.decompress_image(depth_msg.data, is_depth=True)
        else:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')

        depth_img = depth_img.astype(np.float32) / 1000.0
        depth_img = np.expand_dims(depth_img, axis=-1)

        translation = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ])
        quaternion = np.array([
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        ])

        pose_matrix = self.build_pose_matrix(translation, quaternion)
        self.push_data(rgb_img, depth_img, pose_matrix, timestamp)
        self.last_message_time = self.get_clock().now().nanoseconds / 1e9

    def camera_info_callback(self, msg):
        """Populate intrinsics from CameraInfo topic if not already loaded."""
        if self.intrinsics is None:
            self.intrinsics = np.array(msg.k).reshape(3, 3)
            self.logger.warning("[Main] Camera intrinsics received and stored.")

    def run(self):
        """Periodic processing loop triggered by ROS2 timer."""
        # 调用 run_once 来处理 Dualmap 中的数据队列
        self.run_once(lambda: self.get_clock().now().nanoseconds / 1e9)
        # 发布最新的地图和路径信息
        self.publish_executor.submit(self.publisher.publish_all, self.dualmap)

    def shutdown_all_threads(self):
        """Clean up all threads and timers."""
        self.logger.warning("[Main] Shutting down all threads and timers.")
        try:
            self.timer.cancel()
        except Exception as e:
            self.logger.warning(f"[Main] Failed to cancel timer: {e}")
        self.publish_executor.shutdown(wait=True)

    def destroy_node(self):
        """Override base destroy_node with cleanup logic."""
        self.shutdown_all_threads()
        super().destroy_node()

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

        self.logger.warning(f"[VLMapNavROS2][get_nav_path] Received navigation query: '{goal_query}'")
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
            self.logger.warning(f"[VLMapNavROS2][get_nav_path] Updated config file '{config_path}' to trigger path planning.")

        except Exception as e:
            self.logger.warning(f"[VLMapNavROS2][get_nav_path] Failed to update config file: {e}")
            return None

        # 2. Wait for the path to be calculated by the dualmap thread
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if self.dualmap.global_map_manager.has_action_path:
                self.logger.warning("[VLMapNavROS2][get_nav_path] Navigation path found.")
                return self.dualmap.action_path  # a list of 3 elements [(x,y,z), ...]
            time.sleep(0.5)  # Check every 0.5 seconds

        self.logger.warning("[VLMapNavROS2][get_nav_path] Timed out waiting for navigation path.")
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
            self.logger.warning("[VLMapNavROS2][get_cmd_vel] Waypoint or camera_pose is None.")
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
            self.logger.info(f"[VLMapNavROS2][get_cmd_vel] Waypoint reached (distance: {dist_to_goal: .2f}m).")
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
        # todo: query the object from bt, return the candidates infos 
        pass


# ===============================================
# For unit test for dualmap with ros2
# ===============================================
if __name__ == "__main__":
    """Entry point for launching ROS2 runner."""
    rclpy.init()
    runner = VLMapNavROS2()
    runner.logger.warning("[Main] ROS2 Runner started. Waiting for data stream...")
    try:
        while rclpy.ok() and not runner.shutdown_requested:
            rclpy.spin_once(runner, timeout_sec=0.1)
    except KeyboardInterrupt:
        runner.logger.warning("[Main] KeyboardInterrupt received. Shutting down.")
    finally:
        runner.destroy_node()
        rclpy.shutdown()
        runner.logger.warning("[Main] Done.")