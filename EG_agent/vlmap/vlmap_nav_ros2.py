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

        # CameraInfo fallback
        self.create_subscription(CameraInfo, self.cfg.ros_topics.camera_info, self.camera_info_callback, 10)

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
    def get_nav_path(self, goal_query: str, goal_mode: GoalMode = None) -> list | None:
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
        timeout_seconds = 30  # Maximum wait time
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if self.dualmap.global_map_manager.has_action_path:
                self.logger.warning("[VLMapNavROS2][get_nav_path] Navigation path found.")
                return self.dualmap.action_path  # a list of 3 elements [(x,y,z), ...]
            time.sleep(0.5)  # Check every 0.5 seconds

        self.logger.warning("[VLMapNavROS2][get_nav_path] Timed out waiting for navigation path.")
        return None


    def get_cmd_vel(self, path: list, kp: float = 0.5, max_lin_vel: float = 0.5, max_ang_vel: float = 0.8,
                    yaw_error_threshold: float = 0.5, goal_reached_threshold: float = 0.2) -> list:
        # todo: impl a simple velocity command generator based on the planned path
        """
        Generates a sequence of velocity commands to follow a given path using a P-controller logic.

        Args:
            path: A list of 3D waypoints [(x, y, z), ...].
            kp: Proportional gain for the angular velocity controller.
            max_lin_vel: Maximum linear velocity (m/s).
            max_ang_vel: Maximum angular velocity (rad/s).
            yaw_error_threshold: The yaw error (in radians) above which the robot focuses on turning.
            goal_reached_threshold: The distance (in meters) to a waypoint to consider it reached.

        Returns:
            A list of velocity commands [(vx, vy, vz), ...], where vy is always 0.
        """
        if not path:
            self.logger.warning("[VLMapNavROS2][get_cmd_vel] received an empty path.")
            return []

        cmd_vel_sequence = []
        current_pose = self.dualmap.curr_pose
        if current_pose is None:
            self.logger.error(
                "[VLMapNavROS2][get_cmd_vel] Cannot get current pose from dualmap. Cannot generate velocity commands.")
            return []

        # This function generates an open-loop sequence by simulating the robot's movement.
        simulated_pose = np.copy(current_pose)
        dt = 0.1  # Assume 10Hz command rate for simulation

        for waypoint in path:
            # Simulate movement towards each waypoint for a maximum number of steps
            max_steps_per_waypoint = 200
            for _ in range(max_steps_per_waypoint):
                # Get current simulated position and yaw from the 4x4 pose matrix
                current_pos = simulated_pose[:3, 3]
                rotation_matrix = simulated_pose[:3, :3]
                current_yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

                # --- P-Controller Logic based on user's reference ---
                # Calculate distance and angle to the goal (using 2D for navigation)
                dist_to_goal = np.linalg.norm(np.array(waypoint)[:2] - current_pos[:2])
                
                if dist_to_goal <= goal_reached_threshold:
                    break  # Waypoint reached, move to the next one

                goal_yaw = np.arctan2(waypoint[1] - current_pos[1], waypoint[0] - current_pos[0])

                # Calculate heading error, wrapped to the range [-pi, pi]
                yaw_error = goal_yaw - current_yaw
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                # Decide velocity based on distance and heading error
                lin_vel_x = 0.0
                ang_vel_z = 0.0

                # Calculate angular velocity
                ang_vel_z = kp * yaw_error
                ang_vel_z = np.clip(ang_vel_z, -max_ang_vel, max_ang_vel)

                # Calculate linear velocity (decoupling)
                if abs(yaw_error) > yaw_error_threshold:
                    # Robot needs to turn, reduce linear velocity
                    lin_vel_x = max_lin_vel * (1.0 - (abs(yaw_error) - yaw_error_threshold) / (math.pi - yaw_error_threshold))
                    lin_vel_x = np.clip(lin_vel_x, 0.0, max_lin_vel)
                else:
                    # Robot is facing the goal, move forward
                    lin_vel_x = max_lin_vel
                # --- End of P-Controller Logic ---

                # Append the command in (vx, vy, vz) format as requested
                cmd_vel_sequence.append((lin_vel_x, 0.0, ang_vel_z))

                # --- Simulation of pose update for the next step ---
                current_yaw += ang_vel_z * dt
                simulated_pose[0, 0] = np.cos(current_yaw)
                simulated_pose[0, 1] = -np.sin(current_yaw)
                simulated_pose[1, 0] = np.sin(current_yaw)
                simulated_pose[1, 1] = np.cos(current_yaw)
                simulated_pose[0, 3] += lin_vel_x * np.cos(current_yaw) * dt
                simulated_pose[1, 3] += lin_vel_x * np.sin(current_yaw) * dt
                # --- End of simulation ---

        # Add a final stop command
        cmd_vel_sequence.append((0.0, 0.0, 0.0))
        self.logger.info(f"[VLMapNavROS2][get_cmd_vel] Generated {len(cmd_vel_sequence)} velocity commands.")
        return cmd_vel_sequence

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