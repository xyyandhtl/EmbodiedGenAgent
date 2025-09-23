# runner_ros2.py

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from dynaconf import Dynaconf
import numpy as np
import rclpy
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
        self.logger = logging.getLogger(__name__)
        setup_logging(output_path=self.cfg.output_path, config_path=str(self.cfg.logging_config))
        self.logger.info("[Runner ROS2]")
        self.logger.info(self.cfg.as_dict())

        # self.cfg = cfg
        self.dualmap = Dualmap(self.cfg)
        RunnerROSBase.__init__(self, self.cfg, self.dualmap)

        self.bridge = CvBridge()
        self.dataset_cfg = Dynaconf(settings_files=self.cfg.ros_stream_config_path)
        self.intrinsics = self.load_intrinsics(self.dataset_cfg)
        self.extrinsics = self.load_extrinsics(self.dataset_cfg)
        # self.orig_height = self.cfg.camera_params['image_height']
        # self.orig_width = self.cfg.camera_params['image_width']
        # self.fx = self.cfg.camera_params['fx']
        # self.fy = self.cfg.camera_params['fy']
        # self.cx = self.cfg.camera_params['cx']
        # self.cy = self.cfg.camera_params['cy']

        # self.intrinsics = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        # Topic Subscribers
        if self.cfg.use_compressed_topic:
            self.logger.warning("[Main] Using compressed topics.")
            self.rgb_sub = Subscriber(self, CompressedImage, self.dataset_cfg.ros_topics.rgb)
            self.depth_sub = Subscriber(self, CompressedImage, self.dataset_cfg.ros_topics.depth)
        else:
            self.logger.warning("[Main] Using uncompressed topics.")
            self.rgb_sub = Subscriber(self, Image, self.dataset_cfg.ros_topics.rgb)
            self.depth_sub = Subscriber(self, Image, self.dataset_cfg.ros_topics.depth)

        self.odom_sub = Subscriber(self, Odometry, self.dataset_cfg.ros_topics.odom)

        # Sync messages
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            queue_size=10,
            slop=self.cfg.sync_threshold
        )
        self.sync.registerCallback(self.synced_callback)

        # CameraInfo fallback
        self.create_subscription(CameraInfo, self.dataset_cfg.ros_topics.camera_info, self.camera_info_callback, 10)

        # Publisher and timer
        self.publisher = ROSPublisher(self, self.cfg)
        self.publish_executor = ThreadPoolExecutor(max_workers=2)

        timer_period = 1.0 / self.cfg.ros_rate
        self.timer = self.create_timer(timer_period, self.run)

    def synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """Callback for synced RGB-D-Odom input."""
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
        self.run_once(lambda: self.get_clock().now().nanoseconds / 1e9)
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
    def get_nav_path(self):
        # todo: wrap the path planning from dualmap navigation_helper
        pass

    def get_cmd_vel(self):
        # todo: impl a simple velocity command generator based on the planned path
        pass

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