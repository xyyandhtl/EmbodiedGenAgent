import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.spatial.transform import Rotation as R

from EG_agent.vlmap.vlmap import VLMapNav
from EG_agent.vlmap.ros_runner.ros_publisher import ROSPublisher


class VLMapRosRunner(Node, VLMapNav):
    """ROS2-based runner by directly inheriting Node and VLMapNav."""
    def __init__(self):
        Node.__init__(self, "vlmap_unit_test")
        VLMapNav.__init__(self)

        # Setup logging consistent with VLMapNav
        self.logger = logging.getLogger(__name__)
        self.logger.warning("[VLMapRosRunner] start")

        self.bridge = CvBridge()

        # Subscribers and synchronizer (use cfg loaded by VLMapNav)
        if bool(self.cfg.use_compressed_topic):
            self.logger.warning("[VLMapRosRunner] Using compressed topics.")
            self.rgb_sub = Subscriber(self, CompressedImage, self.cfg.ros_topics.rgb)
            self.depth_sub = Subscriber(self, CompressedImage, self.cfg.ros_topics.depth)
        else:
            self.logger.warning("[VLMapRosRunner] Using raw topics.")
            self.rgb_sub = Subscriber(self, Image, self.cfg.ros_topics.rgb)
            self.depth_sub = Subscriber(self, Image, self.cfg.ros_topics.depth)
        self.odom_sub = Subscriber(self, Odometry, self.cfg.ros_topics.odom)

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            queue_size=10,
            slop=float(self.cfg.sync_threshold),
        )
        self.sync.registerCallback(self._synced_callback)

        # Publisher (original ROSPublisher) and timer tick
        self.publisher = ROSPublisher(self, self.cfg)
        self.publish_executor = ThreadPoolExecutor(max_workers=2)
        self.timer = self.create_timer(1.0 / float(self.cfg.ros_rate), self._tick)

    def _synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """Convert RGB/Depth/Odom to numpy and push to VLMap (self)."""
        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9

        if bool(self.cfg.use_compressed_topic):
            rgb_img = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_cv = self.bridge.compressed_imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        else:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")

        depth_img = depth_cv.astype(np.float32) / 1000.0
        depth_img = np.expand_dims(depth_img, axis=-1)

        t = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ], dtype=np.float32)
        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot
        pose[:3, 3] = t

        # Push into VLMapNav queue
        self.push_data(rgb_img, depth_img, pose, timestamp)

    def _tick(self):
        """Periodic processing + publish maps/topics via the original ROSPublisher."""
        self.run_once(lambda: self.get_clock().now().nanoseconds / 1e9)
        self.publish_executor.submit(self.publisher.publish_all, self.dualmap)

    def destroy_node(self):
        try:
            self.timer.cancel()
        except Exception:
            pass
        self.publish_executor.shutdown(wait=True)
        super().destroy_node()


def main():
    rclpy.init()
    node = VLMapRosRunner()
    node.get_logger().warning("[Main] VLMap ROS runner. Ctrl+C to exit.")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warning("[Main] KeyboardInterrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().warning("[Main] Done.")


if __name__ == "__main__":
    main()

