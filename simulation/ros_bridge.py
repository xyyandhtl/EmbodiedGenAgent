import rclpy
from rclpy.node import Node
import numpy as np
import torch
import zmq
import pickle

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster

class ROSBridge(Node):
    """
    Receives sensor data via ZMQ and publishes it to ROS2 topics.
    """
    def __init__(self, camera_params, zmq_port=5555):
        super().__init__('ros_bridge_node')
        
        # --- ZMQ Subscriber Setup ---
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{zmq_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')  # Subscribe to all messages
        self.get_logger().info(f"Connecting to ZMQ publisher on port {zmq_port}...")

        # --- ROS2 Publisher and Broadcaster Setup ---
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        self.rgb_pub = self.create_publisher(Image, "/camera/rgb/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.pose_pub = self.create_publisher(Odometry, "/camera/pose", 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera_info", 10)

        # --- Prepare CameraInfo message (sent once) ---
        self.camera_info_msg = self._prepare_camera_info(camera_params)
        
        self.get_logger().info("ROS2 Bridge initialized. Waiting for data...")

    def _prepare_camera_info(self, params):
        msg = CameraInfo()
        msg.width = params['image_width']
        msg.height = params['image_height']
        msg.k = [params['fx'], 0.0, params['cx'], 0.0, params['fy'], params['cy'], 0.0, 0.0, 1.0]
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Assuming no distortion
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Rectification matrix (identity for undistorted)
        msg.p = [params['fx'], 0.0, params['cx'], 0.0, 0.0, params['fy'], params['cy'], 0.0, 0.0, 0.0, 1.0, 0.0]  # Projection matrix
        return msg

    def run(self):
        """Main loop to receive data and publish to ROS."""
        while rclpy.ok():
            try:
                # Receive data from ZMQ
                message = self.socket.recv(flags=zmq.NOBLOCK)
                sensor_data = pickle.loads(message)
                
                # Process and publish
                self.publish_ros_data(sensor_data)

            except zmq.Again:
                # No message received, continue
                pass
            except Exception as e:
                self.get_logger().error(f"An error occurred: {e}")

    def publish_ros_data(self, sensor_data):
        ros_time = self.get_clock().now().to_msg()

        # --- Pose and TF ---
        pose_tuple = sensor_data.get("pose")
        if pose_tuple is not None:
            pos_np, quat_np_wxyz = pose_tuple
            quat_np_xyzw = np.roll(quat_np_wxyz, -1)

            # Publish PoseStamped
            # pose_msg = PoseStamped()
            # pose_msg.header.stamp = ros_time
            # pose_msg.header.frame_id = "odom"
            # pose_msg.pose.position.x = float(pos_np[0])
            # pose_msg.pose.position.y = float(pos_np[1])
            # pose_msg.pose.position.z = float(pos_np[2])
            # pose_msg.pose.orientation.x = float(quat_np_xyzw[0])
            # pose_msg.pose.orientation.y = float(quat_np_xyzw[1])
            # pose_msg.pose.orientation.z = float(quat_np_xyzw[2])
            # pose_msg.pose.orientation.w = float(quat_np_xyzw[3])
            # self.pose_pub.publish(pose_msg)

            # Publish Odometry
            odom_msg = Odometry()
            odom_msg.header.stamp = ros_time
            odom_msg.header.frame_id = "map"
            odom_msg.child_frame_id = "camera_link"
            odom_msg.pose.pose.position.x = float(pos_np[0])
            odom_msg.pose.pose.position.y = float(pos_np[1])
            odom_msg.pose.pose.position.z = float(pos_np[2])
            odom_msg.pose.pose.orientation.x = float(quat_np_xyzw[0])
            odom_msg.pose.pose.orientation.y = float(quat_np_xyzw[1])
            odom_msg.pose.pose.orientation.z = float(quat_np_xyzw[2])
            odom_msg.pose.pose.orientation.w = float(quat_np_xyzw[3])
            self.pose_pub.publish(odom_msg)

            # Broadcast TF
            t = TransformStamped()
            t.header.stamp = ros_time
            t.header.frame_id = 'map'
            t.child_frame_id = 'camera_link'
            t.transform.translation.x = float(pos_np[0])
            t.transform.translation.y = float(pos_np[1])
            t.transform.translation.z = float(pos_np[2])
            t.transform.rotation.x = float(quat_np_xyzw[0])
            t.transform.rotation.y = float(quat_np_xyzw[1])
            t.transform.rotation.z = float(quat_np_xyzw[2])
            t.transform.rotation.w = float(quat_np_xyzw[3])
            self.tf_broadcaster.sendTransform(t)

        # --- RGB Image ---
        rgb_np = sensor_data.get("rgb")
        if rgb_np is not None:
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_np, "rgb8")
            rgb_msg.header.stamp = ros_time
            rgb_msg.header.frame_id = "camera_link"
            self.rgb_pub.publish(rgb_msg)

        # --- Depth Image ---
        depth_np = sensor_data.get("depth")
        if depth_np is not None:
            depth_msg = self.bridge.cv2_to_imgmsg(depth_np, "16UC1")
            depth_msg.header.stamp = ros_time
            depth_msg.header.frame_id = "camera_link"
            self.depth_pub.publish(depth_msg)
            
        # --- Camera Info ---
        self.camera_info_msg.header.stamp = ros_time
        self.camera_info_msg.header.frame_id = "camera_link"
        self.camera_info_pub.publish(self.camera_info_msg)

def main():
    rclpy.init()

    # These parameters should match the simulation environment
    camera_params = {
        'image_width': 640,
        'image_height': 480,
        'fx': 320.0,
        'fy': 320.0,
        'cx': 320.0,
        'cy': 240.0,
    }

    ros_bridge = ROSBridge(camera_params)
    try:
        ros_bridge.run()
    finally:
        ros_bridge.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
