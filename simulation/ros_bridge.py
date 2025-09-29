import rclpy
from rclpy.node import Node
import numpy as np
import zmq
import pickle

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster

class ROSBridge(Node):
    """
    Receives sensor data via ZMQ and publishes it to ROS2 topics.
    ZMQ schema aligned with ZMQBridge:
      - SUB (tcp://localhost:5555) receives dicts with optional keys:
          "pose": (pos_np(3,), quat_wxyz_np(4,))
          "rgb": np.uint8[H,W,3]
          "depth": np.uint16[H,W] or np.float32[H,W]
      - PUB (tcp://*:5556) sends dicts with keys used by the simulation:
          "cmd_vel": np.float32[6]
          "nav_pose": (pos_np(3,), quat_wxyz_np(4,))
          "enum": int
    Ports must be different: one for sensor stream (SUB), one for commands (PUB).
    """
    def __init__(self, camera_params, zmq_port: int = 5555, command_port: int = 5556):
        super().__init__('ros_bridge_node')
        
        if zmq_port == command_port:
            raise ValueError("zmq_port (sensor SUB) and command_port (command PUB) must be different.")

        # --- ZMQ Context and Sockets ---
        self._zmq_ctx = zmq.Context()
        # Subscriber: Simulation -> ROS (sensor data)
        self.socket = self._zmq_ctx.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{zmq_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')  # Subscribe to all messages
        # Keep only the most recent message to minimize latency
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.get_logger().info(f"Connecting to ZMQ publisher on port {zmq_port}...")

        # Publisher: ROS -> Simulation (commands)
        self.cmd_pub_socket = self._zmq_ctx.socket(zmq.PUB)
        self.cmd_pub_socket.bind(f"tcp://*:{command_port}")
        self.get_logger().info(f"Command PUB bound on port {command_port} (publishes ROS -> Simulation commands)")

        # --- ROS2 Publisher and Broadcaster Setup ---
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        self.rgb_pub = self.create_publisher(Image, "/camera/rgb/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.pose_pub = self.create_publisher(Odometry, "/camera/pose", 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera_info", 10)

        # --- Prepare CameraInfo message (sent every tick with fresh header) ---
        self.camera_info_msg = self._prepare_camera_info(camera_params)
        
        # --- ROS2 Subscribers for commands (forwarded to ZMQ) ---
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 10)
        self.create_subscription(PoseStamped, "/nav_pose", self._nav_pose_callback, 10)
        self.create_subscription(Int32, "/enum_cmd", self._enum_command_callback, 10)

        # --- Timer to poll ZMQ and publish to ROS ---
        # Use a small period for low latency without busy-wait
        self.create_timer(0.01, self._poll_and_publish)

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

    def _poll_and_publish(self):
        """Timer callback to receive ZMQ data (non-blocking) and publish to ROS."""
        try:
            while True:
                message = self.socket.recv(flags=zmq.NOBLOCK)
                sensor_data = pickle.loads(message)
                self.publish_ros_data(sensor_data)
        except zmq.Again:
            # No more messages this tick
            pass
        except Exception as e:
            self.get_logger().error(f"An error occurred while polling ZMQ: {e}")

    def publish_ros_data(self, sensor_data):
        ros_time = self.get_clock().now().to_msg()

        # --- Pose and TF ---
        pose_tuple = sensor_data.get("pose")
        if pose_tuple is not None:
            pos_np, quat_wxyz_np = pose_tuple
            quat_xyzw_np = np.roll(quat_wxyz_np, -1)

            odom_msg = Odometry()
            odom_msg.header.stamp = ros_time
            odom_msg.header.frame_id = "map"
            odom_msg.child_frame_id = "camera_link"
            odom_msg.pose.pose.position.x = float(pos_np[0])
            odom_msg.pose.pose.position.y = float(pos_np[1])
            odom_msg.pose.pose.position.z = float(pos_np[2])
            odom_msg.pose.pose.orientation.x = float(quat_xyzw_np[0])
            odom_msg.pose.pose.orientation.y = float(quat_xyzw_np[1])
            odom_msg.pose.pose.orientation.z = float(quat_xyzw_np[2])
            odom_msg.pose.pose.orientation.w = float(quat_xyzw_np[3])
            self.pose_pub.publish(odom_msg)

            t = TransformStamped()
            t.header.stamp = ros_time
            t.header.frame_id = 'map'
            t.child_frame_id = 'camera_link'
            t.transform.translation.x = float(pos_np[0])
            t.transform.translation.y = float(pos_np[1])
            t.transform.translation.z = float(pos_np[2])
            t.transform.rotation.x = float(quat_xyzw_np[0])
            t.transform.rotation.y = float(quat_xyzw_np[1])
            t.transform.rotation.z = float(quat_xyzw_np[2])
            t.transform.rotation.w = float(quat_xyzw_np[3])
            self.tf_broadcaster.sendTransform(t)

        # --- RGB Image ---
        rgb_np = sensor_data.get("rgb")
        if rgb_np is not None:
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_np, "rgb8")
            rgb_msg.header.stamp = ros_time
            rgb_msg.header.frame_id = "camera_link"
            self.rgb_pub.publish(rgb_msg)

        # --- Depth Image (support uint16 or float32) ---
        depth_np = sensor_data.get("depth")
        if depth_np is not None:
            if depth_np.dtype == np.uint16:
                encoding = "16UC1"
            elif depth_np.dtype == np.float32:
                encoding = "32FC1"
            else:
                # Fallback: try to cast to float32
                depth_np = depth_np.astype(np.float32, copy=False)
                encoding = "32FC1"
            depth_msg = self.bridge.cv2_to_imgmsg(depth_np, encoding)
            depth_msg.header.stamp = ros_time
            depth_msg.header.frame_id = "camera_link"
            self.depth_pub.publish(depth_msg)
            
        # --- Camera Info ---
        self.camera_info_msg.header.stamp = ros_time
        self.camera_info_msg.header.frame_id = "camera_link"
        self.camera_info_pub.publish(self.camera_info_msg)

    # --- Callbacks: forward ROS messages into ZMQ for simulation ---
    def _cmd_vel_callback(self, msg: Twist):
        try:
            data = {
                "cmd_vel": np.array(
                    [msg.linear.x, msg.linear.y, msg.linear.z,
                     msg.angular.x, msg.angular.y, msg.angular.z],
                    dtype=np.float32
                )
            }
            self.cmd_pub_socket.send(pickle.dumps(data))
        except Exception as e:
            self.get_logger().error(f"Error forwarding cmd_vel: {e}")

    def _nav_pose_callback(self, msg: PoseStamped):
        try:
            pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)
            quat_xyzw = np.array(
                [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w],
                dtype=np.float32
            )
            quat_wxyz = np.roll(quat_xyzw, 1)
            data = {"nav_pose": (pos, quat_wxyz)}
            self.cmd_pub_socket.send(pickle.dumps(data))
        except Exception as e:
            self.get_logger().error(f"Error forwarding nav_pose: {e}")

    def _enum_command_callback(self, msg: Int32):
        try:
            cmd = int(msg.data)
            data = {"enum": cmd}
            self.cmd_pub_socket.send(pickle.dumps(data))
        except Exception as e:
            self.get_logger().error(f"Error forwarding enum command: {e}")

    def destroy_node(self):
        # Clean shutdown of ZMQ sockets
        try:
            self.cmd_pub_socket.close(0)
        except Exception:
            pass
        try:
            self.socket.close(0)
        except Exception:
            pass
        try:
            self._zmq_ctx.term()
        except Exception:
            pass
        return super().destroy_node()

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
        # Use ROS2 executor so subscriber callbacks and timers are processed
        rclpy.spin(ros_bridge)
    finally:
        ros_bridge.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
