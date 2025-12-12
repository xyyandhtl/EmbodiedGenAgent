from typing import Callable
import numpy as _np
import math
from concurrent.futures import ThreadPoolExecutor

# --- ROS2 相关导入 ---
import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from std_msgs.msg import Int32
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
# from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.spatial.transform import Rotation as _R

from EG_agent.system.envs.base_env import BaseAgentEnv
from EG_agent.system.module_path import AGENT_ENV_PATH
from EG_agent.vlmap.dualmap.ros_publisher import ROSPublisher

import typing
if typing.TYPE_CHECKING:
    from EG_agent.vlmap.vlmap import VLMapNav

class RealEnv(BaseAgentEnv):
    behavior_lib_path = f"{AGENT_ENV_PATH}/embodied"
    # =========================================================
    # 构造与配置（初始化、ROS 话题与发布者/订阅者配置）
    # =========================================================
    def __init__(self):
        super().__init__()
        # Camera model defaults and tracking
        self.cam_fov_x_deg = 90.0
        self.cam_aspect = 4.0 / 3.0
        self.cam_forward_axis = "z"  # hardcoded: camera's +Z faces forward
        # Real-time visibility state: {goal_name_lower: bool}
        self.goal_inview = {}
        self.near_dist = 3.0  # meters

        # Navigation settings
        self.internal_nav = True
        self.use_action_client = False

        # Defer ROS init to configure_ros
        self.ros_node: Node = None
        self.cmd_vel_pub: Publisher = None
        self.nav_pose_pub: Publisher = None
        self.enum_pub: Publisher = None
        self.mark_pub: Publisher = None

        # Subscribers and sync
        self.cmd_vel_sub: Subscription = None
        self._rgb_sub = None
        self._depth_sub = None
        self._odom_sub = None
        self._sync = None
        self._bridge = CvBridge()

        # ROS spinning
        self._ros_executor: MultiThreadedExecutor = None
        self._ros_thread = None
        self._ros_spin_stop = None
        self._ros_init_owner = False

        # Config reference
        self._cfg = None  # Dynaconf instance

        # --- VLMap 后端引用 + 发布器资源 ---
        self._vlmap_backend: VLMapNav = None
        self._ros_publisher: ROSPublisher = None
        self._ros_pub_executor: ThreadPoolExecutor = None
        self._ros_pub_timer = None

        self._action_count: int = 0
        self.cur_goal_places = dict()  # str -> [x,y,z]
        self.cur_cmd_vel: tuple = (0.0, 0.0, 0.0)  # vx, vy, wz
        self.cur_goal_pose: tuple = ()  # [x,y,z,qw,qx,qy,qz]

        # Action client
        # 状态变量
        self._current_goal_handle = None
        self._current_status = "idle"
        self._failure_reason = None
        if self.use_action_client:
            self._nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 可扩展动作分发表（运行时绑定）
        self._action_dispatch = {
            "walk": self._handle_walk,
            "cmd_vel": self._handle_cmd_vel,
            "goal_pose": self._handle_nav_pose,
            "enum_command": self._handle_enum,
            "mark": self._handle_mark,
        }
    
    # ==========================================
    # Basic Interface
    # ==========================================
    def set_vlmap_backend(self, backend) -> None:
        """Attach VLMap backend so ROSPublisher can publish dualmap outputs."""
        self._vlmap_backend = backend

    def find_path(self, goal_pose):
        """调用 VLMap 后端计算路径"""
        return self._vlmap_backend.get_global_path(goal_pose)
    
    def get_cur_cmd_vel(self) -> tuple:
        """返回当前计算的速度命令 (vx, vy, wz)"""
        if self.internal_nav:
            self.cur_cmd_vel = self._vlmap_backend.get_cmd_vel()
        return self.cur_cmd_vel
    
    def get_target_pos(self, target_name: str) -> list[float]:
        """返回指定目标的当前位置 [x,y,z]，若未知则返回 None。"""
        return self.cur_goal_places.get(target_name, [])
    
    def get_cur_target_pos(self) -> list[float]:
        return self.get_target_pos(self.cur_target)
    
    def get_inview_goals(self) -> list[str]:
        """返回当前在相机视锥内的目标名称列表）。"""
        return [name for name, inview in self.goal_inview.items() if inview]

    def set_object_places(self, places: dict[str, list[float]]):
        """设置/更新目标位置，并更新可视性（每个目标是否在当前相机的视锥内）"""
        # Normalize to lower-case keys to match action args
        self.cur_goal_places.update(places)
        # Recompute visibility when goal set changes
        self._update_goal_inview()

    def get_cur_target(self) -> list[float]:
        return self._vlmap_backend.dualmap.goal_pose

    def reset(self):
        """重置环境：清空内部状态、发布零速以停止，并等待首次同步观测（最多约2秒）。"""
        # Clear internal state
        # self.goal_inview = {}
        self.init_statistics()

    # ==========================================
    # ROS Configuration
    # ==========================================
    def configure_ros(self, cfg) -> None:
        """创建 ROS 节点、发布/订阅与同步器，并在后台线程 spin。"""
        # 保存 Dynaconf 引用并同步相机配置
        self._cfg = cfg
        self.cam_fov_x_deg = float(cfg.camera.fov_x_deg)
        self.cam_aspect = float(cfg.camera.aspect)
        self.use_compressed_topic = bool(cfg.ros.use_compressed_topic)
        self.near_dist = float(cfg.camera.near_dist)
        self.range_sensor: str = cfg.range_sensor

        # Init rclpy if needed
        if not rclpy.ok():
            rclpy.init()
            self._ros_init_owner = True

        # Create node and pubs
        self.ros_node = Node("isaacsim_env_node")
        if self.internal_nav:
            self.cmd_vel_pub = self.ros_node.create_publisher(Twist, cfg.ros.pubs.cmd_vel, 10)
        else:
            self.nav_pose_pub = self.ros_node.create_publisher(PoseStamped, cfg.ros.pubs.nav_pose, 10)
            self.cmd_vel_sub = self.ros_node.create_subscription(Twist, cfg.ros.pubs.cmd_vel, self._cmd_vel_callback, 10)
        self.enum_pub = self.ros_node.create_publisher(Int32, cfg.ros.pubs.enum_cmd, 10)
        # 合并：统一使用 /mark_point (PointStamped)
        self.mark_pub = self.ros_node.create_publisher(PointStamped, "/mark_point", 10)

        # Independent subscribers (no synchronization)
        if self.use_compressed_topic:
            self._rgb_sub = self.ros_node.create_subscription(
                CompressedImage,
                cfg.ros.topics.rgb,
                self._rgb_callback,
                10
            )
        else:
            self._rgb_sub = self.ros_node.create_subscription(
                Image,
                cfg.ros.topics.rgb,
                self._rgb_callback,
                10
            )

        if self.range_sensor == "lidar":
            self._odom_sub = self.ros_node.create_subscription(
                Odometry,
                cfg.ros.topics.odom_lidar,
                self._odom_callback,
                10
            )
            # Subscribe to lidar with a callback that stores latest message
            # Use BEST_EFFORT QoS to match the publisher's configuration
            lidar_qos_profile = rclpy.qos.QoSProfile(
                depth=10,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST
            )
            self._lidar_sub = self.ros_node.create_subscription(
                PointCloud2,
                cfg.ros.topics.lidar,
                self._store_latest_lidar,
                lidar_qos_profile
            )
        elif self.range_sensor == "depth":
            self._odom_sub = self.ros_node.create_subscription(
                Odometry,
                cfg.ros.topics.odom_camera,
                self._odom_callback,
                10
            )
            # Subscribe to depth with a callback that stores latest message
            self._depth_sub = self.ros_node.create_subscription(
                Image,
                cfg.ros.topics.depth,
                self._store_latest_depth,
                10
            )
        else:
            raise NotImplementedError(f"Unsupported range_sensor type: {self.range_sensor}")

        # Create timer for range sensor processing at specified frequency
        self.range_sensor_frequency = float(cfg.ros.range_sensor_frequency)
        range_sensor_period = 1.0 / self.range_sensor_frequency
        self._range_sensor_timer = self.ros_node.create_timer(
            range_sensor_period, self._process_range_sensor_at_frequency)

        # Initialize message buffers for time-based matching
        self._rgb_buffer = {}
        self._lidar_buffer = {}
        self._depth_buffer = {}
        self._odom_buffer = {}
        self._latest_rgb = None
        self._latest_lidar = None
        self._latest_depth = None
        self._latest_odom = None

        # Timer for range sensor subscription at specific frequency
        self._range_sensor_timer = None
        self._latest_range_msg = None

        # Start spinning
        self._ros_executor = MultiThreadedExecutor()
        self._ros_executor.add_node(self.ros_node)

        def _spin():
            while rclpy.ok():
                self._ros_executor.spin_once(timeout_sec=0.1)

        self._ros_thread = __import__("threading").Thread(target=_spin, daemon=True)
        self._ros_thread.start()

        # if cfg.use_rviz:
        self._ros_publisher = ROSPublisher(self.ros_node, cfg)
        self._ros_pub_executor = ThreadPoolExecutor(max_workers=2)
        period = 1.0 / float(cfg.ros.ros_rate)
        self._ros_pub_timer = self.ros_node.create_timer(period, self._ros_pub_tick)

    def _match_and_process_data(self):
        """Match messages by timestamp: find odom, then range_sensor with identical timestamp (using extremely small threshold), then closest cam topic"""
        if self._latest_odom is None:
            return  # Need odom to proceed

        # Find range sensor data that has nearly identical timestamp to odom (using extremely small threshold)
        if self.range_sensor == "lidar":
            range_msg = self._find_identical_timestamp_message(self._latest_odom, self._lidar_buffer)
        elif self.range_sensor == "depth":
            range_msg = self._find_identical_timestamp_message(self._latest_odom, self._depth_buffer)
        else:
            raise NotImplementedError(f"Unsupported range_sensor type: {self.range_sensor}")

        if range_msg is None:
            return  # Wait for range data with nearly identical timestamp to odom

        # Find the camera message that is closest in time to the synchronized odom/range
        cam_msg = self._find_closest_message(self._latest_odom, self._rgb_buffer)
        if cam_msg is None:
            return  # Wait for camera data

        # Process with all three data types
        self._process_matched_messages(cam_msg, range_msg, self._latest_odom)

    def _find_closest_message(self, base_msg, buffer_dict):
        """Find the message in buffer that is closest in time to the base message"""
        if not buffer_dict:
            return None

        base_time = base_msg.header.stamp.sec + base_msg.header.stamp.nanosec * 1e-9
        closest_msg = None
        min_diff = float('inf')

        # Look for messages within the sync threshold
        for msg_time, msg in buffer_dict.items():
            time_diff = abs(msg_time - base_time)
            if time_diff < float(self._cfg.ros.sync_threshold) and time_diff < min_diff:
                min_diff = time_diff
                closest_msg = msg

        return closest_msg

    def _find_identical_timestamp_message(self, base_msg, buffer_dict):
        """Find the message in buffer that has an identical timestamp (using extremely small threshold)"""
        if not buffer_dict:
            return None

        base_time = base_msg.header.stamp.sec + base_msg.header.stamp.nanosec * 1e-9

        # Look for messages with nearly identical timestamps (using extremely small threshold)
        for msg_time, msg in buffer_dict.items():
            time_diff = abs(msg_time - base_time)
            # Use a very small threshold to ensure "identical" timestamps (0.1ms)
            if time_diff < 0.0001:  # 0.1ms threshold for "identical" timestamps
                return msg

        return None

    def _store_latest_lidar(self, msg):
        """Store the latest lidar message without processing immediately"""
        self._latest_range_msg = msg

    def _store_latest_depth(self, msg):
        """Store the latest depth message without processing immediately"""
        self._latest_range_msg = msg

    def _process_range_sensor_at_frequency(self):
        """Process range sensor data at the specified frequency"""
        if self._latest_range_msg is not None:
            # Update the appropriate buffer based on range sensor type
            timestamp = self._latest_range_msg.header.stamp.sec + self._latest_range_msg.header.stamp.nanosec * 1e-9

            if self.range_sensor == "lidar":
                self._lidar_buffer[timestamp] = self._latest_range_msg
                self._latest_lidar = self._latest_range_msg
                # Clean up old messages from buffer
                self._clean_old_messages(self._lidar_buffer, timestamp)
            elif self.range_sensor == "depth":
                self._depth_buffer[timestamp] = self._latest_range_msg
                self._latest_depth = self._latest_range_msg
                # Clean up old messages from buffer
                self._clean_old_messages(self._depth_buffer, timestamp)

            # Try to match and process data
            self._match_and_process_data()

    def _rgb_callback(self, msg):
        """Handle RGB image messages independently"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._rgb_buffer[timestamp] = msg
        self._latest_rgb = msg

        # Clean up old messages from buffer
        self._clean_old_messages(self._rgb_buffer, timestamp)

        # Try to match and process data
        self._match_and_process_data()

    def _odom_callback(self, msg):
        """Handle odometry messages independently"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._odom_buffer[timestamp] = msg
        self._latest_odom = msg

        # Clean up old messages from buffer
        self._clean_old_messages(self._odom_buffer, timestamp)

        # Try to match and process data
        self._match_and_process_data()

    def _process_matched_messages(self, rgb_msg, range_msg, odom_msg):
        """Process matched RGB, range, and odometry messages"""
        # Timestamp
        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9

        # RGB conversion
        if self.use_compressed_topic:
            rgb_img = self._bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        else:
            rgb_img = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')

        # Range data processing
        if self.range_sensor == "lidar":
            lidar_points = _np.vstack(pc2.read_points(range_msg,
                                      field_names=["x", "y", "z"],
                                      skip_nans=True).tolist())
            lidar_points = lidar_points[_np.isfinite(lidar_points).all(1)]

            min_thr = float(self._cfg.lidar.min_thr)
            max_thr = float(self._cfg.lidar.max_thr)

            x_mask = (_np.abs(lidar_points[:, 0]) >= min_thr) & (_np.abs(lidar_points[:, 0]) <= max_thr)
            y_mask = (_np.abs(lidar_points[:, 1]) >= min_thr) & (_np.abs(lidar_points[:, 1]) <= max_thr)
            lidar_points = lidar_points[x_mask & y_mask]
            # print(f"lidar_points {len(lidar_points)}")
            depth_img = None
        elif self.range_sensor == "depth":
            lidar_points = None
            depth_cv = self._bridge.imgmsg_to_cv2(range_msg, desired_encoding='16UC1')
            depth_img = depth_cv.astype(_np.float32) / 1000.0
            depth_img = _np.expand_dims(depth_img, axis=-1)
        else:
            raise NotImplementedError(f"Unsupported range_sensor type: {self.range_sensor}")

        # Pose matrix from odom
        t = _np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ], dtype=_np.float32)
        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w
        Rm = _R.from_quat([qx, qy, qz, qw]).as_matrix()
        pose = _np.eye(4, dtype=_np.float32)
        pose[:3, :3] = Rm
        pose[:3, 3] = t

        # Forward to backend directly
        self._vlmap_backend.push_data(rgb_img, depth_img, lidar_points, pose, timestamp)

        # Recompute real-time FOV visibility check using cam_pose_w
        self._update_goal_inview()

    def _clean_old_messages(self, buffer_dict, current_timestamp):
        """Remove old messages from the buffer to prevent infinite growth"""
        sync_threshold = float(self._cfg.ros.sync_threshold)
        # Keep only messages within a reasonable time window (3x threshold)
        threshold_time = current_timestamp - 3 * sync_threshold
        old_keys = [key for key in buffer_dict.keys() if key < threshold_time]
        for key in old_keys:
            del buffer_dict[key]

    def _cmd_vel_callback(self, msg: Twist):
        v = msg.linear
        w = msg.angular
        self.cur_cmd_vel = (v.x, v.y, w.z)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._current_status = "failed"
            self._failure_reason = "Goal rejected by Nav2"
            print("[RealEnv] Goal rejected by Nav2.")
            return

        self._current_goal_handle = goal_handle
        self._current_status = "navigating"
        print("[RealEnv] Goal accepted, navigating...")

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._result_callback)

        # 同步等待直到导航完成，若要异步处理，可移除此行
        rclpy.spin_until_future_complete(self.ros_node, get_result_future)

    def _feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        if hasattr(feedback, 'distance_remaining'):
            print(f"[Nav Feedback] distance_remaining: {feedback.distance_remaining:.2f}")

    def _result_callback(self, future):
        result = future.result().result
        status = future.result().status

        if status == 4:  # STATUS_SUCCEEDED
            self._current_status = "succeeded"
            print("[RealEnv] Navigation succeeded.")
        else:
            self._current_status = "failed"
            self._failure_reason = f"Navigation failed, status code: {status}"
            print(f"[RealEnv] Navigation failed, status={status}")

    def _ros_pub_tick(self):
        """Timer tick to publish all visualizations via ROSPublisher."""
        # Guard against early timer firing before backend/publisher ready
        self._ros_pub_executor.submit(self._ros_publisher.publish_all, self._vlmap_backend.dualmap)

    def close(self):
        """销毁节点与执行器线程；必要时关闭 rclpy。"""
        # 先清理发布器资源
        if self._ros_pub_timer is not None:
            self._ros_pub_timer.cancel()
            self._ros_pub_timer = None
        if self._ros_pub_executor is not None:
            self._ros_pub_executor.shutdown(wait=False)
            self._ros_pub_executor = None

        # Clear message buffers to free memory
        self._rgb_buffer.clear()
        self._lidar_buffer.clear()
        self._depth_buffer.clear()
        self._odom_buffer.clear()
        self._latest_rgb = None
        self._latest_lidar = None
        self._latest_depth = None
        self._latest_odom = None
        self._latest_range_msg = None

        # Cancel the range sensor timer
        if self._range_sensor_timer is not None:
            self._range_sensor_timer.cancel()
            self._range_sensor_timer = None

    # ==========================================
    # Action Implementation
    # ==========================================
    def run_action(self, action_type: str, action: tuple | None = None, verbose=False):
        """
        严格动作格式：
          - 'cmd_vel': [vx, vy, wz]
          - 'nav_pose': [x,y,z,qw,qx,qy,qz]
          - 'enum'/'enum_command': 单个或多个 Int32
          - 'mark': () 或 [x,y,z]（空表示在机器人前方插旗；否则在指定坐标插旗）
        """
        verbose = verbose and self._action_count % 10 == 0
        self._action_count += 1
        if self.ros_node is None:
            raise RuntimeError("ROS node not initialized; cannot publish actions.")

        key = str(action_type).lower().strip()
        handler = self._action_dispatch[key]
        if handler is None:
            raise ValueError(f"Action type {action_type} not registered and not a known ROS-publishable command.")
        handler(action, verbose)

    # 具体动作处理器（更易扩展）
    def _handle_cmd_vel(self, action, verbose: bool):
        if not isinstance(action, (list, tuple, _np.ndarray)) or len(action) != 3:
            raise ValueError("cmd_vel action must be a list/tuple/ndarray of 3 elements: [vx, vy, wz].")
        vx, vy, wz = float(action[0]), float(action[1]), float(action[2])
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = wz
        self.cmd_vel_pub.publish(twist)
        if verbose:
            print(f"[RealEnv] Published cmd_vel: vx={vx}, vy={vy}, wz={wz}")

    def _handle_nav_pose(self, action, verbose: bool):
        if self.use_action_client:
            if self._current_status == "failed":
                print(f"[RealEnv] NavigateToPose action failed: {self._failure_reason}")
                return
            if self._current_status == "navigating":
                return

        x, y, z = map(float, action[:3])
        qw, qx, qy, qz = map(float, action[3:])

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.stamp = self.ros_node.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = z
        goal_msg.pose.pose.orientation.w = qw
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        if verbose:
            print(f"[RealEnv] Sending nav goal: pos=({x:.3f},{y:.3f},{z:.3f}), quat=({qw:.3f},{qx:.3f},{qy:.3f},{qz:.3f})")
        
        if self.use_action_client:
            self._current_status = "sending"
            self._failure_reason = None

            # 等待action server可用
            if not self._nav_to_pose_client.wait_for_server(timeout_sec=3.0):
                self._current_status = "failed"
                self._failure_reason = "Nav2 action server not available"
                print("[RealEnv] Nav2 action server not available.")
                return

            send_goal_future = self._nav_to_pose_client.send_goal_async(
                goal_msg,
                feedback_callback=self._feedback_callback
            )
            send_goal_future.add_done_callback(lambda fut: self._goal_response_callback(fut))
        else:
            self.nav_pose_pub.publish(goal_msg.pose)

    def _handle_walk(self, action, verbose: bool):
        # a wrapper for walk action
        if self.internal_nav:
            # if use vlmap internal nav system
            cur_cmd_vel = self.get_cur_cmd_vel()
            self._handle_cmd_vel(cur_cmd_vel, verbose)
        elif self.get_cur_target():
            # if use a external nav system
            goal_pose = tuple(self.get_cur_target() + [1, 0, 0, 0])
            print(f"[RealEnv] walk action: {goal_pose}")
            if self.use_action_client or goal_pose != self.cur_goal_pose:
                self._handle_nav_pose(goal_pose, verbose)
                self.cur_goal_pose = goal_pose

    def _handle_enum(self, action, verbose: bool):
        to_publish = []
        if isinstance(action, (list, tuple, _np.ndarray)):
            to_publish = [int(x) for x in action]
        else:
            to_publish = [int(action)]
        for cmd in to_publish:
            msg = Int32()
            msg.data = int(cmd)
            self.enum_pub.publish(msg)
            if verbose:
                print(f"[RealEnv] Published enum command: {cmd}")

    def _handle_mark(self, action, verbose: bool):
        """
        'mark' 动作：
          - action == () / [] / None / 长度==0: 发布默认 mark（PointStamped.x/y/z = NaN）
          - action 为长度3: 在 [x,y,z] 插旗（/mark_point, PointStamped）
        """
        pt = PointStamped()
        pt.header.stamp = self.ros_node.get_clock().now().to_msg()
        pt.header.frame_id = "map"

        # 默认：空参数 -> 用 NaN 作为哨兵
        if not action:
            pt.point.x = math.nan
            pt.point.y = math.nan
            pt.point.z = math.nan
            self.mark_pub.publish(pt)
            # if verbose:
            print("[RealEnv] Published mark (default via NaN).")
            return

        # # 显式坐标
        # if not isinstance(action, (list, tuple, _np.ndarray)) or len(action) != 3:
        #     raise ValueError("mark action must be empty () or a 3-element [x,y,z] coordinate.")
        x, y, z = float(action[0]), float(action[1]), float(action[2])
        pt.point.x = x
        pt.point.y = y
        pt.point.z = z
        self.mark_pub.publish(pt)
        # if verbose:
        print(f"[RealEnv] Published mark point at: ({x}, {y}, {z})")

    # ==========================================
    # Auxiliary Methods
    # ==========================================
    def _update_goal_inview(self):
        """根据当前相机位姿与目标位置，实时更新目标是否在视野内。"""
        # Default: no goals or no pose => all False
        self.goal_inview = {name: False for name in self.cur_goal_places.keys()}
        cam_pose = self._vlmap_backend.realtime_pose
        
        cam_pos = cam_pose[:3, 3]
        cam_rot = cam_pose[:3, :3]
        for name, pt in self.cur_goal_places.items():
            point = _np.array(pt[:3])
            # 先进行距离阀值判断
            dist = float(_np.linalg.norm(point - cam_pos))
            if dist > float(self.near_dist):
                self.goal_inview[name] = False
                continue
            # 再根据当前目标点的3D坐标，判断是否在相机视锥内
            # 计算目标方向向量（世界坐标系）
            delta_world = point - cam_pos
            # 将目标方向投影到相机局部坐标系
            delta_body = cam_rot.T @ delta_world
            # 计算偏航角
            # 在相机坐标系中，+Z轴为前方，X-Z平面的偏航角
            goal_yaw = _np.arctan2(delta_body[0], delta_body[2])  # atan2(x, z) for yaw around Y axis
            # 计算水平FOV的一半（弧度）
            fov_x_rad = _np.radians(self.cam_fov_x_deg / 2.0)
            # 检查目标是否在水平FOV内
            if abs(goal_yaw) <= fov_x_rad:
                self.goal_inview[name] = True
            else:
                self.goal_inview[name] = False
