from typing import Callable
import numpy as _np
import math
from concurrent.futures import ThreadPoolExecutor

# --- ROS2 相关导入 ---
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from std_msgs.msg import Int32
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.spatial.transform import Rotation as _R

from EG_agent.system.envs.base_env import BaseAgentEnv
from EG_agent.system.module_path import AGENT_ENV_PATH
from EG_agent.vlmap.ros_runner.ros_publisher import ROSPublisher

import typing
if typing.TYPE_CHECKING:
    from EG_agent.vlmap.vlmap import VLMapNav

class IsaacsimEnv(BaseAgentEnv):
    agent_num = 1

    behavior_lib_path = f"{AGENT_ENV_PATH}/embodied"

    # Camera model defaults and tracking
    cam_fov_x_deg = 90.0
    cam_aspect = 4.0 / 3.0
    cam_forward_axis = "z"  # hardcoded: camera's +Z faces forward
    # Real-time visibility state: {goal_name_lower: bool}
    goal_inview = {}
    near_dist = 2.0

    # =========================================================
    # 构造与配置（初始化、ROS 话题与发布者/订阅者配置）
    # =========================================================
    def __init__(self):
        super().__init__()
        # Defer ROS init to configure_ros
        self.ros_node: Node | None = None
        self.cmd_vel_pub: Publisher = None
        self.nav_pose_pub: Publisher = None
        self.enum_pub: Publisher = None
        self.mark_pub: Publisher = None

        # Subscribers and sync
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
        # Track last synced observation time
        self._last_obs_time: float | None = None

        # --- 新增: VLMap 后端引用 + 发布器资源 ---
        self._vlmap_backend: VLMapNav = None
        self._ros_publisher: ROSPublisher = None
        self._ros_pub_executor: ThreadPoolExecutor = None
        self._ros_pub_timer = None

        self._cur_path = []
        self._cur_cmd_vel = (0.0, 0.0, 0.0)  # vx, vy, wz
        self._action_count = 0
        # 新增：可扩展动作分发表（运行时绑定）
        self._action_dispatch = {
            "cmd_vel": self._handle_cmd_vel,
            "cmdvel": self._handle_cmd_vel,
            "cmd-vel": self._handle_cmd_vel,
            "nav_pose": self._handle_nav_pose,
            "goal_pose": self._handle_nav_pose,
            "pose": self._handle_nav_pose,
            "enum": self._handle_enum,
            "enum_command": self._handle_enum,
            "command_enum": self._handle_enum,
            "mark": self._handle_mark,
            "mark_point": self._handle_mark,
            "place_flag": self._handle_mark,
        }

    def configure_ros(self, cfg) -> None:
        """创建 ROS 节点、发布/订阅与同步器，并在后台线程 spin。"""
        # 保存 Dynaconf 引用并同步相机配置
        self._cfg = cfg
        self.cam_fov_x_deg = float(cfg.camera.fov_x_deg)
        self.cam_aspect = float(cfg.camera.aspect)
        self.use_compressed_topic = bool(cfg.ros.use_compressed_topic)
        self.near_dist = float(cfg.camera.near_dist)

        # Init rclpy if needed
        if not rclpy.ok():
            rclpy.init()
            self._ros_init_owner = True

        # Create node and pubs
        self.ros_node = Node("isaacsim_env_node")
        self.cmd_vel_pub = self.ros_node.create_publisher(Twist, cfg.ros.pubs.cmd_vel, 10)
        self.nav_pose_pub = self.ros_node.create_publisher(PoseStamped, cfg.ros.pubs.nav_pose, 10)
        self.enum_pub = self.ros_node.create_publisher(Int32, cfg.ros.pubs.enum_cmd, 10)
        # 合并：统一使用 /mark_point (PointStamped)
        self.mark_pub = self.ros_node.create_publisher(PointStamped, "/mark_point", 10)

        # Subs + sync
        if self.use_compressed_topic:
            self._rgb_sub = Subscriber(self.ros_node, CompressedImage, cfg.ros.topics.rgb)
            self._depth_sub = Subscriber(self.ros_node, CompressedImage, cfg.ros.topics.depth)
        else:
            self._rgb_sub = Subscriber(self.ros_node, Image, cfg.ros.topics.rgb)
            self._depth_sub = Subscriber(self.ros_node, Image, cfg.ros.topics.depth)

        self._odom_sub = Subscriber(self.ros_node, Odometry, cfg.ros.topics.odom)

        self._sync = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._odom_sub],
            queue_size=10,
            slop=float(cfg.ros.sync_threshold)
        )
        self._sync.registerCallback(self._synced_callback)

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

    def set_vlmap_backend(self, backend) -> None:
        """Attach VLMap backend so ROSPublisher can publish dualmap outputs."""
        self._vlmap_backend = backend

    def find_path(self, goal_pose):
        """调用 VLMap 后端计算路径"""
        return self._vlmap_backend.get_global_path(goal_pose)

    def _synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """RGB/Depth/Odom 同步回调：解码 -> 位姿矩阵 -> 推送到 VLMap 后端 -> 更新可视状态"""
        # Timestamp
        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        # print(f"Received synced data at time {timestamp}")

        # RGB/Depth conversion
        if self.use_compressed_topic:
            # Minimal compressed support via CvBridge; if not supported fallback to no-op
            rgb_img = self._bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_cv = self._bridge.compressed_imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
        else:
            rgb_img = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_cv = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')

        depth_img = depth_cv.astype(_np.float32) / 1000.0
        depth_img = _np.expand_dims(depth_img, axis=-1)

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
        self._vlmap_backend.push_data(rgb_img, depth_img, pose, timestamp)

        # Record last observation time
        self._last_obs_time = timestamp

        # Optionally update env state for FOV checks
        self.agent_env_update({
            "cam_pose_w": [t[0], t[1], t[2], qw, qx, qy, qz],
            # TODO: if need more status updates
        })

    def _ros_pub_tick(self):
        """Timer tick to publish all visualizations via ROSPublisher."""
        # Guard against early timer firing before backend/publisher ready
        self._ros_pub_executor.submit(self._ros_publisher.publish_all, self._vlmap_backend.dualmap)

    # ==========================================
    # 环境状态与目标管理（状态更新、目标位置、完成判定）
    # ==========================================
    def agent_env_update(self, env_data: dict):
        """更新环境态并实时刷新目标可视性。"""
        self.cur_agent_states = {**env_data}

        # Recompute real-time visibility using cam_pose_w
        self._update_goal_inview()
        self._cur_cmd_vel = self._vlmap_backend.get_cmd_vel()

    def get_inview_goals(self) -> list[str]:
        """返回当前在相机视锥内的目标名称列表）。"""
        return [name for name, inview in self.goal_inview.items() if inview]

    def set_object_places(self, places: dict[str, list[float]]):
        """设置/更新目标位置，并更新可视性（每个目标是否在当前相机的视锥内）"""
        # Normalize to lower-case keys to match action args
        self.cur_goal_places = {str(k): v for k, v in places.items()}
        # Recompute visibility when goal set changes
        self._update_goal_inview()

    def reset(self):
        """重置环境：清空内部状态、发布零速以停止，并等待首次同步观测（最多约2秒）。"""
        # Clear internal state
        self.cur_agent_states = {}
        self.goal_inview = {}
        self._last_obs_time = None

    # ==========================================
    # 动作发布与执行（cmd_vel、nav_pose、枚举命令、mark）
    # ==========================================
    def run_action(self, action_type: str, action: tuple | None, verbose=True):
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
            print(f"[IsaacsimEnv] Published cmd_vel: vx={vx}, vy={vy}, wz={wz}")

    def _handle_nav_pose(self, action, verbose: bool):
        if not isinstance(action, (list, tuple, _np.ndarray)) or len(action) != 7:
            raise ValueError("nav_pose action must be a list/tuple/ndarray of 7 elements: [x,y,z,qw,qx,qy,qz].")
        x, y, z = float(action[0]), float(action[1]), float(action[2])
        qw, qx, qy, qz = float(action[3]), float(action[4]), float(action[5]), float(action[6])
        ps = PoseStamped()
        ps.header.stamp = self.ros_node.get_clock().now().to_msg()
        ps.header.frame_id = "map"
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.w = qw
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        self.nav_pose_pub.publish(ps)
        if verbose:
            print(f"[IsaacsimEnv] Published nav_pose: pos=({x},{y},{z}) quat=({qw},{qx},{qy},{qz})")

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
                print(f"[IsaacsimEnv] Published enum command: {cmd}")

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
            print("[IsaacsimEnv] Published mark (default via NaN).")
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
        print(f"[IsaacsimEnv] Published mark point at: ({x}, {y}, {z})")

    # ==========================================
    # 相机几何与可视性辅助（旋转、视锥检测）
    # ==========================================
    def _quat_to_rotmat(self, qw: float, qx: float, qy: float, qz: float) -> _np.ndarray:
        """单位四元数 (w,x,y,z) -> 旋转矩阵 (3x3)。"""
        # Normalize to be safe
        n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if n == 0.0:
            return _np.eye(3, dtype=_np.float64)
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        R = _np.array([
            [1.0 - 2.0*(yy + zz),     2.0*(xy - wz),         2.0*(xz + wy)],
            [    2.0*(xy + wz),   1.0 - 2.0*(xx + zz),       2.0*(yz - wx)],
            [    2.0*(xz - wy),       2.0*(yz + wx),     1.0 - 2.0*(xx + yy)]
        ], dtype=_np.float64)
        return R

    def _point_in_cam_fov(
        self,
        cam_pose: _np.ndarray | list | tuple,
        point_world: _np.ndarray,
        fov_x_deg: float,
        aspect: float,
        forward_axis: str = "z",
    ) -> bool:
        """判断点是否处于相机视锥内。"""
        cx, cy, cz, qw, qx, qy, qz = map(float, cam_pose)
        cam_pos = _np.array([cx, cy, cz], dtype=_np.float64)
        R_wc = self._quat_to_rotmat(qw, qx, qy, qz)  # world-from-camera
        R_cw = R_wc.T  # camera-from-world
        v = point_world - cam_pos
        v_cam = R_cw @ v  # point in camera coordinates

        # Choose axes according to forward
        if forward_axis.lower().startswith("x"):
            f_idx, h_idx, v_idx = 0, 1, 2
        else:  # "z" default
            f_idx, h_idx, v_idx = 2, 0, 1

        depth = v_cam[f_idx]
        # Only require point to be in front of the camera; ignore near/far planes
        if depth <= 0.0:
            return False

        half_x = math.radians(fov_x_deg) * 0.5
        # Derive vertical half FOV from aspect = width/height
        half_y = math.atan(math.tan(half_x) / float(aspect))

        # Projected half-width/height at current depth
        max_h = math.tan(half_x) * depth
        max_v = math.tan(half_y) * depth
        return (abs(v_cam[h_idx]) <= max_h) and (abs(v_cam[v_idx]) <= max_v)

    def _update_goal_inview(self):
        """根据当前相机位姿与目标位置，实时更新目标是否在视野内。"""
        # Default: no goals or no pose => all False
        self.goal_inview = {name: False for name in self.cur_goal_places.keys()}
        cam_pose = self.cur_agent_states.get("cam_pose_w")
        if not cam_pose or len(cam_pose) != 7:
            return
        
        cam_pos = _np.array(cam_pose[:3])
        for name, pt in self.cur_goal_places.items():
            point = _np.array(pt[:3])
            
            # 先进行距离阀值判断
            dist = float(_np.linalg.norm(point - cam_pos))
            if dist > float(self.near_dist):
                self.goal_inview[name] = False
                continue

            # 再根据当前目标点的3D坐标，判断是否在相机视锥内
            self.goal_inview[name] = self._point_in_cam_fov(
                cam_pose,
                point,
                fov_x_deg=self.cam_fov_x_deg,
                aspect=self.cam_aspect,
                forward_axis=self.cam_forward_axis,
            )

    # ==========================================
    # ROS 生命周期（关闭/清理）
    # ==========================================
    def close(self):
        """销毁节点与执行器线程；必要时关闭 rclpy。"""
        # 先清理发布器资源
        if self._ros_pub_timer is not None:
            self._ros_pub_timer.cancel()
            self._ros_pub_timer = None
        if self._ros_pub_executor is not None:
            self._ros_pub_executor.shutdown(wait=False)
            self._ros_pub_executor = None