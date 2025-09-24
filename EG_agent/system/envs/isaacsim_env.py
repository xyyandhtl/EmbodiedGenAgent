from typing import Callable

from EG_agent.system.envs.base_env import BaseEnv
from EG_agent.system.module_path import AGENT_ENV_PATH

# --- ROS2 相关导入 ---
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Int32
import numpy as _np

class IsaacsimEnv(BaseEnv):
    agent_num = 1

    # launch simulator
    # simulator_path = f'{ROOT_PATH}/../simulators/virtualhome/windows/VirtualHome.exe'
    # simulator_path = f'{ROOT_PATH}/../simulators/virtualhome/linux_exec/linux_exec.v2.3.0.x86_64'

    behavior_lib_path = f"{AGENT_ENV_PATH}/embodied"

    def __init__(self):
        if not self.headless:
            self.launch_simulator()
        super().__init__()
        self.action_callbacks_dict = {}

        # Assume rclpy.init() was called elsewhere in the process.
        # Create node and publishers directly; let exceptions propagate if rclpy not initialized.
        self.ros_node = Node("isaacsim_env_node")
        self.cmd_vel_pub = self.ros_node.create_publisher(Twist, "/cmd_vel", 10)
        self.nav_pose_pub = self.ros_node.create_publisher(PoseStamped, "/nav_pose", 10)
        self.enum_pub = self.ros_node.create_publisher(Int32, "/enum_command", 10)

    def register_action_callbacks(self, type: str, fn: Callable):
        self.action_callbacks_dict[type] = fn

    def reset(self):
        raise NotImplementedError

    def task_finished(self):
        raise NotImplementedError

    def launch_simulator(self):
        # todo: maybe set ros2 topic names or register callbacks
        pass

    def load_scenario(self,scenario_id):
        # todo: maybe do nothing
        pass

    def run_action(self, action_type: str, action: tuple, verbose=False):
        """
        Strict action formats:
          - 'cmd_vel': list/tuple/ndarray of 3 floats -> [vx, vy, wz]
          - 'nav_pose': list/tuple/ndarray of 7 floats -> [x,y,z,qw,qx,qy,qz]
          - 'enum'/'enum_command': int or iterable of ints -> publish each Int32
        """
        # priority to registered callbacks
        if action_type in self.action_callbacks_dict:
            self.action_callbacks_dict[action_type](action)
            return

        # ensure node/publishers exist
        if self.ros_node is None:
            raise RuntimeError("ROS node not initialized; cannot publish actions.")

        # cmd_vel: expect exactly 3 elements [vx, vy, wz]
        if action_type in ("cmd_vel", "cmdvel", "cmd-vel"):
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
            return

        # nav_pose: expect exactly 7 elements [x,y,z,qw,qx,qy,qz]
        if action_type in ("nav_pose", "goal_pose", "pose"):
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
            return

        # enum: accept single int or iterable of ints; publish one Int32 per command
        if action_type in ("enum", "enum_command", "command_enum"):
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
            return

        # unknown action
        raise ValueError(f"Action type {action_type} not registered and not a known ROS-publishable command.")

    def close(self):
        # Destroy only this node; do NOT shutdown rclpy (managed externally)
        if hasattr(self, "ros_node") and self.ros_node is not None:
            try:
                self.ros_node.destroy_node()
            except Exception:
                pass
            self.ros_node = None

    def set_navigator(self, navigator):
        self.navigator = navigator

