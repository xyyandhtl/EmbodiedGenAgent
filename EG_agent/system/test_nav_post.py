import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import argparse

from EG_agent.vlmap.vlmap_nav_ros2 import VLMapNavROS2
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Int32

def main():
    """
    Main entry point for running the full agent system, with different control modes.
    """
    parser = argparse.ArgumentParser(description="Run the EmbodiedGenAgent system with different control modes.")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['cmd_vel', 'nav_pose', 'enum'],
        default='cmd_vel',
        help="The control mode to use: 'cmd_vel' to send a velocity sequence, 'nav_pose' to send waypoints, 'enum' to send a command code."
    )
    parser.add_argument(
        '--goal',
        type=str,
        default='chair',
        help="The natural language goal for navigation (used in cmd_vel and nav_pose modes)."
    )
    parser.add_argument(
        '--cmd',
        type=int,
        help="The integer command to send in 'enum' mode (e.g., 0 for photo, 1 for mark, 2 for report)."
    )
    args = parser.parse_args()

    rclpy.init()
    
    # --- Publisher Node Setup ---
    # This node is responsible for sending commands to the simulation.
    publisher_node = Node('agent_system_publisher')
    cmd_vel_pub = publisher_node.create_publisher(Twist, '/cmd_vel', 10)
    nav_pose_pub = publisher_node.create_publisher(PoseStamped, '/nav_pose', 10)
    enum_pub = publisher_node.create_publisher(Int32, '/enum_command', 10)
    publisher_node.get_logger().warning(f"Publisher node created. Selected mode: {args.mode}")

    # --- Mode 1: Enum Command ---
    # This mode is simple and doesn't require the VLMap system.
    if args.mode == 'enum':
        if args.cmd is None:
            publisher_node.get_logger().error("Error: --cmd <integer> is required for 'enum' mode.")
        else:
            msg = Int32()
            msg.data = args.cmd
            enum_pub.publish(msg)
            publisher_node.get_logger().warning(f"Published enum command: {args.cmd} to /enum_command")
            time.sleep(1) # Allow time for publish
        rclpy.shutdown()
        return

    # --- Modes 2 & 3: Navigation (cmd_vel or nav_pose) ---
    # These modes require the full VLMap system.
    vlmap_node = None
    executor = None
    try:
        # 1. Initialize VLMap node and run it in the background
        vlmap_node = VLMapNavROS2()

        executor = MultiThreadedExecutor()
        executor.add_node(vlmap_node)
        executor.add_node(publisher_node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        vlmap_node.get_logger().warning("VLMap and Publisher nodes are spinning in a background thread.")

        # 2. Wait for VLMap to initialize and build a map
        initial_wait_time = 20
        vlmap_node.get_logger().warning(f"Waiting {initial_wait_time}s for map initialization...")
        time.sleep(initial_wait_time)

        # 3. Get navigation path
        vlmap_node.get_logger().warning(f"Requesting navigation path for goal: '{args.goal}'")
        nav_path = vlmap_node.get_nav_path(args.goal)

        if not nav_path:
            vlmap_node.get_logger().error("Failed to get navigation path. Shutting down.")
            return

        vlmap_node.get_logger().warning(f"Successfully received path with {len(nav_path)} waypoints.")

        # 4. Execute action based on selected mode
        if args.mode == 'cmd_vel':
            publisher_node.get_logger().warning("Executing in 'cmd_vel' mode.")
            cmd_vel_sequence = vlmap_node.get_cmd_vel(nav_path)
            if not cmd_vel_sequence:
                publisher_node.get_logger().error("Failed to generate velocity commands.")
                return
            
            publisher_node.get_logger().warning(f"Sending {len(cmd_vel_sequence)} velocity commands...")
            for vx, vy, wz in cmd_vel_sequence:
                twist = Twist()
                twist.linear.x = float(vx)
                twist.linear.y = float(vy)
                twist.linear.z = 0.0
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = float(wz)
                cmd_vel_pub.publish(twist)
                time.sleep(0.02) # Send at 50Hz
            publisher_node.get_logger().warning("Velocity sequence execution finished.")

        elif args.mode == 'nav_pose':
            publisher_node.get_logger().warning("Executing in 'nav_pose' mode.")
            publisher_node.get_logger().warning(f"Sending {len(nav_path)} waypoints...")
            for i, waypoint in enumerate(nav_path):
                ps = PoseStamped()
                ps.header.stamp = publisher_node.get_clock().now().to_msg()
                ps.header.frame_id = "map"
                ps.pose.position.x = float(waypoint[0])
                ps.pose.position.y = float(waypoint[1])
                ps.pose.position.z = float(waypoint[2])
                ps.pose.orientation.w = 1.0 # Default orientation
                nav_pose_pub.publish(ps)
                publisher_node.get_logger().warning(f"Sent waypoint {i+1}/{len(nav_path)}: {waypoint}")
                time.sleep(3.0) # Wait 3 seconds for the robot to approach the waypoint
            publisher_node.get_logger().warning("Waypoint sequence execution finished.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # --- Shutdown ---
        print("Shutting down system.")
        if executor:
            executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
