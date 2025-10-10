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
    
    # The VLMap node will also serve as our publisher node
    vlmap_node = VLMapNavROS2()

    # Create publishers
    cmd_vel_pub = vlmap_node.create_publisher(Twist, '/cmd_vel', 10)
    nav_pose_pub = vlmap_node.create_publisher(PoseStamped, '/nav_pose', 10)
    enum_pub = vlmap_node.create_publisher(Int32, '/enum_command', 10)
    
    vlmap_node.get_logger().warning(f"VLMap node created and will also publish commands. Selected mode: {args.mode}")

    # --- Mode 1: Enum Command (Simple, no executor needed) ---
    if args.mode == 'enum':
        if args.cmd is None:
            vlmap_node.get_logger().error("Error: --cmd <integer> is required for 'enum' mode.")
        else:
            # Give publisher a moment to connect
            time.sleep(1.0)
            msg = Int32()
            msg.data = args.cmd
            enum_pub.publish(msg)
            vlmap_node.get_logger().warning(f"Published enum command: {args.cmd} to /enum_command")
            time.sleep(1.0) # Allow time for message to be sent
        
        vlmap_node.destroy_node()
        rclpy.shutdown()
        return

    # --- Modes 2 & 3: Navigation (cmd_vel or nav_pose) ---
    executor = MultiThreadedExecutor()
    executor.add_node(vlmap_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    
    try:
        # 1. Spin the VLMap node in the background. It handles both mapping and publishing.
        spin_thread.start()
        vlmap_node.get_logger().warning("VLMap node is spinning in a background thread.")

        # 2. Wait for VLMap to initialize and build a map
        initial_wait_time = 60
        vlmap_node.get_logger().warning(f"Waiting {initial_wait_time}s for map initialization...")
        time.sleep(initial_wait_time)

        # 3. Get navigation path (with exploration loop)
        vlmap_node.get_logger().warning(f"Starting search and exploration loop for goal: '{args.goal}'")
        nav_path = None
        while rclpy.ok() and not nav_path:
            vlmap_node.get_logger().info(f"Attempting to find path for '{args.goal}'...")
            # Use a shorter timeout for each attempt inside the loop
            path_attempt = vlmap_node.get_nav_path(args.goal, timeout_seconds=30)  # in ROS frame

            if path_attempt:
                nav_path = path_attempt
                vlmap_node.get_logger().warning(f"Goal found! Successfully received path with {len(nav_path)} waypoints.")
                # Stop any potential rotation before starting navigation
                cmd_vel_pub.publish(Twist())
                break  # Exit the while loop
            else:
                # If path is not found, explore by rotating
                vlmap_node.get_logger().warning(f"Goal '{args.goal}' not found in map. Exploring by rotating one full circle...")
                exploration_twist = Twist()
                exploration_twist.angular.z = 1.0  # rad/s
                
                # Rotate for 2π radians (one full circle) at 1.0 rad/s
                # This should take approximately 2π seconds (~6.28 seconds)
                rotation_duration = 2 * 3.14159  # 2π seconds for a full circle
                rotation_start_time = time.time()
                cmd_vel_pub.publish(exploration_twist)
                
                # Keep publishing rotation command until one full circle is completed
                while time.time() - rotation_start_time < rotation_duration:
                    cmd_vel_pub.publish(exploration_twist)
                    time.sleep(0.2)  # Small delay to prevent flooding

                # Stop rotating
                cmd_vel_pub.publish(Twist())
                vlmap_node.get_logger().info("Exploration rotation finished. Retrying in 5 seconds...")
                time.sleep(5) # Pause before the next attempt

        if not nav_path:
            vlmap_node.get_logger().error("Could not find navigation path after exploration. Shutting down.")
            return

        # 4. Execute action based on selected mode
        if args.mode == 'cmd_vel':
            # High-frequency closed-loop control performed by this script
            vlmap_node.get_logger().warning("Executing in 'cmd_vel' mode (high-frequency control).")
            
            control_frequency = 5 # Hz
            control_period = 1.0 / control_frequency

            for i, waypoint in enumerate(nav_path):
                vlmap_node.get_logger().warning(f"Moving to waypoint {i+1}/{len(nav_path)}: {waypoint}")
                count = 0
                while rclpy.ok():
                    # Calculate command and check if waypoint is reached
                    (vx, vy, wz), is_reached = vlmap_node.get_cmd_vel(waypoint)
                    count += 1
                    if count % 100 == 0:
                        vlmap_node.get_logger().warning(f"Moving to waypoint {i}, vx={vx}, vy={vy}, vz={wz}")

                    if is_reached:
                        vlmap_node.get_logger().warning(f"Waypoint {i+1} reached.")
                        break # Exit inner loop and move to next waypoint

                    # Publish the velocity command
                    twist = Twist()
                    twist.linear.x = float(vx)
                    twist.linear.y = float(vy)
                    twist.linear.z = 0.0
                    twist.angular.x = 0.0
                    twist.angular.y = 0.0
                    twist.angular.z = float(wz)
                    cmd_vel_pub.publish(twist)
                    
                    time.sleep(control_period)

            # Stop the robot after the last waypoint
            cmd_vel_pub.publish(Twist())
            vlmap_node.get_logger().warning("Final waypoint reached. Navigation finished.")

        elif args.mode == 'nav_pose':
            # Low-frequency goal posting; simulation handles the control loop
            vlmap_node.get_logger().warning("Executing in 'nav_pose' mode (low-frequency goal posting).")
            vlmap_node.get_logger().warning(f"Sending {len(nav_path)} waypoints one by one...")

            for i, waypoint in enumerate(nav_path):
                # Just publish the goal pose
                ps = PoseStamped()
                ps.header.stamp = vlmap_node.get_clock().now().to_msg()
                ps.header.frame_id = "map"
                ps.pose.position.x = float(waypoint[0])
                ps.pose.position.y = float(waypoint[1])
                ps.pose.position.z = float(waypoint[2])
                # Assuming a default orientation is acceptable for navigation goals
                ps.pose.orientation.w = 1.0 
                nav_pose_pub.publish(ps)
                vlmap_node.get_logger().warning(f"Sent waypoint {i+1}/{len(nav_path)}: {waypoint}. Handing over control to simulation.")
                
                # Wait for a significant time, assuming the simulation is now responsible for reaching the goal.
                # This is a simple way to sequence goals. A more robust implementation would use feedback.
                time.sleep(10.0) 
            
            vlmap_node.get_logger().warning("All waypoints have been sent.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # --- Shutdown ---
        print("Shutting down system.")
        if executor:
            executor.shutdown()
        # The node is part of the executor, which is already shut down.
        # No need to destroy node explicitly if it was added to an executor.
        rclpy.shutdown()

if __name__ == '__main__':
    main()
