import rclpy
from rclpy.node import Node
import threading
import time

from EG_agent.vlmap.vlmap_nav_ros2 import VLMapNavROS2

def main():
    """
    Tests VLMap navigation by periodically polling the get_nav_path() method.
    """
    rclpy.init()

    # 1. Instantiate the VLMapNavROS2 node
    vlmap_node = VLMapNavROS2()
    vlmap_node.get_logger().warning("[test_nav_polling] VLMapNavROS2 node created for polling test.")

    # 2. Run the node's spin logic in a background thread
    # This is crucial for the node to process data from the simulation
    spin_thread = threading.Thread(target=rclpy.spin, args=(vlmap_node,), daemon=True)
    spin_thread.start()
    vlmap_node.get_logger().warning("[test_nav_polling] Node spinning in a background thread.")

    # 3. Wait for initialization and initial map building
    initial_wait_time = 20.0
    vlmap_node.get_logger().warning(f"[test_nav_polling] Waiting for {initial_wait_time} seconds for initialization and map building...")
    time.sleep(initial_wait_time)
    vlmap_node.get_logger().warning("[test_nav_polling] Initial wait finished. Starting to poll for navigation path.")

    # 4. Periodically poll the get_nav_path() method
    polling_interval = 5.0 # seconds
    goal_query = "a table"
    
    try:
        while rclpy.ok():
            vlmap_node.get_logger().warning(f"[test_nav_polling] Attempting to call get_nav_path() for: '{goal_query}'")
            
            # This is a blocking call, but the node is spinning in the background
            nav_path = vlmap_node.get_nav_path(goal_query)

            if nav_path is not None and len(nav_path) > 0:
                print("\n--- [test_nav_polling] get_nav_path Test SUCCESS ---")
                print(f"[test_nav_polling] Successfully received navigation path with {len(nav_path)} waypoints.")
                print(f"[test_nav_polling] Data type of navigation path: {type(nav_path)}")
                if nav_path:
                    print(f"[test_nav_polling] Data type of a path element: {type(nav_path[0])}")
                    print(f"[test_nav_polling] Example path element: {nav_path[0]}")
                print("--- [test_nav_polling] End of get_nav_path Test ---\n")

                # --- Test get_cmd_vel ---
                print("--- [test_nav_polling] Starting get_cmd_vel Test ---")
                cmd_vel_sequence = vlmap_node.get_cmd_vel(nav_path)
                if cmd_vel_sequence:
                    print(f"[test_nav_polling] Successfully generated command velocity sequence with {len(cmd_vel_sequence)} commands.")
                    print(f"[test_nav_polling] Data type of sequence: {type(cmd_vel_sequence)}")
                    print(f"[test_nav_polling] Data type of a command element: {type(cmd_vel_sequence[0])}")
                    print(f"[test_nav_polling] Example command element (vx, vy, wz): {cmd_vel_sequence[0]}")
                    print(f"[test_nav_polling] Note: The command is a 3-element tuple (vx, vy, wz), where vy is expected to be 0.")
                else:
                    print("[test_nav_polling] get_cmd_vel returned an empty or None sequence.")
                print("--- [test_nav_polling] End of get_cmd_vel Test ---\n")

                break # Exit the loop on success
            else:
                vlmap_node.get_logger().warning(f"Failed to get path. Retrying in {polling_interval} seconds...")
                time.sleep(polling_interval)

    except KeyboardInterrupt:
        vlmap_node.get_logger().warning("Keyboard interrupt received.")
    except Exception as e:
        vlmap_node.get_logger().error(f"An unexpected error occurred: {e}")
    finally:
        # 5. Shutdown
        print("Test finished. Shutting down.")
        rclpy.shutdown()
        # Wait for the spin thread to finish
        spin_thread.join(timeout=2)

if __name__ == '__main__':
    main()