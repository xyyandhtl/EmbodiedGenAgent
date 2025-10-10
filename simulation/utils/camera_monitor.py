import numpy as np

class CamDataMonitor:
    """A simple, single-threaded class to process and display camera data."""

    def __init__(self):
        self.frame_count = 0
        self.monitor_camera = False
        self.monitor_pose = True

    def process_frame(self, sensor_data: dict):
        """Receives sensor data, processes it, and prints information."""
        self.frame_count += 1

        # Print info every 30 frames to avoid spamming the console
        if self.monitor_camera and self.frame_count % 50 == 0:
            rgb_np = sensor_data.get("rgb")
            depth_np = sensor_data.get("depth")
            pose_wxyz_np = sensor_data.get("pose")
            pose_agent_wxyz_np = sensor_data.get("pose_agent")

            if rgb_np is None or depth_np is None or pose_wxyz_np is None:
                print("\n[CameraViewer] Incomplete sensor data received.")
                return

            valid_depth = depth_np[np.isfinite(depth_np)]
            depth_90p = np.percentile(valid_depth, 90) if len(valid_depth) > 0 else -1.0

            pos_w, quat_wxyz_ros = pose_wxyz_np
            pos = pos_w.flatten()
            quat_wxyz = quat_wxyz_ros.flatten()

            pos_agent_w, quat_agent_wxyz_ros = pose_agent_wxyz_np
            pos_agent = pos_agent_w.flatten()
            quat_agent_wxyz = quat_agent_wxyz_ros.flatten()

            log_msg = (
                f"\n[Camera] Frame: {self.frame_count:<5} | "
                f"RGB: {tuple(rgb_np.shape)} | "
                f"Depth: {tuple(depth_np.shape)} | "
                f"Depth 90%: {depth_90p:.2f}mm == {depth_90p * 0.001:.2f}m| "
                f"Pos: [x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}] | "
                f"Quat: [w={quat_wxyz[0]:.3f}, x={quat_wxyz[1]:.3f}, y={quat_wxyz[2]:.3f}, z={quat_wxyz[3]:.3f}], "
                f"Pos_Agent: [x={pos_agent[0]:.3f}, y={pos_agent[1]:.3f}, z={pos_agent[2]:.3f}] | "
                f"Quat_Agent: [w={quat_agent_wxyz[0]:.3f}, x={quat_agent_wxyz[1]:.3f}, y={quat_agent_wxyz[2]:.3f}, z={quat_agent_wxyz[3]:.3f}]"
            )
            print(log_msg, flush=True)
        
        if self.monitor_pose and self.frame_count % 30 == 0:
            pose_agent_wxyz_np = sensor_data.get("pose_agent")
            pos_agent_w, quat_agent_wxyz_ros = pose_agent_wxyz_np
            pos_agent = pos_agent_w.flatten()
            log_msg = f"Pos_Agent: [x={pos_agent[0]:.3f}, y={pos_agent[1]:.3f}, z={pos_agent[2]:.3f}]"
            print(log_msg, flush=True)