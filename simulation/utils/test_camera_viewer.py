import numpy as np

class SimpleCameraViewer:
    """A simple, single-threaded class to process and display camera data."""

    def __init__(self):
        self.frame_count = 0

    def process_frame(self, sensor_data: dict):
        """Receives sensor data, processes it, and prints information."""
        self.frame_count += 1

        # Print info every 30 frames to avoid spamming the console
        if self.frame_count % 50 == 0:
            rgb_np = sensor_data.get("rgb")
            depth_np = sensor_data.get("depth")
            pose_np = sensor_data.get("pose")

            if rgb_np is None or depth_np is None or pose_np is None:
                print("\n[CameraViewer] Incomplete sensor data received.")
                return

            valid_depth = depth_np[np.isfinite(depth_np)]
            depth_90p = np.percentile(valid_depth, 90) if len(valid_depth) > 0 else -1.0

            pos_w_np, quat_w_ros_np = pose_np
            pos_np = pos_w_np.flatten()
            quat_np = quat_w_ros_np.flatten()

            log_msg = (
                f"\n[Camera] Frame: {self.frame_count:<5} | "
                f"RGB: {tuple(rgb_np.shape)} | "
                f"Depth: {tuple(depth_np.shape)} | "
                f"Depth 90%: {depth_90p:.2f}mm == {depth_90p * 0.001:.2f}m| "
                f"Pos: [x={pos_np[0]:.3f}, y={pos_np[1]:.3f}, z={pos_np[2]:.3f}] | "
                f"Quat: [w={quat_np[0]:.3f}, x={quat_np[1]:.3f}, y={quat_np[2]:.3f}, z={quat_np[3]:.3f}]"
            )
            print(log_msg, flush=True)
