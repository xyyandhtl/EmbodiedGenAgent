import numpy as np

class SimpleCameraViewer:
    """A simple, single-threaded class to process and display camera data."""

    def __init__(self):
        self.frame_count = 0

    def process_frame(self, sensor_data: dict):
        """Receives sensor data, processes it, and prints information."""
        self.frame_count += 1

        # Print info every 30 frames to avoid spamming the console
        if self.frame_count % 30 == 0:
            rgb_full = sensor_data.get("rgb")
            depth_full = sensor_data.get("depth")
            pose_full = sensor_data.get("pose")

            if rgb_full is None or depth_full is None or pose_full is None:
                print("\n[Viewer] Incomplete sensor data received.")
                return

            # Extract data for the first environment
            rgb = rgb_full[0]
            depth = depth_full[0]
            cam_pos_w, cam_quat_w_ros = pose_full
            cam_pos = cam_pos_w[0]
            cam_quat = cam_quat_w_ros[0]

            depth_np = depth.cpu().numpy()
            valid_depth = depth_np[np.isfinite(depth_np)]
            depth_90p = np.percentile(valid_depth, 90) if len(valid_depth) > 0 else -1.0

            pos_np = cam_pos.cpu().numpy().flatten()
            quat_np = cam_quat.cpu().numpy().flatten()

            log_msg = (
                f"\n[Viewer] Frame: {self.frame_count:<5} | "
                f"RGB: {tuple(rgb.shape)} | "
                f"Depth 90%: {depth_90p:.2f}m | "
                f"Pos: [x={pos_np[0]:.2f}, y={pos_np[1]:.2f}, z={pos_np[2]:.2f}] | "
                f"Quat: [x={quat_np[0]:.2f}, y={quat_np[1]:.2f}, z={quat_np[2]:.2f}, w={quat_np[3]:.2f}]"
            )
            print(log_msg, flush=True)
