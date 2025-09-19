import time
import threading
import numpy as np

from simulation.utils.sensors import IsaacLabSensorHandler


class SimpleCameraViewer:
    """A simple class to test the camera data stream from Isaac Lab."""

    def __init__(self):
        self.frame_count = 0
        self.sensor = None
        self.event = None

    def connect_to_simulation(self, handler: IsaacLabSensorHandler):
        """Connects to the Isaac Lab sensor handler."""
        self.sensor = handler
        self.event = handler.new_frame_event
        print("[SimpleCameraViewer] Successfully connected to sensor handler.")

    def start_viewing_stream(self):
        """The main loop to wait for and process data frames."""
        while True:
            if self.event is None:
                time.sleep(0.1)
                continue

            self.event.wait()  # Wait for a new frame signal

            # 1. Get data from the sensor handler
            rgb = self.sensor.get_rgb_frame()
            depth = self.sensor.get_depth_frame()
            pose = self.sensor.get_camera_pose()

            # 2. Check if data is valid
            if rgb is None or depth is None or pose is None:
                self.event.clear()
                continue

            self.frame_count += 1

            # 3. Print info every 60 frames to avoid spamming the console
            if self.frame_count % 60 == 0:
                cam_pos, _ = pose
                pos_np = cam_pos.cpu().numpy().flatten()
                depth_np = depth.cpu().numpy()
                valid_depth = depth_np[np.isfinite(depth_np)]
                depth_90p = np.percentile(valid_depth, 90) if len(valid_depth) > 0 else -1.0

                log_msg = (
                    f"\r[Viewer] Frame: {self.frame_count:<5} | "
                    f"RGB: {tuple(rgb.shape)} | "
                    f"Depth 90%: {depth_90p:.2f}m | "
                    f"Pose: [x={pos_np[0]:.2f}, y={pos_np[1]:.2f}, z={pos_np[2]:.2f}]"
                )
                print(log_msg, end="", flush=True)

            self.event.clear()  # Clear the event to wait for the next signal
