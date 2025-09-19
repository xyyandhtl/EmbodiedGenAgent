from __future__ import annotations

import torch
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors.camera.camera import Camera


class IsaacLabSensorHandler:
    """
    A handler to interface Isaac Lab sensors with an external agent.

    This class retrieves data (RGB, depth, pose, intrinsics) from a specified
    camera in the Isaac Lab environment and uses a threading event to signal
    when a new frame is available.
    """

    def __init__(self, env: ManagerBasedRLEnv, camera_name: str = "rgbd_camera"):
        """
        Initializes the sensor handler.

        Args:
            env: The Isaac Lab ManagerBasedRLEnv instance.
            camera_name: The name of the camera entity in the scene.
        """
        self.env = env
        self.camera_name = camera_name
        self.camera: Camera = self.env.unwrapped.scene[self.camera_name]
        self.new_frame_event = threading.Event()

    def update(self):
        """Called by the simulation loop to signal a new frame is ready."""
        self.new_frame_event.set()

    def get_rgb_frame(self) -> torch.Tensor | None:
        """Returns the latest RGB frame."""
        if "rgb" in self.camera.data.output:
            return self.camera.data.output["rgb"]
        return None

    def get_depth_frame(self) -> torch.Tensor | None:
        """Returns the latest depth frame."""
        if "distance_to_image_plane" in self.camera.data.output:
            return self.camera.data.output["distance_to_image_plane"]
        return None

    def get_camera_pose(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Returns the camera's world pose as (position, quaternion).
            - pos_w: camera position in world frame (following ROS convention)
            - quat_w_ros: camera quaternion in world frame (w, x, y, z, following ROS convention)
        """
        if self.camera.data.pos_w is not None and self.camera.data.quat_w_ros is not None:
            return self.camera.data.pos_w, self.camera.data.quat_w_ros
        return None

    def get_intrinsics(self) -> torch.Tensor | None:
        """Returns the camera's intrinsic matrix."""
        if self.camera.data.intrinsic_matrices is not None:
            return self.camera.data.intrinsic_matrices
        return None

    def __str__(self) -> str:
        """Provides a descriptive string of the current sensor data."""
        log_info = "  "
        output = self.camera.data.output

        if "rgb" in output:
            rgb_data = output['rgb']
            log_info += f"rgb-shape: {tuple(rgb_data.shape)}"

        if "distance_to_image_plane" in output:
            depth_data = output['distance_to_image_plane']
            log_info += f", depth-shape: {tuple(depth_data.shape)}"

            valid_depth = depth_data[torch.isfinite(depth_data)]
            if len(valid_depth) > 0:
                depth_90_percentile = torch.quantile(valid_depth, 0.9).item()
                depth_min = valid_depth.min().item()
                log_info += f", 90% percentile: {depth_90_percentile:.2f} m, min: {depth_min:.2f} m"
            else:
                log_info += ", all depth values are invalid (inf or nan)"

        return log_info if len(log_info.strip()) > 0 else "No data available."