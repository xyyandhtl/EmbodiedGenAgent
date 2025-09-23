from __future__ import annotations

import torch
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors.camera.camera import Camera

from isaaclab.utils.math import (
    quat_mul,
    quat_apply,
    convert_camera_frame_orientation_convention
)


class IsaacLabSensorHandler:
    """
    A thread-safe handler to interface Isaac Lab sensors with an external agent.

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
        self.data_lock = threading.Lock()

    def update(self):
        """Called by the simulation loop to signal a new frame is ready."""
        self.new_frame_event.set()

    def get_rgb_frame(self) -> torch.Tensor | None:
        """Returns a thread-safe clone of the latest RGB frame."""
        with self.data_lock:
            if "rgb" in self.camera.data.output:
                return self.camera.data.output["rgb"].clone()
        return None

    def get_depth_frame(self) -> torch.Tensor | None:
        """Returns a thread-safe clone of the latest depth frame."""
        with self.data_lock:
            if "distance_to_image_plane" in self.camera.data.output:
                return self.camera.data.output["distance_to_image_plane"].clone()
        return None

    def get_camera_pose(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Computes and returns a clone of the camera's world pose as (position, quaternion)."""
        with self.data_lock:
            # Get the robot articulation object from the environment
            asset = self.env.unwrapped.scene["robot"]

            # Get the world pose of the robot's root body, which is guaranteed to be up-to-date
            base_pos_w = asset.data.root_pos_w
            base_quat_w = asset.data.root_quat_w

            # Get the camera's static offset from its configuration
            offset_pos = torch.tensor(self.camera.cfg.offset.pos, device=asset.device).unsqueeze(0)
            offset_quat = torch.tensor(self.camera.cfg.offset.rot, device=asset.device).unsqueeze(0)

            # Manually compute the camera's world pose
            # 1. Rotate the position offset by the base orientation and add to the base position
            cam_pos_w = base_pos_w + quat_apply(base_quat_w, offset_pos)
            # 2. Multiply the quaternions to get the final orientation
            cam_quat_w_world = quat_mul(base_quat_w, offset_quat)

            # Convert the world-frame quaternion to ROS convention for compatibility
            cam_quat_w_ros = convert_camera_frame_orientation_convention(cam_quat_w_world, origin="world", target="ros")

            return cam_pos_w.clone(), cam_quat_w_ros.clone()
        return None

    def get_intrinsics(self) -> torch.Tensor | None:
        """Returns a thread-safe clone of the camera's intrinsic matrix."""
        with self.data_lock:
            if self.camera.data.intrinsic_matrices is not None:
                return self.camera.data.intrinsic_matrices.clone()
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