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
        # self.new_frame_event = threading.Event()  # remove event usage, kept for backward compatibility
        self.data_lock = threading.Lock()

        # Internal storage for an atomic snapshot of a frame (protected by data_lock)
        self._rgb: torch.Tensor | None = None
        self._depth: torch.Tensor | None = None
        self._pose: tuple[torch.Tensor, torch.Tensor] | None = None
        self._intrinsics: torch.Tensor | None = None

    def capture_frame(self):
        """
        Atomically snapshot camera outputs and pose and return them immediately as
        (rgb, depth, pose, intrinsics). Each returned tensor is a clone and safe
        to use outside the handler.
        """
        with self.data_lock:
            # Snapshot camera outputs if available
            output = self.camera.data.output
            rgb = output["rgb"].clone() if "rgb" in output else None
            depth = output["distance_to_image_plane"].clone() if "distance_to_image_plane" in output else None
            # intrinsics = self.camera.data.intrinsic_matrices.clone() if self.camera.data.intrinsic_matrices is not None else None

            # Snapshot camera pose based on the robot base, using same device as the asset
            asset = self.env.unwrapped.scene["robot"]
            base_pos_w = asset.data.root_pos_w
            base_quat_w = asset.data.root_quat_w

            offset_pos = torch.tensor(self.camera.cfg.offset.pos, device=asset.device).unsqueeze(0)
            offset_quat = torch.tensor(self.camera.cfg.offset.rot, device=asset.device).unsqueeze(0)

            cam_pos_w = base_pos_w + quat_apply(base_quat_w, offset_pos)
            cam_quat_w_world = quat_mul(base_quat_w, offset_quat)
            cam_quat_w_ros = convert_camera_frame_orientation_convention(cam_quat_w_world, origin="world", target="ros")

            pose = (cam_pos_w.clone(), cam_quat_w_ros.clone())

        return rgb, depth, pose

    def get_rgb_frame(self) -> torch.Tensor | None:
        """Returns the latest snapshot RGB frame (non-blocking)."""
        with self.data_lock:
            return self._rgb.clone() if self._rgb is not None else None

    def get_depth_frame(self) -> torch.Tensor | None:
        """Returns the latest snapshot depth frame (non-blocking)."""
        with self.data_lock:
            return self._depth.clone() if self._depth is not None else None

    def get_camera_pose(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Returns the latest snapshot camera pose (non-blocking)."""
        with self.data_lock:
            if self._pose is not None:
                return self._pose[0].clone(), self._pose[1].clone()
        return None

    def get_intrinsics(self) -> torch.Tensor | None:
        """Returns the latest snapshot intrinsics (non-blocking)."""
        with self.data_lock:
            return self._intrinsics.clone() if self._intrinsics is not None else None

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