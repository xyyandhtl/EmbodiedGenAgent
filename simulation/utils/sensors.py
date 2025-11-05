from __future__ import annotations

import torch
import threading
import time
from typing import TYPE_CHECKING
from dynaconf import Dynaconf

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors.camera.camera import Camera
    from simulation.assets.sensors.lidar_sensor import LidarSensor

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

    def __init__(self, env: ManagerBasedRLEnv, enable_depth: bool = True, enable_lidar: bool = False):
        """
        Initializes the sensor handler.

        Args:
            env: The Isaac Lab ManagerBasedRLEnv instance.
            camera_name: The name of the camera entity in the scene.
        """
        self.env = env
        self.enable_depth = enable_depth
        self.enable_lidar = enable_lidar
        self.camera: Camera = self.env.unwrapped.scene["rgbd_camera"]
        self.cam_offset_pos = torch.tensor(self.camera.cfg.offset.pos, device=self.env.device).unsqueeze(0)
        self.cam_offset_quat = torch.tensor(self.camera.cfg.offset.rot, device=self.env.device).unsqueeze(0)
        if self.enable_lidar:
            self.lidar: LidarSensor = self.env.unwrapped.scene["lidar_sensor"]
            self.lidar_offset_pos = torch.tensor(self.lidar.cfg.offset.pos, device=self.env.device).unsqueeze(0)
            self.lidar_offset_quat = torch.tensor(self.lidar.cfg.offset.rot, device=self.env.device).unsqueeze(0)
        self.data_lock = threading.Lock()

        # Internal storage for an atomic snapshot of a frame (protected by data_lock)
        self._rgb: torch.Tensor | None = None
        self._depth: torch.Tensor | None = None
        self._pose_camera: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pose_lidar: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pose_agent: tuple[torch.Tensor, torch.Tensor] | None = None
        self._intrinsics: torch.Tensor | None = None
        self._ts: float | None = None  # timestamp of the last atomic capture

    def capture_frame(self, gpu_sync: bool = False) -> tuple:
        """
        Atomically snapshot camera outputs and pose and return them immediately as
        (rgb, depth, pose). Optionally:
          - gpu_sync: synchronize CUDA to avoid async updates overlapping.
          - sync_to_cpu: move all tensors to CPU together to enforce a single sync point.
        """
        with self.data_lock:
            if gpu_sync:
                torch.cuda.synchronize()

            asset = self.env.unwrapped.scene["robot"]
            # base_pos_w = asset.data.root_pos_w
            # base_quat_w = asset.data.root_quat_w

            # 找到 base 的索引（可能叫 "base" 或 "base_link" 或 "trunk"）
            base_idx = asset.data.body_names.index("base")
            base_pos_w = asset.data.body_pos_w[:, base_idx]
            base_quat_w = asset.data.body_quat_w[:, base_idx]
            base_quat_ros = convert_camera_frame_orientation_convention(base_quat_w, origin="world", target="ros")

            # Snapshot camera outputs
            output = self.camera.data.output
            self._rgb = output["rgb"].clone() if "rgb" in output else None

            if self.enable_depth:
                self._depth  = output["distance_to_image_plane"].clone() if "distance_to_image_plane" in output else None
            if self.enable_lidar:
                self._lidar = self.lidar.data.pointcloud.clone()
                # self._lidar = self.lidar.data.ray_hits_w.clone()
                # compute lidar pose from base + offset
                lidar_pos_w = base_pos_w + quat_apply(base_quat_w, self.lidar_offset_pos)
                lidar_quat_w_world = quat_mul(base_quat_w, self.lidar_offset_quat)
                lidar_quat_w_ros = convert_camera_frame_orientation_convention(lidar_quat_w_world, origin="world", target="ros")
                # lidar_pos_w = self.lidar.data.pos_w[0].unsqueeze(0)
                # lidar_quat_w_world = self.lidar.data.quat_w[0].unsqueeze(0)
                # lidar_quat_w_ros = convert_camera_frame_orientation_convention(lidar_quat_w_world, origin="world", target="ros")
                self._pose_lidar = (lidar_pos_w.clone(), base_quat_w.clone())
                
            # Compute camera pose from base + offset
            cam_pos_w = base_pos_w + quat_apply(base_quat_w, self.cam_offset_pos)
            cam_quat_w_world = quat_mul(base_quat_w, self.cam_offset_quat)
            cam_quat_w_ros = convert_camera_frame_orientation_convention(cam_quat_w_world, origin="world", target="ros")
            # pos_cam = self.camera.data.pos_w[0].unsqueeze(0)
            # quat_cam_w_world = self.camera.data.quat_w_world[0].unsqueeze(0)
            # quat_cam_w_ros = convert_camera_frame_orientation_convention(quat_cam_w_world, origin="world", target="ros")

            # Store atomically
            self._pose_camera = (cam_pos_w.clone(), cam_quat_w_ros.clone())
            self._pose_agent = (base_pos_w.clone(), base_quat_ros.clone())
            self._ts = time.time()
        
        return self._rgb, self._depth, self._lidar, self._pose_camera, self._pose_lidar, self._pose_agent

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
            if self._pose_camera is not None:
                return self._pose_camera[0].clone(), self._pose_camera[1].clone()
        return None

    def get_agent_pose(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Returns the latest snapshot camera pose (non-blocking)."""
        with self.data_lock:
            if self._pose_agent is not None:
                return self._pose_agent[0].clone(), self._pose_agent[1].clone()
        return None

    def get_intrinsics(self) -> torch.Tensor | None:
        """Returns the latest snapshot intrinsics (non-blocking)."""
        with self.data_lock:
            return self._intrinsics.clone() if self._intrinsics is not None else None

    def get_lidar_pointcloud(self) -> torch.Tensor | None:
        """Returns the latest snapshot LiDAR point cloud (non-blocking)."""
        with self.data_lock:
            return self._lidar.clone() if self._lidar is not None else None

    def get_timestamp(self) -> float | None:
        """Returns the capture timestamp of the latest snapshot."""
        with self.data_lock:
            return self._ts

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