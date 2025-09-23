import math
import torch

import isaaclab.utils.math as math_utils
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors.camera import Camera


def camera_follow(
        env: ManagerBasedRLEnv,
        camera_offset_: tuple[float, float, float] = (-2.0, -2.0, 1.0),
        smoothing_window: int = 50
):
    """
    Make the camera track the robot's movement.

    Args:
        env: Isaac Lab environment instance.
        camera_offset_: Camera offset relative to the robot (x, y, z).
        smoothing_window: Smoothing window size to reduce camera jitter.
    """
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []

    asset = env.unwrapped.scene["robot"]
    robot_pos = asset.data.root_pos_w[0]
    robot_quat = asset.data.root_quat_w[0]

    camera_offset_tensor = torch.tensor(camera_offset_, dtype=torch.float32, device=env.device)

    camera_pos = math_utils.transform_points(
        camera_offset_tensor.unsqueeze(0),
        pos=robot_pos.unsqueeze(0),
        quat=robot_quat.unsqueeze(0)
    ).squeeze(0)
    camera_pos[2] = torch.clamp(camera_pos[2], min=0.1)

    camera_follow.smooth_camera_positions.append(camera_pos)

    if len(camera_follow.smooth_camera_positions) > smoothing_window:
        camera_follow.smooth_camera_positions.pop(0)

    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)

    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(),
        lookat=robot_pos.cpu().numpy()
    )


def compute_cam_cfg(W=640, H=480, fov_deg_x=90.0):
    # rgbd_camera settings
    fov_deg_y = 2 * math.degrees(math.atan((H / W) * math.tan(math.radians(fov_deg_x) / 2.0)))

    fx = W / (2.0 * math.tan(math.radians(fov_deg_x) / 2.0))
    fy = H / (2.0 * math.tan(math.radians(fov_deg_y) / 2.0))
    cx, cy = W / 2.0, H / 2.0
    intrinsic_matrix = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    cam_cfg = PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=intrinsic_matrix,
        width=W,
        height=H,
        clipping_range=(0.6, 30.0),  # 近、远平面，m
        focal_length=12,  # 物理透视焦距，cm
        f_stop=0.0,  # 光圈（F值），用于模拟景深
    )
    # PinholeCameraCfg 中的其他参数
    # - horizontal_aperture、vertical_aperture 感光面尺寸，cm，在 USD/Omniverse 中是实际传感器的感光面尺寸的 10 倍
    # - focus_distance 焦平面距离，m

    print(f"[INFO] Computed fx={fx:.2f}, fy={fy:.2f}, HFOV={fov_deg_x:.2f} deg, VFOV={fov_deg_y:.2f} deg")
    print(f"[INFO] Sensor width={cam_cfg.horizontal_aperture} cm, height={cam_cfg.vertical_aperture} cm")
    print(f"[INFO] Physical focal length ≈ {cam_cfg.focal_length:.2f} cm")
    return cam_cfg
