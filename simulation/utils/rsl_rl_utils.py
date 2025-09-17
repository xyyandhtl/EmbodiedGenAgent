import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv


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