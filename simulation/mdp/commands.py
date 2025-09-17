import math
import torch
from torch import Tensor
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv


def compute_velocity_with_goalPoint(
        env: ManagerBasedRLEnv,
        goal_position: Tensor,
        kp: float = 0.25,
        max_lin_vel: float = 1.0,  # m/s
        max_ang_vel: float = 0.5,  # rad/s
        yaw_error_threshold: float = 0.3,  # rad
        goal_reached_threshold: float = 0.2,  # m
)->tuple[Tensor|float, Tensor|float]:
    # Get current robot state from the environment
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]  # qw, qx, qy, qz
    # Convert quaternion to yaw angle for heading
    qw, qx, qy, qz = robot_quat[0], robot_quat[1], robot_quat[2], robot_quat[3]
    current_yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    # Calculate distance and angle to the goal
    dist_to_goal = torch.linalg.norm(goal_position - robot_pos)  # Now 3D distance
    goal_yaw = torch.atan2(goal_position[1] - robot_pos[1], goal_position[0] - robot_pos[0])  # Still 2D heading

    # Calculate heading error, wrapped to the range [-pi, pi]
    yaw_error = goal_yaw - current_yaw
    yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))

    # Decide velocity based on distance and heading error
    lin_vel_x = 0.0
    ang_vel_z = 0.0
    if dist_to_goal > goal_reached_threshold:
        # Calculate angular velocity
        ang_vel_z = kp * yaw_error
        ang_vel_z = torch.clamp(ang_vel_z, -max_ang_vel, max_ang_vel)

        # Calculate linear velocity (decoupling)
        if torch.abs(yaw_error) > yaw_error_threshold:
            # Reduce linear velocity significantly during sharp turns
            lin_vel_x = max_lin_vel * (1.0 - (torch.abs(yaw_error) - yaw_error_threshold) / (math.pi - yaw_error_threshold))
            lin_vel_x = torch.clamp(lin_vel_x, 0.0, max_lin_vel)  # Ensure it doesn't go negative
        else:
            lin_vel_x = max_lin_vel

    return lin_vel_x, ang_vel_z


def update_observation_with_velocity_command(
        env: ManagerBasedRLEnv,
        obs: Tensor,
        velocity_command: Tensor,
):
    num_envs = obs.shape[0]
    history_len = env.history_len
    base_obs_dim = env.base_obs_dim

    obs_reshaped = obs.view(num_envs, history_len, base_obs_dim)
    obs_reshaped[:, 0, 6:9] = velocity_command
    obs = obs_reshaped.view(num_envs, -1)
    return obs