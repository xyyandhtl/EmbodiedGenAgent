import torch
import os
import sys
import time
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

PROJECT_PATH = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_PATH)

from simulation.assets import SIMULATION_DIR, SIMULATION_DATA_DIR

CONFIG_PATH = os.path.join(SIMULATION_DIR, "settings.yaml")
CFG = OmegaConf.load(CONFIG_PATH)
# prepare output directories (capture/report)
CAPTURED_DIR = os.path.join(SIMULATION_DIR, "..", "app/captured")
REPORTS_DIR = os.path.join(SIMULATION_DIR, "..", "app/reports")

from isaaclab.app import AppLauncher

# --- Launch Omniverse App ---
simulation_app = AppLauncher(
    headless=CFG.sim_app.headless,
    enable_cameras=CFG.sim_app.record_video,  # if record_video=True, should set enable_cameras=True in simulation_app
    anti_aliasing=CFG.sim_app.anti_aliasing,
    width=CFG.sim_app.width,
    height=CFG.sim_app.height,
    hide_ui=CFG.sim_app.hide_ui,
).app

import carb
import omni.usd
# Use SimulaitonApp directly should set carb_settings
# carb_settings_iface = carb.settings.get_settings()
# carb_settings_iface.set("/persistent/isaac/asset_root/cloud", "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0")

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from simulation.assets.terrains.usd_scene import ASSET_DICT, add_collision_and_material
import simulation.mdp as mdp
from simulation.env.go2w_locomotion_env_cfg import LocomotionVelocityEnvCfg
from simulation.utils import (
    camera_follow,
    handle_enum_action,
    LabGo2WEnvHistoryWrapper,
    IsaacLabSensorHandler,
    SimpleCameraViewer,
    ZMQBridge,
)

def main():
    """Main function to run the Go2W locomotion demo and publish sensor data via ZMQ."""

    # --- 1. Get Environment Configs ---
    env_cfg = LocomotionVelocityEnvCfg()
    # update configs
    env_cfg.scene.terrain = ASSET_DICT[CFG.terrain]

    # if CFG.controller == "keyboard":
    # env_cfg.commands.base_velocity.debug_vis = False
    controller = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1] * 2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1] * 2,
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
    )
    print(controller)
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device)
    )

    # --- 2. Create Environment ---
    # The environment wrapper for Isaac Lab
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if CFG.sim_app.record_video else None)

    print(f"[INFO] joint names (IsaacLab Default): {env.scene['robot'].joint_names}")
    print(f"[INFO] joint_pos names (IsaacLab Actual): {env.action_manager.get_term('joint_pos')._joint_names}")
    if hasattr(env_cfg.actions, 'joint_vel'):
        print(f"[INFO] joint_vel names (IsaacLab Actual): {env.action_manager.get_term('joint_vel')._joint_names}")
    print("[INFO] env.observation_manager.active_terms[policy]: ", env.observation_manager.active_terms['policy'])
    print("[INFO] env.observation_manager.group_obs_term_dim: ", env.observation_manager.group_obs_term_dim)

    # wrap around environment for rsl-rl
    if CFG.policy == "eilab":
        env = LabGo2WEnvHistoryWrapper(env, history_len=CFG.observation_len)
    else:
        raise NotImplementedError(f"Policy '{CFG.policy}' not implemented.")

    stage = omni.usd.get_context().get_stage()
    if CFG.add_physics:
        terrain_prim = stage.GetPrimAtPath("/World/Terrain")
        add_collision_and_material(terrain_prim, static_friction=0.8, dynamic_friction=0.6)

    # --- 3. Initialize Sensor Handlers and ZMQ Bridge ---
    sensor_handler = IsaacLabSensorHandler(env, camera_name="rgbd_camera")
    print(f"[INFO] SensorHandler: {sensor_handler}")

    # One bridge for pub/sub
    zmq = ZMQBridge(pub_port=5555, sub_port=5556)

    # --- Control State ---
    # Default to "cmd_vel" mode with zero velocity. Mode is switched by incoming ROS commands.
    # control_mode = "cmd_vel"  # cmd_vel / goal_point
    latest_cmd_vel = (0.0, 0.0)  # tuple (lin_x, ang_z)
    goal_position = None
    marks = []

    # [DEBUG] Use a simple class to test camera data
    camera_viewer = SimpleCameraViewer()

    # --- 4. Load Policy ---
    # Path to the pre-trained low-level locomotion policy
    policy_ckpt_path = os.path.join(SIMULATION_DATA_DIR, "ckpts/go2w/blind/policy_roughRecover.jit")
    if not check_file_path(policy_ckpt_path):
        raise FileNotFoundError(f"Checkpoint file '{policy_ckpt_path}' not found. Please place your trained policy model (.jit file) in the 'ckpts' directory.")
    file_bytes = read_file(policy_ckpt_path)
    policy = torch.jit.load(file_bytes).to(CFG.policy_device).eval()

    # --- 5. Run Simulation Loop (reordered) ---
    policy_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    obs, _ = env.reset()
    print(f"[INFO] init-observation shape: {obs.shape}")
    
    frame_count = 0
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # 1) Capture a fresh snapshot and use it immediately to minimize desync.
            rgb_tensor, depth_tensor, pose_tuple = sensor_handler.capture_frame()

            # Poll commands
            zmq.poll()
            # 2) Execute control logic based on the current mode.
            #    The keyboard controller is always running in the background, but its output
            #    will be overwritten by the logic below if ROS commands are being received,
            #    cmd_vel has higher priority than nav_pose if both are received.
            if zmq.latest_cmd_vel is not None:
                latest_cmd_vel = zmq.latest_cmd_vel
                print(f"[CMD] Received cmd_vel -> lin_x={latest_cmd_vel[0]}, ang_z={latest_cmd_vel[1]}")
                lin_vel_x, ang_vel_z = latest_cmd_vel[0], latest_cmd_vel[1]
                velocity_command = torch.tensor([[lin_vel_x, 0.0, ang_vel_z]], device=CFG.policy_device)
                obs = mdp.update_observation_with_velocity_command(env, obs, velocity_command)
            elif zmq.nav_pose is not None:
                pos_np, quat_np = zmq.nav_pose
                goal_position = torch.tensor([float(pos_np[0]), float(pos_np[1]), float(pos_np[2])], device=CFG.policy_device)
                print(f"[CMD] Received nav_pose -> new goal_position={goal_position.cpu().numpy()}")
                lin_vel_x, ang_vel_z = mdp.compute_velocity_with_goalPoint(env, goal_position)
                velocity_command = torch.tensor([[lin_vel_x, 0.0, ang_vel_z]], device=CFG.policy_device)
                # Update observation with the new velocity command
                obs = mdp.update_observation_with_velocity_command(env, obs, velocity_command)
            # 3) Handle enum actions (capture/mark/report)
            if zmq.enum_cmd is not None:
                handle_enum_action(
                    enum_cmd=zmq.enum_cmd,
                    rgb_tensor=rgb_tensor,
                    pose_tuple=pose_tuple,
                    stage=stage,
                    captured_dir=CAPTURED_DIR,
                    reports_dir=REPORTS_DIR,
                    marks=marks,
                )            
  
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            camera_follow(env, camera_offset_=(-2.0, 0.0, 0.5))

            # --- 发布 sensor 数据（保持原逻辑） ---
            data_to_send = {}
            if rgb_tensor is not None:
                data_to_send['rgb'] = rgb_tensor[0, :, :, :3].cpu().numpy().astype(np.uint8)
            if depth_tensor is not None:
                depth_data = (depth_tensor[0] * 1000).cpu().numpy()
                depth_data[depth_data > 65535] = 0
                data_to_send['depth'] = depth_data.astype(np.uint16)
            if pose_tuple is not None:
                data_to_send['pose'] = (pose_tuple[0][0].cpu().numpy(), pose_tuple[1][0].cpu().numpy())

            if data_to_send:
                zmq.publish_data(data_to_send)
            camera_viewer.process_frame(data_to_send)

            # Time delay for real-time evaluation
            elapsed_time = time.time() - start_time
            sleep_time = policy_step_dt - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Print policy step time info periodically to avoid spamming
        # frame_count += 1
        # if frame_count % 50 == 1:
        #     actual_loop_time = time.time() - start_time
        #     rtf = min(1.0, policy_step_dt / elapsed_time)
        #     print(f"[INFO] Policy Step time: {actual_loop_time * 1000:.2f}ms, Real Time Factor: {rtf:.2f}", flush=True)

    # --- 6. Cleanup ---
    print("Simulation finished.")
    zmq.close()
    simulation_app.close()


if __name__ == "__main__":
    main()