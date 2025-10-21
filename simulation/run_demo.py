import torch
import os
import sys
import time
import numpy as np
from pathlib import Path
from dynaconf import Dynaconf

PROJECT_PATH = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_PATH)

from simulation.assets import SIMULATION_DIR, SIMULATION_DATA_DIR

CONFIG_PATH = os.path.join(SIMULATION_DIR, "settings.yaml")
CFG = Dynaconf(settings_files=[CONFIG_PATH], lowercase_read=True)
CAPTURED_DIR = os.path.join(SIMULATION_DIR, "..", "app/captured")
REPORTS_DIR = os.path.join(SIMULATION_DIR, "..", "app/reports")

from isaaclab.app import AppLauncher

# --- Launch Omniverse App ---
simulation_app = AppLauncher(
    headless=CFG.sim_app.headless,
    enable_cameras=CFG.sim_app.record_video,
    anti_aliasing=CFG.sim_app.anti_aliasing,
    width=CFG.sim_app.width,
    height=CFG.sim_app.height,
).app

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import check_file_path, read_file

from simulation.assets.terrains.usd_scene import ASSET_DICT, add_collision_and_material
import simulation.mdp as mdp
from simulation.env.go2w_locomotion_env_cfg import LocomotionVelocityEnvCfg
from simulation.utils import (
    setup_isaacsim_settings,
    camera_follow,
    get_current_stage,
    handle_enum_action,
    handle_mark_action,
    LabGo2WEnvHistoryWrapper,
    IsaacLabSensorHandler,
    CamDataMonitor,
    ZMQBridge,
)

def main():
    # --- 0. Setup Isaac Sim Settings ---
    if not CFG.sim_app.headless:
        setup_isaacsim_settings()
    # hide_workspace_windows()

    """Main function to run the Go2W locomotion demo and publish sensor data via ZMQ."""
    # --- 1. Get Environment Configs ---
    env_cfg = LocomotionVelocityEnvCfg()
    # update configs
    env_cfg.scene.terrain = ASSET_DICT[CFG.terrain]

    # if CFG.controller == "keyboard":
    # env_cfg.commands.base_velocity.debug_vis = False
    controller = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
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

    stage = get_current_stage()
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
    camera_monitor = CamDataMonitor()

    # --- 4. Load Policy ---
    # Path to the pre-trained low-level locomotion policy
    policy_ckpt_path = os.path.join(SIMULATION_DATA_DIR, "ckpts/go2w/blind", str(CFG.policy_name))
    if not check_file_path(policy_ckpt_path):
        raise FileNotFoundError(f"Checkpoint file '{policy_ckpt_path}' not found. Please place your trained policy model (.jit file) in the 'ckpts' directory.")
    file_bytes = read_file(policy_ckpt_path)
    policy = torch.jit.load(file_bytes).to(CFG.policy_device).eval()

    # --- 5. Run Simulation Loop (reordered) ---
    policy_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    obs, _ = env.reset()
    print(f"[INFO] init-observation shape: {obs.shape}")
    
    # 从CFG读取发布间隔（秒），未设置则沿用每个policy step发布
    sensor_publish_dt = float(CFG.publish_dt)
    print(f"[INFO] Sensor publish interval: {sensor_publish_dt:.3f}s")

    frame_count = 0
    # 让第一次循环就发布一次
    last_sensor_pub_ts = time.time() - sensor_publish_dt

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # Poll commands
            zmq.poll()
            # 1) Execute control logic based on the current mode.
            #    The keyboard controller is always running in the background, but its output
            #    will be overwritten by the logic below if ROS commands are being received,
            #    cmd_vel has higher priority than nav_pose if both are received.
            if zmq.latest_cmd_vel is not None:
                latest_cmd_vel = zmq.latest_cmd_vel
                # print(f"[CMD] Received cmd_vel -> lin_x={latest_cmd_vel[0]}, ang_z={latest_cmd_vel[1]}")
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
            # 2) Handle enum actions (capture/report)
            if zmq.enum_cmd is not None:
                handle_enum_action(
                    enum_cmd=zmq.enum_cmd,
                    rgb_tensor=sensor_handler.get_rgb_frame(),
                    pose_tuple=sensor_handler.get_camera_pose(),
                    stage=stage,
                    captured_dir=CAPTURED_DIR,
                    reports_dir=REPORTS_DIR,
                    marks=marks,
                )
            # 新增：处理独立的 mark 动作
            if zmq.mark_pos is not None:
                handle_mark_action(
                    mark_pos_or_none=zmq.mark_pos,
                    pose_tuple=sensor_handler.get_camera_pose(),
                    stage=stage,
                    marks=marks,
                )

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            if CFG.camera_follow:
                camera_follow(env, camera_offset_=(-2.0, 0.0, 0.5))

            # --- 发布 sensor 数据（按CFG间隔） ---
            now = time.time()
            if now - last_sensor_pub_ts >= sensor_publish_dt:
                # 仅在到达发布间隔时采集并发布（一次性同步到CPU，减小不同步）
                rgb_tensor, depth_tensor, pose_camera_tuple, pose_agent_tuple = sensor_handler.capture_frame(gpu_sync=True)
                data_to_send = {}
                if rgb_tensor is not None:
                    data_to_send['rgb'] = rgb_tensor[0, :, :, :3].to("cpu", non_blocking=False).numpy().astype(np.uint8)
                if depth_tensor is not None:
                    depth_data = (depth_tensor[0] * 1000).to("cpu", non_blocking=False).numpy()
                    depth_data[depth_data > 65535] = 0
                    data_to_send['depth'] = depth_data.astype(np.uint16)
                if pose_camera_tuple is not None:
                    data_to_send['pose'] = (pose_camera_tuple[0][0].cpu().numpy(), pose_camera_tuple[1][0].cpu().numpy())
                # [DEBUG]
                if pose_agent_tuple is not None:
                    data_to_send['pose_agent'] = (pose_agent_tuple[0][0].cpu().numpy(), pose_agent_tuple[1][0].cpu().numpy())
                # 可按需获取时间戳用于下游对齐：sensor_handler.get_timestamp()
                if data_to_send:
                    zmq.publish_data(data_to_send)

                if CFG.monitor_camera:
                    camera_monitor.process_frame(data_to_send)
                last_sensor_pub_ts = now

            # Time delay for real-time evaluation
            elapsed_time = time.time() - start_time
            sleep_time = policy_step_dt - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Print policy step time info periodically to avoid spamming
        frame_count += 1
        if frame_count % 50 == 1:
            actual_loop_time = time.time() - start_time
            rtf = min(1.0, policy_step_dt / elapsed_time)
            print(f"[INFO] Policy Step time: {actual_loop_time * 1000:.2f}ms, Real Time Factor: {rtf:.2f}", flush=True)

    # --- 6. Cleanup ---
    print("Simulation finished.")
    zmq.close()
    simulation_app.close()


if __name__ == "__main__":
    main()