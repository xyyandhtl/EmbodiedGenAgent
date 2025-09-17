import argparse
import numpy as np
import torch
import os
import sys
import hydra
import time

from pathlib import Path
# Add the project root to the python path to allow for absolute imports
PROJECT_PATH = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_PATH)

from simulation.assets import SIMULATION_DIR, SIMULATION_DATA_DIR
from isaacsim import SimulationApp
import carb


@hydra.main(config_path=SIMULATION_DIR, config_name="settings", version_base=None)
def main(cfg):
    """Main function to run the Go2W locomotion demo."""

    # --- 1. Launch Omniverse App ---
    simulation_app = SimulationApp({
        "headless": cfg.sim_app.headless,
        "anti_aliasing": cfg.sim_app.anti_aliasing,
        "width": cfg.sim_app.width,
        "height": cfg.sim_app.height,
        "hide_ui": cfg.sim_app.hide_ui,
    })
    # for not setting
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set("/persistent/isaac/asset_root/cloud",
                           "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0")

    from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.utils.assets import check_file_path, read_file
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import simulation.mdp as mdp
    from simulation.env.go2w_locomotion_env_cfg import LocomotionVelocityEnvCfg
    from simulation.utils import (
        LabGo2WEnvHistoryWrapper,
        camera_follow,
    )

    # --- 2. Get Environment Configs ---
    env_cfg = LocomotionVelocityEnvCfg()

    if cfg.controller == "keyboard":
        # env_cfg.commands.base_velocity.debug_vis = False
        controller = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
                v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
                omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
            )
        )
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device)
        )

    # --- 3. Create Environment ---
    # The environment wrapper for Isaac Lab
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if cfg.sim_app.record_video else None)

    print(f"[INFO] joint names (IsaacLab Default): {env.scene['robot'].joint_names}")
    print(f"[INFO] joint_pos names (IsaacLab Actual): {env.action_manager.get_term('joint_pos')._joint_names}")
    if hasattr(env_cfg.actions, 'joint_vel'):
        print(f"[INFO] joint_vel names (IsaacLab Actual): {env.action_manager.get_term('joint_vel')._joint_names}")
    print("[INFO] env.observation_manager.active_terms[policy]: ", env.observation_manager.active_terms['policy'])
    print("[INFO] env.observation_manager.group_obs_term_dim: ", env.observation_manager.group_obs_term_dim)

    # wrap around environment for rsl-rl
    if cfg.policy == "eilab":
        env = LabGo2WEnvHistoryWrapper(env, history_len=cfg.observation_len)
    else:
        env = RslRlVecEnvWrapper(env)

    # --- 4. Load Policy ---
    # Path to the pre-trained low-level locomotion policy
    policy_ckpt_path = os.path.join(SIMULATION_DATA_DIR, "ckpts/go2w/blind/policy_roughRecover.jit")
    if not check_file_path(policy_ckpt_path):
        raise FileNotFoundError(f"Checkpoint file '{policy_ckpt_path}' not found. Please place your trained policy model (.jit file) in the 'ckpts' directory.")
    file_bytes = read_file(policy_ckpt_path)
    policy = torch.jit.load(file_bytes).to(cfg.policy_device).eval()

    # --- 5. Run Simulation Loop ---
    policy_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    obs, _ = env.reset()
    print(f"[INFO] init-observation shape: {obs.shape}")

    # --- Set goal position ---
    goal_position = torch.tensor([-3.0, 8.0, 0.4], device=cfg.policy_device)
    
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # Controller to generate velocity command
            if not cfg.controller == "keyboard":
                lin_vel_x, ang_vel_z = mdp.compute_velocity_with_goalPoint(env, goal_position)
                velocity_command = torch.tensor([[lin_vel_x, 0.0, ang_vel_z]], device=cfg.policy_device)
                # Update observation with the new velocity command
                obs = mdp.update_observation_with_velocity_command(env, obs, velocity_command)

            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)

            camera_follow(env, camera_offset_=(-2.0, -2.0, 1.0))

            # time delay for real-time evaluation
            elapsed_time = time.time() - start_time
            sleep_time = policy_step_dt - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        actual_loop_time = time.time() - start_time
        rtf = min(1.0, policy_step_dt / elapsed_time)
        print(f"\rPolicy Step time: {actual_loop_time * 1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)

    print("Simulation finished.")
    simulation_app.close()


if __name__ == "__main__":
    main()