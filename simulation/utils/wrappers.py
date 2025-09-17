import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class LabGo2WEnvHistoryWrapper(RslRlVecEnvWrapper):
    """Wraps around Isaac Lab environment for RSL-RL to add history buffer to the proprioception observations.
    """
    def __init__(self, env: ManagerBasedRLEnv, history_len: int = 5):
        """Initializes the wrapper."""
        super().__init__(env)

        self.history_len = history_len + 1
        self.base_obs_dim = 57
        # (num_envs, 6, 45)
        self.his_obs_buf = torch.zeros(self.num_envs, self.history_len, self.base_obs_dim, dtype=torch.float, device=self.unwrapped.device)

        self.actions_clip = 100
        self.obs_pos_start_idx = 9

        """ dof_names
            Legged-Loco:
                0 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'
                3 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'
                6 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
                9 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
                12 'FR_foot_joint', 'FL_foot_joint', 'RR_foot_joint', 'RL_foot_joint'
            IsaacLab Default:
                0 'FL_hip_joint',   'FR_hip_joint',   'RL_hip_joint',   'RR_hip_joint'
                4 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
                8 'FL_calf_joint',  'FR_calf_joint',  'RL_calf_joint',  'RR_calf_joint',
                12 'FL_foot_joint', 'FR_foot_joint',  'RL_foot_joint',  'RR_foot_joint'
            IsaacLab Actual (because assign joint_names list in go2w's env_cfg, and set preserve_order=True):
                0 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'
                3 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'
                6 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
                9 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
                12 'FR_foot_joint', 'FL_foot_joint', 'RR_foot_joint', 'RL_foot_joint'
        """
        self.lab_to_policy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.policy_to_lab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    def reset(self):
        # (num_envs, 57)
        obs, info = super().reset()  # info: {'observations': {'policy': tensor(num_envs, 57)}}
        obs = obs[:, :self.base_obs_dim]  # (num_envs, 57)
        obs_policy = self._permute_obs_to_policy(obs)
        curr_obs = torch.cat([obs_policy] * self.history_len, dim=1)  # (num_envs, 6 * 57)
        return curr_obs, info

    def step(self, actions_policy):
        actions = actions_policy[:, self.policy_to_lab]
        actions = torch.clamp(actions, -self.actions_clip, self.actions_clip)

        # record step information (执行一个control步 (包含4个物理仿真步)，计算 奖励，计算新的 obs_dict)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)

        obs = obs_dict["policy"]
        obs = obs[:, :self.base_obs_dim]  # (num_envs, 57)
        obs = self._permute_obs_to_policy(obs)  # ==> policy

        extras["observations"] = obs_dict
        # move time out information to the extras dict this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # update observation history buffer & reset the history buffer for done environments
        # (num_envs, 6, 57)
        self.his_obs_buf = torch.where(
            (self.episode_length_buf < 1)[:, None, None],
            torch.stack([torch.zeros_like(obs)] * self.history_len, dim=1),
            torch.cat([
                obs.unsqueeze(1),
                self.his_obs_buf[:, :-1],
            ], dim=1)
        )
        curr_obs = self.his_obs_buf.view(self.num_envs, -1)  # (num_envs, 6*57)
        extras["observations"]["policy"] = curr_obs

        # return the step information
        return curr_obs, rew, dones, extras

    def _permute_obs_to_policy(self, obs):
        # joint_pos
        obs[:, self.obs_pos_start_idx:self.obs_pos_start_idx + 16] = obs[:, self.obs_pos_start_idx:self.obs_pos_start_idx + 16][:, self.lab_to_policy]
        # joint_vel
        obs[:, self.obs_pos_start_idx + 16:self.obs_pos_start_idx + 32] = obs[:, self.obs_pos_start_idx + 16:self.obs_pos_start_idx + 32][:, self.lab_to_policy]
        # last_action
        obs[:, self.obs_pos_start_idx + 32:self.obs_pos_start_idx + 48] = obs[:, self.obs_pos_start_idx + 32:self.obs_pos_start_idx + 48][:, self.lab_to_policy]
        return obs


