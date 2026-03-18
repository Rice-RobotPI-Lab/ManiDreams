"""
Observation and Action mappers for RL policy execution.

Migrated from plate_catcher_demo/models/mappers.py

Contains:
- ObsMapper: Maps raw observation to motion frame (single ball)
- ObsDRISMapper: Maps raw observation to DRIS in motion frame
- ActionMapperSequence: Maps action sequence to robot base frame
- ActionMapperStep: Maps single-step action to robot base frame
"""

import math
import gymnasium as gym
import torch

from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_quaternion,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    quaternion_multiply
)
from mani_skill.utils.structs import Pose
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from .geometry import (
    quaternion_to_axis_angle,
    find_nearest_quat_to_normal,
    find_minimum_delta_rot_to_up,
    find_minimum_delta_rot_from_up
)


class ObsMapper(object):
    """
    Map the raw observation to a new observation (single-ball position and velocity)
    in the motion frame (centered at tcp, x-axis points towards the ball, z-axis up).

    Used by Full-Horizon (open-loop) policy.
    """

    def __init__(self, envs, device, use_full_obs=True):
        if isinstance(envs, ManiSkillVectorEnv):
            self.num_objs = envs._env.num_objs
        elif isinstance(envs, gym.vector.SyncVectorEnv):
            self.num_objs = envs.envs[0].num_objs
        elif isinstance(envs, gym.vector.AsyncVectorEnv):
            self.num_objs = 200  # HARDCODED, need to be fixed
        else:
            self.num_objs = envs.num_objs

        self.device = device
        self.use_full_obs = use_full_obs

    def map(self, obs):
        """
        Map observation to motion frame.

        Args:
            obs: Raw observation in shape (num_envs, obs_dim)

        Returns:
            Mapped observation in shape (num_envs, 6)
        """
        num_envs = obs.shape[0]
        if self.use_full_obs:
            num_objs = int((obs.shape[1] - 7) / 6)
            obj_pos = obs[:, 0:3]  # shape (num_envs, 3)
            obj_vel = obs[:, 3*num_objs:3*(num_objs+1)]  # shape (num_envs, 3)
            tcp_pos = obs[:, 3*(2*num_objs):3*(2*num_objs)+3]  # shape (num_envs, 3)
        else:
            obj_pos = obs[:, 0:3]  # shape (num_envs, 3)
            obj_vel = obs[:, 3:6]  # shape (num_envs, 3)
            tcp_pos = obs[:, 6:9]  # shape (num_envs, 3)
        vec = obj_pos - tcp_pos  # direction from the plate to the ball

        # rotation (about z-axis) from the robot base frame to the motion frame
        rot_bm = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        rot_bm[:, 2] = torch.atan2(vec[:, 1], vec[:, 0])
        quat_bm = axis_angle_to_quaternion(rot_bm)
        pose_mb = Pose.create_from_pq(tcp_pos, quat_bm).inv()
        tmat_mb = pose_mb.to_transformation_matrix()  # (num_envs, 4, 4)

        obs_m = torch.zeros((num_envs, 6), dtype=torch.float32, device=self.device)
        obj_pos_h = torch.hstack((obj_pos, torch.ones((num_envs, 1)).to(obj_pos.device)))
        obj_pos_m = (tmat_mb @ obj_pos_h.unsqueeze(-1)).squeeze(-1)[:, 0:3]
        obj_vel_m = (tmat_mb[:, 0:3, 0:3] @ obj_vel.unsqueeze(-1)).squeeze(-1)
        obs_m[:, 0:3] = obj_pos_m  # object position in motion frame
        obs_m[:, 3:6] = obj_vel_m  # object velocity in motion frame
        return obs_m


class ObsDRISMapper(ObsMapper):
    """
    Map the raw observation to DRIS (object position and velocity, represented
    as a complete set or Gaussian) in the motion frame.

    Motion frame: centered at tcp, x-axis points towards the ball, z-axis up.
    """

    def __init__(self, envs, device, use_full_obs=True, use_guassian_distr=False):
        super(ObsDRISMapper, self).__init__(envs, device, use_full_obs)
        self.tmat_mb = None  # transformation matrix from motion frame to robot base frame
        self.use_gaussian_distr = use_guassian_distr

    def register_ref_obs(self, obs):
        """Register reference observation to compute motion frame transformation."""
        num_envs = obs.shape[0]
        if self.use_full_obs:
            num_objs = int((obs.shape[1] - 7) / 6)
            obj_pos = obs[:, 0:3]
            tcp_pos = obs[:, 3*(2*num_objs):3*(2*num_objs)+3]
        else:
            obj_pos = obs[:, 0:3]
            tcp_pos = obs[:, 6:9]
        vec = obj_pos - tcp_pos

        rot_bm = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        rot_bm[:, 2] = torch.atan2(vec[:, 1], vec[:, 0])
        quat_bm = axis_angle_to_quaternion(rot_bm)
        pose_mb = Pose.create_from_pq(tcp_pos, quat_bm).inv()
        self.tmat_mb = pose_mb.to_transformation_matrix()

    def map(self, obs):
        """
        Map observation to DRIS in motion frame.

        Args:
            obs: Raw observation in shape (num_envs, obs_dim)

        Returns:
            If use_gaussian_distr: (dris_m, tcpn_m) where dris_m is (num_envs, 12)
            Else: (obs_m, tcpn_m) where obs_m is (num_envs, num_objs, 6)
        """
        if self.tmat_mb is None:
            raise ValueError("Reference observation is not registered. Call register_ref_obs() first.")

        num_envs = obs.shape[0]
        if self.use_full_obs:
            num_objs = int((obs.shape[1] - 7) / 6)
            objs_pos = torch.transpose(
                obs[:, 0:3*num_objs].view(-1, num_objs, 3),
                1, 2)  # shape (num_envs, 3, num_objs)
            objs_vel = torch.transpose(
                obs[:, 3*num_objs:3*(2*num_objs)].view(-1, num_objs, 3),
                1, 2)  # shape (num_envs, 3, num_objs)
            tcp_pos = obs[:, 3*(2*num_objs):3*(2*num_objs)+3].unsqueeze(-1)
            tcp_quat = obs[:, 3*(2*num_objs)+3:3*(2*num_objs)+7]
        else:
            objs_pos = obs[:, 0:3].unsqueeze(-1)
            objs_vel = obs[:, 3:6].unsqueeze(-1)
            tcp_pos = obs[:, 6:9].unsqueeze(-1)
            tcp_quat = obs[:, 9:13]

        objs_disp = objs_pos - tcp_pos
        objs_disp_m = self.tmat_mb[:, 0:3, 0:3] @ objs_disp
        objs_vel_m = self.tmat_mb[:, 0:3, 0:3] @ objs_vel

        tcp_rotmat = quaternion_to_matrix(tcp_quat)
        tcpn = tcp_rotmat[:, :, 2]  # normal vector (z-axis)
        tcpn_m = self.tmat_mb[:, 0:3, 0:3] @ tcpn.unsqueeze(-1)
        tcpn_m = tcpn_m.squeeze(-1)

        tcpn_m = tcpn_m / tcpn_m.norm(dim=-1, keepdim=True)
        tcpn_m = find_minimum_delta_rot_from_up(tcpn_m)
        tcpn_m = tcpn_m[:, 0:2]  # only keep x and y components

        if self.use_gaussian_distr:
            disp_m_mean = torch.mean(objs_disp_m, dim=-1)
            vel_m_mean = torch.mean(objs_vel_m, dim=-1)

            if objs_disp_m.shape[-1] == 1:
                disp_m_std = torch.zeros_like(disp_m_mean)
                vel_m_std = torch.zeros_like(vel_m_mean)
            else:
                disp_m_std = torch.std(objs_disp_m, dim=-1)
                vel_m_std = torch.std(objs_vel_m, dim=-1)

            dris_m = torch.hstack([disp_m_mean, disp_m_std, vel_m_mean, vel_m_std])
            return dris_m, tcpn_m
        else:
            obs_m = torch.cat((objs_disp_m, objs_vel_m), dim=1)
            obs_m = torch.transpose(obs_m, 1, 2)  # shape (num_envs, num_objs, 6)
            return obs_m, tcpn_m


class ActionMapperSequence(object):
    """
    Project a sequence of 6D actions (e.g., delta pose control)
    defined in the object's motion plane (vertical) into robot base frame.
    """

    def __init__(self, envs, device, vel_match_steps=0, use_full_obs=True):
        if isinstance(envs, gym.vector.AsyncVectorEnv):
            self.num_steps = 20
            self.num_objs = 200
            self.control_freq = 20
            self.pos_limit = 0.025
            self.rot_limit = 0.3
        else:
            if isinstance(envs, ManiSkillVectorEnv):
                env = envs._env
            elif isinstance(envs, gym.vector.SyncVectorEnv):
                env = envs.envs[0]
            else:
                env = envs
            self.num_steps = env.get_wrapper_attr("_max_episode_steps")
            self.num_objs = env.num_objs
            self.control_freq = env.control_freq
            self._set_control_limit(env)

        self.device = device
        self.vel_match_steps = vel_match_steps
        assert self.vel_match_steps >= 0 & self.vel_match_steps <= self.num_steps
        self.use_full_obs = use_full_obs

        self.workspace_min = [-0.15, -0.15, -0.1]
        self.workspace_max = [0.15, 0.15, 0.1]

    def _set_control_limit(self, env):
        self.pos_limit = env.agent.controller.controllers["arm"].config.pos_limit
        self.rot_limit = env.agent.controller.controllers["arm"].config.rot_limit

    def to_pos_ratio(self, vel):
        delta_pos = vel / self.control_freq
        ratio = delta_pos / self.pos_limit
        scale_max = torch.max(torch.abs(ratio), dim=-1)[0]
        scale_target = torch.clamp(scale_max, min=1.0).unsqueeze(-1)
        return ratio / scale_target

    def map(self, obs, action):
        """
        Map action sequence to robot base frame.

        Args:
            obs: First raw observation in shape (num_envs, obs_dim)
            action: Action ratio in [-1, 1]^d

        Returns:
            Control commands in robot base frame
        """
        num_envs = obs.shape[0]
        if self.use_full_obs:
            num_objs = int((obs.shape[1] - 7) / 6)
            obj_pos = obs[:, 0:3].unsqueeze(1)
            obj_vel = obs[:, 3*num_objs:3*(num_objs+1)].unsqueeze(1)
            tcp_pos = obs[:, 3*(2*num_objs):3*(2*num_objs)+3].unsqueeze(1)
            tcp_quat = obs[:, 3*(2*num_objs)+3:3*(2*num_objs)+7].unsqueeze(1)
        else:
            obj_pos = obs[:, 0:3].unsqueeze(1)
            obj_vel = obs[:, 3:6].unsqueeze(1)
            tcp_pos = obs[:, 6:9].unsqueeze(1)
            tcp_quat = obs[:, 9:13].unsqueeze(1)
        vec = obj_pos - tcp_pos

        rot_bm = torch.zeros((num_envs, self.num_steps, 3), dtype=torch.float32, device=self.device)
        rot_bm[:, :, 2] = torch.atan2(vec[:, :, 1], vec[:, :, 0])
        quat_bm = axis_angle_to_quaternion(rot_bm.view(-1, 3))
        pose_bm = Pose.create_from_pq(tcp_pos.expand(-1, self.num_steps, -1).view(-1, 3), quat_bm)
        pose_bm_0 = Pose.create_from_pq(tcp_pos[:, 0, :], quat_bm.view(num_envs, self.num_steps, 4)[:, 0, :])

        tcp_pose = Pose.create_from_pq(tcp_pos[:, 0, :], tcp_quat[:, 0, :])
        tcp_pose_m = pose_bm_0.inv() * tcp_pose

        action = action.clone().view(num_envs, self.num_steps, -1)

        if self.vel_match_steps > 0:
            rotmat_bm = pose_bm.to_transformation_matrix()[:, 0:3, 0:3]
            rotmat_bm = rotmat_bm.view(num_envs, self.num_steps, *rotmat_bm.shape[1:])
            rotmat_bm = rotmat_bm[:, 0, :, :]
            obj_vel_m = (torch.transpose(rotmat_bm, 1, 2) @ obj_vel.view(-1, 3, 1)).view(-1, 1, 3)
            base_action = torch.zeros_like(action)
            base_action[:, 0:self.vel_match_steps, 0:3] = self.to_pos_ratio(obj_vel_m)
            action += base_action

        # translational action
        pos_action = action[:, :, 0:3] * self.pos_limit
        pos_action[:, 0, :] += tcp_pose_m.p
        for i in range(1, self.num_steps):
            pos_action[:, i, :] += pos_action[:, i-1, :]
            pos_action[:, i, :] = torch.clamp(
                pos_action[:, i, :],
                min=torch.tensor(self.workspace_min, device=pos_action.device),
                max=torch.tensor(self.workspace_max, device=pos_action.device)
            )
        pos_action = pos_action.view(-1, 3)

        # rotational action
        rot_action = action[:, :, 3:5].view(-1, 2) * self.rot_limit
        rot_action = torch.hstack((rot_action, torch.zeros((rot_action.shape[0], 1), device=rot_action.device)))
        delta_rot = find_minimum_delta_rot_to_up(quaternion_to_axis_angle(tcp_pose_m.q))
        delta_quat = axis_angle_to_quaternion(delta_rot)
        quat_ref = quaternion_multiply(delta_quat, tcp_pose_m.q)
        quat_ref = quat_ref.unsqueeze(1).expand(-1, self.num_steps, -1).contiguous().view(-1, 4)
        quat_action = axis_angle_to_quaternion(rot_action)
        quat = quaternion_multiply(quat_action, quat_ref)

        pose_m = Pose.create_from_pq(pos_action, quat)
        pose_b = pose_bm * pose_m
        ctrl = torch.hstack([pose_b.p, quaternion_to_axis_angle(pose_b.q)]).view(num_envs, self.num_steps, -1)
        return ctrl


class ActionMapperStep(ActionMapperSequence):
    """
    Project a single 6D action (e.g., delta pose control)
    defined in the object's motion plane (vertical) into robot base frame.
    """

    def __init__(self, envs, device, vel_match=False, use_full_obs=True):
        super(ActionMapperStep, self).__init__(envs, device, 0, use_full_obs)
        self.vel_match = vel_match
        self.pose_bm = None
        self.obj_vel_m = None

        self.pos_limit = 0.025
        self.rot_limit = 0.3

    def register_ref_obs(self, obs):
        """Register reference observation to compute motion frame transformation."""
        num_envs = obs.shape[0]
        if self.use_full_obs:
            num_objs = int((obs.shape[1] - 7) / 6)
            obj_pos = obs[:, 0:3]
            obj_vel = obs[:, 3*num_objs:3*(num_objs+1)]
            tcp_pos = obs[:, 3*(2*num_objs):3*(2*num_objs)+3]
        else:
            obj_pos = obs[:, 0:3]
            obj_vel = obs[:, 3:6]
            tcp_pos = obs[:, 6:9]
        vec = obj_pos - tcp_pos

        rot_bm = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        rot_bm[:, 2] = torch.atan2(vec[:, 1], vec[:, 0])
        quat_bm = axis_angle_to_quaternion(rot_bm)
        self.pose_bm = Pose.create_from_pq(tcp_pos, quat_bm)

        rotmat_bm = self.pose_bm.to_transformation_matrix()[:, 0:3, 0:3]
        self.obj_vel_m = (torch.transpose(rotmat_bm, 1, 2) @ obj_vel.view(-1, 3, 1)).squeeze(-1)

    def map(self, obs, action):
        """
        Map single-step action to robot base frame.

        Args:
            obs: Raw observation in shape (num_envs, obs_dim)
            action: Action ratio in [-1, 1]^d

        Returns:
            Control command in robot base frame
        """
        if self.pose_bm is None or self.obj_vel_m is None:
            raise ValueError("Reference observation is not registered. Call register_ref_obs() first.")

        if self.use_full_obs:
            num_objs = int((obs.shape[1] - 7) / 6)
            tcp_pos = obs[:, 3*(2*num_objs):3*(2*num_objs)+3]
            tcp_quat = obs[:, 3*(2*num_objs)+3:3*(2*num_objs)+7]
        else:
            tcp_pos = obs[:, 6:9]
            tcp_quat = obs[:, 9:13]

        tcp_pose = Pose.create_from_pq(tcp_pos, tcp_quat)
        tcp_pose_m = self.pose_bm.inv() * tcp_pose

        action = action.clone()

        if self.vel_match:
            base_action = torch.zeros_like(action)
            base_action[:, 0:3] = self.to_pos_ratio(self.obj_vel_m)
            action += base_action

        # translational action
        pos_action = action[:, 0:3] * self.pos_limit + tcp_pose_m.p
        pos_action = torch.clamp(
            pos_action,
            min=torch.tensor(self.workspace_min, device=pos_action.device),
            max=torch.tensor(self.workspace_max, device=pos_action.device)
        )

        # rotational action
        rot_action = action[:, 3:5] * self.rot_limit
        rot_action = torch.hstack((rot_action, torch.zeros((action.shape[0], 1), device=action.device)))

        delta_rot = find_minimum_delta_rot_to_up(quaternion_to_axis_angle(tcp_pose_m.q))
        delta_quat = axis_angle_to_quaternion(delta_rot)
        quat_action = axis_angle_to_quaternion(rot_action)
        quat = quaternion_multiply(quat_action, quaternion_multiply(delta_quat, tcp_pose_m.q))

        pose_m = Pose.create_from_pq(pos_action, quat)
        pose_b = self.pose_bm * pose_m
        ctrl = torch.hstack([pose_b.p, quaternion_to_axis_angle(pose_b.q)])
        return ctrl
