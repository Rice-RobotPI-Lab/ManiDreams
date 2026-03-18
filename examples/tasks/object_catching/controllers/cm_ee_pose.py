"""
Custom End-Effector Pose Controllers for robotic manipulation.

Migrated from plate_catcher_demo/controller/cm_ee_pose.py

Contains:
- cmEEPosController: PD EE Position controller
- cmEEPoseController: PD EE Pose controller (position + orientation)
"""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
)
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, DriveMode

from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController


class cmEEPosController(PDJointPosController):
    """
    PD EE Position controller.

    NOTE: On GPU it is assumed the controlled robot is not a merged
    articulation and is the same across every sub-scene.
    """

    config: "cmEEPosControllerConfig"
    _target_pose = None

    def _check_gpu_sim_works(self):
        assert (
            self.config.frame == "root_translation"
        ), "currently only translation in the root frame for EE control is supported in GPU sim"

    def _initialize_joints(self):
        self.initial_qpos = None
        super()._initialize_joints()
        if self.device.type == "cuda":
            self._check_gpu_sim_works()
        self.kinematics = Kinematics(
            self.config.urdf_path,
            self.config.ee_link,
            self.articulation,
            self.active_joint_indices,
        )

        self.ee_link = self.kinematics.end_link

        if self.config.root_link_name is not None:
            self.root_link = sapien_utils.get_obj_by_name(
                self.articulation.get_links(), self.config.root_link_name
            )
        else:
            self.root_link = self.articulation.root

    def _initialize_action_space(self):
        low = np.float32(np.broadcast_to(-self.config.pos_limit, 3))
        high = np.float32(np.broadcast_to(self.config.pos_limit, 3))
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    @property
    def ee_pos(self):
        return self.ee_link.pose.p

    @property
    def ee_pose(self):
        return self.ee_link.pose

    @property
    def ee_pose_at_base(self):
        to_base = self.root_link.pose.inv()
        return to_base * (self.ee_pose)

    def reset(self):
        super().reset()
        if self._target_pose is None:
            self._target_pose = self.ee_pose_at_base
        else:
            self._target_pose.raw_pose[self.scene._reset_mask] = (
                self.ee_pose_at_base.raw_pose[self.scene._reset_mask]
            )

    def _preprocess_action(self, action: Array):
        if self.scene.num_envs > 1:
            action_dim = self.action_space.shape[1]
        else:
            action_dim = self.action_space.shape[0]
        assert action.shape == (self.scene.num_envs, action_dim), (
            action.shape,
            action_dim,
        )

        action = self._clip_and_scale_action(action)
        return action

    def _clip_and_scale_action(self, action):
        pos_action = action[:, :3].clone()
        if self.config.use_delta:
            pos_norm = torch.linalg.norm(pos_action, axis=1)
            pos_action[pos_norm > self.config.pos_limit] = torch.mul(
                pos_action, self.config.pos_limit / pos_norm[:, None]
            )[pos_norm > self.config.pos_limit]

        rot_action = action[:, 3:].clone()
        return torch.hstack([pos_action, rot_action])

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        if self.config.use_delta:
            delta_pose = Pose.create(action)
            if self.config.frame == "root_translation":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "body_translation":
                target_pose = prev_ee_pose_at_base * delta_pose
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "root_translation", self.config.frame
            target_pose = Pose.create(action)
        return target_pose

    def compute_target_qpos(self, action: Array):
        action = self._preprocess_action(action)

        if self.config.use_target:
            prev_ee_pose_at_base = self._target_pose
        else:
            prev_ee_pose_at_base = self.ee_pose_at_base

        target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
        pos_only = type(self.config) == cmEEPosControllerConfig
        target_qpos = self.kinematics.compute_ik(
            target_pose,
            self.articulation.get_qpos(),
            pos_only=pos_only,
            action=action,
            use_delta_ik_solver=False
        )
        if target_qpos is None:
            target_qpos = self._start_qpos
        return target_qpos

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_target:
            prev_ee_pose_at_base = self._target_pose
        else:
            prev_ee_pose_at_base = self.ee_pose_at_base

        self._target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
        pos_only = type(self.config) == cmEEPosControllerConfig
        self._target_qpos = self.kinematics.compute_ik(
            self._target_pose,
            self.articulation.get_qpos(),
            pos_only=pos_only,
            action=action,
            use_delta_ik_solver=False
        )
        if self._target_qpos is None:
            self._target_qpos = self._start_qpos
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def get_state(self) -> dict:
        if self.config.use_target:
            return {"target_pose": self._target_pose.raw_pose}
        return {}

    def set_state(self, state: dict):
        if self.config.use_target:
            target_pose = state["target_pose"]
            self._target_pose = Pose.create_from_pq(
                target_pose[:, :3], target_pose[:, 3:]
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dof={self.single_action_space.shape[0]}, "
            f"active_joints={len(self.joints)}, end_link={self.config.ee_link}, "
            f"joints=({', '.join([x.name for x in self.joints])}))"
        )


@dataclass
class cmEEPosControllerConfig(ControllerConfig):
    """Configuration for cmEEPosController."""

    pos_limit: float

    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0

    ee_link: str = None
    """The name of the end-effector link to control."""
    urdf_path: str = None
    """Path to the URDF file defining the robot to control."""

    frame: Literal[
        "body_translation",
        "root_translation",
    ] = "root_translation"
    """Choice of frame for translational control."""
    root_link_name: Optional[str] = None
    """Optionally set different root link for root translation control."""
    use_delta: bool = True
    """Whether to use delta-action control."""
    use_target: bool = False
    """Whether to use the most recent target end-effector pose for control."""
    interpolate: bool = False
    normalize_action: bool = False
    """Whether to normalize each action dimension into a range of [-1, 1]."""
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = cmEEPosController


class cmEEPoseController(cmEEPosController):
    """PD EE Pose controller with position and orientation control."""

    config: "cmEEPoseControllerConfig"

    def _check_gpu_sim_works(self):
        assert (
            self.config.frame == "root_translation:root_aligned_body_rotation"
        ), "currently only translation in the root frame for EE control is supported in GPU sim"

    def _initialize_action_space(self):
        low = np.float32(
            np.hstack(
                [
                    np.broadcast_to(-self.config.pos_limit, 3),
                    np.broadcast_to(-self.config.rot_limit, 3),
                ]
            )
        )
        high = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_limit, 3),
                    np.broadcast_to(self.config.rot_limit, 3),
                ]
            )
        )
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def _clip_and_scale_action(self, action):
        pos_action = action[:, :3].clone()
        if self.config.use_delta:
            pos_norm = torch.linalg.norm(pos_action, axis=1)
            pos_action[pos_norm > self.config.pos_limit] = torch.mul(
                pos_action, self.config.pos_limit / pos_norm[:, None]
            )[pos_norm > self.config.pos_limit]

        rot_action = action[:, 3:].clone()
        if self.config.use_delta:
            rot_norm = torch.linalg.norm(rot_action, axis=1)
            rot_action[rot_norm > self.config.rot_limit] = torch.mul(
                rot_action, self.config.rot_limit / rot_norm[:, None]
            )[rot_norm > self.config.rot_limit]
        return torch.hstack([pos_action, rot_action])

    def compute_target_pose(self, prev_ee_pose_at_base: Pose, action):
        if self.config.use_delta:
            delta_pos, delta_rot = action[:, 0:3], action[:, 3:6]
            delta_quat = axis_angle_to_quaternion(delta_rot)
            delta_pose = Pose.create_from_pq(delta_pos, delta_quat)

            if "root_aligned_body_rotation" in self.config.frame:
                q = quaternion_multiply(delta_pose.q, prev_ee_pose_at_base.q)
            if "body_aligned_body_rotation" in self.config.frame:
                q = quaternion_multiply(prev_ee_pose_at_base.q, delta_pose.q)
            if "root_translation" in self.config.frame:
                p = prev_ee_pose_at_base.p + delta_pos
            if "body_translation" in self.config.frame:
                p = prev_ee_pose_at_base.p + quaternion_apply(
                    prev_ee_pose_at_base.q, delta_pose.p
                )
            target_pose = Pose.create_from_pq(p, q)
        else:
            assert (
                self.config.frame == "root_translation:root_aligned_body_rotation"
            ), self.config.frame
            target_pos, target_rot = action[:, 0:3], action[:, 3:6]
            target_quat = axis_angle_to_quaternion(target_rot)
            target_pose = Pose.create_from_pq(target_pos, target_quat)

        return target_pose


@dataclass
class cmEEPoseControllerConfig(cmEEPosControllerConfig):
    """Configuration for cmEEPoseController."""

    rot_limit: float = None

    stiffness: Union[float, Sequence[float]] = None
    damping: Union[float, Sequence[float]] = None
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0

    frame: Literal[
        "body_translation:root_aligned_body_rotation",
        "root_translation:root_aligned_body_rotation",
        "body_translation:body_aligned_body_rotation",
        "root_translation:body_aligned_body_rotation",
    ] = "root_translation:root_aligned_body_rotation"
    """Choice of frame for translational and rotational control."""

    controller_cls = cmEEPoseController
