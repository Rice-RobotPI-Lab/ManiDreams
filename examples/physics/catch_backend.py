"""
Catch Backend for ManiDreams Framework

Task-specific backend for object catching using PlateCatch-v1 environment.

This module includes:
- PandaCatcher: Custom robot agent with plate end-effector
- TaskSceneBuilder: Scene builder for the catching task
- PlateCatchEnv: ManiSkill environment for plate catching
- CatchBackend: Backend wrapper supporting both executor and CAGE evaluation modes
"""

from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union
import copy
import logging
import os
import warnings

logger = logging.getLogger(__name__)

import numpy as np
import sapien
import sapien.physx as physx
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
    MANISKILL_AVAILABLE = True
except ImportError:
    MANISKILL_AVAILABLE = False
    warnings.warn("ManiSkill not available. Install with: pip install mani-skill>=2.0.0")

if MANISKILL_AVAILABLE:
    from transforms3d.euler import euler2quat

    from mani_skill.agents.base_agent import BaseAgent, Keyframe
    from mani_skill.agents.controllers import *
    from mani_skill.agents.registration import register_agent
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.envs.utils import randomization
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import common, sapien_utils
    from mani_skill.utils.building import actors
    from mani_skill.utils.building.ground import build_ground
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder import SceneBuilder
    from mani_skill.utils.structs import Actor, Link, Pose
    from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
    from mani_skill.utils.geometry.rotation_conversions import (
        axis_angle_to_quaternion,
        quaternion_to_matrix,
    )
    from sapien.physx import PhysxMaterial

    from examples.tasks.object_catching.controllers import (
        cmEEPosController,
        cmEEPosControllerConfig,
        cmEEPoseController,
        cmEEPoseControllerConfig,
    )
    from examples.samplers.catching.geometry import find_minimum_delta_rot_from_up

from .maniskill_base import ManiSkillBaseBackend


# Get the path to the assets directory
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
ROBOT_DIR = os.path.join(ASSETS_DIR, "robots", "panda_catcher")


if MANISKILL_AVAILABLE:
    @register_agent()
    class PandaCatcher(BaseAgent):
        """
        Panda robot with a plate end-effector for catching tasks.

        This robot has 7 DOF arm joints with a fixed plate attached
        to the end-effector for catching falling objects.
        """

        uid = "panda_catcher"
        urdf_path = os.path.join(ROBOT_DIR, "panda_catcher.urdf")
        urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
                plate=dict(static_friction=0.0, dynamic_friction=0.0, restitution=0.2),
            ),
            link=dict(
                panda_leftfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                panda_rightfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                plate=dict(material="plate")
            ),
        )
        keyframes = dict(
            rest=Keyframe(
                qpos=np.array(
                    [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4]
                ),
                pose=sapien.Pose(),
            ),
            home=Keyframe(
                qpos=np.array(
                    [0.0, -np.pi / 4, 0.0, -np.pi * 3 / 4, 0.0, np.pi / 2, np.pi / 4]
                ),
                pose=sapien.Pose(),
            ),
        )

        arm_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        ee_link_name = "plate"

        # PD controller gains (after system ID)
        arm_stiffness = [120, 70, 70, 120, 30, 20, 40]
        arm_damping = [15, 11, 10, 14, 4, 3, 5]
        arm_force_limit = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]

        @property
        def _controller_configs(self):
            # Arm controllers
            arm_pd_joint_pos = PDJointPosControllerConfig(
                self.arm_joint_names,
                lower=None,
                upper=None,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                normalize_action=False,
            )
            arm_pd_joint_delta_pos = PDJointPosControllerConfig(
                self.arm_joint_names,
                lower=-0.1,
                upper=0.1,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                use_delta=True,
            )
            arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
            arm_pd_joint_target_delta_pos.use_target = True

            # PD ee position
            arm_pd_ee_delta_pos = cmEEPosControllerConfig(
                joint_names=self.arm_joint_names,
                pos_limit=0.025,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                ee_link=self.ee_link_name,
                urdf_path=self.urdf_path,
            )
            arm_pd_ee_delta_pose = cmEEPoseControllerConfig(
                joint_names=self.arm_joint_names,
                pos_limit=0.025,
                rot_limit=0.3,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                ee_link=self.ee_link_name,
                urdf_path=self.urdf_path,
            )
            arm_pd_ee_pose = cmEEPoseControllerConfig(
                joint_names=self.arm_joint_names,
                pos_limit=0.1,
                rot_limit=0.3,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                ee_link=self.ee_link_name,
                urdf_path=self.urdf_path,
                use_delta=False,
                normalize_action=False,
            )

            arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
            arm_pd_ee_target_delta_pos.use_target = True
            arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
            arm_pd_ee_target_delta_pose.use_target = True

            # PD ee position (for human-interaction/teleoperation)
            arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
            arm_pd_ee_delta_pose_align.frame = "ee_align"

            # PD joint velocity
            arm_pd_joint_vel = PDJointVelControllerConfig(
                self.arm_joint_names,
                -1.0,
                1.0,
                self.arm_damping,
                self.arm_force_limit,
            )

            # PD joint position and velocity
            arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
                self.arm_joint_names,
                None,
                None,
                self.arm_stiffness,
                self.arm_damping,
                self.arm_force_limit,
                normalize_action=False,
            )
            arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
                self.arm_joint_names,
                -0.1,
                0.1,
                self.arm_stiffness,
                self.arm_damping,
                self.arm_force_limit,
                use_delta=True,
            )

            controller_configs = dict(
                pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos),
                pd_joint_pos=dict(arm=arm_pd_joint_pos),
                pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos),
                pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose),
                pd_ee_delta_pose_align=dict(arm=arm_pd_ee_delta_pose_align),
                pd_ee_pose=dict(arm=arm_pd_ee_pose),
                pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos),
                pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos),
                pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose),
                pd_joint_vel=dict(arm=arm_pd_joint_vel),
                pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel),
                pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel),
            )

            return deepcopy_dict(controller_configs)

        def _after_init(self):
            self.tcp = sapien_utils.get_obj_by_name(
                self.robot.get_links(), self.ee_link_name
            )
            self.queries: Dict[
                str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
            ] = dict()

        @property
        def tcp_pos(self):
            return self.tcp.pose.p

        @property
        def tcp_quat(self):
            return self.tcp.pose.q

        @property
        def tcp_pose(self):
            return self.tcp.pose

        @property
        def qpos(self):
            return self.robot.get_qpos()

        @property
        def qvel(self):
            return self.robot.get_qvel()

        def set_qpos(self, qpos):
            self.robot.set_qpos(qpos)

        def set_qvel(self, qvel):
            self.robot.set_qvel(qvel)

        def set_qpos_and_qvel(self, qpos, qvel):
            self.set_qpos(qpos)
            self.set_qvel(qvel)
            self.robot.set_qf(torch.zeros(self.robot.max_dof, device=self.device))

        def is_static(self, threshold: float = 0.2):
            qvel = self.robot.get_qvel()[..., :-2]
            return torch.max(torch.abs(qvel), 1)[0] <= threshold

        def get_target_qpos(self):
            if self.controller.controllers["arm"] is None:
                return None
            else:
                return self.controller.controllers["arm"]._target_qpos


    class TaskSceneBuilder(SceneBuilder):
        """Scene builder for the plate catching task."""

        def build(self):
            floor_width = 100
            if self.scene.parallel_in_single_scene:
                floor_width = 500
            self.ground = build_ground(
                self.scene, floor_width=floor_width, altitude=0, texture_square_len=4
            )
            self.scene_objects: List[sapien.Entity] = [self.ground]

        def initialize(self, env_idx: torch.Tensor):
            if self.env.robot_uids == "panda_catcher":
                qpos = self.env.agent.keyframes["home"].qpos
                self.env.agent.reset(qpos)
                if self.env.backend.sim_backend == "physx_cuda":
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene._gpu_fetch_all()
                self.env.agent.robot.set_pose(sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))


    @register_env("PlateCatch-v1", max_episode_steps=20)
    class PlateCatchEnv(BaseEnv):
        """
        Plate catching environment for ManiSkill.

        The robot must catch falling balls using a plate attached
        to its end-effector.
        """

        SUPPORTED_ROBOTS = ["panda_catcher"]

        ball_radius = 0.03

        def __init__(
            self,
            *args,
            robot_uids="panda_catcher",
            robot_init_qpos_noise=0.02,
            num_objs=200,
            **kwargs
        ):
            self.robot_init_qpos_noise = robot_init_qpos_noise
            self.num_objs = num_objs
            self.dist_thresh = 0.1
            self.vel_thresh = 0.05
            super().__init__(*args, robot_uids=robot_uids, **kwargs)

        @property
        def _default_sim_config(self):
            return SimConfig(
                spacing=1,
                sim_freq=100,
                control_freq=20,
                gpu_memory_config=GPUMemoryConfig(
                    max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
                    max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
                    found_lost_pairs_capacity=2**25,
                ),
            )

        @property
        def _default_sensor_configs(self):
            pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
            return [
                CameraConfig(
                    "base_camera",
                    pose=pose,
                    width=128,
                    height=128,
                    fov=np.pi / 2,
                    near=0.01,
                    far=100,
                )
            ]

        @property
        def _default_human_render_camera_configs(self):
            pose = sapien_utils.look_at([1.2, -0.5, 1.2], [0.0, 0.0, 0.3])
            return CameraConfig(
                "render_camera", pose=pose, width=1920, height=1920, fov=1, near=0.01, far=100
            )

        def _clear_sim_state(self):
            """Clear simulation state (velocities)."""
            for obj in self.objs:
                obj.set_linear_velocity(torch.zeros(3, device=self.device))
                obj.set_angular_velocity(torch.zeros(3, device=self.device))
            for articulation in self.scene.articulations.values():
                articulation.set_qvel(torch.zeros(articulation.max_dof, device=self.device))
                articulation.set_root_linear_velocity(torch.zeros(3, device=self.device))
                articulation.set_root_angular_velocity(torch.zeros(3, device=self.device))
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene._gpu_fetch_all()

        def _load_agent(self, options: dict):
            super()._load_agent(options, sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
            # Disable collision between robot links (except plate) and balls
            for link in self.agent.robot.get_links():
                if link.name != "plate":
                    link.set_collision_group(1, 0xFFFFFFFE)

            if "controller_config" in options.keys():
                controller_config = options["controller_config"]
                stiffness = controller_config["stiffness"]
                damping = controller_config["damping"]
                force_limit = controller_config["force_limit"]

                self.agent.controller.controllers["arm"].config.stiffness = stiffness
                self.agent.controller.controllers["arm"].config.damping = damping
                self.agent.controller.controllers["arm"].config.force_limit = force_limit
                self.agent.controller.controllers["arm"].set_drive_property()

        def _load_scene(self, options: dict):
            self.task_scene = TaskSceneBuilder(
                env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.task_scene.build()

            self.objs = []
            for i in range(self.num_objs):
                obj_i = []
                for e in range(self.num_envs):
                    ball_ei = self.spawn_ball(e, i)
                    self.remove_from_state_dict_registry(ball_ei)
                    obj_i.append(ball_ei)
                obj_i = Actor.merge(obj_i, name="ball_%d" % i)
                self.add_to_state_dict_registry(obj_i)
                self.objs.append(obj_i)

        def spawn_ball(self, env_id, ball_id):
            builder = self.scene.create_actor_builder()
            radius = float(self.ball_radius + self._batched_episode_rng[0].uniform(low=-0.003, high=0.003))

            physical_material = PhysxMaterial(0.0, 0.0, 0.0)
            builder.add_sphere_collision(radius=radius, material=physical_material, density=0.5)
            # Blue base color with per-ball RGB variation
            base_color = np.array([30.0, 80.0, 240.0, 100.0]) / 255.0
            noise = np.random.uniform(-0.08, 0.08, size=3)
            ball_color = base_color.copy()
            ball_color[:3] = np.clip(base_color[:3] + noise, 0.0, 1.0)
            builder.add_sphere_visual(radius=radius, material=sapien.render.RenderMaterial(
                base_color=ball_color,
            ))
            builder.set_initial_pose(sapien.Pose(p=[0, -1.0, radius]))
            builder.set_scene_idxs([env_id])
            sphere = builder.build(name="ball_%d_%d" % (env_id, ball_id))
            sphere.set_collision_group(1, 0xFFFFFFFE)
            return sphere

        def initialize_robot(self, env_idx: torch.Tensor, options: dict, txyz=None, tvxyz=None):
            b = len(env_idx)
            dof = self.agent.robot.max_dof
            if "qpos" in options.keys():
                target_qpos = torch.zeros((b, dof))
                target_qpos[:, :] = torch.tensor(options["qpos"])
            else:
                tcp_pose = self.agent.tcp_pose
                target_pose = copy.deepcopy(tcp_pose)
                if (txyz is not None) and (tvxyz is not None):
                    target_pose.p = txyz
                    normal = -tvxyz / torch.norm(tvxyz, dim=-1, keepdim=True)
                    delta_rot = find_minimum_delta_rot_from_up(normal)
                    target_pose.q = axis_angle_to_quaternion(delta_rot)
                try:
                    target_qpos = self.agent.controller.controllers["arm"].kinematics.compute_ik(
                        target_pose,
                        self.agent.qpos,
                        pos_only=False,
                        use_delta_ik_solver=False,
                    )
                except:
                    target_qpos = torch.zeros((b, dof))
                    target_qpos[:, :] = torch.tensor(self.agent.keyframes["home"].qpos)
                    logger.warning("IK failed, use home position")

                target_qpos += torch.normal(0, self.robot_init_qpos_noise, target_qpos.shape).to(self.device)

            target_qvel = torch.zeros((b, dof))
            if "qvel" in options.keys():
                target_qvel[:, :] = torch.tensor(options["qvel"])

            self.agent.set_qpos_and_qvel(target_qpos, target_qvel)
            if self.backend.sim_backend == "physx_cuda":
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()

        def initialize_objs(self, env_idx: torch.Tensor, options: dict):
            b = len(env_idx)

            if "bstate" in options.keys():
                bstate = torch.tensor(options["bstate"])
                dxyz = bstate[0:3].unsqueeze(0).expand(b, -1)
                vxyz = bstate[3:6].unsqueeze(0).expand(b, -1)
                txyz = None
                tvxyz = None
            else:
                tcp_pos = self.agent.tcp_pos

                # Target position for ball (within a circle centered at tcp)
                r = torch.rand((b,)) * 0.2
                alpha = torch.rand((b,)) * 2 * torch.pi
                txyz = tcp_pos.clone()
                txyz[:, 0] += r * torch.cos(alpha)
                txyz[:, 1] += r * torch.sin(alpha)

                # Sample starting position of ball before thrown
                l = torch.rand((b,)) + 1.0
                beta = torch.rand((b,)) * torch.pi - torch.pi / 2

                xyz = torch.zeros((b, 3))
                xyz[:, 0] = txyz[:, 0] + torch.cos(beta) * l
                xyz[:, 1] = txyz[:, 1] + torch.sin(beta) * l
                xyz[:, 2] = txyz[:, 2]

                Tf = torch.rand((b,)) * 0.5 + 1.0
                vxyz = torch.zeros((b, 3))
                vxyz[:, 0] = (txyz[:, 0] - xyz[:, 0]) / Tf
                vxyz[:, 1] = (txyz[:, 1] - xyz[:, 1]) / Tf
                vxyz[:, 2] = 9.81 * Tf / 2

                # Position of ball at 0.1 second before falling onto the plate
                Tst = torch.rand((b,)) * 0.04 + 0.08
                dxyz = torch.zeros((b, 3))
                dxyz[:, 0] = xyz[:, 0] + vxyz[:, 0] * (Tf - Tst)
                dxyz[:, 1] = xyz[:, 1] + vxyz[:, 1] * (Tf - Tst)
                dxyz[:, 2] = xyz[:, 2] + vxyz[:, 2] * (Tf - Tst) - 9.81 / 2 * (Tf - Tst) ** 2

                # Modify velocities at this position
                vxyz[:, 2] = vxyz[:, 2] - 9.81 * (Tf - Tst)

                # Ball velocity at the catch position
                tvxyz = vxyz.clone()
                tvxyz[:, 2] = vxyz[:, 2] - 9.81 * Tst

            obj_pose = Pose.create_from_pq(p=dxyz, q=None)
            for obj in self.objs:
                for i, _obj in enumerate(obj._objs):
                    rigid_component = _obj.find_component_by_type(sapien.physx.PhysxRigidBodyComponent)
                    for shape in rigid_component.collision_shapes:
                        shape.physical_material.dynamic_friction = self._batched_episode_rng[i].uniform(low=0.0, high=0.1)
                        shape.physical_material.static_friction = self._batched_episode_rng[i].uniform(low=0.0, high=0.1)
                        shape.physical_material.restitution = self._batched_episode_rng[i].uniform(low=0.4, high=0.7)

                obj.set_pose(obj_pose)
                obj.set_linear_velocity(vxyz)
            return txyz, tvxyz

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
            with torch.device(self.device):
                self.task_scene.initialize(env_idx)
                txyz, tvxyz = self.initialize_objs(env_idx, options)
                self.initialize_robot(env_idx, options, txyz=txyz, tvxyz=tvxyz)

        def set_state(self, options):
            env_idx = torch.arange(0, self.num_envs, device=self.device)
            with torch.device(self.device):
                self.initialize_robot(env_idx, options)
                self.initialize_objs(env_idx, options)
            info = self.get_info()
            obs = self.get_obs(info)
            return obs, info

        def evaluate(self):
            with torch.device(self.device):
                if self.num_objs > 0:
                    objs_pos = torch.stack([obj.pose.p for obj in self.objs])
                    objs_vel = torch.stack([obj.linear_velocity for obj in self.objs])
                    tcp_pos = self.agent.tcp_pos.reshape(1, self.num_envs, -1)
                    tcp_quat = self.agent.tcp_quat

                    tcp_normals = quaternion_to_matrix(tcp_quat)[:, 0:3, 2].reshape(1, self.num_envs, -1)
                    tcp_normals = tcp_normals / torch.norm(tcp_normals, dim=-1, keepdim=True)

                    diff = objs_pos - tcp_pos
                    w = torch.sum(diff * tcp_normals, dim=-1, keepdim=True)
                    u = torch.norm(diff - w * tcp_normals, dim=-1)
                    w = w.squeeze(-1)

                    wdot = torch.sum(objs_vel * tcp_normals, dim=-1, keepdim=True)
                    udot = torch.norm(objs_vel - wdot * tcp_normals, dim=-1)
                    wdot = wdot.squeeze(-1)

                    dist = torch.norm(diff, dim=-1)
                    vel = objs_vel.norm(dim=-1)

                    success_dist = (u <= self.dist_thresh) & (w > 0)
                    success_vel = (vel <= 0.2)
                    success = success_dist & success_vel
                    success = success.sum(0) / self.num_objs
                    return {
                        "success": success > 0.5,
                        "success_rate": success,
                        "success_dist": success_dist.sum(0) / self.num_objs,
                        "success_vel": success_vel.sum(0) / self.num_objs,
                        "dist_mean": dist.mean(0),
                        "vel_mean": vel.mean(0),
                        "u_mean": u.mean(0),
                        "w_mean": w.abs().mean(0),
                        "udot_mean": udot.mean(0),
                        "wdot_mean": wdot.abs().mean(0),
                    }
                else:
                    return {}

        def _get_obs_extra(self, info: Dict):
            if self.num_objs > 0:
                objs_pos = torch.stack([obj.pose.p for obj in self.objs])
                objs_vel = torch.stack([obj.get_linear_velocity() for obj in self.objs])
                objs_pos = torch.swapaxes(objs_pos, 0, 1)
                objs_vel = torch.swapaxes(objs_vel, 0, 1)
                objs_pos = objs_pos.reshape(-1, self.num_objs * 3)
                objs_vel = objs_vel.reshape(-1, self.num_objs * 3)

                obs = dict(
                    obj_pos=objs_pos,
                    obj_vel=objs_vel,
                    tcp_pos=self.agent.tcp_pos,
                    tcp_quat=self.agent.tcp_quat
                )
            else:
                obs = dict(
                    tcp_pos=self.agent.tcp_pos,
                    tcp_quat=self.agent.tcp_quat
                )
            return obs

        def _get_obs_agent(self):
            return dict()

        def thresh_reward(self, values, thresh, decay=1.0):
            """Threshold values to be either 0 or 1 based on the given threshold."""
            rewards = torch.exp(-(values / decay)**2)
            return rewards

        def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
            if self.num_objs > 0:
                obj_pos = obs["extra"]["obj_pos"].reshape(self.num_envs, self.num_objs, -1)
                obj_vel = obs["extra"]["obj_vel"].reshape(self.num_envs, self.num_objs, -1)
                tcp_pos = obs["extra"]["tcp_pos"].reshape(self.num_envs, 1, -1)

                tcp_normals = quaternion_to_matrix(obs["extra"]["tcp_quat"])[:, 0:3, 2].reshape(self.num_envs, 1, -1)
                tcp_normals = tcp_normals / torch.norm(tcp_normals, dim=-1, keepdim=True)

                diff = obj_pos - tcp_pos
                w = torch.sum(diff * tcp_normals, dim=-1, keepdim=True)
                u = torch.norm(diff - w * tcp_normals, dim=-1)
                w = w.squeeze(-1)

                wdot = torch.sum(obj_vel * tcp_normals, dim=-1, keepdim=True)
                udot = torch.norm(obj_vel - wdot * tcp_normals, dim=-1)
                wdot = wdot.squeeze(-1)

                r_udot = self.thresh_reward(udot, self.vel_thresh, decay=0.25).mean(-1)
                r_w = 0.0
                r_wdot = self.thresh_reward(torch.clamp(wdot, min=-0.1), self.vel_thresh, decay=0.25).mean(-1)

                # Penalty if the ball is below the plate and outside plate region
                caged = (w > 0) * (u <= self.dist_thresh)
                r_c = -(~caged).float().mean(-1)

                r_u = self.thresh_reward(torch.clamp(u, min=self.dist_thresh/2), self.dist_thresh, decay=0.1).mean(-1)
                r = 0.4 * r_u + 0.3 * r_udot + 0.3 * r_wdot + r_c
            else:
                r = torch.zeros((self.num_envs,), device=self.device)
            return r

        def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
            max_reward = 1.0
            return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


class CatchBackend(ManiSkillBaseBackend):
    """
    Task-specific backend for object catching.

    Supports both single-environment executor mode and multi-environment
    CAGE evaluation mode (parallel action evaluation via TSIP).
    """

    def __init__(self):
        """Initialize CatchBackend."""
        super().__init__()

        # Task parameters
        self.ball_radius = 0.025
        self.num_objs = 200
        self.dist_thresh = 0.1
        self.vel_thresh = 0.05

        # Device tracking
        self.device = None

    def create_environment(self, env_config: Dict[str, Any]) -> gym.Env:
        """
        Create PlateCatch-v1 environment.

        Args:
            env_config: Environment configuration with keys:
                - num_envs: Number of parallel environments (default: 1)
                - num_objs: Number of objects/balls (default: 1)
                - sim_backend: Simulation backend ('cpu', 'gpu', or 'physx_cuda')
                - render_mode: 'human' for visualization, None for headless
                - render_backend: Render backend ('cpu' or 'gpu')
                - shader: Shader type ('default', 'rt-fast', 'rt')

        Returns:
            ManiSkill PlateCatch-v1 environment
        """
        if not MANISKILL_AVAILABLE:
            raise ImportError("ManiSkill not available. Install with: pip install mani-skill>=2.0.0")

        # Get configuration with defaults
        num_envs = env_config.get('num_envs', 1)
        num_objs = env_config.get('num_objs', 1)
        sim_backend = env_config.get('sim_backend', 'cpu')
        render_mode = env_config.get('render_mode', 'human')
        render_backend = env_config.get('render_backend', 'gpu')
        shader = env_config.get('shader', 'rt-fast')

        # Determine if we should use parallel_in_single_scene for multi-env visualization
        parallel_in_single_scene = (render_mode == 'human' and num_envs > 1)

        # Build environment kwargs
        env_kwargs = {
            'num_envs': num_envs,
            'num_objs': num_objs,
            'obs_mode': 'state',
            'control_mode': 'pd_ee_pose',
            'render_mode': render_mode,
            'sim_backend': sim_backend,
        }

        # Add render options if rendering is enabled
        if render_mode == 'human':
            env_kwargs['parallel_in_single_scene'] = parallel_in_single_scene
            env_kwargs['render_backend'] = render_backend
            env_kwargs['sensor_configs'] = dict(shader_pack=shader)
            env_kwargs['human_render_camera_configs'] = dict(shader_pack=shader)
            env_kwargs['viewer_camera_configs'] = dict(shader_pack=shader)
            env_kwargs['enable_shadow'] = True
            if parallel_in_single_scene:
                env_kwargs['sim_config'] = dict(spacing=1.0)

        # Create environment
        env = gym.make("PlateCatch-v1", **env_kwargs)

        # Store configuration
        self.context.update(env_config)
        self.device = env.unwrapped.device if hasattr(env.unwrapped, 'device') else 'cpu'

        # Initialize viewer before first reset (required by SAPIEN for proper viewer setup)
        if render_mode == 'human':
            try:
                viewer = env.render()
                if viewer is not None and hasattr(viewer, 'paused'):
                    viewer.paused = env_config.get('pause_on_start', True)
                env.render()
            except Exception as e:
                warnings.warn(f"Failed to initialize viewer pause: {e}")

        return env

    def reset_environment(self, env: gym.Env, seed: int = None, options: Dict = None):
        """
        Reset environment without rendering.

        Overrides ManiSkillBaseBackend.reset_environment() which tries to render
        after reset. For CatchBackend, the viewer is already initialized
        in create_environment(), so we just need the raw reset.
        """
        if options is None:
            options = {}
        return env.reset(seed=seed, options=options)

    def get_action_space(self, env: gym.Env) -> gym.Space:
        """Get action space from environment."""
        return env.action_space

    def load_env(self, context: Dict[str, Any]) -> None:
        """Load simulation environment (handled by ManiSkill)."""
        pass

    def load_object(self, context: Dict[str, Any]) -> None:
        """Load objects (handled by ManiSkill)."""
        pass

    def load_robot(self, context: Dict[str, Any]) -> None:
        """Load robot (handled by ManiSkill)."""
        pass

    def get_state(self, env: gym.Env) -> Dict[str, Any]:
        """
        Get current state from environment.

        Returns dictionary with:
        - obj_pos: Ball positions (num_envs, 3)
        - obj_vel: Ball velocities (num_envs, 3)
        - tcp_pos: TCP positions (num_envs, 3)
        - tcp_quat: TCP quaternions (num_envs, 4)
        - qpos: Robot joint positions (num_envs, dof)
        - qvel: Robot joint velocities (num_envs, dof)
        """
        actual_env = env.unwrapped

        state = {}

        # Get ball state (first ball only)
        if hasattr(actual_env, 'objs') and len(actual_env.objs) > 0:
            ball = actual_env.objs[0]
            state['obj_pos'] = ball.pose.p.cpu().numpy() if hasattr(ball.pose.p, 'cpu') else ball.pose.p
            state['obj_vel'] = ball.linear_velocity.cpu().numpy() if hasattr(ball.linear_velocity, 'cpu') else ball.linear_velocity

        # Get robot/TCP state
        if hasattr(actual_env, 'agent'):
            agent = actual_env.agent
            state['tcp_pos'] = agent.tcp_pos.cpu().numpy() if hasattr(agent.tcp_pos, 'cpu') else agent.tcp_pos
            state['tcp_quat'] = agent.tcp_quat.cpu().numpy() if hasattr(agent.tcp_quat, 'cpu') else agent.tcp_quat
            state['qpos'] = agent.qpos.cpu().numpy() if hasattr(agent.qpos, 'cpu') else agent.qpos
            state['qvel'] = agent.qvel.cpu().numpy() if hasattr(agent.qvel, 'cpu') else agent.qvel

        return state

    def set_state(self, env: gym.Env, state: Dict[str, Any]) -> None:
        """
        Set environment state from state dictionary.

        Broadcasts single ball state to ALL balls in ALL environments.
        This creates the DRIS representation where each evaluation env has multiple balls
        all at the same position (representing the single ball from executor).
        """
        actual_env = env.unwrapped
        num_envs = actual_env.num_envs
        num_objs = len(actual_env.objs) if hasattr(actual_env, 'objs') else 0
        device = actual_env.device

        # Set ball state - broadcast to ALL balls in ALL envs
        if 'obj_pos' in state and 'obj_vel' in state:
            obj_pos = state['obj_pos']
            obj_vel = state['obj_vel']

            # Convert to torch if needed
            if isinstance(obj_pos, np.ndarray):
                obj_pos = torch.from_numpy(obj_pos).to(device).float()
            if isinstance(obj_vel, np.ndarray):
                obj_vel = torch.from_numpy(obj_vel).to(device).float()

            # Ensure obj_pos/obj_vel are 1D (single ball position)
            if obj_pos.dim() > 1:
                obj_pos = obj_pos.squeeze()
            if obj_vel.dim() > 1:
                obj_vel = obj_vel.squeeze()

            # Take only first 3 components if needed
            obj_pos = obj_pos[:3]
            obj_vel = obj_vel[:3]

            # Broadcast to all environments: [3] -> [num_envs, 3]
            obj_pos_broadcast = obj_pos.unsqueeze(0).expand(num_envs, -1).contiguous()
            obj_vel_broadcast = obj_vel.unsqueeze(0).expand(num_envs, -1).contiguous()

            # Set ALL balls in ALL environments to the same position (DRIS representation)
            if hasattr(actual_env, 'objs') and num_objs > 0:
                for ball in actual_env.objs:
                    ball_pose = Pose.create_from_pq(p=obj_pos_broadcast)
                    ball.set_pose(ball_pose)
                    ball.set_linear_velocity(obj_vel_broadcast)

        # Set robot state
        if 'qpos' in state and 'qvel' in state:
            qpos = state['qpos']
            qvel = state['qvel']

            # Convert to torch if needed
            if isinstance(qpos, np.ndarray):
                qpos = torch.from_numpy(qpos).to(device)
            if isinstance(qvel, np.ndarray):
                qvel = torch.from_numpy(qvel).to(device)

            # Broadcast single state to all environments if needed
            if len(qpos.shape) == 1:
                qpos = qpos.unsqueeze(0).expand(num_envs, -1)
                qvel = qvel.unsqueeze(0).expand(num_envs, -1)
            elif qpos.shape[0] == 1 and num_envs > 1:
                qpos = qpos.expand(num_envs, -1)
                qvel = qvel.expand(num_envs, -1)

            if hasattr(actual_env, 'agent'):
                actual_env.agent.set_qpos_and_qvel(qpos, qvel)

        # Apply GPU state if needed
        if device.type == "cuda":
            actual_env.scene._gpu_apply_all()
            actual_env.scene.px.gpu_update_articulation_kinematics()
            actual_env.scene._gpu_fetch_all()

    def step_act(self, actions: Union[Any, List[Any]], env: gym.Env = None,
                 cage=None, single_action=False) -> Any:
        """
        Execute actions in parallel environments.

        Args:
            actions: List of actions (one per environment) or single action
            env: PlateCatch environment
            cage: Cage object (not used for catching)
            single_action: Whether this is a single action to replicate

        Returns:
            Observations after stepping
        """
        actual_env = env.unwrapped
        num_envs = actual_env.num_envs

        # Convert actions to tensor
        if isinstance(actions, list):
            if len(actions) == 1 or single_action:
                # Replicate single action for all environments
                action = actions[0] if isinstance(actions, list) else actions
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action).to(actual_env.device)
                while action.dim() > 1:
                    action = action.squeeze(0)
                action_tensor = action.unsqueeze(0).expand(num_envs, -1)
            else:
                # Stack multiple actions
                action_list = []
                for a in actions:
                    if isinstance(a, np.ndarray):
                        a = torch.from_numpy(a).to(actual_env.device)
                    while a.dim() > 1:
                        a = a.squeeze(0)
                    action_list.append(a)
                action_tensor = torch.stack(action_list)
        else:
            # Single action tensor
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(actual_env.device)
            while actions.dim() > 1:
                actions = actions.squeeze(0)
            action_tensor = actions.unsqueeze(0).expand(num_envs, -1)

        # Ensure correct shape
        if action_tensor.shape[0] != num_envs:
            if action_tensor.shape[0] == 1:
                action_tensor = action_tensor.expand(num_envs, -1)
            else:
                raise ValueError(f"Action batch size {action_tensor.shape[0]} != num_envs {num_envs}")

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_tensor)

        return obs

    def state2dris(self, observations: Any, env_indices: List[int] = None,
                   env_config: Dict = None) -> List['DRIS']:
        """
        Convert observations to DRIS format for CAGE evaluation.

        Args:
            observations: Observation dictionary or state dictionary
            env_indices: Optional indices to filter (not used)
            env_config: Environment configuration

        Returns:
            List of DRIS objects (one per environment)
        """
        from manidreams.base.dris import DRIS

        dris_list = []

        # Get number of environments
        num_envs = env_config.get('num_envs', 8) if env_config else 8

        # Extract state components
        obj_pos = None
        obj_vel = None
        tcp_pos = None
        tcp_quat = None

        if isinstance(observations, dict):
            if 'extra' in observations:
                extra = observations['extra']
                obj_pos = extra.get('obj_pos')
                obj_vel = extra.get('obj_vel')
                tcp_pos = extra.get('tcp_pos')
                tcp_quat = extra.get('tcp_quat')
            else:
                obj_pos = observations.get('obj_pos')
                obj_vel = observations.get('obj_vel')
                tcp_pos = observations.get('tcp_pos')
                tcp_quat = observations.get('tcp_quat')

        # Convert to numpy
        def to_numpy(x):
            if x is None:
                return None
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
            return np.array(x)

        obj_pos = to_numpy(obj_pos)
        obj_vel = to_numpy(obj_vel)
        tcp_pos = to_numpy(tcp_pos)
        tcp_quat = to_numpy(tcp_quat)

        # Create DRIS for each environment
        for i in range(num_envs):
            if obj_pos is not None and len(obj_pos.shape) >= 2:
                env_obj_pos = obj_pos[i, :3] if obj_pos.shape[1] >= 3 else obj_pos[i]
            elif obj_pos is not None:
                env_obj_pos = obj_pos[:3]
            else:
                env_obj_pos = np.zeros(3)

            if obj_vel is not None and len(obj_vel.shape) >= 2:
                env_obj_vel = obj_vel[i, :3] if obj_vel.shape[1] >= 3 else obj_vel[i]
            elif obj_vel is not None:
                env_obj_vel = obj_vel[:3]
            else:
                env_obj_vel = np.zeros(3)

            if tcp_pos is not None and len(tcp_pos.shape) >= 2:
                env_tcp_pos = tcp_pos[i]
            elif tcp_pos is not None:
                env_tcp_pos = tcp_pos
            else:
                env_tcp_pos = np.zeros(3)

            if tcp_quat is not None and len(tcp_quat.shape) >= 2:
                env_tcp_quat = tcp_quat[i]
            elif tcp_quat is not None:
                env_tcp_quat = tcp_quat
            else:
                env_tcp_quat = np.array([1.0, 0.0, 0.0, 0.0])

            # Format: [obj_pos(3), obj_vel(3), tcp_pos(3), tcp_quat(4)] = 13 dims
            observation = np.concatenate([
                env_obj_pos.flatten()[:3],
                env_obj_vel.flatten()[:3],
                env_tcp_pos.flatten()[:3],
                env_tcp_quat.flatten()[:4]
            ])

            dris = DRIS(
                observation=observation,
                representation_type="state",
                context={
                    'obj_pos': env_obj_pos,
                    'obj_vel': env_obj_vel,
                    'tcp_pos': env_tcp_pos,
                    'tcp_quat': env_tcp_quat
                }
            )
            dris_list.append(dris)

        return dris_list

    def dris2state(self, dris: Any) -> Dict[str, np.ndarray]:
        """
        Convert DRIS to state dictionary for set_state.
        """
        state = {}

        if hasattr(dris, 'context') and dris.context:
            ctx = dris.context
            for key in ['obj_pos', 'obj_vel', 'tcp_pos', 'tcp_quat', 'qpos', 'qvel']:
                if key in ctx:
                    state[key] = np.array(ctx[key])
        elif hasattr(dris, 'observation') and dris.observation is not None:
            obs = dris.observation
            if len(obs) >= 13:
                state['obj_pos'] = obs[0:3]
                state['obj_vel'] = obs[3:6]
                state['tcp_pos'] = obs[6:9]
                state['tcp_quat'] = obs[9:13]

        return state

    def sync_state_from_obs(self, env: gym.Env, obs: Dict[str, Any]) -> None:
        """
        Synchronize environment state from observation dictionary.
        Converts observation format to state format and calls set_state.
        """
        state = {}

        # Handle different observation formats
        if 'extra' in obs:
            extra = obs['extra']
            for key in ['obj_pos', 'obj_vel', 'tcp_pos', 'tcp_quat']:
                if key in extra:
                    val = extra[key]
                    if hasattr(val, 'cpu'):
                        val = val.cpu().numpy()
                    if key in ('obj_pos', 'obj_vel') and len(val.shape) == 1:
                        val = val[:3]
                    elif len(val.shape) == 2:
                        val = val[0, :3] if key in ('obj_pos', 'obj_vel') else val[0]
                    state[key] = val

        # Direct format
        for key in ['obj_pos', 'obj_vel', 'tcp_pos', 'tcp_quat', 'qpos', 'qvel']:
            if key in obs and key not in state:
                val = obs[key]
                if hasattr(val, 'cpu'):
                    val = val.cpu().numpy()
                if len(val.shape) == 2:
                    val = val[0]
                state[key] = val

        # Get robot state from environment if not in obs
        if 'qpos' not in state or 'qvel' not in state:
            actual_env = env.unwrapped
            if hasattr(actual_env, 'agent'):
                agent = actual_env.agent
                qpos = agent.qpos
                qvel = agent.qvel
                if hasattr(qpos, 'cpu'):
                    qpos = qpos.cpu().numpy()
                    qvel = qvel.cpu().numpy()
                if len(qpos.shape) == 2:
                    qpos = qpos[0]
                    qvel = qvel[0]
                state['qpos'] = qpos
                state['qvel'] = qvel

        self.set_state(env, state)
