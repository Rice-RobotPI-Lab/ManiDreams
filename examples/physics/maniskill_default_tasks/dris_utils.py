"""
DRIS Mixin for ManiSkill Environments

Provides Domain-Randomized Intuitive State (DRIS) functionality:
- Create n copies of target object with collision isolation
- Randomize copy poses based on pose_noise (position + orientation)
- Override reward to use mean of per-copy rewards
- Compute DRIS variance for uncertainty estimation
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import sapien
from transforms3d.euler import euler2quat, quat2euler

from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.building import actors

from .task_config import TaskConfig, ObjectConfig


class DRISMixin:
    """
    Mixin class that adds DRIS functionality to ManiSkill environments.

    Must be mixed with a BaseEnv subclass.

    Attributes (set by factory):
        n_dris_copies: Number of DRIS copies
        pose_noise: (dx, dy, dz, droll, dpitch, dyaw) pose randomization range
        task_config: TaskConfig for this task
    """

    # These will be set by the factory
    n_dris_copies: int
    pose_noise: Tuple[float, float, float, float, float, float]
    physics_noise: Tuple[float, float]   # (dfric, dmass_ratio)
    task_config: TaskConfig

    # DRIS objects storage
    dris_objects: List[Actor]
    _dris_materials: List  # sapien.physx.PhysxMaterial, one per copy
    _dris_base_mass: List[float]  # nominal mass, one per copy

    # Collision group for all DRIS objects (same group = no inter-collision)
    # g0/g1 use low bits for broadphase collision filtering.
    # g2 is the self-collision exclusion mask: objects sharing any g2 bit
    # are excluded from colliding.  Robot self-collision uses low bits 0-3,
    # so we must use a HIGH bit (bit 20) for DRIS exclusion to avoid
    # accidentally disabling robot-vs-DRIS collisions.
    DRIS_COLLISION_GROUP = 6
    DRIS_SELF_EXCLUSION_BIT = 1 << 20  # 0x00100000 — no overlap with robot g2

    def _set_target_transparent(self, alpha: float = 0.5) -> None:
        """Make the original target object semi-transparent."""
        target = getattr(self, self.task_config.target_attr, None)
        if target is None:
            return
        try:
            objs = target._objs if hasattr(target, '_objs') else [target._obj]
            for obj in objs:
                render_body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
                if render_body is None:
                    continue
                for shape in render_body.render_shapes:
                    mat = shape.material
                    c = mat.base_color
                    mat.base_color = [c[0], c[1], c[2], alpha]
        except Exception:
            pass

    def _set_target_collision_group(self) -> None:
        """
        Set the original target object to use the same collision group as DRIS copies.

        This allows DRIS copies to overlap with the target object when pose_noise=0.
        """
        target = getattr(self, self.task_config.target_attr, None)
        if target is None:
            return

        # Set collision group for all actors in the target
        collision_group = self.DRIS_COLLISION_GROUP  # 6
        collision_mask = 1 | collision_group  # 7
        exclusion = self.DRIS_SELF_EXCLUSION_BIT  # high bit for g2

        # Handle both single actor and merged actors
        if hasattr(target, '_objs'):
            # Merged actor - iterate over all internal objects
            for obj in target._objs:
                for body in obj.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).collision_shapes:
                    body.set_collision_groups([collision_group, collision_mask, exclusion, 0])
        else:
            # Single actor
            try:
                for body in target.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).collision_shapes:
                    body.set_collision_groups([collision_group, collision_mask, exclusion, 0])
            except Exception:
                pass  # Ignore if unable to set collision groups

    def _create_dris_copies(self) -> None:
        """
        Create DRIS copies of the target object.

        Called in _load_scene after the original object is created.
        All copies use the same collision group to prevent inter-collision.
        """
        self.dris_objects = []
        self._dris_materials = []
        self._dris_base_mass = []

        # First, set the original target object to use the same collision group
        # so DRIS copies can overlap with it when pose_noise=0
        self._set_target_collision_group()

        obj_config = self.task_config.object_config
        builder_type = obj_config.builder_type
        params = obj_config.params

        for copy_idx in range(self.n_dris_copies):
            # Get distinct color for this copy
            color = self._get_dris_copy_color(copy_idx)

            # Create object based on type
            if builder_type == "cube":
                dris_obj = self._create_cube_copy(copy_idx, params, color)
            elif builder_type == "sphere":
                dris_obj = self._create_sphere_copy(copy_idx, params, color)
            elif builder_type == "tee":
                dris_obj = self._create_tee_copy(copy_idx, params, color)
            else:
                raise ValueError(f"Unknown builder type: {builder_type}")

            self.dris_objects.append(dris_obj)

    def _create_cube_copy(
        self,
        copy_idx: int,
        params: Dict[str, Any],
        color: List[float]
    ) -> Actor:
        """Create a cube DRIS copy."""
        half_size = params.get("half_size", 0.02)

        # Use ManiSkill's actors.build_cube but we need to set collision groups
        # So we build manually
        builder = self.scene.create_actor_builder()

        # Physics material
        material = sapien.physx.PhysxMaterial(
            static_friction=params.get("static_friction", 0.5),
            dynamic_friction=params.get("dynamic_friction", 0.5),
            restitution=params.get("restitution", 0.0),
        )

        # Add collision
        builder.add_box_collision(
            half_size=[half_size, half_size, half_size],
            material=material,
        )

        # Add visual
        builder.add_box_visual(
            half_size=[half_size, half_size, half_size],
            material=sapien.render.RenderMaterial(base_color=color),
        )

        # Set mass
        base_mass = params.get("mass", None)
        if base_mass is not None:
            builder._mass = base_mass

        # Set collision group
        collision_group = self.DRIS_COLLISION_GROUP  # 6
        collision_mask = 1 | collision_group  # 7
        exclusion = self.DRIS_SELF_EXCLUSION_BIT  # high bit for g2
        builder.collision_groups = [collision_group, collision_mask, exclusion, 0]

        # Initial pose (will be set in _initialize_episode)
        builder.initial_pose = sapien.Pose(p=[0, 0, half_size + 0.5])  # High up initially

        actor = builder.build(name=f"dris_copy_{copy_idx}")

        # Store material reference (for per-episode friction randomization)
        self._dris_materials.append(material)
        # Store base mass (read back from component after build)
        try:
            component = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            self._dris_base_mass.append(float(component.mass))
        except Exception:
            self._dris_base_mass.append(base_mass if base_mass is not None else 1.0)

        return actor

    def _create_sphere_copy(
        self,
        copy_idx: int,
        params: Dict[str, Any],
        color: List[float]
    ) -> Actor:
        """Create a sphere DRIS copy."""
        radius = params.get("radius", 0.035)

        builder = self.scene.create_actor_builder()

        material = sapien.physx.PhysxMaterial(
            static_friction=params.get("static_friction", 0.5),
            dynamic_friction=params.get("dynamic_friction", 0.5),
            restitution=params.get("restitution", 0.0),
        )

        builder.add_sphere_collision(
            radius=radius,
            material=material,
        )

        builder.add_sphere_visual(
            radius=radius,
            material=sapien.render.RenderMaterial(base_color=color),
        )

        base_mass = params.get("mass", None)
        if base_mass is not None:
            builder._mass = base_mass

        # Set collision group
        collision_group = self.DRIS_COLLISION_GROUP  # 6
        collision_mask = 1 | collision_group  # 7
        exclusion = self.DRIS_SELF_EXCLUSION_BIT  # high bit for g2
        builder.collision_groups = [collision_group, collision_mask, exclusion, 0]

        builder.initial_pose = sapien.Pose(p=[0, 0, radius + 0.5])

        actor = builder.build(name=f"dris_copy_{copy_idx}")

        self._dris_materials.append(material)
        try:
            component = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            self._dris_base_mass.append(float(component.mass))
        except Exception:
            self._dris_base_mass.append(base_mass if base_mass is not None else 1.0)

        return actor

    def _create_tee_copy(
        self,
        copy_idx: int,
        params: Dict[str, Any],
        color: List[float]
    ) -> Actor:
        """Create a T-shaped DRIS copy."""
        builder = self.scene.create_actor_builder()

        material = sapien.physx.PhysxMaterial(
            static_friction=params.get("static_friction", 3.0),
            dynamic_friction=params.get("dynamic_friction", 3.0),
            restitution=params.get("restitution", 0.0),
        )

        # Box 1 (horizontal part of T)
        box1_half_size = params.get("box1_half_size", [0.1, 0.025, 0.02])
        box1_offset = params.get("box1_offset", [0.0, -0.0375, 0.0])
        box1_pose = sapien.Pose(p=box1_offset)

        builder.add_box_collision(
            pose=box1_pose,
            half_size=box1_half_size,
            material=material,
        )
        builder.add_box_visual(
            pose=box1_pose,
            half_size=box1_half_size,
            material=sapien.render.RenderMaterial(base_color=color),
        )

        # Box 2 (vertical part of T)
        box2_half_size = params.get("box2_half_size", [0.025, 0.075, 0.02])
        box2_offset = params.get("box2_offset", [0.0, 0.0625, 0.0])
        box2_pose = sapien.Pose(p=box2_offset)

        builder.add_box_collision(
            pose=box2_pose,
            half_size=box2_half_size,
            material=material,
        )
        builder.add_box_visual(
            pose=box2_pose,
            half_size=box2_half_size,
            material=sapien.render.RenderMaterial(base_color=color),
        )

        base_mass = params.get("mass", None)
        if base_mass is not None:
            builder._mass = base_mass

        # Set collision group
        collision_group = self.DRIS_COLLISION_GROUP  # 6
        collision_mask = 1 | collision_group  # 7
        exclusion = self.DRIS_SELF_EXCLUSION_BIT  # high bit for g2
        builder.collision_groups = [collision_group, collision_mask, exclusion, 0]

        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])

        actor = builder.build(name=f"dris_copy_{copy_idx}")

        self._dris_materials.append(material)
        try:
            component = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            self._dris_base_mass.append(float(component.mass))
        except Exception:
            self._dris_base_mass.append(base_mass if base_mass is not None else 1.0)

        return actor

    def _get_dris_copy_color(self, copy_idx: int) -> List[float]:
        """
        Get color for DRIS copy based on original object color.

        Uses the original object's color with ~10% RGB randomness and 0.8 alpha.
        """
        # Get original color from task config
        original_color = self.task_config.object_config.params.get("color", [0.5, 0.5, 0.5, 1.0])

        # Convert to list if numpy array
        if hasattr(original_color, 'tolist'):
            original_color = original_color.tolist()
        else:
            original_color = list(original_color)

        # Ensure we have 4 values (RGBA)
        if len(original_color) == 3:
            original_color = original_color + [1.0]

        # Add ~10% randomness to RGB (not alpha)
        randomness = 0.1
        color = []
        for i in range(3):  # RGB only
            noise = (np.random.random() * 2 - 1) * randomness  # [-0.1, 0.1]
            value = original_color[i] + noise
            value = max(0.0, min(1.0, value))  # Clamp to [0, 1]
            color.append(value)

        # Set alpha for transparency
        color.append(0.6)

        return color

    def _randomize_dris_poses(self, env_idx: torch.Tensor) -> None:
        """
        Randomize DRIS copy poses (position + orientation) around the target object.

        Called in _initialize_episode after the target object is positioned.

        pose_noise format: (dx, dy, dz, droll, dpitch, dyaw)
        - dx, dy, dz: position noise range (meters)
        - droll, dpitch, dyaw: orientation noise range (radians)
        """
        # Check if DRIS objects exist
        if not hasattr(self, 'dris_objects') or self.dris_objects is None:
            return

        b = len(env_idx)

        # Get target object pose
        target = getattr(self, self.task_config.target_attr)
        target_pose = target.pose

        # Get base position and quaternion - keep on device
        if hasattr(target_pose, 'raw_pose'):
            raw_pose = target_pose.raw_pose
            # Keep on original device, don't move to cpu
            base_positions = raw_pose[..., :3].clone()
            base_quats = raw_pose[..., 3:7].clone()
        else:
            base_positions = target_pose.p
            base_quats = target_pose.q

        # Convert to torch if needed and ensure on correct device
        if not isinstance(base_positions, torch.Tensor):
            base_positions = torch.tensor(base_positions, device=self.device)
            base_quats = torch.tensor(base_quats, device=self.device)
        else:
            # Ensure on correct device
            base_positions = base_positions.to(self.device)
            base_quats = base_quats.to(self.device)

        # Ensure correct shape [b, 3] and [b, 4]
        if len(base_positions.shape) == 1:
            base_positions = base_positions.unsqueeze(0).expand(b, -1).contiguous()
            base_quats = base_quats.unsqueeze(0).expand(b, -1).contiguous()

        # Parse pose_noise: (dx, dy, dz, droll, dpitch, dyaw)
        dx, dy, dz, droll, dpitch, dyaw = self.pose_noise

        # Randomize each DRIS copy
        for copy_idx, dris_obj in enumerate(self.dris_objects):
            # Generate random position offsets
            pos_offsets = torch.zeros((b, 3), device=self.device)
            pos_offsets[:, 0] = torch.rand(b, device=self.device) * 2 * dx - dx
            pos_offsets[:, 1] = torch.rand(b, device=self.device) * 2 * dy - dy
            pos_offsets[:, 2] = torch.rand(b, device=self.device) * 2 * dz - dz

            # Compute new positions
            new_positions = base_positions + pos_offsets

            # Generate random orientation offsets (euler angles)
            euler_offsets = torch.zeros((b, 3), device=self.device)
            euler_offsets[:, 0] = torch.rand(b, device=self.device) * 2 * droll - droll    # roll
            euler_offsets[:, 1] = torch.rand(b, device=self.device) * 2 * dpitch - dpitch  # pitch
            euler_offsets[:, 2] = torch.rand(b, device=self.device) * 2 * dyaw - dyaw      # yaw

            # Apply orientation noise by converting to quaternion and multiplying
            new_quats = self._apply_euler_noise_to_quat(base_quats, euler_offsets)

            # Set pose
            dris_pose = Pose.create_from_pq(p=new_positions, q=new_quats)
            dris_obj.set_pose(dris_pose)

        self._randomize_dris_physics()

    def _randomize_dris_physics(self) -> None:
        """
        Randomize friction and mass of each DRIS copy.

        Called at the end of _randomize_dris_poses() so that physics parameters
        are re-drawn whenever poses are re-drawn (episode init + set_state sync).

        physics_noise format: (dfric, dmass_ratio)
        - dfric: ±delta added to static and dynamic friction
        - dmass_ratio: mass multiplied by U[1-r, 1+r]
        """
        if not hasattr(self, 'physics_noise'):
            return
        dfric, dmass_ratio = self.physics_noise
        if dfric == 0.0 and dmass_ratio == 0.0:
            return

        base_fric_s = self.task_config.object_config.params.get("static_friction", 0.5)
        base_fric_d = self.task_config.object_config.params.get("dynamic_friction", 0.5)

        for i, dris_obj in enumerate(self.dris_objects):
            # --- Friction ---
            if dfric != 0.0 and i < len(self._dris_materials):
                noise_s = (np.random.rand() * 2 - 1) * dfric
                noise_d = (np.random.rand() * 2 - 1) * dfric
                self._dris_materials[i].static_friction = float(
                    np.clip(base_fric_s + noise_s, 0.0, 10.0)
                )
                self._dris_materials[i].dynamic_friction = float(
                    np.clip(base_fric_d + noise_d, 0.0, 10.0)
                )

            # --- Mass ---
            if dmass_ratio != 0.0 and i < len(self._dris_base_mass):
                ratio = 1.0 + (np.random.rand() * 2 - 1) * dmass_ratio
                new_mass = float(np.clip(self._dris_base_mass[i] * ratio, 1e-4, None))
                try:
                    component = dris_obj.find_component_by_type(
                        sapien.physx.PhysxRigidDynamicComponent
                    )
                    component.mass = new_mass
                except Exception:
                    pass

        # Note: no manual _gpu_apply_all() here.
        # component.mass is CPU-side; it will be pushed to GPU by the
        # next _gpu_apply_all() call made by the caller (reset() or set_state()).

    def _apply_euler_noise_to_quat(
        self,
        base_quats: torch.Tensor,
        euler_offsets: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply euler angle noise to quaternions.

        Args:
            base_quats: Base quaternions [b, 4] in (w, x, y, z) format
            euler_offsets: Euler angle offsets [b, 3] in (roll, pitch, yaw)

        Returns:
            New quaternions [b, 4]
        """
        b = base_quats.shape[0]
        new_quats = torch.zeros_like(base_quats)

        for i in range(b):
            # Convert euler offset to quaternion
            roll, pitch, yaw = euler_offsets[i].cpu().numpy()
            offset_quat = euler2quat(roll, pitch, yaw, 'sxyz')  # returns (w, x, y, z)

            # Convert to torch tensor
            offset_quat_tensor = torch.tensor(offset_quat, device=self.device, dtype=base_quats.dtype)

            # Quaternion multiplication: q_new = q_offset * q_base
            new_quats[i] = self._quat_multiply(offset_quat_tensor, base_quats[i])

        return new_quats

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiply two quaternions (w, x, y, z format).

        Args:
            q1, q2: Quaternions in (w, x, y, z) format

        Returns:
            Product quaternion (w, x, y, z)
        """
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z])

    def _get_dris_positions(self) -> torch.Tensor:
        """
        Get positions of all DRIS copies.

        Returns:
            Tensor of shape [num_envs, n_dris_copies, 3]
        """
        positions = []
        for dris_obj in self.dris_objects:
            pos = dris_obj.pose.p  # [num_envs, 3]
            positions.append(pos)

        # Stack: [n_copies, num_envs, 3] -> [num_envs, n_copies, 3]
        stacked = torch.stack(positions, dim=0)
        return stacked.permute(1, 0, 2)

    def _get_dris_poses(self) -> torch.Tensor:
        """
        Get full poses of all DRIS copies.

        Returns:
            Tensor of shape [num_envs, n_dris_copies, 7]
        """
        poses = []
        for dris_obj in self.dris_objects:
            pose = dris_obj.pose.raw_pose  # [num_envs, 7]
            poses.append(pose)

        stacked = torch.stack(poses, dim=0)
        return stacked.permute(1, 0, 2)

    def _compute_dris_variance(self) -> torch.Tensor:
        """
        Compute position variance across DRIS copies.

        Returns:
            Tensor of shape [num_envs, 3] - variance in x, y, z
        """
        positions = self._get_dris_positions()  # [num_envs, n_copies, 3]
        variance = positions.var(dim=1)  # [num_envs, 3]
        return variance

    def _compute_dris_mean_position(self) -> torch.Tensor:
        """
        Compute mean position across DRIS copies.

        Returns:
            Tensor of shape [num_envs, 3]
        """
        positions = self._get_dris_positions()  # [num_envs, n_copies, 3]
        mean_pos = positions.mean(dim=1)  # [num_envs, 3]
        return mean_pos

    def _sync_dris_to_pose(self, target_pose: torch.Tensor) -> None:
        """
        Sync all DRIS copies to a target pose.

        Args:
            target_pose: Tensor of shape [num_envs, 7] or [num_envs, n_copies, 7]
        """
        if len(target_pose.shape) == 2:
            # Single pose - broadcast to all copies
            for dris_obj in self.dris_objects:
                dris_obj.set_pose(Pose.create_from_pq(
                    p=target_pose[..., :3],
                    q=target_pose[..., 3:7]
                ))
        else:
            # Per-copy poses
            for copy_idx, dris_obj in enumerate(self.dris_objects):
                pose = target_pose[:, copy_idx, :]
                dris_obj.set_pose(Pose.create_from_pq(
                    p=pose[..., :3],
                    q=pose[..., 3:7]
                ))

    def _compute_per_copy_goal_distances(self) -> torch.Tensor:
        """
        Compute distance to goal for each DRIS copy.

        Returns:
            Tensor of shape [num_envs, n_copies]
        """
        if self.task_config.goal_attr is None:
            # No goal object - return zeros
            return torch.zeros(
                self.num_envs, self.n_dris_copies,
                device=self.device
            )

        # Get goal position
        goal_obj = getattr(self, self.task_config.goal_attr)
        goal_pos = goal_obj.pose.p  # [num_envs, 3]

        # Get DRIS positions
        dris_positions = self._get_dris_positions()  # [num_envs, n_copies, 3]

        # Compute distances (2D - x,y only for most tabletop tasks)
        distances = torch.linalg.norm(
            dris_positions[..., :2] - goal_pos[..., :2].unsqueeze(1),
            dim=-1
        )  # [num_envs, n_copies]

        return distances

    def _compute_dris_rewards(self, base_reward_func, obs, action, info) -> torch.Tensor:
        """
        Compute reward using mean of per-copy rewards.

        Temporarily moves each DRIS copy to the target position,
        computes reward, then restores.

        Args:
            base_reward_func: Original reward function
            obs: Observation
            action: Action
            info: Info dict

        Returns:
            Mean reward across DRIS copies [num_envs]
        """
        # Get original target object
        target = getattr(self, self.task_config.target_attr)
        original_pose = target.pose.raw_pose.clone()

        rewards = []

        for dris_obj in self.dris_objects:
            # Move target to DRIS copy position
            dris_pose = dris_obj.pose
            target.set_pose(dris_pose)

            # Compute reward
            reward = base_reward_func(obs, action, info)
            rewards.append(reward)

        # Restore original pose
        target.set_pose(Pose.create_from_pq(
            p=original_pose[..., :3],
            q=original_pose[..., 3:7]
        ))

        # Stack and mean
        rewards_stacked = torch.stack(rewards, dim=1)  # [num_envs, n_copies]
        mean_reward = rewards_stacked.mean(dim=1)  # [num_envs]

        return mean_reward

    def get_dris_info(self) -> Dict[str, torch.Tensor]:
        """
        Get DRIS-related information for info dict.

        Returns:
            Dict with:
            - dris_poses: [num_envs, n_copies, 7]
            - dris_variance: [num_envs, 3]
            - dris_mean_position: [num_envs, 3]
            - dris_goal_distances: [num_envs, n_copies] (if goal exists)
        """
        info = {
            'dris_poses': self._get_dris_poses(),
            'dris_variance': self._compute_dris_variance(),
            'dris_mean_position': self._compute_dris_mean_position(),
        }

        if self.task_config.goal_attr is not None:
            info['dris_goal_distances'] = self._compute_per_copy_goal_distances()

        return info
