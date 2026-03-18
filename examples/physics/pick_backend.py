"""
Pick Backend for ManiDreams Framework

Task-specific backend for picking colored box target objects.
Adapted for ManiDreams framework.
"""

from typing import Any, Dict
import logging
import numpy as np
import warnings

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    MANISKILL_AVAILABLE = True
except ImportError:
    MANISKILL_AVAILABLE = False
    warnings.warn("ManiSkill not available. Install with: pip install mani-skill>=2.0.0")

import math
from .maniskill_base import ManiSkillBaseBackend, TableSetupMixin


class PickBackend(ManiSkillBaseBackend, TableSetupMixin):
    """
    Task-specific backend for picking colored box target objects.

    Features:
    - Multiple colored box (0.04 x 0.06 x 0.01 m) instances as target objects with random colors
    - Simplified action control (approach -> position, no grasp)
    - Inherits table setup from TableSetupMixin
    """

    def __init__(self):
        ManiSkillBaseBackend.__init__(self)
        TableSetupMixin.__init__(self)

        # Task parameters similar to pushing task
        self.device = None
        self.sequence_len = 100  # Total steps per action sequence (4 * 25)
        self.step_len = 6       # Steps per phase
        self.r_approach = 0.15   # Approach distance (shorter for picking)
        self.Kp_pos = 1.0        # Position control gain
        self.Kp_ori = 2.0        # Orientation control gain
        self.Kp_pos_grasp = 1.5  # Reduced gain for grasping
        self.num_directions = 16 # 16 approach directions (0-15)

        # Picking-specific parameters
        self.grasp_height = 0.05  # Height to grasp objects
        self.lift_height = 0.15   # Height to lift objects

        # Target object configuration - red box with specific dimensions
        self.target_object_type = "red_box"  # Custom red box
        self.target_box_half_size = [0.03, 0.04, 0.001]  # Half-size for (0.04, 0.06, 0.01)
        self.num_target_objects = 16  # Multiple target instances
        self.num_clutter_objects = 0  # No clutter objects

        # Wall obstacle parameters
        self.wall_enabled = True  # Enable wall obstacle
        self.wall_position = [0.0, 0.5, 0.2]  # Position on table edge [x, y, z]
        self.wall_half_size = [0.5, 0.1, 0.2]  # Half-size: 30cm wide x 2cm deep x 15cm tall
        self.wall_color = [0.7, 0.7, 0.7, 1.0]  # Light gray color

        self.current_context = {}
        
    def step_act(self, actions, env: gym.Env = None, cage=None, single_action=False,
                 visualize_cage=True) -> Any:
        """
        Execute 6D offset actions for MPPI-based picking.

        Complete workflow:
        1. Compute reference action from cage center and target position
        2. Apply offset: final_action = reference + offset
        3. Execute final_action using PD control

        Args:
            actions: List of [6] offsets [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
                    or single [6] offset array
            env: ManiSkill environment
            cage: Cage object (used to compute reference action)
            single_action: Whether this is a single action (for rendering control)
            visualize_cage: Whether to show green cage center sphere (default True)

        Returns:
            Final observation after executing the offset action
        """
        import torch

        # === Step 0: Initialize ===
        self.device = env.unwrapped.device if hasattr(env.unwrapped, 'device') else None
        num_envs = len(env.agent.tcp.pose.p) if hasattr(env, 'agent') and hasattr(env.agent, 'tcp') else 1

        # Ensure actions is a list
        if not isinstance(actions, list):
            actions = [actions]

        # === Step 1: Compute Reference Action ===
        reference_pos, reference_ori = self._compute_reference_action(cage, num_envs, env)
        # reference_pos: [num_envs, 3] torch.Tensor
        # reference_ori: [num_envs, 3] torch.Tensor (euler angles)

        # === Step 2: Prepare Offsets ===
        # Ensure we have enough offsets for all parallel environments
        while len(actions) < num_envs:
            actions.append(actions[0])

        # Convert offsets to torch tensor [num_envs, 6]
        # First convert to numpy array to avoid inefficient tensor creation from list of arrays
        import numpy as np
        offsets_np = np.array(actions[:num_envs], dtype=np.float32)
        offsets_tensor = torch.from_numpy(offsets_np).to(self.device)

        # === Step 3: Apply Offset to Reference ===
        # Position: final_pos = reference_pos + offset[:3]
        final_positions = reference_pos + offsets_tensor[:, :3]

        # Orientation: final_ori = reference_ori + offset[3:]
        final_orientations = reference_ori + offsets_tensor[:, 3:]

        # === Step 4: Visualize Cage Center (green sphere) ===
        if visualize_cage and cage is not None and hasattr(env, 'goal_site'):
            from mani_skill.utils.structs import Pose
            # Offset visualization position: cage.center + [0, 0.1, 0]
            cage_center = torch.tensor(
                cage.center, device=self.device, dtype=torch.float32
            )
            cage_center[1] += 0.08  # Add 0.1 to y coordinate
            env.goal_site.set_pose(Pose.create_from_pq(p=cage_center))

        # === Step 5: Execute Action using PD Control ===
        for _ in range(self.step_len):
            # Compute end-effector control command
            action_vec = self.control_ee_pose(
                env,
                final_positions,
                final_orientations,
                Kp_pos=self.Kp_pos,
                Kp_ori=self.Kp_ori
            )

            # Keep gripper open (0.9 = open)
            action_vec[:, -1] = 0.9

            # Step environment
            obs, _, _, _, _ = env.step(action_vec)

            # Render if in single action mode and GUI enabled
            if hasattr(env, 'render_mode') and env.render_mode == 'human' and single_action:
                try:
                    env.render()
                except AttributeError:
                    pass

        return obs

    def _compute_reference_action(self, cage, num_envs, env):
        """
        Compute reference action from cage center and orientation.

        Strategy:
        - Reference position: Approach point between cage center and target
        - Reference orientation: Use cage's fixed orientation (if available) or compute from target

        Args:
            cage: Cage object (provides cage.center and optionally cage.orientation)
            num_envs: Number of parallel environments
            env: ManiSkill environment

        Returns:
            reference_pos: [num_envs, 3] torch.Tensor - Reference position
            reference_ori: [num_envs, 3] torch.Tensor - Reference orientation (euler angles)
        """
        import torch

        # === Get Cage Center ===
        cage_center = torch.tensor(
            cage.center, device=self.device, dtype=torch.float32
        )

        # === Get Target Position (closest target object) ===
        target_positions = self.calculate_target_position(cage, num_envs, env)
        # target_positions: [num_envs, 3]

        # === Compute Approach Direction ===
        # Direction from cage center to target (in xy plane)
        direction_to_target = target_positions - cage_center.unsqueeze(0)
        direction_to_target[:, 2] = 0  # Ignore z component (horizontal only)

        # Normalize direction vector
        direction_norm = torch.norm(direction_to_target, dim=1, keepdim=True) + 1e-6
        direction_normalized = direction_to_target / direction_norm

        # === Reference Position ===
        # Strategy: Position between cage center and target
        # Distance from cage center: r_approach * 0.5
        reference_pos = (
            cage_center.unsqueeze(0) +
            direction_normalized * (self.r_approach * 0.5)
        )

        # Adjust height: slightly below cage center
        reference_pos[:, 2] = cage_center[2] + 0.14

        # === Reference Orientation ===
        # Check if cage has orientation
        if hasattr(cage, 'orientation') and cage.orientation is not None:
            # Use cage's fixed orientation [roll, pitch, yaw]
            cage_ori = torch.tensor(
                cage.orientation, device=self.device, dtype=torch.float32
            )
            # Expand to match batch size
            reference_ori = cage_ori.unsqueeze(0).expand(num_envs, -1)
            logger.debug("Using cage orientation: %s", cage.orientation)
        else:
            # Fallback: Compute orientation from target direction
            yaw = torch.atan2(direction_to_target[:, 1], direction_to_target[:, 0])
            reference_ori = torch.stack([
                torch.ones(num_envs, device=self.device) * (np.pi / 3),  # roll: 60° (downward)
                torch.zeros(num_envs, device=self.device),               # pitch: 0° (horizontal)
                yaw + np.pi / 2                                          # yaw: toward target + 90°
            ], dim=1)
            logger.debug("Using computed orientation (cage has no orientation)")

        return reference_pos, reference_ori

    def calculate_action_angle(self, action: int) -> float:
        """Convert discrete action index to approach angle in radians."""
        if isinstance(action, (np.integer, np.int64, np.int32)):
            action = int(action)
        if not isinstance(action, int) or action < 0 or action >= self.num_directions:
            action = 0
        return action * (2 * math.pi / self.num_directions)
    
    def calculate_target_position(self, cage, num_envs, env, fallback_positions=None):
        """Calculate target position for picking (closest target object to cage center)"""
        import torch
        
        try:
            if cage is not None and hasattr(cage, 'center'):
                cage_center = torch.tensor(cage.center, device=self.device, dtype=torch.float32)
                
                # Get target object positions from environment
                if hasattr(env, 'target_objects') and hasattr(env.target_objects, 'pose'):
                    target_poses = env.target_objects.pose.p  # [num_target_objects, 3]
                    
                    # Find closest target object to cage center for each env
                    distances = torch.norm(target_poses[:, :2] - cage_center[:2].unsqueeze(0), dim=1)
                    closest_idx = torch.argmin(distances)
                    target_pos = target_poses[closest_idx]
                    
                    # Expand to match batch size
                    target_positions = target_pos.unsqueeze(0).expand(num_envs, -1)
                    
                elif hasattr(env, 'objects') and hasattr(env.objects, 'pose'):
                    # Fallback: use all objects and find target objects by name/properties
                    obj_poses = env.objects.pose.p
                    if len(obj_poses.shape) >= 2:
                        obj_poses = obj_poses.reshape(num_envs, -1, 3)
                        # Use center of objects as rough target area
                        target_positions = obj_poses.mean(dim=1)
                    else:
                        target_positions = torch.tensor(cage.center, device=self.device).unsqueeze(0).expand(num_envs, -1)
                        if target_positions.shape[1] == 2:
                            z_coord = torch.full((num_envs, 1), self.grasp_height, device=self.device)
                            target_positions = torch.cat([target_positions, z_coord], dim=1)
                else:
                    # Use cage center as fallback
                    target_positions = cage_center.unsqueeze(0).expand(num_envs, -1)
                    if target_positions.shape[1] == 2:
                        z_coord = torch.full((num_envs, 1), self.grasp_height, device=self.device)
                        target_positions = torch.cat([target_positions, z_coord], dim=1)
            else:
                # No cage - use environment center or fallback
                target_positions = torch.zeros((num_envs, 3), device=self.device)
                target_positions[:, 2] = self.grasp_height
        
        except Exception as e:
            logger.warning("Error calculating target position: %s", e)
            target_positions = torch.zeros((num_envs, 3), device=self.device)
            target_positions[:, 2] = self.grasp_height
        
        return target_positions
    
    def load_object(self, context: Dict[str, Any]) -> list:
        """
        Task-specific load_object implementation for picking.

        Creates multiple colored box target instances.

        Args:
            context: Object context parameters

        Returns:
            List of object configurations for picking task
        """
        object_configs = []

        # Get parameters from context (overriding defaults)
        num_target_objects = context.get('num_target_objects', self.num_target_objects)
        target_object_type = context.get('target_object_type', self.target_object_type)
        target_box_half_size = context.get('target_box_half_size', self.target_box_half_size)

        logger.debug("load_object: Creating %d target objects", num_target_objects)
        logger.debug("load_object: Target object type: %s, half_size: %s", target_object_type, target_box_half_size)

        # Add target objects - create colored boxes in a moderate pile
        pile_center = context.get('pile_center', [-0.0, -0.02])
        center_x, center_y = pile_center[0], pile_center[1]
        pile_spacing = context.get('pile_spacing', 0.012)

        for i in range(num_target_objects):
            # Create a tight grid pattern for target objects
            if num_target_objects <= 4:
                # 2x2 grid
                row = i // 2
                col = i % 2
                spawn_x = center_x + (col - 0.5) * pile_spacing
                spawn_y = center_y + (row - 0.5) * pile_spacing
            else:
                # Circular tight arrangement
                if i == 0:
                    spawn_x, spawn_y = center_x, center_y  # Center object
                else:
                    angle = ((i - 1) / (num_target_objects - 1)) * 2 * np.pi
                    spawn_x = center_x + pile_spacing * np.cos(angle)
                    spawn_y = center_y + pile_spacing * np.sin(angle)

            # Randomize thickness for each card
            random_thickness = np.random.uniform(0.0007, 0.0012)  # Half-size: 0.001 to 0.004m (1-4mm full thickness)
            card_half_size = [target_box_half_size[0], target_box_half_size[1], random_thickness]

            # Spawn height should be half_size[2] to sit on table
            spawn_z = random_thickness
            spawn_rotation = np.random.uniform(0, 0.03*np.pi)  # Random orientation

            cos_half = np.cos(spawn_rotation / 2)
            sin_half = np.sin(spawn_rotation / 2)
            quaternion = [cos_half, 0, 0, sin_half]

            # Generate random color for each card
            # RGB range: [220, 19, 22] to [240, 160, 160] normalized to [0, 1]
            random_color = [
                np.random.uniform(220/255, 240/255),  # R: [220, 240] / 255
                np.random.uniform(19/255, 160/255),   # G: [19, 160] / 255
                np.random.uniform(22/255, 160/255),   # B: [22, 160] / 255
                1.0                                    # Alpha
            ]

            # Configure colored box target object
            target_config = {
                'id': f'target_object_{i}',
                'type': 'box',
                'half_size': card_half_size,
                'initial_pose': {
                    'position': [spawn_x, spawn_y, spawn_z],
                    'quaternion': quaternion
                },
                'manipulation_properties': {
                    'pushable': True,
                    'graspable': True,  # Target objects can be grasped
                    'rollable': False  # Boxes don't roll
                },
                'physics_properties': {
                    'mass': 0.05,  # Light weight for easy manipulation
                    'friction': 0.5,
                    'density': 1000
                },
                'visual_properties': {
                    'color': random_color  # Random color for each card
                }
            }
            object_configs.append(target_config)
        
        return object_configs
    
    def create_environment(self, env_config: Dict[str, Any]) -> gym.Env:
        """
        Create custom picking environment from scratch using TableSetupMixin and colored box targets.

        Uses our own TableSetupMixin for the white table.
        """
        logger.debug("PickBackend.create_environment() called")
        logger.debug("create_environment: env_config keys: %s", env_config.keys())
        try:
            import mani_skill
            from mani_skill.envs.sapien_env import BaseEnv
            from mani_skill.utils.building import actors
            from mani_skill.utils.structs import Actor, Pose
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils import sapien_utils
            from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
            import sapien
            import torch
            import numpy as np
            
            # Get context configuration
            context_config = env_config.get('context_info', env_config)
            
            # Create custom picking environment class
            class CustomPickingEnv(BaseEnv):
                """Custom picking environment using TableSetupMixin with colored box targets"""
                
                SUPPORTED_REWARD_MODES = ["none"]
                SUPPORTED_ROBOTS = ["panda", "floating_panda_gripper_fin"]
                
                def __init__(self, **kwargs):
                    # Extract robot_init_qpos_noise before passing to parent
                    self.robot_init_qpos_noise = kwargs.pop('robot_init_qpos_noise', 0.02)
                    super().__init__(**kwargs)
                
                @property
                def _default_sim_config(self):
                    return SimConfig(
                        gpu_memory_config=GPUMemoryConfig(
                            max_rigid_contact_count=2**21, 
                            max_rigid_patch_count=2**19
                        )
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
                    pose = sapien_utils.look_at([0.5, 0.0, 0.6], [0.0, 0.0, 0.35])
                    return CameraConfig(
                        "render_camera", pose=pose, width=512, height=512, 
                        fov=1, near=0.01, far=100
                    )
                
                def _load_agent(self, options: dict):
                    """Override to provide initial robot pose."""
                    initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
                    super()._load_agent(options, initial_agent_poses=initial_pose)

                    # Disable collision between gripper and table using collision group bits
                    # Set bit 30 on group 2 for all gripper links to match table's bit
                    if hasattr(self, 'agent') and hasattr(self.agent, 'robot'):
                        for link in self.agent.robot.get_links():
                            link.set_collision_group_bit(group=2, bit_idx=30, bit=1)
                
                def _load_scene(self, options: dict):
                    logger.debug("_load_scene: Starting scene loading...")
                    try:
                        # Use our WhiteTableSceneBuilder from TableSetupMixin
                        from .maniskill_base import WhiteTableSceneBuilder
                        logger.debug("_load_scene: Creating WhiteTableSceneBuilder...")
                        self.scene_builder = WhiteTableSceneBuilder(
                            env=self,
                            robot_init_qpos_noise=self.robot_init_qpos_noise
                        )
                        logger.debug("_load_scene: Building scene...")
                        self.scene_builder.build()

                        # Set collision group bit for table to disable collision with gripper
                        # Both table and gripper have bit 30 on group 2 set to 1
                        if hasattr(self.scene_builder, 'table') and self.scene_builder.table is not None:
                            self.scene_builder.table.set_collision_group_bit(group=2, bit_idx=30, bit=1)
                            logger.debug("_load_scene: Set table collision group bit (group=2, bit_idx=30)")

                        # Create wall obstacle on table
                        logger.debug("_load_scene: Creating wall obstacle...")
                        backend = PickBackend()
                        if backend.wall_enabled:
                            wall_builder = self.scene.create_actor_builder()

                            # Add collision shape for physical interaction
                            wall_builder.add_box_collision(
                                half_size=backend.wall_half_size
                            )

                            # Add visual shape for rendering
                            wall_builder.add_box_visual(
                                half_size=backend.wall_half_size,
                                material=sapien.render.RenderMaterial(
                                    base_color=backend.wall_color
                                )
                            )

                            # Set initial pose
                            wall_builder.initial_pose = sapien.Pose(
                                p=backend.wall_position,
                                q=[1, 0, 0, 0]  # Identity quaternion
                            )

                            # Build as static actor (non-movable)
                            self.wall = wall_builder.build_static(name="wall")
                            logger.debug("_load_scene: Wall created at position %s", backend.wall_position)

                            # Disable collision between gripper and wall using same collision group bit
                            self.wall.set_collision_group_bit(group=2, bit_idx=30, bit=1)
                            logger.debug("_load_scene: Set wall collision group bit (group=2, bit_idx=30)")
                        
                        # Create objects based on our load_object configuration
                        logger.debug("_load_scene: Creating backend and loading objects...")
                        backend = PickBackend()
                        object_configs = backend.load_object(context_config)
                        
                        logger.debug("_load_scene: Got %d object configs", len(object_configs))
                        
                        # ⭐ Create per-env actors (following pushing backend pattern)
                        all_objects = []
                        target_objects = []

                        logger.debug("_load_scene: Creating per-env actors for %d environments", self.num_envs)

                        # Outer loop: iterate over environments
                        for env_idx in range(self.num_envs):
                            # Inner loop: iterate over objects
                            for obj_idx, obj_config in enumerate(object_configs):
                                try:
                                    if obj_config['type'] == 'box':
                                        # Colored box target objects
                                        builder = self.scene.create_actor_builder()
                                        half_size = obj_config['half_size']

                                        # Add collision shape
                                        builder.add_box_collision(
                                            half_size=half_size,
                                            density=obj_config['physics_properties']['density']
                                        )

                                        # Add visual shape with color
                                        color = obj_config['visual_properties']['color']
                                        builder.add_box_visual(
                                            half_size=half_size,
                                            material=sapien.render.RenderMaterial(
                                                base_color=color
                                            )
                                        )
                                    else:
                                        logger.warning("_load_scene: Unknown object type: %s, skipping", obj_config['type'])
                                        continue

                                    # Set initial pose
                                    init_pose = obj_config['initial_pose']
                                    builder.initial_pose = sapien.Pose(
                                        p=init_pose['position'],
                                        q=init_pose['quaternion']
                                    )

                                    # ⭐ Key: Only set to this specific environment
                                    builder.set_scene_idxs([env_idx])

                                    # Set collision groups for target objects
                                    if 'target' in obj_config['id']:
                                        collision_group = 2
                                        collision_mask = 1 | 4
                                        builder.collision_groups = [collision_group, collision_mask, collision_group, 0]

                                    # ⭐ Key: Per-env naming format: object_{env_idx}_{obj_idx}
                                    obj_name = f"object_{env_idx}_{obj_idx}"
                                    obj = builder.build(name=obj_name)
                                    all_objects.append(obj)

                                    # Track target objects from first env for reference
                                    if 'target' in obj_config['id'] and env_idx == 0:
                                        target_objects.append(obj)

                                except Exception as e:
                                    logger.error("_load_scene: Failed to create object env=%d, obj=%d: %s", env_idx, obj_idx, e)
                                    import traceback
                                    traceback.print_exc()
                                    obj_type = obj_config.get('type', 'unknown')
                                    obj_id = obj_config.get('model_id', obj_config.get('id', 'unknown'))
                                    raise RuntimeError(f"Cannot create {obj_type} object {obj_id}") from e

                        total_objects = len(all_objects)
                        objects_per_env = len(object_configs)
                        logger.debug("_load_scene: Created %d objects total (%d envs x %d objects/env)", total_objects, self.num_envs, objects_per_env)

                        # Ensure we actually created objects
                        if len(all_objects) == 0:
                            raise RuntimeError("No objects were created!")

                        # ⭐ Still merge, but with per-env naming preserved
                        self.all_objects = Actor.merge(all_objects, name="all_objects")
                        logger.debug("_load_scene: Merged all_objects with per-env naming")
                            
                        # Create target object group
                        if target_objects:
                            self.target_object = Actor.merge(target_objects, name="target_object")
                            logger.debug("_load_scene: Created target_object with %d objects", len(target_objects))
                        elif all_objects:
                            # Fallback: use first few objects as targets
                            num_targets = min(8, len(all_objects))
                            self.target_object = Actor.merge(all_objects[:num_targets], name="target_object")
                            logger.debug("_load_scene: Fallback: Created target_object with first %d objects", num_targets)
                        else:
                            logger.error("_load_scene: No target objects could be created!")
                        
                        # Create goal site for visualization
                        logger.debug("_load_scene: Creating goal site...")
                        self.goal_site = actors.build_sphere(
                            self.scene,
                            radius=0.1,  # Larger radius for better visibility
                            color=[0, 1, 0, 0.5],  # Green with 50% transparency
                            name="goal_site",
                            body_type="kinematic",
                            add_collision=False,
                            initial_pose=sapien.Pose(),
                        )
                        self._hidden_objects.append(self.goal_site)
                        logger.debug("_load_scene: Scene loading completed successfully!")

                    except Exception as e:
                        logger.error("_load_scene: Failed during scene loading: %s", e)
                        import traceback
                        traceback.print_exc()
                        # Don't re-raise the exception - let the environment continue
                
                def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
                    with torch.device(self.device):
                        b = len(env_idx)
                        self.scene_builder.initialize(env_idx)

                        # Set gripper initial state aligned with trajectory start
                        # Trajectory starts at gripper_init_pos = [0.0, -0.5, 0.3], orientation = [π/3, 0, 0]
                        # control_ee_pose applies offsets: pos += [0, 0, -0.33], ori += [0, π, 0]
                        # keyframe.pose provides base z≈0.33 which compensates pos_offset
                        # So qpos = desired + pos_offset, and world_z = keyframe.z + qpos.z ≈ 0.33 + (-0.03) = 0.30
                        if hasattr(self, 'agent') and hasattr(self.agent, 'robot'):
                            # For floating_panda_gripper_fin: qpos = [x, y, z, roll, pitch, yaw, gripper1, gripper2]
                            custom_qpos = torch.zeros((b, 8), device=self.device)
                            custom_qpos[:, 0] = 0.0                           # x position
                            custom_qpos[:, 1] = -0.5                          # y position
                            custom_qpos[:, 2] = 0.3 + (-0.33)                 # z = desired_z + pos_offset_z = -0.03
                            custom_qpos[:, 3] = np.pi / 3                     # roll = 60°
                            custom_qpos[:, 4] = 0.0 + np.pi                   # pitch = desired + ori_offset (π)
                            custom_qpos[:, 5] = 0.0                           # yaw = 0°
                            custom_qpos[:, 6] = 0.04                          # gripper finger 1 (open)
                            custom_qpos[:, 7] = 0.04                          # gripper finger 2 (open)

                            # Only reset qpos; keep keyframe.pose from scene_builder (provides z≈0.33 compensation)
                            self.agent.reset(custom_qpos)
                            logger.debug("_initialize_episode: Set gripper qpos: pos=[0, -0.5, -0.03], ori=[pi/3, pi, 0] (keyframe base_pose preserved)")
                        
                        # Set goal positions
                        goal_pos = torch.rand(size=(b, 3)) * torch.tensor(
                            [0.3, 0.5, 0.1]
                        ) + torch.tensor([-0.15, -0.25, 0.35])
                        self.goal_pos = goal_pos
                        
                        # Only set goal site pose if it exists
                        if hasattr(self, 'goal_site') and self.goal_site is not None:
                            self.goal_site.set_pose(Pose.create_from_pq(self.goal_pos))

                        # Note: Object poses are initialized by scene_builder.initialize(env_idx) above
                        # No manual reset needed - ManiSkill's SceneBuilder handles vectorized initialization
                
                def evaluate(self):
                    return {
                        "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
                        "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
                    }
                
                def _get_obs_extra(self, info: Dict):
                    """Return observation dict with object poses for state2dris."""
                    # Get object poses for state observation
                    obs_dict = {}

                    # Include object poses if available
                    if hasattr(self, 'all_objects') and self.all_objects is not None:
                        try:
                            # Get poses from all objects
                            obj_poses = self.all_objects.pose.raw_pose
                            obs_dict['obj_pose'] = obj_poses
                        except Exception as e:
                            logger.debug("Warning: Could not get object poses: %s", e)
                            # Fallback: create dummy poses
                            obs_dict['obj_pose'] = torch.zeros((self.num_envs, 43, 7), device=self.device)

                    # Include target object poses specifically
                    if hasattr(self, 'target_objects') and self.target_objects is not None:
                        try:
                            target_poses = self.target_objects.pose.raw_pose
                            obs_dict['target_pose'] = target_poses
                        except Exception:
                            pass

                    # Include robot TCP pose
                    if hasattr(self, 'agent') and hasattr(self.agent, 'tcp'):
                        try:
                            tcp_pose = self.agent.tcp.pose.raw_pose
                            obs_dict['tcp_pose'] = tcp_pose
                        except Exception:
                            pass

                    return obs_dict
            
            # GUI configuration
            enable_gui = context_config.get('enable_gui', False)
            render_mode = context_config.get('render_mode', None)
            if enable_gui:
                render_mode = 'human'

            # Parallel rendering configuration (like ManiSkillBaseBackend)
            parallel_in_single_scene = False
            if render_mode == 'human':
                obs_mode = context_config.get('obs_mode', 'state_dict')
                num_envs = context_config.get('num_envs', 16)
                # Enable parallel rendering in GUI if not using visual observations
                if obs_mode not in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"] and num_envs > 1:
                    parallel_in_single_scene = True

            # Environment creation parameters
            env_kwargs = {
                'num_envs': context_config.get('num_envs', 16),
                'obs_mode': context_config.get('obs_mode', 'state_dict'),
                'control_mode': context_config.get('control_mode', 'pd_joint_delta_pos'),
                'render_mode': render_mode,
                'sim_backend': context_config.get('sim_backend', 'gpu'),
                'robot_uids': context_config.get('robot_uids', 'floating_panda_gripper_fin'),
                'robot_init_qpos_noise': context_config.get('robot_init_qpos_noise', 0.02),
                'parallel_in_single_scene': parallel_in_single_scene,
            }

            # Add shader configuration for GUI
            shader = context_config.get('shader', 'default')
            env_kwargs['sensor_configs'] = dict(shader_pack=shader)
            env_kwargs['human_render_camera_configs'] = dict(shader_pack=shader)
            env_kwargs['viewer_camera_configs'] = dict(shader_pack=shader)

            # Add spacing for parallel environments (like ManiSkillBaseBackend)
            num_envs = context_config.get('num_envs', 16)
            if 'sim_config' in context_config:
                env_kwargs['sim_config'] = context_config['sim_config']
            elif num_envs > 1:
                env_kwargs['sim_config'] = dict(spacing=1.6)
            
            # Create environment instance
            env = CustomPickingEnv(**env_kwargs)
            
            # Initialize GUI if needed
            if render_mode == 'human':
                obs, _ = env.reset(seed=2)
                env.action_space.seed(2)
                viewer = env.render()
                if hasattr(viewer, 'paused'):
                    viewer.paused = context_config.get('pause_on_start', True)
                env.render()
            
            # Store environment parameters
            self.context.update(env_kwargs)
            self.context['enable_gui'] = enable_gui
            
            return env
            
        except ImportError as e:
            import warnings
            warnings.warn(f"ManiSkill not available: {e}")
            return self._create_mock_environment(env_config)
        except Exception as e:
            import warnings
            logger.error("Failed to create custom picking environment: %s", e)
            import traceback
            traceback.print_exc()
            logger.critical("Cannot create picking environment. Exiting.")
            raise RuntimeError(f"Failed to create picking environment: {e}") from e
    
    def extract_state(self, env: gym.Env) -> Dict[str, np.ndarray]:
        """
        Override parent's extract_state to use our custom picking environment logic.
        This is the method actually called by SimulationBasedTSIP.
        """
        logger.debug("PickBackend.extract_state() called")
        return self.get_state(env)
    
    def get_state(self, env):
        """Get current state from our custom picking environment - returns FIRST environment only"""
        try:
            if hasattr(env, 'unwrapped'):
                actual_env = env.unwrapped
            else:
                actual_env = env

            # Try our custom environment's all_objects first
            if hasattr(actual_env, 'all_objects') and hasattr(actual_env.all_objects, 'pose'):
                obj_poses = actual_env.all_objects.pose.raw_pose

                if hasattr(obj_poses, 'cpu'):
                    obj_poses_np = obj_poses.cpu().numpy()
                else:
                    obj_poses_np = obj_poses

                # ⭐ Extract ONLY first environment's state (following pushing backend pattern)
                # With per-env actors: total_objects = num_envs × num_objects_per_env
                num_envs = actual_env.num_envs if hasattr(actual_env, 'num_envs') else 1
                total_objects = obj_poses_np.shape[0]
                num_objects_per_env = total_objects // num_envs

                # Get first env's objects (target objects only)
                first_env_poses = obj_poses_np[:num_objects_per_env]
                first_env_positions = first_env_poses[:, :3]

                logger.debug("get_state: Retrieved %d total poses, returning first env's %d poses", total_objects, num_objects_per_env)

                return {
                    'obj_pose': first_env_poses,  # [22, 7]
                    'obj_positions': first_env_positions  # [22, 3]
                }

            # Fallback to generic object access
            elif hasattr(actual_env, 'objects') and hasattr(actual_env.objects, 'pose'):
                obj_poses = actual_env.objects.pose.raw_pose

                if hasattr(obj_poses, 'cpu'):
                    obj_poses_np = obj_poses.cpu().numpy()
                else:
                    obj_poses_np = obj_poses

                if len(obj_poses_np.shape) >= 2:
                    obj_positions = obj_poses_np[..., :3]
                else:
                    obj_positions = obj_poses_np

                return {
                    'obj_pose': obj_poses_np,
                    'obj_positions': obj_positions
                }

            # Fallback to 'te' objects (push_t2multi style)
            elif hasattr(actual_env, 'te') and hasattr(actual_env.te, 'pose'):
                obj_poses = actual_env.te.pose.raw_pose

                if hasattr(obj_poses, 'cpu'):
                    obj_poses_np = obj_poses.cpu().numpy()
                else:
                    obj_poses_np = obj_poses

                if len(obj_poses_np.shape) >= 2:
                    obj_positions = obj_poses_np[..., :3]
                else:
                    obj_positions = obj_poses_np

                return {
                    'obj_pose': obj_poses_np,
                    'obj_positions': obj_positions
                }
            else:
                return {}

        except Exception as e:
            warnings.warn(f"Failed to get state: {e}")
            return {}
    
    def set_state(self, env, state):
        """Set state to our custom picking environment using state_dict approach (following pushing backend)"""
        try:
            # Check if we need to access unwrapped environment
            if hasattr(env, 'unwrapped'):
                actual_env = env.unwrapped
            else:
                actual_env = env

            if isinstance(state, dict):
                import torch
                from mani_skill.utils.structs import Pose

                # Handle 'obj_pose' or 'obj_poses' keys
                obj_poses = state.get('obj_pose')
                if obj_poses is None:
                    obj_poses = state.get('obj_poses')

                if obj_poses is not None:
                    # Get device
                    if hasattr(actual_env, 'device'):
                        device = actual_env.unwrapped.device
                    else:
                        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

                    # Convert to torch tensor if needed
                    if isinstance(obj_poses, np.ndarray):
                        obj_poses = torch.from_numpy(obj_poses).to(device)

                    # ⭐ Use state_dict approach like pushing backend (PROPER STATE SYNCHRONIZATION)
                    # Get current scene state
                    sim_state = actual_env.scene.get_sim_state()
                    actors_state = sim_state['actors']

                    # Get number of environments
                    num_envs = getattr(actual_env, 'num_envs', 64)

                    # Handle different input shapes
                    if len(obj_poses.shape) == 2:  # [num_objects, 7] - single env state
                        num_objects = obj_poses.shape[0]  # Number of target objects

                        logger.debug("set_state: Setting %d objects to ALL %d environments using state_dict", num_objects, num_envs)

                        # ⭐ For cage evaluation: all environments should start from the same state
                        # This ensures parallel action evaluation from identical initial conditions
                        for env_idx in range(num_envs):
                            for obj_idx in range(num_objects):
                                # ⭐ Use picking backend's per-env naming: object_{env_idx}_{obj_idx}
                                actor_name = f"object_{env_idx}_{obj_idx}"
                                if actor_name in actors_state:
                                    # Copy the same pose to all environments
                                    new_pose = actors_state[actor_name].clone()
                                    # Use proper indexing for the state tensor
                                    new_pose[..., :3] = obj_poses[obj_idx, :3].clone()  # position
                                    new_pose[..., 3:7] = obj_poses[obj_idx, 3:7].clone()  # quaternion
                                    actors_state[actor_name] = new_pose

                    elif len(obj_poses.shape) == 3:  # [num_envs, num_objects, 7] - multi-env state
                        num_envs_in_state, num_objects = obj_poses.shape[:2]

                        logger.debug("set_state: Setting %d objects to %d environments using state_dict", num_objects, num_envs_in_state)

                        # Update actors_state for each environment
                        for env_idx in range(min(num_envs_in_state, num_envs)):
                            for obj_idx in range(num_objects):
                                actor_name = f"object_{env_idx}_{obj_idx}"
                                if actor_name in actors_state:
                                    # Clone the pose for this object
                                    new_pose = torch.zeros_like(actors_state[actor_name])
                                    # Use proper indexing for the state tensor
                                    new_pose[..., :3] = obj_poses[env_idx, obj_idx, :3]  # position
                                    new_pose[..., 3:7] = obj_poses[env_idx, obj_idx, 3:7]  # quaternion
                                    actors_state[actor_name] = new_pose

                    # ⭐ Apply the updated state using set_state_dict (KEY SYNCHRONIZATION METHOD)
                    state_dict = {'actors': actors_state}
                    actual_env.set_state_dict(state_dict)

                    # Step physics to ensure poses are applied (following pushing backend pattern)
                    for _ in range(2):
                        actual_env.scene.step()

                    logger.debug("set_state: State applied successfully via set_state_dict and physics stepped")

        except Exception as e:
            warnings.warn(f"Failed to set state: {e}")
    
    def obs_to_dris(self, env, observations):
        """Convert ManiSkill observations to DRIS for picking task"""
        _ = env  # Mark parameter as intentionally unused
        if isinstance(observations, dict):
            return self.state2dris(observations)
        else:
            from manidreams.base.dris import DRIS
            return [DRIS(observation=observations, representation_type="state")]