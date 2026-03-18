"""
Push Backend (Simulation) for ManiDreams Framework

Task-specific backend for object pushing with all ManiSkill functions.
Separated from the generic ManiSkillBaseBackend to keep task-specific logic isolated.
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
from .maniskill_base import ManiSkillBaseBackend, TableSetupMixin, WhiteTableSceneBuilder

# ManiSkill imports for custom environment creation
try:
    import torch
    import sapien
    from transforms3d.euler import euler2quat
    from assets.robots import FloatingPandaGripperFin
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import common, sapien_utils
    from mani_skill.utils.building import actors
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    from mani_skill.utils.structs import Actor
    from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
except ImportError:
    logger.warning("ManiSkill not available")
    pass


# Custom Environment based on push_t2multi.py
@register_env("CustomPushT-v2multi", max_episode_steps=100)
class CustomPushT2MultiEnv(BaseEnv):
    """Custom pushing environment based on ManiSkill's PushT-v2multi."""

    SUPPORTED_ROBOTS = ["floating_panda_gripper_fin"]
    agent: FloatingPandaGripperFin

    # Environment parameters from push_t2multi.py
    NUM_OBJECTS = 64
    tee_spawnbox_xlength = 0.2
    tee_spawnbox_ylength = 0.3
    tee_spawnbox_xoffset = -0.06
    tee_spawnbox_yoffset = 0
    goal_offset = torch.tensor([-0.156, -0.1])
    goal_z_rot = (5 / 3) * np.pi
    ee_starting_pos2D = torch.tensor([0, 0, 0.5])
    ee_starting_pos3D = torch.tensor([0, 0, 0.5])
    intersection_thresh = 0.90
    T_mass = 0.8
    T_dynamic_friction = 0.3
    T_static_friction = 0.3

    def __init__(self, *args, robot_uids="floating_panda_gripper_fin", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.08, 0, 0.45], target=[-0.08, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=1.4,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # Use the same camera configuration as original push_t2multi.py
        pose = sapien_utils.look_at(eye=[2, 0, 3.6], target=[-0.05, 0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=0.5, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        self.ee_starting_pos2D = self.ee_starting_pos2D.to(self.device)
        self.ee_starting_pos3D = self.ee_starting_pos3D.to(self.device)

        self.table_scene = WhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

    def _load_agent(self, options: dict):
        """Override to provide initial robot pose during building phase."""
        initial_pose = sapien.Pose(p=[0, 0, 0.5])
        super()._load_agent(options, initial_agent_poses=initial_pose)

        # Create polygon objects based on push_t2multi.py create_te function
        def create_te(name="te", base_color=np.array([194, 19, 22, 255]) / 255, r=0.07, n_min=3, n_max=10, seed=66, num_objs=128):
            if seed is not None:
                torch.manual_seed(seed)

            te_objects = []
            for i in range(self.num_envs):
                for obj_idx in range(num_objs):
                    builder = self.scene.create_actor_builder()
                    builder._mass = self.T_mass
                    te_material = sapien.pysapien.physx.PhysxMaterial(
                        static_friction=self.T_dynamic_friction,
                        dynamic_friction=self.T_static_friction,
                        restitution=0,
                    )

                    n_vertices = torch.randint(n_min, n_max + 1, (1,), device=self.device)[0].item()

                    # Generate angles for vertices
                    if n_vertices < 5:
                        section_size = 2 * np.pi / n_vertices
                        angles = []
                        for j in range(n_vertices):
                            section_start = j * section_size
                            section_end = (j + 1) * section_size
                            angle = section_start + torch.rand(1, device=self.device).item() * (section_end - section_start)
                            angles.append(angle)
                        angles = np.array(angles)
                    else:
                        angles = torch.sort(torch.rand(n_vertices, device=self.device) * (2 * torch.pi))[0].cpu().numpy()

                    # Calculate vertices
                    vertices = []
                    for angle in angles:
                        x = r * np.cos(angle)
                        y = r * np.sin(angle)
                        vertices.append([x, y])
                    vertices = np.array(vertices)

                    # Create boxes for each edge
                    half_thickness = 0.03
                    box_width = 0.02

                    for j in range(n_vertices):
                        v1 = vertices[j]
                        v2 = vertices[(j + 1) % n_vertices]

                        center = (v1 + v2) / 2
                        edge = v2 - v1
                        length = np.linalg.norm(edge)
                        angle = np.arctan2(edge[1], edge[0])

                        box_pose = sapien.Pose(
                            p=np.array([center[0]*0.9, center[1]*0.9, 0.0]),
                            q=euler2quat(0, 0, angle)
                        )

                        box_half_size = [length/2, box_width/2, half_thickness]
                        builder.add_box_collision(
                            pose=box_pose,
                            half_size=box_half_size,
                            material=te_material,
                        )

                        # Color interpolation
                        red_color = np.array([194, 19, 22, 255]) / 255
                        white_color = np.array([240, 160, 160, 255]) / 255
                        t = 0.0
                        object_color = red_color * (1 - t) + white_color * t

                        builder.add_box_visual(
                            pose=box_pose,
                            half_size=box_half_size,
                            material=sapien.render.RenderMaterial(
                                base_color=object_color,
                            ),
                        )

                    # Set collision groups
                    collision_group = 6
                    collision_mask = 1 | collision_group
                    builder.collision_groups = [collision_group, collision_mask, collision_group, 0]

                    # Set initial pose to avoid warnings
                    initial_pose = sapien.Pose(
                        p=np.array([0.0, 0.0, 0.03]),  # Slightly above ground
                        q=np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
                    )
                    builder.initial_pose = initial_pose

                    # Set scene index
                    builder.set_scene_idxs([i])
                    te_objects.append(builder.build(name=f"{name}_{i}_{obj_idx}"))

            return Actor.merge(te_objects, name=name)

        self.te = create_te(name="Te", num_objs=self.NUM_OBJECTS)

    def _initialize_episode(self, env_idx, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize object positions
            target_region_xyz = torch.zeros((b * self.NUM_OBJECTS, 3))
            for i in range(self.NUM_OBJECTS):
                start_idx = i * b
                end_idx = (i + 1) * b

                offset_x = self.tee_spawnbox_xoffset
                offset_y = self.tee_spawnbox_yoffset

                target_region_xyz[start_idx:end_idx, 0] += (
                    torch.rand(b) * 0.01 + offset_x
                )
                target_region_xyz[start_idx:end_idx, 1] += (
                    torch.rand(b) * 0.01 + offset_y
                )
                target_region_xyz[start_idx:end_idx, 2] = 0.06 / 2 + 1e-3

            # Initialize object orientations
            q_euler_angle = torch.rand(b * self.NUM_OBJECTS) * (2 * torch.pi)
            q = torch.zeros((b * self.NUM_OBJECTS, 4))
            q[:, 0] = (q_euler_angle / 2).cos()
            q[:, -1] = (q_euler_angle / 2).sin()

            # Set object poses
            from mani_skill.utils.structs import Pose
            obj_pose = Pose.create_from_pq(p=target_region_xyz, q=q)
            self.te.set_pose(obj_pose)

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.te.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        num_envs = len(self.agent.tcp.pose.p)
        obj_positions = self.te.pose.p.reshape(num_envs, self.NUM_OBJECTS, 3)
        mean_obj_positions = obj_positions.mean(dim=1)
        tcp_to_push_pose = mean_obj_positions - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reward = (1 - torch.tanh(5 * tcp_to_push_pose_dist))
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


class PushBackend(ManiSkillBaseBackend, TableSetupMixin):
    """
    Task-specific backend for object pushing with all ManiSkill functions.

    Consolidates all pushing-related functionality and inherits table setup
    from TableSetupMixin and basic ManiSkill functionality from ManiSkillBaseBackend.
    """

    def __init__(self):
        ManiSkillBaseBackend.__init__(self)
        TableSetupMixin.__init__(self)
        
        # DiscretePush parameters merged into this class
        self.device = None
        # Parameters matching gpu_demo_gui_float_gripper_multiple_obj.py
        self.sequence_len = 20   # Push action steps (matching reference)
        self.step_len = 10      # Steps per push action (matching reference script)
        self.r_approach = 0.26   # Approach distance (matching reference)
        self.Kp_pos = 3.0        # Position control gain
        self.Kp_ori = 0.5        # Orientation control gain (matching reference)
        self.Kp_pos_push = 2  # Reduced gain for pushing (matching reference)
        self.num_directions = 16  # 16 push directions (0-15)
        
        self.current_context = {}
        
    def step_act(self, actions, env: gym.Env = None, cage=None, single_action=False) -> Any:
        """
        Task-specific step_act implementation using direct teleportation + push approach.
        Based on gpu_demo_gui_float_gripper_multiple_obj.py reference implementation.
        
        Executes: 1) Direct teleportation to approach position, 2) Controlled push action
        
        Args:
            actions: Single action or list of actions (0-15 for 16 push directions) 
            env: ManiSkill environment
            cage: Cage object containing center position and constraints
            
        Returns:
            Final observation after completing the action sequence
        """
        import torch
        
        # Handle parallel action execution for cage evaluation
        if not isinstance(actions, list):
            # Single action case - replicate for all environments
            actions = [actions]
        
        # Set device from environment
        self.device = env.unwrapped.device if hasattr(env.unwrapped, 'device') else None
        
        # Get number of environments
        num_envs = len(env.agent.tcp.pose.p) if hasattr(env, 'agent') and hasattr(env.agent, 'tcp') else 1
        # Ensure we have the right number of actions for parallel execution
        if len(actions) == 1:
            # Single action - replicate for all environments
            actions = actions * num_envs
        elif len(actions) != num_envs:
            # Mismatch - pad or truncate to match num_envs
            if len(actions) < num_envs:
                # Pad with the last action
                actions = actions + [actions[-1]] * (num_envs - len(actions))
            else:
                # Truncate to num_envs
                actions = actions[:num_envs]
            
        
        # Calculate object center positions as fallback
        fallback_positions = None
        if hasattr(env, 'objects'):
            try:
                obj_positions = env.objects.pose.p
                if len(obj_positions.shape) >= 2:
                    obj_positions = obj_positions.reshape(num_envs, -1, 3)
                    fallback_positions = obj_positions.mean(dim=1)  # Mean object position
            except Exception:
                pass
        
        # Get target position (cage center or object center) - updated each timestep
        target_positions = self.calculate_target_position(cage, num_envs, fallback_positions)
        logger.debug("Target positions: %s", target_positions[0])
        
        # Debug: Print cage center for verification
        if cage is not None and hasattr(cage, 'center'):
            logger.debug("Using cage center: %s -> target: %s", cage.center, target_positions[0][:2].tolist())
        else:
            logger.warning("No valid cage object with center attribute passed to step_act!")
        
        # Calculate push directions and orientations for all actions in parallel
        push_angles = [self.calculate_action_angle(action) for action in actions]
        push_angle_tensor = torch.tensor(push_angles, device=self.device, dtype=torch.float32)  # [num_envs]
        
        # Calculate push directions for each environment
        push_directions = torch.stack([
            torch.cos(push_angle_tensor),
            torch.sin(push_angle_tensor),
            torch.zeros_like(push_angle_tensor)
        ], dim=1)  # [num_envs, 3]
        
        # Calculate target orientations like reference script
        # theta_ori = theta + pi (pointing toward push direction)
        theta_ori = push_angle_tensor + torch.pi  # [num_envs]
        
        # Create target orientation tensor: [roll, pitch, yaw] = [0, 0, -theta_ori]
        target_ori = torch.stack([
            torch.zeros_like(theta_ori),  # roll = 0
            torch.zeros_like(theta_ori),  # pitch = 0
            -theta_ori                    # yaw = -theta_ori
        ], dim=1)  # [num_envs, 3]
        
        # Step 1: Direct teleportation to approach position (like reference script)
        approach_positions = target_positions + push_directions * self.r_approach
        approach_positions[:, 2] = target_positions[:, 2] + 0.06  # Lift above objects
        
        logger.debug("Teleporting to approach positions: %s", approach_positions[0])
        self.direct_set_gripper_position(env, approach_positions, target_ori)
        
        # Step 2: Controlled push action (like reference script)
        # Calculate push target: 1.2 * object_positions - 0.2 * approach_positions
        push_target = 0.75 * target_positions + 0.25 * approach_positions
        push_target[:, 2] = target_positions[:, 2] + 0.01  # Keep height above objects
        
        logger.debug("Executing push to: %s", push_target[0])
        
        # Execute push action with controller (matching reference parameters)
        for _ in range(self.step_len):
            action_vec = self.control_ee_pose(env, push_target,
                                            target_ori,
                                            self.Kp_pos_push, self.Kp_ori)
            obs, _, _, _, _ = env.step(action_vec)
            # Render if GUI mode
            if hasattr(env, 'render_mode') and env.render_mode == 'human' and single_action:
                try:
                    env.render()
                except AttributeError:
                    pass  # Ignore viewer-related errors
        
        return obs
    
    def calculate_action_angle(self, action: int) -> float:
        """Convert discrete action index to angle in radians."""
        if not isinstance(action, int) or action < 0 or action >= self.num_directions:
            action = 0
        return action * (2 * math.pi / self.num_directions)
    
    def calculate_target_position(self, cage, num_envs, fallback_positions=None):
        """Calculate target position from cage center or fallback to object positions (merged from DiscretePush)"""
        import torch
        
        if cage is not None and hasattr(cage, 'center'):
            # Use current cage center as the target position for pushing
            # IMPORTANT: This gets the CURRENT cage position, which updates with time-varying cage
            current_cage_center = cage.center  # This should be the updated center
            cage_center = torch.tensor(current_cage_center, device=self.device, dtype=torch.float32)
            # Expand to match batch size
            target_positions = cage_center.unsqueeze(0).expand(num_envs, -1)
            # Add z-coordinate (assume objects are on table surface)
            if len(cage_center) == 2:
                z_coord = torch.full((num_envs, 1), 0.06, device=self.device)  # Table height
                target_positions = torch.cat([target_positions, z_coord], dim=1)
        else:
            # Fallback: use provided fallback positions or origin
            if fallback_positions is not None:
                target_positions = fallback_positions
            else:
                target_positions = torch.zeros((num_envs, 3), device=self.device)
                target_positions[:, 2] = 0.06  # Default table height
        
        return target_positions
    
    def load_object(self, context: Dict[str, Any]) -> list:
        """
        Task-specific load_object implementation for pushing task.
        
        Creates polygon objects optimized for pushing manipulation, similar to 
        push_t2multi.py create_te function.
        
        Args:
            context: Object context parameters including masses, frictions, object_type, etc.
            
        Returns:
            List of object configurations for pushing task
        """
        num_objects = context.get('num_objects', 64)
        object_type = context.get('object_type', 'polygon_te')  # Default to polygon_te type
        masses = context.get('masses', np.random.uniform(0.5, 1.0, num_objects))
        frictions = context.get('frictions', np.random.uniform(0.2, 0.5, num_objects))
        
        # Ensure arrays are the right length
        if not isinstance(masses, (list, np.ndarray)) or len(masses) < num_objects:
            masses = np.random.uniform(0.5, 1.0, num_objects)
        if not isinstance(frictions, (list, np.ndarray)) or len(frictions) < num_objects:
            frictions = np.random.uniform(0.2, 0.5, num_objects)
        
        try:
            # ManiSkill components for object creation (imports moved to specific usage)
            
            object_configs = []
            
            for i in range(num_objects):
                # Polygon parameters matching push_t2multi.py
                n_vertices = context.get('n_vertices', np.random.randint(4, 8))  # 4-8 vertices for stability
                radius = context.get('object_radius', 0.07)  # Matching push_t2multi.py
                
                # Physics properties optimized for pushing
                mass = masses[i] if hasattr(masses, '__getitem__') else masses
                friction = frictions[i] if hasattr(frictions, '__getitem__') else frictions
                restitution = context.get('restitutions', 0.0)  # No bounce
                
                # Spawn positions in tighter cluster for pushing
                spawn_x = context.get('spawn_x', np.random.uniform(-0.08, 0.08))
                spawn_y = context.get('spawn_y', np.random.uniform(-0.12, 0.12))
                spawn_rotation = context.get('spawn_rotation', np.random.uniform(0, 2*np.pi))
                
                # Color scheme matching push_t2multi.py
                red_color = np.array([194, 19, 22, 255]) / 255
                white_color = np.array([240, 160, 160, 255]) / 255
                color_factor = context.get('color_factor', 0.0)  # Mostly red
                object_color = red_color * (1 - color_factor) + white_color * color_factor
                
                # Convert rotation to quaternion [w, x, y, z] to avoid warning
                import math
                cos_half = math.cos(spawn_rotation / 2)
                sin_half = math.sin(spawn_rotation / 2)
                quaternion = [cos_half, 0, 0, sin_half]  # Rotation around Z-axis
                
                object_config = {
                    'id': f'push_object_{i}',
                    'type': object_type,  # Use object_type from context
                    'n_vertices': n_vertices,
                    'radius': radius,
                    'mass': mass,
                    'friction': friction,
                    'restitution': restitution,
                    'initial_pose': {
                        'position': [spawn_x, spawn_y, 0.025],  # On table surface
                        'quaternion': quaternion  # Proper quaternion instead of rotation angle
                    },
                    'color': object_color.tolist(),
                    'thickness': 0.06,  # Half thickness = 0.03 as in push_t2multi.py
                    
                    # ManiSkill-specific parameters
                    'material_config': {
                        'static_friction': friction,
                        'dynamic_friction': friction,
                        'restitution': restitution
                    },
                    'manipulation_properties': {
                        'pushable': True,
                        'graspable': False,  # Focus on pushing
                        'rollable': True
                    },
                    # Explicit pose to avoid "No initial pose set" warning
                    'builder_pose': {
                        'p': [spawn_x, spawn_y, 0.025],  # position
                        'q': quaternion  # quaternion [w, x, y, z]
                    }
                }
                
                object_configs.append(object_config)
            
            return object_configs

        except ImportError:
            # Fallback to basic config if ManiSkill not available
            object_configs = []
            for i in range(num_objects):
                object_config = {
                    'id': f'push_object_{i}',
                    'type': 'polygon',
                    'n_vertices': 6,
                    'radius': 0.07,
                    'mass': 0.8,
                    'friction': 0.3,
                    'restitution': 0.0
                }
                object_configs.append(object_config)
            return object_configs

    def create_environment(self, env_config: Dict[str, Any]):
        """Create custom environment instance using registered environment for proper multi-env visualization"""
        try:
            import gymnasium as gym

            # Determine if we should use parallel_in_single_scene for multi-env visualization
            render_mode = env_config.get('render_mode', 'human')
            num_envs = env_config.get('num_envs', 16)
            parallel_in_single_scene = (render_mode == 'human' and num_envs > 1)
            parallel_in_single_scene = True

            # Use our registered custom environment for proper multi-env visualization
            env_kwargs = {
                'num_envs': num_envs,
                'obs_mode': env_config.get('obs_mode', 'state_dict'),
                'control_mode': env_config.get('control_mode', 'pd_joint_delta_pos'),
                'render_mode': render_mode,
                'sim_backend': env_config.get('sim_backend', 'gpu'),
                'robot_uids': env_config.get('robot_uids', 'floating_panda_gripper_fin'),
                'robot_init_qpos_noise': env_config.get('robot_init_qpos_noise', 0.02),
                'parallel_in_single_scene': parallel_in_single_scene,
                'sim_config': dict(spacing=1.0),  # Tighter spacing for better visualization
                'viewer_camera_configs': dict(shader_pack='default'),  # Add viewer camera config
            }

            logger.info("Using registered custom environment: CustomPushT-v2multi (parallel_in_single_scene=%s)", parallel_in_single_scene)
            env = gym.make("CustomPushT-v2multi", **env_kwargs)

            return env

        except Exception as e:
            logger.error("Failed to create custom registered environment: %s", e)
            logger.info("Falling back to direct instantiation...")
            try:
                # Fallback to direct instantiation
                env = CustomPushT2MultiEnv(
                    num_envs=num_envs,
                    obs_mode=env_config.get('obs_mode', 'state_dict'),
                    control_mode=env_config.get('control_mode', 'pd_joint_delta_pos'),
                    render_mode=render_mode,
                    sim_backend=env_config.get('sim_backend', 'gpu'),
                    robot_uids=env_config.get('robot_uids', 'floating_panda_gripper_fin'),
                    robot_init_qpos_noise=env_config.get('robot_init_qpos_noise', 0.02),
                    parallel_in_single_scene=parallel_in_single_scene,
                    sim_config=dict(spacing=1.6),
                    viewer_camera_configs=dict(shader_pack='default'),
                )
                return env
            except Exception as e2:
                logger.error("Direct instantiation also failed: %s", e2)
                # Final fallback to original registered environment
                return super().create_environment(env_config)

    def load_robot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load robot configuration for pushing task"""
        return {
            'robot_uid': 'floating_panda_gripper_fin',
            'robot_config': {
                'control_mode': context.get('control_mode', 'pd_joint_delta_pos'),
                'init_qpos_noise': context.get('robot_init_qpos_noise', 0.02),
                'keyframe': 'open_facing_down'  # Default keyframe for pushing
            },
            'initial_pose': {
                'position': [0, 0, 0.5],  # Above table to avoid collisions
                'orientation': [1, 0, 0, 0]  # Identity quaternion
            }
        }
    
    def get_state(self, env):
        """Get current state from ManiSkill environment - focus on object states for caging"""
        try:
            # Access unwrapped environment to get to actual ManiSkill env
            if hasattr(env, 'unwrapped'):
                actual_env = env.unwrapped
            else:
                actual_env = env
            
            # Extract only object poses for caging tasks
            obj_poses = None
            
            if hasattr(actual_env, 'te'):
                obj_poses = actual_env.te.pose.raw_pose  # Object poses [num_envs, num_objects, 7]
            elif hasattr(actual_env, 'objects'):
                obj_poses = actual_env.objects.pose.raw_pose  # Object poses [num_envs, num_objects, 7]
            
            if obj_poses is not None:
                # Convert to numpy and focus on object positions for caging
                if hasattr(obj_poses, 'cpu'):
                    obj_poses_np = obj_poses.cpu().numpy()
                else:
                    obj_poses_np = obj_poses
                
                # Extract only positions (first 3 components) for caging
                if len(obj_poses_np.shape) >= 2:
                    obj_positions = obj_poses_np[..., :3]  # [num_envs, num_objects, 3]
                else:
                    obj_positions = obj_poses_np
                
                # Return object-focused state dictionary consistent with state2dris
                state = {
                    'obj_pose': obj_poses_np,  # Full poses for context
                    'obj_positions': obj_positions  # Positions only for caging
                }
                return state
            else:
                return {}
        except Exception as e:
            logger.error("Failed to get state: %s", e)
            return {}
    
    def set_state(self, env, state):
        """Set state to ManiSkill environment using state_dict approach from reference code"""
        try:

            # Check if we need to access the unwrapped environment
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

                # Check for executor format (single object from executor.get_obs())
                if 'object_pos' in state and 'object_quat' in state:
                    logger.info("set_state: Detected executor observation format - converting single YCB to 64 polygons")

                    # Extract single object pose
                    obj_pos = state['object_pos']  # [x, y, z]
                    obj_quat = state['object_quat']  # [w, x, y, z]

                    # Create single pose vector [x, y, z, qw, qx, qy, qz]
                    single_pose = np.concatenate([obj_pos, obj_quat])  # shape: (7,)

                    # Replicate for all 64 objects in planning environment
                    num_objects = 64
                    obj_poses = np.tile(single_pose, (num_objects, 1))  # shape: [64, 7]

                    logger.info("set_state: Replicated single object to %d objects at position %s", num_objects, obj_pos)

                if obj_poses is not None:

                    # Get device
                    if hasattr(actual_env, 'device'):
                        device = actual_env.unwrapped.device
                    else:
                        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

                    # Convert to torch tensor if needed
                    if isinstance(obj_poses, np.ndarray):
                        obj_poses = torch.from_numpy(obj_poses).to(device)

                    # Use state_dict approach like reference code
                    # Get current scene state
                    sim_state = actual_env.scene.get_sim_state()
                    actors_state = sim_state['actors']

                    # Get number of environments
                    num_envs = getattr(actual_env, 'num_envs', 16)

                    # Handle different input shapes
                    if len(obj_poses.shape) == 2:  # [num_objects, 7] - single env state
                        num_objects = obj_poses.shape[0]

                        # For cage evaluation: all environments should start from the same state
                        # This is intentional - we evaluate different actions from the same initial state
                        for env_idx in range(num_envs):
                            for obj_idx in range(num_objects):
                                actor_name = f"Te_{env_idx}_{obj_idx}"
                                if actor_name in actors_state:
                                    # Copy the same pose to all environments
                                    # This ensures parallel action evaluation from identical initial conditions
                                    new_pose = actors_state[actor_name].clone()
                                    # Use proper indexing for the state tensor
                                    new_pose[..., :3] = obj_poses[obj_idx, :3].clone()  # position
                                    new_pose[..., 3:7] = obj_poses[obj_idx, 3:7].clone()  # quaternion
                                    actors_state[actor_name] = new_pose


                    elif len(obj_poses.shape) == 3:  # [num_envs, num_objects, 7] - multi-env state
                        num_envs_in_state, num_objects = obj_poses.shape[:2]

                        # Update actors_state for each environment
                        for env_idx in range(min(num_envs_in_state, num_envs)):
                            for obj_idx in range(num_objects):
                                actor_name = f"Te_{env_idx}_{obj_idx}"
                                if actor_name in actors_state:
                                    # Clone the pose for this object
                                    new_pose = torch.zeros_like(actors_state[actor_name])
                                    # Use proper indexing for the state tensor
                                    new_pose[..., :3] = obj_poses[env_idx, obj_idx, :3]  # position
                                    new_pose[..., 3:7] = obj_poses[env_idx, obj_idx, 3:7]  # quaternion
                                    actors_state[actor_name] = new_pose


                    # Apply the updated state using set_state_dict
                    state_dict = {'actors': actors_state}
                    actual_env.set_state_dict(state_dict)

                    # Step physics to ensure poses are applied (optional, but helps with stability)
                    for _ in range(2):
                        actual_env.scene.step()

        except Exception as e:
            warnings.warn(f"Failed to set state: {e}")
    
    def direct_set_gripper_position(self, env, target_pos, target_ori):
        """
        Directly set gripper position (teleportation), not through controller control.
        Based on the reference implementation from gpu_demo_gui_float_gripper_multiple_obj.py
        
        Args:
            env: ManiSkill environment
            target_pos: Target position [x, y, z] [num_envs, 3]
            target_ori: Target orientation [roll, pitch, yaw] [num_envs, 3]
        """
        import torch
        
        robot = env.unwrapped.agent.robot
        
        # Convert target position and orientation to torch tensor
        if not isinstance(target_pos, torch.Tensor):
            target_pos = torch.tensor(target_pos, device=env.device, dtype=torch.float32)
        if not isinstance(target_ori, torch.Tensor):
            target_ori = torch.tensor(target_ori, device=env.device, dtype=torch.float32)
        
        # Apply offset (consistent with control_ee_pose)
        pos_offset = torch.tensor([0, 0.0, -0.41], device=target_pos.device)
        ori_offset = torch.tensor([0, torch.pi, 0.0], device=target_ori.device)
        
        # Apply offset
        target_pos = target_pos + pos_offset.expand_as(target_pos)
        target_ori = target_ori + ori_offset.expand_as(target_ori)
        
        # Get current qpos
        current_qpos = robot.get_qpos()
        
        # Directly set position and orientation (for floating gripper, first 3 are position, next 3 are orientation)
        new_qpos = current_qpos.clone()
        new_qpos[..., :3] = target_pos  # Set position
        new_qpos[..., 3:6] = target_ori  # Set orientation
        
        # Directly set robot state (teleportation)
        robot.set_qpos(new_qpos)
        
        # If GPU simulation, need manual update
        if env.device.type == "cuda":
            env.unwrapped.scene._gpu_apply_all()
            env.unwrapped.scene.px.gpu_update_articulation_kinematics()
            env.unwrapped.scene._gpu_fetch_all()

    def compute_dense_reward(self, env, obs, action, info):
        """Compute dense reward for pushing task - placeholder implementation"""
        import torch
        
        # Simple placeholder reward - parameters intentionally unused
        _ = env, obs, action, info  # Mark parameters as intentionally unused
        return torch.ones(1)
    
    def obs_to_dris(self, env, observations):
        """Convert ManiSkill observations to DRIS"""
        # env parameter intentionally unused - method uses observations directly
        _ = env  # Mark parameter as intentionally unused
        if isinstance(observations, dict):
            # Use the backend's state2dris method
            return self.state2dris(observations)
        else:
            # Handle raw observations
            from manidreams.base.dris import DRIS
            return [DRIS(observation=observations, representation_type="state")]

    def state2dris(self, observations, env_indices=None, env_config=None):
        """
        Convert state observation(s) to DRIS format.

        Supports two input formats:
        1. Backend format (from planning env): {'obj_pose': [num_envs, num_objects, 7]}
        2. Executor format (from executor.get_obs()): {'object_pos': [3], 'object_quat': [4]}

        Args:
            observations: Observation dict in either format
            env_indices: Optional list of environment indices to process
            env_config: Environment configuration containing num_envs

        Returns:
            List of DRIS objects (one per environment)
        """
        from manidreams.base.dris import DRIS
        import numpy as np

        # Check if this is executor format (single YCB object)
        if 'object_pos' in observations and 'object_quat' in observations:
            logger.info("state2dris: Detected executor observation format - converting single YCB to 64 polygons")

            # Extract YCB object pose
            ycb_pos = observations['object_pos']      # [x, y, z]
            ycb_quat = observations['object_quat']    # [w, x, y, z]

            # Create single pose vector [x, y, z, qw, qx, qy, qz]
            single_pose = np.concatenate([ycb_pos, ycb_quat])  # shape: (7,)

            # Replicate this pose for all 64 objects
            # All objects will be at the same position
            num_objects = 64  # Planning environment has 64 objects
            obj_poses = np.tile(single_pose, (num_objects, 1))  # shape: [64, 7]

            logger.info("state2dris: Created %d objects at position %s", num_objects, ycb_pos)

            # Convert to backend format and continue processing
            observations = {'obj_pose': obj_poses}

        # Now use parent class implementation for backend format
        return super().state2dris(observations, env_indices, env_config)