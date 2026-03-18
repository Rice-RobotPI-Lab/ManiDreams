"""
ManiSkill Backend for ManiDreams Framework
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

from manidreams.physics.simulation_tsip import SimulationBackend


class WhiteTableSceneBuilder:
    """
    White Table Scene Builder for ManiSkill environments.

    Extends TableSceneBuilder to create a smooth white table and handle
    robot keyframe initialization.
    """

    def __init__(self, env, robot_init_qpos_noise=0.02):
        try:
            from mani_skill.utils.scene_builder.table import TableSceneBuilder
            self.env = env
            self.robot_init_qpos_noise = robot_init_qpos_noise
            self.table_scene = TableSceneBuilder(env=env, robot_init_qpos_noise=robot_init_qpos_noise)
            self.table = None
        except ImportError as e:
            warnings.warn(f"ManiSkill components not available: {e}")
            self.table_scene = None

    def initialize(self, env_idx):
        if self.table_scene is None:
            return
        try:
            self.table_scene.initialize(env_idx)
            if hasattr(self.env, 'robot_uids') and self.env.robot_uids == "floating_panda_gripper_fin":
                if hasattr(self.env, 'agent') and hasattr(self.env.agent, 'keyframes'):
                    keyframe = self.env.agent.keyframes['open_facing_down']
                    self.env.agent.reset(keyframe.qpos)
                    self.env.agent.robot.set_pose(keyframe.pose)
        except Exception as e:
            warnings.warn(f"Failed to initialize robot keyframe: {e}")

    def build(self):
        if self.table_scene is None:
            return
        try:
            import sapien
            self.table_scene.build()
            self.table = self.table_scene.table
            if hasattr(self.table_scene, 'table') and self.table_scene.table is not None:
                for part in self.table_scene.table._objs:
                    render_component = part.find_component_by_type(sapien.render.RenderBodyComponent)
                    if render_component and render_component.render_shapes:
                        for render_shape in render_component.render_shapes:
                            if hasattr(render_shape, 'parts'):
                                for triangle in render_shape.parts:
                                    material = triangle.material
                                    material.set_base_color(np.array([255, 255, 255, 255]) / 255)
                                    material.set_base_color_texture(None)
                                    material.set_normal_texture(None)
                                    material.set_emission_texture(None)
                                    material.set_transmission_texture(None)
                                    material.set_metallic_texture(None)
                                    material.set_roughness_texture(None)
        except Exception as e:
            warnings.warn(f"Failed to create white table: {e}")


class ManiSkillBaseBackend(SimulationBackend):
    """
    Minimal ManiSkill backend.

    """

    def __init__(self):
        """Initialize ManiSkill backend"""
        self.context = {}
        if not MANISKILL_AVAILABLE:
            raise ImportError("ManiSkill is required for ManiSkillBaseBackend")
    
    def create_environment(self, env_config: Dict[str, Any]) -> gym.Env:
        """
        Create ManiSkill environment directly using TableSetupMixin as default.
        
        Args:
            env_config: Environment configuration including context_info
            
        Returns:
            ManiSkill gym environment created directly
        """
        logger.debug("ManiSkillBaseBackend.create_environment() called")
        try:
            # Import ManiSkill for environment registration
            import mani_skill
            import sapien
            
            # Default to TableSetupMixin configuration
            context_config = env_config.get('context_info', env_config)
            
            # Get environment type from config (default: PushT-v2multi)
            env_id = context_config.get('env_id', 'PushT-v2multi')
            
            # GUI configuration
            enable_gui = context_config.get('enable_gui', False)
            render_mode = context_config.get('render_mode', None)
            
            # Override render_mode if enable_gui is True
            if enable_gui:
                render_mode = 'human'
            
            # Shader configuration for rendering quality
            shader = 'default'

            
            # Parallel rendering configuration
            parallel_in_single_scene = True
            if render_mode == 'human':
                obs_mode = context_config.get('obs_mode', 'state_dict')
                num_envs = context_config.get('num_envs', 16)

            
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
            
            # Add shader configuration (always add for GUI compatibility)
            env_kwargs['sensor_configs'] = dict(shader_pack=shader)
            env_kwargs['human_render_camera_configs'] = dict(shader_pack=shader) 
            env_kwargs['viewer_camera_configs'] = dict(shader_pack=shader)
            
            # Add spacing for parallel environments
            num_envs = context_config.get('num_envs', 16)
            if 'sim_config' in context_config:
                env_kwargs['sim_config'] = context_config['sim_config']
            elif num_envs > 1:
                env_kwargs['sim_config'] = dict(spacing=1.6)
            
            # Create environment directly using ManiSkill registry
            env = gym.make(env_id, **env_kwargs)
            
            # Initialize GUI immediately like reference script
            if render_mode == 'human':
                # Reset environment to initialize properly
                obs, _ = env.reset(seed=2)
                env.action_space.seed(2)
                
                # Initialize viewer
                viewer = env.render()
                if hasattr(viewer, 'paused'):
                    viewer.paused = context_config.get('pause_on_start', True)
                env.render()  # Additional render call like reference script
            
            # Store environment parameters
            self.context.update(env_kwargs)
            self.context['enable_gui'] = enable_gui
            
            return env
            
        except ImportError as e:
            warnings.warn(f"ManiSkill not available: {e}")
            # Create mock environment for testing
            return self._create_mock_environment(env_config)
            
        except Exception as e:
            warnings.warn(f"Failed to create ManiSkill environment: {e}")
            raise
    

    
    def set_context(self, context: Dict[str, Any]):
        """
        Set context parameters for environment creation.
        
        Args:
            context: Context dictionary with environment parameters
        """
        self.context = context
    
    
    def reset_environment(self, env: gym.Env, seed: int = None, options: Dict = None):
        """Reset environment and initialize GUI if needed"""
        result = env.reset(seed=seed, options=options)
        
        # Initialize viewer after reset to avoid render order issues
        if hasattr(env, 'render_mode') and env.render_mode == 'human':
            try:
                viewer = env.render()
                if viewer is not None and hasattr(viewer, 'paused'):
                    # Configure viewer settings
                    viewer.paused = self.context.get('pause_on_start', True)
                env.render()  # Render initial frame
            except Exception as e:
                warnings.warn(f"Failed to initialize GUI viewer: {e}")
        
        return result
    
    
    def get_state(self, env: gym.Env) -> Dict[str, np.ndarray]:
        """
        Extract state from ManiSkill environment for DRIS.
        
        For pushing tasks, this extracts object poses directly from the environment,
        not from observations.
        
        Args:
            env: ManiSkill environment instance
            
        Returns:
            Dictionary containing object poses and positions
        """
        logger.debug("get_state called with env type: %s", type(env))
        try:
            # Check if we need to access the unwrapped environment
            if hasattr(env, 'unwrapped'):
                actual_env = env.unwrapped
                logger.debug("extract_state: Using unwrapped environment")
            else:
                actual_env = env
                
            # Check for push_t2multi style environment with 'te' objects
            if hasattr(actual_env, 'te'):
                logger.debug("extract_state: Found 'te' attribute")
                # push_t2multi uses env.te for objects
                obj_poses = actual_env.te.pose.raw_pose  # [num_envs * num_objects, 7]
                
                # Convert to numpy if needed
                if hasattr(obj_poses, 'cpu'):
                    obj_poses_np = obj_poses.cpu().numpy()
                else:
                    obj_poses_np = np.asarray(obj_poses)
                
                logger.debug("extract_state: Raw poses shape: %s", obj_poses_np.shape)
                
                # Determine dimensions
                total_items = obj_poses_np.shape[0]
                
                # Try to infer num_envs and num_objects
                # Assuming 16 envs and 64 objects as per task specification
                num_envs = 16
                num_objects = total_items // num_envs
                
                logger.debug("extract_state: Inferred: %d envs, %d objects", num_envs, num_objects)
                
                # Reshape to [num_envs, num_objects, 7]
                if len(obj_poses_np.shape) == 2 and total_items == num_envs * num_objects:
                    obj_poses_reshaped = obj_poses_np.reshape(num_envs, num_objects, 7)
                else:
                    obj_poses_reshaped = obj_poses_np
                
                logger.debug("extract_state: Reshaped poses: %s", obj_poses_reshaped.shape)
                
                # Extract positions (first 3 components) for caging
                obj_positions = obj_poses_reshaped[..., :3]
                
                # Return dictionary format expected by state2dris
                return {
                    'obj_pose': obj_poses_reshaped,  # [num_envs, num_objects, 7]
                    'obj_positions': obj_positions  # [num_envs, num_objects, 3]
                }
                
            # Fallback to generic 'objects' attribute
            elif hasattr(actual_env, 'objects'):
                # Get object poses directly from environment
                obj_poses = actual_env.objects.pose.raw_pose  # [num_envs, num_objects, 7]
                
                # Convert to numpy if needed
                if hasattr(obj_poses, 'cpu'):
                    obj_poses_np = obj_poses.cpu().numpy()
                else:
                    obj_poses_np = np.asarray(obj_poses)
                
                # Extract positions (first 3 components) for caging
                if len(obj_poses_np.shape) >= 2:
                    # Handle both [num_envs, num_objects, 7] and [num_objects, 7] shapes
                    obj_positions = obj_poses_np[..., :3]
                else:
                    obj_positions = obj_poses_np
                
                # Return dictionary format expected by state2dris
                return {
                    'obj_pose': obj_poses_np,  # Full poses [x,y,z,qw,qx,qy,qz]
                    'obj_positions': obj_positions  # Positions only [x,y,z]
                }
            
            # Fallback: try to get from observation if available
            elif hasattr(actual_env, 'get_obs'):
                logger.debug("extract_state: Trying get_obs fallback")
                obs = actual_env.get_obs()
                if isinstance(obs, dict) and 'obj_pose' in obs:
                    logger.debug("extract_state: Found obj_pose in get_obs")
                    return obs
                else:
                    logger.debug("extract_state: get_obs returned: %s, keys: %s", type(obs), list(obs.keys()) if isinstance(obs, dict) else 'not dict')
                    
            # Last resort: return empty dict
            logger.debug("extract_state: No object attributes found (te, objects, get_obs), returning empty")
            logger.debug("extract_state: Available attributes: %s", [attr for attr in dir(actual_env) if not attr.startswith('_')])
            return {'obj_pose': np.array([])}
        except Exception as e:
            logger.error("Error extracting state: %s", e)
            import traceback
            traceback.print_exc()
            return {'obj_pose': np.array([])}
    
    def set_state(self, env: gym.Env, state: Dict[str, np.ndarray]) -> None:
        """Set environment state from state dictionary"""
        if 'obj_pose' in state:
            # Use existing complex implementation from other backends
            # For now, delegate to basic set_state if available
            if hasattr(env, 'set_state'):
                env.set_state(state)
            else:
                warnings.warn("Environment does not support set_state operation")
        else:
            warnings.warn("State dictionary missing obj_pose")
    
    def get_observation_space(self, env: gym.Env) -> gym.Space:
        """Get observation space from environment"""
        return env.observation_space
    
    def get_action_space(self, env: gym.Env) -> gym.Space:
        """Get action space from environment"""
        return env.action_space
    
    def step_act(self, actions, env: gym.Env = None, cage=None, single_action=False):
        """
        Generic step_act interface for ManiSkill backend.

        This is a generic interface that should be overridden by task-specific backends.

        Args:
            actions: Action(s) to execute
            env: ManiSkill environment
            cage: Optional cage object for context
            single_action: If True, a single action is being broadcast to all envs

        Returns:
            Environment observation after stepping
        """
        if env is not None:
            # Generic step - just pass actions directly to environment
            if isinstance(actions, list) and len(actions) == 1:
                actions = actions[0]
            obs, _, _, _, _ = env.step(actions)
            return obs
        else:
            return actions
    
    def load_env(self, context: Dict[str, Any]) -> None:
        """
        Load simulation environment (table, walls, etc).
        
        Args:
            context: Context containing environment parameters
        """
        # Store environment parameters
        self.env_params = context.get('env_params', {})
        
    def load_object(self, context: Dict[str, Any]) -> None:
        """
        Load objects into the simulation based on context.
        
        Args:
            context: Context containing object parameters
        """
        # Store object parameters
        self.object_params = context.get('object_params', {})
        
    def load_robot(self, context: Dict[str, Any]) -> None:
        """
        Load robot configuration based on context.
        
        Args:
            context: Context containing robot parameters
        """
        # Store robot parameters
        self.robot_params = context.get('robot_params', {})
    
    def state2dris(self, observations, env_indices=None, env_config=None):
        """
        Convert state observation(s) to DRIS format with unified vectorized processing.
        
        Args:
            observations: Observation dict with 'obj_pose' containing vectorized states
            env_indices: Optional list of environment indices to process  
            env_config: Environment configuration containing num_envs
            
        Returns:
            List of DRIS objects (one per environment)
        """
        from manidreams.base.dris import DRIS

        # Handle list input (e.g., [obs] from TSIP reset)
        if isinstance(observations, list):
            if len(observations) == 1:
                # Single observation wrapped in list - unwrap it
                observations = observations[0]
            else:
                # Multiple observations - process each and combine
                all_dris = []
                for obs in observations:
                    dris_list = self.state2dris(obs, env_indices, env_config)
                    all_dris.extend(dris_list)
                return all_dris

        # Always treat as vectorized - if single observation, treat as 1-env vectorized
        if isinstance(observations, dict):

            # Check for different possible object pose keys
            obj_pose_key = None
            if 'obj_pose' in observations:
                obj_pose_key = 'obj_pose'
            elif 'extra' in observations and isinstance(observations['extra'], dict):
                extra = observations['extra']
                # Look for object-related keys in extra
                for key in extra.keys():
                    if 'obj' in key.lower() or 'te_' in key.lower():
                        obj_pose_key = ('extra', key)
                        break

            if obj_pose_key:
                if isinstance(obj_pose_key, tuple):
                    # Key is in extra dict
                    obj_poses = observations[obj_pose_key[0]][obj_pose_key[1]]
                else:
                    # Key is in main dict
                    obj_poses = observations[obj_pose_key]


                # Get environment configuration
                num_envs = env_config.get('num_envs', 1) if env_config else 1
                total_objects = obj_poses.shape[0]
                num_objects_per_env = total_objects // num_envs


                if total_objects % num_envs != 0:
                    logger.warning("Total objects (%d) not evenly divisible by num_envs (%d)", total_objects, num_envs)
                    # Treat as single environment
                    num_envs = 1
                    num_objects_per_env = total_objects

                # Reshape to [num_envs, num_objects_per_env, 7]
                if num_envs == 1:
                    obj_poses_reshaped = obj_poses.reshape(1, total_objects, 7)
                else:
                    obj_poses_reshaped = obj_poses.reshape(num_envs, num_objects_per_env, 7)

                # Create individual DRIS for each environment
                dris_list = []

                # Determine which environments to process
                if env_indices is None:
                    env_indices = list(range(num_envs))

                for env_idx in env_indices:
                    if env_idx < num_envs:
                        # Extract poses for this environment
                        env_obj_poses = obj_poses_reshaped[env_idx]  # [num_objects_per_env, 7]

                        # Create observation from positions (flatten for compatibility)
                        observation = env_obj_poses.flatten()

                        # Create DRIS with full pose information in context
                        dris = DRIS(
                            observation=observation,
                            representation_type="state",
                            context={'obj_poses': env_obj_poses}
                        )
                        dris_list.append(dris)

                return dris_list
        
        # If we reach here, the observations format is not supported
        raise ValueError(f"Unsupported observations format: {type(observations)}. "
                        f"Expected dict with 'obj_pose' key containing vectorized object states.")
    
        """
        Convert image observation(s) to DRIS format.
        
        Handles RGB/RGBD observations from ManiSkill environments.
        
        Args:
            observations: Single observation or list of observations
            
        Returns:
            List of DRIS objects
        """
        from manidreams.base.dris import DRIS
        
        if not isinstance(observations, list):
            observations = [observations]
        
        dris_list = []
        for obs in observations:
            if isinstance(obs, dict):
                # Check for image keys in dict observation
                if 'rgb' in obs:
                    image = obs['rgb']
                elif 'image' in obs:
                    image = obs['image']
                elif 'camera' in obs and isinstance(obs['camera'], dict):
                    # Nested camera observations
                    if 'rgb' in obs['camera']:
                        image = obs['camera']['rgb']
                    else:
                        image = list(obs['camera'].values())[0]  # First camera view
                else:
                    # No image found - create empty image
                    image = np.zeros((128, 128, 3), dtype=np.uint8)
                    
                # Ensure image is numpy array with proper shape
                if isinstance(image, np.ndarray):
                    if len(image.shape) == 4:  # [batch, H, W, C] -> [H, W, C]
                        image = image[0]
                    elif len(image.shape) == 2:  # [H, W] -> [H, W, 1]
                        image = np.expand_dims(image, axis=2)
                else:
                    image = np.array(image)
                
                dris = DRIS(observation=image, representation_type="image")
                dris_list.append(dris)
                
            elif isinstance(obs, np.ndarray):
                # Direct image array
                if len(obs.shape) >= 2:  # Valid image shape
                    dris = DRIS(observation=obs, representation_type="image")
                    dris_list.append(dris)
                else:
                    # Invalid image - create default
                    image = np.zeros((128, 128, 3), dtype=np.uint8)
                    dris = DRIS(observation=image, representation_type="image")
                    dris_list.append(dris)
                    
            else:
                # Fallback - create default image
                image = np.zeros((128, 128, 3), dtype=np.uint8)
                dris = DRIS(observation=image, representation_type="image")
                dris_list.append(dris)
        
        return dris_list


class TableSetupMixin:
    """
    ManiSkill table environment setup for manipulation tasks.
    
    Directly imports ManiSkill components like push_t2multi.py:
    - FloatingPandaGripperFin robot
    - TableSceneBuilder for white table setup
    """
    
    def __init__(self):
        """Initialize the table manipulation environment"""
        self.table_scene = None
        self.robot_config = None
        
    def load_env(self, context: Dict[str, Any]) -> Dict:
        """
        Load environment using ManiSkill TableSceneBuilder like push_t2multi.py.
        
        Args:
            context: Environment context parameters
            
        Returns:
            Environment configuration dict
        """
        try:
            # Import ManiSkill components
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils import sapien_utils
            from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig
            
            env_config = {
                # Simulation configuration matching push_t2multi.py
                'sim_config': SimConfig(
                    gpu_memory_config=GPUMemoryConfig(
                        found_lost_pairs_capacity=2**25, 
                        max_rigid_patch_count=2**18
                    )
                ),
                
                # Camera configuration matching push_t2multi.py
                'sensor_configs': [
                    CameraConfig(
                        "base_camera",
                        pose=sapien_utils.look_at(eye=[-0.08, 0, 0.45], target=[-0.08, 0, 0.1]),
                        width=128,
                        height=128,
                        fov=1.4,
                        near=0.01,
                        far=100,
                    )
                ],
                
                # Human render camera
                'human_render_camera_config': CameraConfig(
                    "render_camera", 
                    pose=sapien_utils.look_at(eye=[1, 0, 2], target=[-0.05, 0, 0.1]),
                    width=512, 
                    height=512, 
                    fov=0.5, 
                    near=0.01, 
                    far=100
                ),
                
                # Table scene builder configuration
                'table_scene_builder_class': WhiteTableSceneBuilder,  # Custom white table class
                'robot_init_qpos_noise': context.get('robot_qpos_noise', 0.02),
            }
            
            return env_config
            
        except ImportError:
            # Fallback configuration if ManiSkill not available
            return {
                'table_height': context.get('table_height', 0.6),
                'workspace_bounds': context.get('workspace_bounds', [-0.3, 0.3, -0.3, 0.3]),
                'camera_configs': {
                    'base_camera': {
                        'position': [-0.08, 0, 0.45],
                        'target': [-0.08, 0, 0.1],
                        'fov': 1.4,
                        'resolution': [128, 128]
                    }
                }
            }
    
    def load_robot(self, context: Dict[str, Any]) -> Dict:
        """
        Load robot using FloatingPandaGripperFin like push_t2multi.py.
        
        Args:
            context: Robot context parameters
            
        Returns:
            Robot configuration dict
        """
        try:
            # Import ManiSkill robot
            from assets.robots import FloatingPandaGripperFin
            
            robot_config = {
                'robot_class': FloatingPandaGripperFin,
                'robot_uids': 'floating_panda_gripper_fin',  # Matching push_t2multi.py
                'robot_init_qpos_noise': context.get('robot_qpos_noise', 0.02),
                'control_mode': context.get('control_mode', 'pd_joint_delta_pos'),
                
                # Default keyframe configuration
                'keyframe': 'open_facing_down',  # Matches push_t2multi.py
                
                # Controller configuration
                'controller_config': {
                    'kp_pos': 3.0,
                    'kp_ori': 2.0,
                    'kp_gripper': 100.0,
                    'damping': 0.1
                },
                
                # Initial starting position
                'ee_starting_pos2D': [0, 0, 0.5],  # Matching push_t2multi.py
                'ee_starting_pos3D': [0, 0, 0.5],
            }
            
            return robot_config
            
        except ImportError:
            # Fallback configuration if ManiSkill not available  
            return {
                'type': 'floating_panda_gripper',
                'initial_qpos': context.get('robot_init_qpos', [0.0] * 9),
                'qpos_noise': context.get('robot_qpos_noise', 0.02),
                'control_mode': context.get('control_mode', 'pd_joint_delta_pos'),
                'tcp_offset': [0.0, 0.0, 0.103]
            }
    
    def control_ee_pose(self, env, target_pos, target_ori, Kp_pos=3.0, Kp_ori=2.0):
        """
        End-effector pose control using ManiSkill's control interface.
        
        Task-agnostic method for controlling robot end-effector position and orientation.
        Follows the reference script gpu_demo_gui_float_gripper_multiple_obj.py implementation.
        
        Args:
            env: ManiSkill environment
            target_pos: Target position [x, y, z]
            target_ori: Target orientation [roll, pitch, yaw]
            Kp_pos: Position control gain
            Kp_ori: Orientation control gain
            
        Returns:
            Joint action array
        """
        import torch
        
        # Convert inputs to torch tensors
        if not isinstance(target_pos, torch.Tensor):
            target_pos = torch.tensor(target_pos, device=env.unwrapped.device, dtype=torch.float32)
        if not isinstance(target_ori, torch.Tensor):
            target_ori = torch.tensor(target_ori, device=env.unwrapped.device, dtype=torch.float32)
        
        # Apply offsets like reference script
        pos_offset = torch.tensor([0, 0.0, -0.33], device=target_pos.device)
        ori_offset = torch.tensor([0, torch.pi, 0.0], device=target_ori.device)
        
        # Apply offset
        target_pos = target_pos + pos_offset.expand_as(target_pos)
        target_ori = target_ori + ori_offset.expand_as(target_ori)
        
        # Get current joint states (for floating gripper, first 3 are position, next 3 are orientation)
        robot = env.agent.robot
        current_qpos = robot.get_qpos()
        
        # Extract position and orientation from joint states
        current_pos = current_qpos[..., :3]
        current_ori = current_qpos[..., 3:6]
        
        # Handle both single env and multi-env cases for target
        if target_pos.dim() == 2:
            # Multi-env case: target_pos is [num_envs, 3]
            pos_error = target_pos - current_pos
            ori_error = target_ori - current_ori
        else:
            # Single target for all envs: target_pos is [3]
            pos_error = target_pos.unsqueeze(0) - current_pos
            ori_error = target_ori.unsqueeze(0) - current_ori
        
        # Handle angle wrapping for orientation
        ori_error = torch.where(ori_error > torch.pi, ori_error - 2*torch.pi, ori_error)
        ori_error = torch.where(ori_error < -torch.pi, ori_error + 2*torch.pi, ori_error)
        
        # Apply gains
        pos_action = Kp_pos * pos_error
        ori_action = Kp_ori * ori_error
        
        # Combine into joint action for all environments
        joint_actions = torch.cat([pos_action, ori_action], dim=-1)
        
        # Create full action tensor with gripper action
        num_envs = env.num_envs if hasattr(env, 'num_envs') else 1
        
        # Ensure joint_actions has the right shape
        if joint_actions.dim() == 1:
            joint_actions = joint_actions.unsqueeze(0).expand(num_envs, -1)
        
        # Create full actions with gripper
        gripper_actions = torch.full((num_envs, 1), -0.6, device=env.unwrapped.device, dtype=torch.float32)
        full_actions = torch.cat([joint_actions, gripper_actions], dim=-1)
        
        # Clamp actions to valid range
        full_actions = torch.clamp(full_actions, -1, 1)
        
        return full_actions


