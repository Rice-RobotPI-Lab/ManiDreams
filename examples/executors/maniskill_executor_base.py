"""
ManiSkill Executor Base - Independent ManiSkill simulator executor.

Copied from ManiSkillBaseBackend but completely independent for executor use.
Creates its own simulation environment separate from planning.
"""

import logging
from typing import Any, Dict
import numpy as np
import warnings

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    MANISKILL_AVAILABLE = True
except ImportError:
    MANISKILL_AVAILABLE = False
    warnings.warn("ManiSkill not available. Install with: pip install mani-skill>=2.0.0")

# Add parent directories to path for imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # examples/
manidreams_dir = os.path.dirname(parent_dir)  # ManiDreams/
if manidreams_dir not in sys.path:
    sys.path.insert(0, manidreams_dir)
sys.path.insert(0, os.path.join(manidreams_dir, "src"))

from manidreams.executors.simulation_executor import SimulationExecutor


class WhiteTableSceneBuilder:
    """
    White Table Scene Builder for ManiSkill environments.
    Copied from ManiSkillBaseBackend.
    """

    def __init__(self, env, robot_init_qpos_noise=0.02):
        """Initialize WhiteTableSceneBuilder"""
        try:
            from mani_skill.utils.scene_builder.table import TableSceneBuilder
            import torch
            import sapien

            self.env = env
            self.robot_init_qpos_noise = robot_init_qpos_noise
            self.table_scene = TableSceneBuilder(env=env, robot_init_qpos_noise=robot_init_qpos_noise)
            self.table = None

        except ImportError as e:
            warnings.warn(f"ManiSkill components not available: {e}")
            self.table_scene = None

    def initialize(self, env_idx):
        """Initialize environment with robot keyframe setup"""
        if self.table_scene is None:
            return

        try:
            import torch

            # Call parent initialization
            self.table_scene.initialize(env_idx)

            b = len(env_idx)

            # Handle floating panda gripper keyframe setup
            if hasattr(self.env, 'robot_uids') and self.env.robot_uids == "floating_panda_gripper_fin":
                if hasattr(self.env, 'agent') and hasattr(self.env.agent, 'keyframes'):
                    keyframe = self.env.agent.keyframes['open_facing_down']
                    self.env.agent.reset(keyframe.qpos)
                    self.env.agent.robot.set_pose(keyframe.pose)

        except Exception as e:
            warnings.warn(f"Failed to initialize robot keyframe: {e}")

    def build(self):
        """Build scene with white table"""
        if self.table_scene is None:
            return

        try:
            import sapien
            import numpy as np

            # Build parent scene
            self.table_scene.build()
            self.table = self.table_scene.table

            # Make table white
            if hasattr(self.table_scene, 'table') and self.table_scene.table is not None:
                for part in self.table_scene.table._objs:
                    render_component = part.find_component_by_type(sapien.render.RenderBodyComponent)
                    if render_component and render_component.render_shapes:
                        for render_shape in render_component.render_shapes:
                            if hasattr(render_shape, 'parts'):
                                for triangle in render_shape.parts:
                                    material = triangle.material
                                    # Set white color
                                    material.set_base_color(np.array([255, 255, 255, 255]) / 255)
                                    # Remove all textures
                                    material.set_base_color_texture(None)
                                    material.set_normal_texture(None)
                                    material.set_emission_texture(None)
                                    material.set_transmission_texture(None)
                                    material.set_metallic_texture(None)
                                    material.set_roughness_texture(None)

        except Exception as e:
            warnings.warn(f"Failed to create white table: {e}")


class ManiSkillExecutorBase(SimulationExecutor):
    """
    Base ManiSkill executor providing common functionality.
    Independent from backends - creates its own simulation.
    """

    def __init__(self):
        """Initialize ManiSkill executor base"""
        super().__init__()
        self.context = {}
        self.env = None
        self.device = None

        if not MANISKILL_AVAILABLE:
            raise ImportError("ManiSkill is required for ManiSkillExecutorBase")

    def create_environment(self, env_config: Dict[str, Any]) -> gym.Env:
        """
        Create ManiSkill environment independently.

        Args:
            env_config: Environment configuration

        Returns:
            ManiSkill gym environment
        """
        try:
            import mani_skill
            import sapien

            context_config = env_config.get('context_info', env_config)

            # Get environment type
            env_id = context_config.get('env_id', 'PushT-v2multi')

            # GUI configuration
            enable_gui = context_config.get('enable_gui', False)
            render_mode = context_config.get('render_mode', None)

            if enable_gui:
                render_mode = 'human'

            # Shader configuration - use ray tracing for beautiful visuals
            shader = context_config.get('shader', 'rt-fast')  # Default to ray tracing

            # Parallel rendering configuration
            parallel_in_single_scene = False
            if render_mode == 'human':
                obs_mode = context_config.get('obs_mode', 'state_dict')
                num_envs = context_config.get('num_envs', 1)  # Single env for executor
                if obs_mode not in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"] and num_envs > 1:
                    parallel_in_single_scene = True

            # Environment creation parameters
            env_kwargs = {
                'num_envs': context_config.get('num_envs', 1),  # Default to single env
                'obs_mode': context_config.get('obs_mode', 'state_dict'),
                'control_mode': context_config.get('control_mode', 'pd_joint_delta_pos'),
                'render_mode': render_mode,
                'sim_backend': context_config.get('sim_backend', 'gpu'),
                'robot_uids': context_config.get('robot_uids', 'floating_panda_gripper_fin'),
                'robot_init_qpos_noise': context_config.get('robot_init_qpos_noise', 0.02),
                'parallel_in_single_scene': parallel_in_single_scene,
            }

            # Add shader configuration
            env_kwargs['sensor_configs'] = dict(shader_pack=shader)
            env_kwargs['human_render_camera_configs'] = dict(shader_pack=shader)
            env_kwargs['viewer_camera_configs'] = dict(shader_pack=shader)

            # Add spacing for parallel environments
            num_envs = context_config.get('num_envs', 1)
            if 'sim_config' in context_config:
                env_kwargs['sim_config'] = context_config['sim_config']
            elif num_envs > 1:
                env_kwargs['sim_config'] = dict(spacing=1.6)

            # Create environment
            env = gym.make(env_id, **env_kwargs)

            # Initialize GUI if needed
            if render_mode == 'human':
                obs, _ = env.reset(seed=context_config.get('seed', 2))
                env.action_space.seed(2)

                viewer = env.render()
                if hasattr(viewer, 'paused'):
                    viewer.paused = context_config.get('pause_on_start', False)
                env.render()

            # Store environment parameters
            self.context.update(env_kwargs)
            self.context['enable_gui'] = enable_gui

            return env

        except Exception as e:
            warnings.warn(f"Failed to create ManiSkill environment: {e}")
            raise

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize executor with configuration.

        Args:
            config: Configuration dictionary
        """
        # Store configuration
        self.config = config

        # Get device
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Call parent initialization
        super().initialize(config)

    def control_ee_pose(self, env, target_pos, target_ori, Kp_pos=3.0, Kp_ori=2.0):
        """
        End-effector pose control.

        Args:
            env: ManiSkill environment
            target_pos: Target position
            target_ori: Target orientation
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

        # Apply offsets
        pos_offset = torch.tensor([0, 0.0, -0.33], device=target_pos.device)
        ori_offset = torch.tensor([0, torch.pi, 0.0], device=target_ori.device)

        target_pos = target_pos + pos_offset.expand_as(target_pos)
        target_ori = target_ori + ori_offset.expand_as(target_ori)

        # Get current joint states
        robot = env.agent.robot
        current_qpos = robot.get_qpos()

        current_pos = current_qpos[..., :3]
        current_ori = current_qpos[..., 3:6]

        # Handle both single env and multi-env cases
        if target_pos.dim() == 2:
            pos_error = target_pos - current_pos
            ori_error = target_ori - current_ori
        else:
            pos_error = target_pos.unsqueeze(0) - current_pos
            ori_error = target_ori.unsqueeze(0) - current_ori

        # Handle angle wrapping
        ori_error = torch.where(ori_error > torch.pi, ori_error - 2*torch.pi, ori_error)
        ori_error = torch.where(ori_error < -torch.pi, ori_error + 2*torch.pi, ori_error)

        # Apply gains
        pos_action = Kp_pos * pos_error
        ori_action = Kp_ori * ori_error

        # Combine into joint action
        joint_actions = torch.cat([pos_action, ori_action], dim=-1)

        # Create full action tensor with gripper
        num_envs = env.num_envs if hasattr(env, 'num_envs') else 1

        if joint_actions.dim() == 1:
            joint_actions = joint_actions.unsqueeze(0).expand(num_envs, -1)

        gripper_actions = torch.full((num_envs, 1), -0.6, device=env.unwrapped.device, dtype=torch.float32)
        full_actions = torch.cat([joint_actions, gripper_actions], dim=-1)

        # Clamp actions
        full_actions = torch.clamp(full_actions, -1, 1)

        return full_actions

    def get_state(self, env: gym.Env) -> Dict[str, np.ndarray]:
        """Extract state from ManiSkill environment"""
        try:
            if hasattr(env, 'unwrapped'):
                actual_env = env.unwrapped
            else:
                actual_env = env

            if hasattr(actual_env, 'te'):
                obj_poses = actual_env.te.pose.raw_pose
            elif hasattr(actual_env, 'objects'):
                obj_poses = actual_env.objects.pose.raw_pose
            else:
                return {'obj_pose': np.array([])}

            if hasattr(obj_poses, 'cpu'):
                obj_poses_np = obj_poses.cpu().numpy()
            else:
                obj_poses_np = np.asarray(obj_poses)

            if len(obj_poses_np.shape) >= 2:
                obj_positions = obj_poses_np[..., :3]
            else:
                obj_positions = obj_poses_np

            return {
                'obj_pose': obj_poses_np,
                'obj_positions': obj_positions
            }
        except Exception as e:
            logger.error("Error extracting state: %s", e)
            return {'obj_pose': np.array([])}

    def set_state(self, env: gym.Env, state: Dict[str, np.ndarray]) -> None:
        """Set environment state"""
        if 'obj_pose' in state:
            if hasattr(env, 'set_state'):
                env.set_state(state)
            else:
                warnings.warn("Environment does not support set_state operation")
        else:
            warnings.warn("State dictionary missing obj_pose")

    def close(self) -> None:
        """Close environment and clean up resources."""
        if self.env is not None:
            self.env.close()
            self.env = None

        super().close()