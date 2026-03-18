"""
Pushing Task Executor - Independent executor for object pushing with YCB objects.

Copied from PushBackend but creates its own simulation with random YCB objects.
Completely independent from planning environment.
"""

import numpy as np
import torch
import random
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
import warnings

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
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
    MANISKILL_AVAILABLE = True
except ImportError:
    MANISKILL_AVAILABLE = False
    warnings.warn("ManiSkill not available")

# Import base class
from .maniskill_executor_base import ManiSkillExecutorBase, WhiteTableSceneBuilder


# Custom environment for executor with single YCB object
@register_env("ExecutorPushYCB-v1", max_episode_steps=100)
class ExecutorPushYCBEnv(BaseEnv):
    """
    Single YCB object pushing environment for executor.
    Independent from planning environment.
    """

    SUPPORTED_ROBOTS = ["floating_panda_gripper_fin"]
    agent: FloatingPandaGripperFin

    # YCB objects list
    YCB_OBJECTS = [
        "002_master_chef_can",
        "004_sugar_box",
        "005_tomato_soup_can",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "010_potted_meat_can",
        "024_bowl",
        "025_mug",
        "036_wood_block",
        "061_foam_brick"
    ]

    def __init__(self, *args, robot_uids="floating_panda_gripper_fin", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Randomly select YCB object
        self.selected_ycb = random.choice(self.YCB_OBJECTS)
        logger.info("Selected YCB object: %s", self.selected_ycb)
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
        pose = sapien_utils.look_at(eye=[1, 0, 2], target=[-0.05, 0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=0.5, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        self.table_scene = WhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

    def _load_agent(self, options: dict):
        """Load robot agent."""
        initial_pose = sapien.Pose(p=[0, 0, 0.5])
        super()._load_agent(options, initial_agent_poses=initial_pose)

        # Load single YCB object
        self._load_ycb_object()

    def _load_ycb_object(self):
        """Load single YCB object for pushing."""
        try:
            # Create YCB object using ManiSkill's actor builder (like pick_backend)
            logger.info("Loading YCB object: ycb:%s", self.selected_ycb)
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.selected_ycb}")

            # Set initial pose before building (same as cage initial center)
            builder.initial_pose = sapien.Pose(
                p=[-0.2, 0.0, 0.05],  # Same x,y as cage center [-0.2, 0.0]
                q=[1, 0, 0, 0]  # Identity quaternion
            )

            # Set scene indices for single environment
            builder.set_scene_idxs([0])

            # Build the object
            self.ycb_object = builder.build(name="ycb_object")
            logger.info("Successfully loaded YCB object: %s", self.selected_ycb)

        except Exception as e:
            logger.warning("Failed to load YCB object %s: %s", self.selected_ycb, e)
            logger.warning("Falling back to simple cube...")

            # Fallback to simple cube
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.025, 0.025, 0.025])
            builder.add_box_visual(half_size=[0.025, 0.025, 0.025], color=[1, 0, 0, 1])
            builder.initial_pose = sapien.Pose(
                p=[0, 0, 0.05],
                q=[1, 0, 0, 0]
            )
            builder.set_scene_idxs([0])
            self.ycb_object = builder.build(name="fallback_cube")
            logger.warning("Using fallback cube object")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode."""
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

            # Random initial position for YCB object around cage center [-0.2, 0.0]
            cage_center_x, cage_center_y = 0.0, 0.0
            random_x = cage_center_x + np.random.uniform(-0.01, 0.01)
            random_y = cage_center_y + np.random.uniform(-0.01, 0.01)

            # Set pose using ManiSkill's Pose structure for proper vectorized handling
            from mani_skill.utils.structs import Pose
            obj_pose = Pose.create_from_pq(
                p=torch.tensor([[random_x, random_y, 0.05]], device=self.device, dtype=torch.float32),
                q=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)  # w,x,y,z
            )
            self.ycb_object.set_pose(obj_pose)

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.ycb_object.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        """No reward needed for execution."""
        return torch.zeros((self.num_envs,), device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        """No reward needed for execution."""
        return torch.zeros((self.num_envs,), device=self.device)


class PushingTaskExecutor(ManiSkillExecutorBase):
    """
    Task-specific executor for object pushing.
    Creates independent simulation with random YCB object.
    """

    def __init__(self):
        """Initialize pushing task executor."""
        super().__init__()

        # Same parameters as PushBackend
        self.device = None
        self.sequence_len = 20
        self.step_len = 25
        self.r_approach = 0.24
        self.Kp_pos = 3.0
        self.Kp_ori = 0.5
        self.Kp_pos_push = 1
        self.num_directions = 16

        self.current_context = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize executor with independent simulation.

        Args:
            config: Configuration dictionary
        """
        # Create independent environment with YCB object
        logger.info("=== Creating Independent Executor Environment ===")
        logger.info("Creating new simulation with random YCB object...")

        env_config = {
            'env_id': 'ExecutorPushYCB-v1',  # Use custom YCB environment
            'num_envs': 1,  # Single environment for execution
            'render_mode': config.get('render_mode', 'human'),
            'sim_backend': config.get('sim_backend', 'gpu'),
            'control_mode': config.get('control_mode', 'pd_joint_delta_pos'),
            'robot_uids': 'floating_panda_gripper_fin',
            'seed': config.get('seed', random.randint(0, 10000)),
            # Ray tracing shader configuration for beautiful rendering
            'shader': 'rt-fast',  # Use ray tracing shader for best visual quality

        }

        # Create environment
        self.env = self.create_environment(env_config)

        # Set device
        if hasattr(self.env, 'unwrapped'):
            self.device = self.env.unwrapped.device
        else:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Reset environment
        self.env.reset()

        logger.info("Executor environment ready")

        # Call parent initialization
        super().initialize(config)

    def step_act(self, actions, env=None, cage=None) -> Any:
        """
        Execute pushing action (copied from PushBackend).

        Args:
            actions: Single action or list of actions (0-15 for 16 push directions)
            env: ManiSkill environment (uses self.env if not provided)
            cage: Cage object containing center position

        Returns:
            Final observation after completing the action
        """
        import torch

        # Use provided env or self.env
        if env is None:
            env = self.env

        # Handle parallel action execution
        if not isinstance(actions, list):
            actions = [actions]

        # Set device from environment
        self.device = env.unwrapped.device if hasattr(env.unwrapped, 'device') else self.device

        # Get number of environments
        num_envs = len(env.agent.tcp.pose.p) if hasattr(env, 'agent') and hasattr(env.agent, 'tcp') else 1

        # Ensure we have the right number of actions
        if len(actions) == 1:
            actions = actions * num_envs


        # Get target position (cage center or object center)
        target_positions = self.calculate_target_position(cage, num_envs)

        # Calculate push directions and orientations
        push_angles = [self.calculate_action_angle(action) for action in actions]
        push_angle_tensor = torch.tensor(push_angles, device=self.device, dtype=torch.float32)

        # Calculate push directions
        push_directions = torch.stack([
            torch.cos(push_angle_tensor),
            torch.sin(push_angle_tensor),
            torch.zeros_like(push_angle_tensor)
        ], dim=1)

        # Calculate target orientations
        theta_ori = push_angle_tensor + torch.pi
        target_ori = torch.stack([
            torch.zeros_like(theta_ori),
            torch.zeros_like(theta_ori),
            -theta_ori
        ], dim=1)

        # Step 1: Direct teleportation to approach position
        approach_positions = target_positions + push_directions * self.r_approach
        approach_positions[:, 2] = target_positions[:, 2] + 0.06

        self.direct_set_gripper_position(env, approach_positions, target_ori)

        # Step 2: Controlled push action
        push_target = 0.75 * target_positions + 0.25 * approach_positions
        push_target[:, 2] = target_positions[:, 2] + 0.01

        # Execute push action with controller
        for _ in range(self.step_len):
            action_vec = self.control_ee_pose(env, push_target, target_ori,
                                            self.Kp_pos_push, self.Kp_ori)
            obs, _, _, _, _ = env.step(action_vec)

            # Render if GUI mode
            if hasattr(env, 'render_mode') and env.render_mode == 'human':
                try:
                    env.render()
                except AttributeError:
                    pass

        return obs

    def get_obs(self) -> Dict[str, Any]:
        """
        Get state-based observation from ManiSkill simulator.

        Returns:
            Dictionary containing:
                - object_pos: [x, y, z] position of YCB object
                - object_quat: [w, x, y, z] quaternion of object
                - gripper_pos: [x, y, z] position of gripper TCP
                - gripper_quat: [w, x, y, z] quaternion of gripper
        """
        if not self.initialized or self.env is None:
            raise RuntimeError("Executor not initialized. Call initialize() first.")

        try:
            import torch
            import numpy as np

            # Get unwrapped environment
            actual_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env

            # Extract YCB object state
            if hasattr(actual_env, 'ycb_object'):
                obj_pose = actual_env.ycb_object.pose.raw_pose  # [pos(3), quat(4)]

                # Convert to numpy
                if hasattr(obj_pose, 'cpu'):
                    obj_pose_np = obj_pose.cpu().numpy()
                else:
                    obj_pose_np = np.asarray(obj_pose)

                # Handle vectorized environments (shape: [num_envs, 7])
                if obj_pose_np.ndim == 2:
                    obj_pose_np = obj_pose_np[0]  # Take first environment

                obj_pos = obj_pose_np[:3]
                obj_quat = obj_pose_np[3:7]  # [w, x, y, z]
            else:
                raise AttributeError("Environment does not have 'ycb_object' attribute")

            # Extract gripper TCP state
            if hasattr(actual_env, 'agent') and hasattr(actual_env.agent, 'tcp'):
                tcp_pose = actual_env.agent.tcp.pose.raw_pose  # [pos(3), quat(4)]

                # Convert to numpy
                if hasattr(tcp_pose, 'cpu'):
                    tcp_pose_np = tcp_pose.cpu().numpy()
                else:
                    tcp_pose_np = np.asarray(tcp_pose)

                # Handle vectorized environments
                if tcp_pose_np.ndim == 2:
                    tcp_pose_np = tcp_pose_np[0]  # Take first environment

                gripper_pos = tcp_pose_np[:3]
                gripper_quat = tcp_pose_np[3:7]  # [w, x, y, z]
            else:
                raise AttributeError("Environment does not have gripper TCP")

            return {
                'object_pos': obj_pos,
                'object_quat': obj_quat,
                'gripper_pos': gripper_pos,
                'gripper_quat': gripper_quat,
                'mode': 'state'
            }

        except Exception as e:
            raise RuntimeError(f"Failed to get observation: {e}")

    def _get_object_position(self, env):
        """Get YCB object position."""
        try:
            if hasattr(env, 'unwrapped'):
                actual_env = env.unwrapped
            else:
                actual_env = env

            # Try to get YCB object position
            if hasattr(actual_env, 'ycb_object'):
                obj_pos = actual_env.ycb_object.pose.p
                if hasattr(obj_pos, 'cpu'):
                    obj_pos = obj_pos.cpu()
                if not isinstance(obj_pos, torch.Tensor):
                    obj_pos = torch.tensor(obj_pos, device=self.device, dtype=torch.float32)
                return obj_pos.unsqueeze(0) if obj_pos.dim() == 1 else obj_pos
            else:
                # Fallback to center of table
                return torch.tensor([[0.0, 0.0, 0.06]], device=self.device, dtype=torch.float32)
        except Exception:
            return torch.tensor([[0.0, 0.0, 0.06]], device=self.device, dtype=torch.float32)

    def calculate_action_angle(self, action: int) -> float:
        """Convert discrete action to angle."""
        import torch
        if not isinstance(action, int) or action < 0 or action >= self.num_directions:
            action = 0

        theta_min = 0
        interval_size = 2 * torch.pi / self.num_directions
        return theta_min + action * interval_size

    def calculate_target_position(self, cage, num_envs, fallback_positions=None):
        """Calculate target position from cage center or fallback."""
        import torch

        if cage is not None and hasattr(cage, 'center'):
            current_cage_center = cage.center
            cage_center = torch.tensor(current_cage_center, device=self.device, dtype=torch.float32)
            target_positions = cage_center.unsqueeze(0).expand(num_envs, -1)

            if len(cage_center) == 2:
                z_coord = torch.full((num_envs, 1), 0.06, device=self.device)
                target_positions = torch.cat([target_positions, z_coord], dim=1)
        else:
            if fallback_positions is not None:
                target_positions = fallback_positions
            else:
                target_positions = torch.zeros((num_envs, 3), device=self.device)
                target_positions[:, 2] = 0.06

        return target_positions

    def direct_set_gripper_position(self, env, target_pos, target_ori):
        """Directly set gripper position."""
        import torch

        robot = env.unwrapped.agent.robot

        # Convert to tensors if needed
        if not isinstance(target_pos, torch.Tensor):
            target_pos = torch.tensor(target_pos, device=env.device, dtype=torch.float32)
        if not isinstance(target_ori, torch.Tensor):
            target_ori = torch.tensor(target_ori, device=env.device, dtype=torch.float32)

        # Apply offset
        pos_offset = torch.tensor([0, 0.0, -0.41], device=target_pos.device)
        ori_offset = torch.tensor([0, torch.pi, 0.0], device=target_ori.device)

        target_pos = target_pos + pos_offset.expand_as(target_pos)
        target_ori = target_ori + ori_offset.expand_as(target_ori)

        # Get current qpos
        current_qpos = robot.get_qpos()

        # Set new position and orientation
        new_qpos = current_qpos.clone()
        new_qpos[..., :3] = target_pos
        new_qpos[..., 3:6] = target_ori

        # Set robot state
        robot.set_qpos(new_qpos)

        # GPU update if needed
        if env.device.type == "cuda":
            env.unwrapped.scene._gpu_apply_all()
            env.unwrapped.scene.px.gpu_update_articulation_kinematics()
            env.unwrapped.scene._gpu_fetch_all()

    def execute(self,
                actions: Union[Any, List[Any]],
                get_feedback: bool = True,
                cage_history: List[Dict[str, Any]] = None) -> Union[Tuple[Any, Dict], Tuple[List[Any], List[Dict]]]:
        """
        Execute actions in independent simulation using saved cage positions.

        Args:
            actions: Action or list of actions to execute
            get_feedback: Whether to return feedback
            cage_history: List of cage states from planning (center, radius, timestep)

        Returns:
            Observations and feedback from execution
        """
        if not isinstance(actions, list):
            actions = [actions]
            single_action = True
        else:
            single_action = False

        observations = []
        feedbacks = []

        logger.info("Executing %d actions using saved cage positions...", len(actions))

        for i, action in enumerate(actions):
            # Use saved cage position if available
            cage_center = None
            if cage_history and i < len(cage_history):
                cage_info = cage_history[i]
                cage_center = cage_info['center']
                logger.info("  Action %d/%d: %s, cage_center: %s", i+1, len(actions), action, cage_center)
            else:
                logger.info("  Action %d/%d: %s (no cage info)", i+1, len(actions), action)

            # Create simple cage object with saved center position
            cage = None
            if cage_center is not None:
                # Create a simple cage-like object with the saved center
                cage = type('SimpleCage', (), {
                    'center': cage_center,
                    'radius': cage_history[i].get('radius', 0.28) if cage_history else 0.28
                })()

            # Execute using step_act
            obs = self.step_act(action, env=self.env, cage=cage)
            observations.append(obs)

            if get_feedback:
                feedbacks.append({
                    'action': action,
                    'timestep': i,
                    'cage_center': cage_center,
                    'cage_radius': cage.radius if cage else None
                })

        logger.info("Execution complete!")

        if single_action:
            return observations[0], feedbacks[0] if get_feedback else {}
        else:
            return observations, feedbacks if get_feedback else [{}] * len(observations)