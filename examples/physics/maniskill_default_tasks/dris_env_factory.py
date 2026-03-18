"""
DRIS Environment Factory

Dynamically creates DRIS-enabled versions of ManiSkill environments.
"""

from typing import Any, Dict, Optional, Tuple, Type
import importlib
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

from .task_config import TASK_CONFIGS, TaskConfig, get_task_config
from .dris_utils import DRISMixin


def _import_class(class_path: str) -> Type:
    """Import a class from its full module path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_dris_env_class(
    task_id: str,
    n_dris_copies: int = 8,
    pose_noise: Tuple[float, float, float, float, float, float] = (0.01, 0.01, 0.0, 0.0, 0.0, 0.0),
    physics_noise: Tuple[float, float] = (0.0, 0.0),
) -> Type[BaseEnv]:
    """
    Dynamically create a DRIS-enabled version of a ManiSkill environment.

    This function creates a new class that:
    1. Inherits from both DRISMixin and the original task environment
    2. Overrides _load_scene to create DRIS copies
    3. Overrides _initialize_episode to randomize DRIS poses
    4. Overrides compute_dense_reward to use mean of per-copy rewards
    5. Adds DRIS info to the step output

    Args:
        task_id: ManiSkill task ID (e.g., "PushCube-v1")
        n_dris_copies: Number of DRIS copies to create
        pose_noise: (dx, dy, dz, droll, dpitch, dyaw) - Random pose offset range

    Returns:
        DRISEnv class (not instantiated)
    """
    # Get task configuration
    config = get_task_config(task_id)

    # Import base environment class
    BaseEnvClass = _import_class(config.base_class_path)

    # Create new class dynamically
    class DRISEnv(DRISMixin, BaseEnvClass):
        """
        DRIS-enabled version of the base environment.

        Automatically creates DRIS copies of the target object and
        computes reward as mean of per-copy rewards.
        """

        def __init__(self, **kwargs):
            # Store DRIS configuration
            self.n_dris_copies = n_dris_copies
            self.pose_noise = pose_noise
            self.physics_noise = physics_noise
            self.task_config = config

            # Initialize base class
            super().__init__(**kwargs)

        def _load_scene(self, options: dict):
            """Load scene and create DRIS copies."""
            # Call base class to load original scene
            super()._load_scene(options)

            # Make original target semi-transparent (same alpha as DRIS copies)
            self._set_target_transparent(alpha=0.6)

            # Create DRIS copies
            self._create_dris_copies()

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
            """Initialize episode and randomize DRIS poses."""
            # Call base class to initialize
            super()._initialize_episode(env_idx, options)

            # Randomize DRIS poses around target
            self._randomize_dris_poses(env_idx)

        def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
            """
            Compute reward as mean of per-copy rewards.

            For each DRIS copy, temporarily moves the target object to that
            copy's position, computes the original reward, then averages.
            """
            # Get the base class reward function
            base_reward_func = super().compute_dense_reward

            # Compute mean reward across DRIS copies
            return self._compute_dris_rewards(base_reward_func, obs, action, info)

        def step(self, action):
            """Step environment and add DRIS info."""
            obs, reward, terminated, truncated, info = super().step(action)

            # Add DRIS information to info
            dris_info = self.get_dris_info()
            info.update(dris_info)

            return obs, reward, terminated, truncated, info

        def _get_obs_extra(self, info: Dict):
            """Get extra observations including DRIS info."""
            obs = super()._get_obs_extra(info)

            # Add DRIS mean position as observation
            if self.obs_mode_struct.use_state:
                obs['dris_mean_pos'] = self._compute_dris_mean_position()
                obs['dris_variance'] = self._compute_dris_variance()

            return obs

    # Set a meaningful name for the class
    DRISEnv.__name__ = f"DRIS{BaseEnvClass.__name__}"
    DRISEnv.__qualname__ = f"DRIS{BaseEnvClass.__name__}"

    return DRISEnv


# Cache for created classes to avoid re-creation
_dris_env_cache: Dict[str, Type[BaseEnv]] = {}


def make_dris_env(
    task_id: str,
    n_dris_copies: int = 8,
    pose_noise: Tuple[float, float, float, float, float, float] = (0.01, 0.01, 0.0, 0.0, 0.0, 0.0),
    physics_noise: Tuple[float, float] = (0.0, 0.0),
    **env_kwargs
) -> BaseEnv:
    """
    Create an instance of a DRIS-enabled ManiSkill environment.

    Args:
        task_id: ManiSkill task ID (e.g., "PushCube-v1")
        n_dris_copies: Number of DRIS copies to create
        pose_noise: (dx, dy, dz, droll, dpitch, dyaw) - Random pose offset range
        **env_kwargs: Additional arguments passed to environment constructor
            Common ones:
            - num_envs: Number of parallel environments
            - render_mode: "human", "rgb_array", or None
            - obs_mode: "state_dict", "rgbd", etc.
            - control_mode: "pd_ee_delta_pose", etc.

    Returns:
        Instantiated DRIS environment

    Example:
        env = make_dris_env(
            "PushCube-v1",
            n_dris_copies=8,
            pose_noise=(0.05, 0.05, 0.0, 0.0, 0.0, 0.1),
            num_envs=16,
            render_mode="human",
        )
    """
    # Create cache key
    cache_key = f"{task_id}_{n_dris_copies}_{pose_noise}_{physics_noise}"

    # Get or create DRIS env class
    if cache_key not in _dris_env_cache:
        _dris_env_cache[cache_key] = create_dris_env_class(
            task_id, n_dris_copies, pose_noise, physics_noise
        )

    DRISEnvClass = _dris_env_cache[cache_key]

    # Instantiate and return
    return DRISEnvClass(**env_kwargs)


def register_dris_env(
    task_id: str,
    n_dris_copies: int = 8,
    pose_noise: Tuple[float, float, float, float, float, float] = (0.01, 0.01, 0.0, 0.0, 0.0, 0.0),
    max_episode_steps: Optional[int] = None,
) -> str:
    """
    Register a DRIS environment with ManiSkill's registry.

    This allows using gym.make() to create the environment.

    Args:
        task_id: Base ManiSkill task ID
        n_dris_copies: Number of DRIS copies
        pose_noise: Pose randomization range (dx, dy, dz, droll, dpitch, dyaw)
        max_episode_steps: Override max episode steps (default: same as base)

    Returns:
        Registered environment ID (e.g., "DRISPushCube-v1")

    Example:
        env_id = register_dris_env("PushCube-v1", n_dris_copies=8)
        env = gym.make(env_id, num_envs=16)
    """
    # Get base config
    config = get_task_config(task_id)

    # Create DRIS env class
    DRISEnvClass = create_dris_env_class(task_id, n_dris_copies, pose_noise)

    # Determine env ID
    base_name = task_id.replace("-v1", "").replace("-v2", "")
    dris_env_id = f"DRIS{base_name}-v1"

    # Get max episode steps from base if not specified
    if max_episode_steps is None:
        BaseEnvClass = _import_class(config.base_class_path)
        # Try to get from class default
        max_episode_steps = getattr(BaseEnvClass, '_max_episode_steps', 100)

    # Register
    register_env(dris_env_id, max_episode_steps=max_episode_steps)(DRISEnvClass)

    return dris_env_id


def list_supported_tasks():
    """List all supported task IDs for DRIS conversion."""
    return list(TASK_CONFIGS.keys())
