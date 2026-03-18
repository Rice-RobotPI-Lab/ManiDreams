"""
DRIS Backend for ManiDreams Framework

Provides SimulationBackend interface for DRIS-enabled ManiSkill environments.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import gymnasium as gym

from manidreams.physics.simulation_tsip import SimulationBackend
from manidreams.base.dris import DRIS

from .task_config import get_task_config, list_supported_tasks
from .dris_env_factory import make_dris_env


class DRISBackend(SimulationBackend):
    """
    Backend that provides DRIS-enabled ManiSkill environments.

    This backend:
    1. Creates DRIS copies of the target object
    2. Randomizes copy poses based on pose_noise (position + orientation)
    3. Computes reward as mean of per-copy rewards
    4. Provides DRIS variance for uncertainty estimation

    Usage:
        backend = DRISBackend(
            task_id="PushCube-v1",
            n_dris_copies=8,
            pose_noise=(0.05, 0.05, 0.0, 0.0, 0.0, 0.1),
        )

        # As TSIP backend
        tsip = SimulationBasedTSIP(backend=backend, env_config={...})

        # Or directly use environment
        env = backend.create_environment({'num_envs': 16})
    """

    def __init__(
        self,
        task_id: str,
        n_dris_copies: int = 8,
        pose_noise: Tuple[float, float, float, float, float, float] = (0.01, 0.01, 0.0, 0.0, 0.0, 0.0),
        physics_noise: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Initialize DRIS Backend.

        Args:
            task_id: ManiSkill task ID (e.g., "PushCube-v1")
            n_dris_copies: Number of DRIS copies to create
            pose_noise: (dx, dy, dz, droll, dpitch, dyaw) - Random pose offset range
            physics_noise: (dfric, dmass_ratio) - Friction delta and mass ratio noise
        """
        self.task_id = task_id
        self.n_dris_copies = n_dris_copies
        self.pose_noise = pose_noise
        self.physics_noise = physics_noise
        self.task_config = get_task_config(task_id)

        # Will be set when environment is created
        self._env: Optional[gym.Env] = None
        self.device = None

    def create_environment(self, env_config: Dict[str, Any]) -> gym.Env:
        """
        Create DRIS-enabled environment.

        Args:
            env_config: Environment configuration
                - num_envs: Number of parallel environments
                - render_mode: "human", "rgb_array", or None
                - obs_mode: "state_dict", "rgbd", etc.
                - control_mode: "pd_ee_delta_pose", etc.

        Returns:
            DRIS-enabled ManiSkill environment
        """
        # Create DRIS environment
        env = make_dris_env(
            self.task_id,
            n_dris_copies=self.n_dris_copies,
            pose_noise=self.pose_noise,
            physics_noise=self.physics_noise,
            **env_config
        )

        self._env = env
        self.device = env.unwrapped.device if hasattr(env.unwrapped, 'device') else 'cuda:0'

        return env

    def reset_environment(
        self,
        env: gym.Env,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """Reset environment."""
        obs, info = env.reset(seed=seed, options=options)

        # Render if GUI mode
        if hasattr(env, 'render_mode') and env.render_mode == 'human':
            env.render()

        return obs, info

    def get_state(self, env: gym.Env) -> Dict[str, np.ndarray]:
        """
        Get full state from DRIS environment (robot + object + DRIS copies).

        Returns:
            Dictionary containing:
            - qpos: [num_envs, dof] robot joint positions
            - qvel: [num_envs, dof] robot joint velocities
            - target_pose: [num_envs, 7] target object pose (p + q)
            - dris_poses: [num_envs, n_copies, 7]
            - variance: [num_envs, 3]
            - mean_position: [num_envs, 3]
        """
        actual_env = env.unwrapped

        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                return t.cpu().numpy()
            return np.array(t)

        # Robot state
        robot = actual_env.agent.robot
        qpos = robot.get_qpos()
        qvel = robot.get_qvel()

        # Target object pose
        target = getattr(actual_env, self.task_config.target_attr)
        target_pose = target.pose.raw_pose

        # Goal marker pose
        goal_obj = getattr(actual_env, self.task_config.goal_attr)
        goal_pose = goal_obj.pose.raw_pose

        # DRIS copies
        dris_poses = actual_env._get_dris_poses()
        variance = actual_env._compute_dris_variance()
        mean_pos = actual_env._compute_dris_mean_position()

        return {
            'qpos': to_numpy(qpos),
            'qvel': to_numpy(qvel),
            'target_pose': to_numpy(target_pose),
            'goal_pose': to_numpy(goal_pose),
            'dris_poses': to_numpy(dris_poses),
            'variance': to_numpy(variance),
            'mean_position': to_numpy(mean_pos),
        }

    def set_state(self, env: gym.Env, state: Dict[str, np.ndarray]) -> None:
        """
        Set full state to DRIS environment (robot + object + re-randomize DRIS).

        Handles broadcasting: if state has fewer envs than the target env
        (e.g., 1 executor → N eval envs), broadcasts to all envs.

        Args:
            state: Dictionary with any combination of:
                - qpos, qvel: robot joint state
                - target_pose: target object pose
                - goal_pose: goal object pose (visual marker)
                - dris_poses: explicit DRIS poses (skips re-randomization)
        """
        actual_env = env.unwrapped
        num_envs = actual_env.num_envs

        def to_tensor(arr):
            t = torch.from_numpy(np.asarray(arr)).to(self.device)
            return t.float() if t.dtype != torch.float32 else t

        def broadcast(t):
            """Broadcast [1, ...] → [num_envs, ...]"""
            if t.shape[0] == 1 and num_envs > 1:
                return t.expand(num_envs, *t.shape[1:]).contiguous()
            return t

        from mani_skill.utils.structs import Pose

        # 1. Robot state
        if 'qpos' in state:
            actual_env.agent.robot.set_qpos(broadcast(to_tensor(state['qpos'])))
        if 'qvel' in state:
            actual_env.agent.robot.set_qvel(broadcast(to_tensor(state['qvel'])))

        # 2. Target object pose
        if 'target_pose' in state:
            target = getattr(actual_env, self.task_config.target_attr)
            target_pose = broadcast(to_tensor(state['target_pose']))

            target.set_pose(Pose.create_from_pq(
                p=target_pose[..., :3],
                q=target_pose[..., 3:7]
            ))

        # 3. Goal marker pose (sync across all parallel envs)
        if 'goal_pose' in state:
            goal_obj = getattr(actual_env, self.task_config.goal_attr)
            goal_pose = broadcast(to_tensor(state['goal_pose']))
            goal_obj.set_pose(Pose.create_from_pq(
                p=goal_pose[..., :3],
                q=goal_pose[..., 3:7]
            ))

        # GPU sync: flush pose writes so subsequent reads see updated state
        if self.device.type == "cuda":
            actual_env.scene._gpu_apply_all()
            actual_env.scene.px.gpu_update_articulation_kinematics()
            actual_env.scene._gpu_fetch_all()

        # 3. DRIS copies: explicit poses or re-randomize around target
        if 'dris_poses' in state:
            actual_env._sync_dris_to_pose(broadcast(to_tensor(state['dris_poses'])))
        elif 'target_pose' in state:
            # Re-randomize DRIS copies around the new target position
            actual_env._randomize_dris_poses(
                torch.arange(num_envs, device=self.device)
            )

        # GPU sync: flush DRIS pose writes for rendering and subsequent reads
        if self.device.type == "cuda":
            actual_env.scene._gpu_apply_all()
            actual_env.scene.px.gpu_update_articulation_kinematics()
            actual_env.scene._gpu_fetch_all()

    def dris2state(self, dris: DRIS) -> Dict[str, np.ndarray]:
        """
        Convert DRIS back to state dictionary for set_state().

        Extracts robot state + target pose from DRIS context so that
        tsip.next() can sync evaluation environments before stepping.
        """
        if not hasattr(dris, 'context') or not dris.context:
            return {}

        ctx = dris.context
        state = {}

        for key in ('qpos', 'qvel', 'target_pose', 'goal_pose', 'dris_poses'):
            if key in ctx:
                val = ctx[key]
                if hasattr(val, 'cpu'):
                    val = val.cpu().numpy()
                arr = np.asarray(val)
                # state2dris extracts per-env data (no batch dim).
                # set_state()'s broadcast() needs shape[0]==1 to expand to
                # [num_envs, ...], so add a leading batch dim for all keys.
                if key == 'dris_poses' and arr.ndim == 2:
                    arr = arr[np.newaxis]  # [n_copies, 7] → [1, n_copies, 7]
                elif key != 'dris_poses' and arr.ndim == 1:
                    arr = arr[np.newaxis]  # [dim] → [1, dim]
                state[key] = arr

        # dris_poses is included so that within a chunk plan the best env's
        # DRIS copies are carried forward (continuous evolution) rather than
        # re-randomized at every planning step.  Initial randomization still
        # happens at need_replan time via get_executor_state() which does not
        # contain dris_poses, triggering _randomize_dris_poses() in set_state().
        return state

    def state2dris(
        self,
        observations: Any,
        env_indices: Optional[List[int]] = None,
        env_config: Optional[Dict] = None
    ) -> List[DRIS]:
        """
        Read current environment state and convert to DRIS objects.

        Each DRIS carries the full state (robot + object + DRIS info) so it
        can be round-tripped through dris2state() → set_state().

        Args:
            observations: Unused (state is read directly from env)
            env_indices: Optional list of environment indices
            env_config: Environment configuration

        Returns:
            List of DRIS objects (one per environment)
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized")

        state = self.get_state(self._env)
        num_envs = state['dris_poses'].shape[0]

        if env_indices is None:
            env_indices = list(range(num_envs))

        dris_list = []
        for env_idx in env_indices:
            mean_pos = state['mean_position'][env_idx]

            dris = DRIS(
                observation=mean_pos.flatten(),
                representation_type="state",
                context={
                    'qpos': state['qpos'][env_idx],
                    'qvel': state['qvel'][env_idx],
                    'target_pose': state['target_pose'][env_idx],
                    'goal_pose': state['goal_pose'][env_idx],
                    'dris_poses': state['dris_poses'][env_idx],
                    'variance': state['variance'][env_idx],
                    'mean_position': mean_pos,
                }
            )
            dris_list.append(dris)

        return dris_list

    def step_act(
        self,
        actions: Union[Any, List[Any]],
        env: gym.Env = None,
        cage: Any = None,
        single_action: bool = False
    ) -> Any:
        """
        Execute actions on DRIS environment.

        Args:
            actions: Action(s) to execute
            env: Environment (uses self._env if None)
            cage: Optional cage object (unused)
            single_action: Whether this is a single action

        Returns:
            Observation from environment
        """
        if env is None:
            env = self._env

        # Handle action format
        if isinstance(actions, list):
            if len(actions) == 1:
                action = actions[0]
            else:
                action = np.stack(actions, axis=0)
        else:
            action = actions

        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)

        # Render if needed
        if single_action and hasattr(env, 'render_mode') and env.render_mode == 'human':
            try:
                env.render()
            except Exception:
                pass

        return obs

    def get_action_space(self, env: gym.Env) -> gym.Space:
        """Get action space from environment."""
        return env.action_space

    def get_observation_space(self, env: gym.Env) -> gym.Space:
        """Get observation space from environment."""
        return env.observation_space

    # ========== DRIS-Specific Methods ==========

    def compute_dris_variance(self, env: gym.Env = None) -> np.ndarray:
        """
        Compute position variance across DRIS copies.

        Returns:
            [num_envs, 3] variance in x, y, z
        """
        if env is None:
            env = self._env

        variance = env.unwrapped._compute_dris_variance()

        if isinstance(variance, torch.Tensor):
            return variance.cpu().numpy()
        return np.array(variance)

    def sync_dris_copies(
        self,
        env: gym.Env,
        source_pose: np.ndarray
    ) -> None:
        """
        Sync all DRIS copies to a specific pose.

        Args:
            env: Environment
            source_pose: Pose to sync to [num_envs, 7] or [num_envs, n_copies, 7]
        """
        actual_env = env.unwrapped
        pose_tensor = torch.from_numpy(source_pose).to(self.device)
        actual_env._sync_dris_to_pose(pose_tensor)

    def get_dris_positions(self, env: gym.Env = None) -> np.ndarray:
        """
        Get positions of all DRIS copies.

        Returns:
            [num_envs, n_copies, 3]
        """
        if env is None:
            env = self._env

        positions = env.unwrapped._get_dris_positions()

        if isinstance(positions, torch.Tensor):
            return positions.cpu().numpy()
        return np.array(positions)

    # ========== Required SimulationBackend Methods ==========

    def load_env(self, context: Dict[str, Any]) -> None:
        """Load environment configuration."""
        pass

    def load_object(self, context: Dict[str, Any]) -> None:
        """Load object configuration."""
        pass

    def load_robot(self, context: Dict[str, Any]) -> None:
        """Load robot configuration."""
        pass

    def close_environment(self, env: gym.Env) -> None:
        """Close environment."""
        if env is not None:
            env.close()
        self._env = None

    @staticmethod
    def list_supported_tasks() -> List[str]:
        """List all supported task IDs."""
        return list_supported_tasks()
