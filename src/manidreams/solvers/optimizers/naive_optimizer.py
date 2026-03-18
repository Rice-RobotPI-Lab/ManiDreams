"""
Naive Planner for angle-based action selection.

Selects actions based on the angle between cage motion direction and
the vector from cage center to DRIS center.
"""

import logging
from typing import Any, List, Tuple, Optional
import numpy as np
import gymnasium as gym
from ...base.solver import SolverBase
from ...base.cage import Cage
from ...base.dris import DRIS
from ...base.tsip import TSIPBase

logger = logging.getLogger(__name__)


class NaiveOptimizer(SolverBase):
    """
    Naive planner that selects actions based on angle between:
    - Cage motion direction (from trajectory)
    - Vector from cage center to DRIS center

    Action 0 corresponds to cage trajectory direction (with optional offset).
    Actions 1-15 are uniformly distributed around action 0.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize NaiveOptimizer.

        Args:
            config: Configuration dict with optional parameters:
                - angle_offset: Offset angle for action 0 in radians (default: 0.0)
        """
        self.config = config or {}
        self.angle_offset = self.config.get('angle_offset', 1.57)

    def action_generator(self, action_space: gym.Space) -> List[Any]:
        """
        Generate all actions from action space.

        Args:
            action_space: Discrete action space

        Returns:
            List of action indices
        """
        if isinstance(action_space, gym.spaces.Discrete):
            return list(range(action_space.n))
        else:
            raise ValueError("NaiveOptimizer only supports Discrete action space")

    def solve(self,
              action_space: gym.Space,
              cage: Cage,
              tsip: TSIPBase,
              current_dris: DRIS,
              verbose: bool = False) -> Tuple[int, None, None]:
        """
        Select best action based on angle relative to cage motion direction.

        NaiveOptimizer uses geometric angle-based selection and does not compute
        costs or validations.

        Args:
            action_space: Discrete action space
            cage: Cage constraint with trajectory
            tsip: TSIP instance (not used in naive planner)
            current_dris: Current DRIS state
            verbose: Whether to print debug information

        Returns:
            Tuple of (best_action, None, None)
            - best_action: Selected action index (0-15)
            - costs: None (not computed by this planner)
            - validations: None (not computed by this planner)
        """
        # 1. Get cage motion direction (3D -> project to 2D for action selection)
        if not hasattr(cage, 'get_trajectory_direction'):
            if verbose:
                logger.info("Cage has no trajectory, returning action 0")
            return 0, None, None

        cage_direction_3d = cage.get_trajectory_direction(cage.current_timestep)
        cage_direction_2d = cage_direction_3d[:2]  # [x, y] projection

        if np.linalg.norm(cage_direction_2d) < 1e-6:
            # No motion, return action 0
            if verbose:
                logger.info("No cage motion, returning action 0")
            return 0, None, None

        # Calculate cage angle (direction of action 0)
        cage_angle = np.arctan2(cage_direction_2d[1], cage_direction_2d[0])
        action_0_angle = cage_angle + self.angle_offset

        # 2. Get DRIS center position (average of object positions)
        dris_center = self._get_dris_center(current_dris)

        # 3. Calculate vector from cage to DRIS (2D projection)
        cage_center_2d = cage.center[:2]
        cage_center_2d = [cage_center_2d[1], cage_center_2d[0]]
        dris_center_2d = dris_center[:2]
        logger.debug(f"Cage center: {cage_center_2d}, DRIS center: {dris_center_2d}")
        v_to_dris = dris_center_2d - cage_center_2d

        if np.linalg.norm(v_to_dris) < 1e-6:
            # DRIS at cage center, return action 0
            if verbose:
                logger.info("DRIS at cage center, returning action 0")
            return 0, None, None

        # 4. Calculate angle of v_to_dris
        dris_angle = np.arctan2(v_to_dris[1], v_to_dris[0])

        

        # 6. Map to action index (0-15)
        # Action 0 = 0°, Action 1 = 22.5°, ..., Action 15 = 337.5°
        num_actions = action_space.n if isinstance(action_space, gym.spaces.Discrete) else 16
        action_index = int(round((dris_angle / (2 * np.pi)) * num_actions)) % num_actions

        if verbose:
            logger.info(f"Cage direction: {cage_direction_2d}, angle: {np.degrees(cage_angle):.1f}°")
            logger.info(f"Action 0 angle (with offset): {np.degrees(action_0_angle):.1f}°")
            logger.info(f"DRIS direction: {v_to_dris}, angle: {np.degrees(dris_angle):.1f}°")
            logger.info(f"Selected action: {action_index}")

        return action_index, None, None

    def _get_dris_center(self, dris: DRIS) -> np.ndarray:
        """
        Extract center position from DRIS observation.

        Args:
            dris: DRIS state

        Returns:
            3D center position [x, y, z]
        """
        # Convert torch tensor to numpy if needed
        obs = dris.observation
        if hasattr(obs, 'cpu'):
            # It's a torch tensor
            obs = obs.cpu().numpy()

        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 1:
                # Single object or flattened: take first 3 dimensions
                return obs[:3]
            elif len(obs.shape) == 2:
                # Multiple objects: take average position
                positions = obs[:, :3]  # First 3 dims = [x, y, z]
                return np.mean(positions, axis=0)
            else:
                raise ValueError(f"Unexpected DRIS observation shape: {obs.shape}")
        else:
            raise ValueError(f"DRIS observation must be numpy array or torch tensor, got {type(obs)}")
