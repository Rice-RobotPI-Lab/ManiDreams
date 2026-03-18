"""
Geometric 3D cage implementation.

Provides Geometric3DCage for 3D state spaces with trajectory-based motion.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import numpy as np
import gymnasium as gym
from ..base.cage import Cage
from ..base.dris import DRIS
from .utils import evaluate_distance_to_center, evaluate_position_variance, validate_within_radius

logger = logging.getLogger(__name__)


class Geometric3DCage(Cage):
    """
    3D geometric cage with trajectory-based motion.

    Generic cage for 3D state spaces. Supports spherical boundary constraints
    and time-varying behavior through externally-provided trajectory.
    """

    def __init__(self,
                 state_space: gym.Space,
                 trajectory: List[List[float]],
                 radius: float = 0.15,
                 time_varying: bool = True,
                 cost_type: str = 'distance'):
        """
        Initialize 3D geometric cage with external trajectory.

        Args:
            state_space: 3D state space
            trajectory: Pre-computed trajectory as list of [x, y, z] or [x, y, z, roll, pitch, yaw]
            radius: Cage radius (spherical boundary)
            time_varying: Whether cage moves along trajectory
            cost_type: Cost function type ('distance' or 'variance')
        """
        self.cost_type = cost_type
        # Initialize with trajectory parameters
        trajectory_params = {
            'trajectory': trajectory
        } if time_varying else {}

        super().__init__(state_space, time_varying, trajectory_params)

        # Validate trajectory
        if not trajectory or len(trajectory) == 0:
            raise ValueError("Trajectory must be non-empty")

        # Store trajectory - supports both 3D [x,y,z] and 6D [x,y,z,roll,pitch,yaw]
        self.trajectory = [np.array(pose) for pose in trajectory]

        # Detect trajectory dimension (3D position-only or 6D pose)
        first_pose = self.trajectory[0]
        if len(first_pose) == 6:
            self.has_orientation = True
            logger.info("Using 6D trajectory (position + orientation)")
        elif len(first_pose) == 3:
            self.has_orientation = False
            logger.info("Using 3D trajectory (position only)")
        else:
            raise ValueError(f"Trajectory must have 3 (position) or 6 (pose) dimensions, got {len(first_pose)}")

        # Initialize parameters
        self.parameters = self._define_parameters()
        self.parameters['center'] = self.trajectory[0][:3].copy()  # Always extract position
        self.parameters['radius'] = radius

        # Add orientation parameter if trajectory has it
        if self.has_orientation:
            self.parameters['orientation'] = self.trajectory[0][3:6].copy()  # [roll, pitch, yaw]

        # Store initial parameters
        self.initial_parameters = self.parameters.copy()

        # Direct access attributes
        self.center = self.parameters['center']
        self.radius = self.parameters['radius']
        self.initial_center = self.initial_parameters['center'].copy()
        self.initial_radius = self.initial_parameters['radius']

        # Orientation attributes (if available)
        if self.has_orientation:
            self.orientation = self.parameters['orientation']
            self.initial_orientation = self.initial_parameters['orientation'].copy()
        else:
            self.orientation = None
            self.initial_orientation = None

        # Trajectory tracking
        self.current_timestep = 0

    def _define_parameters(self) -> Dict[str, Any]:
        """Define parameter schema for Geometric3DCage."""
        return {
            'center': np.array([0.0, 0.0, 0.0]),
            'radius': 0.15
        }

    def _update_from_parameters(self) -> None:
        """Update internal cage representation from parameters."""
        # Ensure radius is non-negative
        if 'radius' in self.parameters:
            self.parameters['radius'] = max(0.0, self.parameters['radius'])

        # Update direct access attributes
        self.center = self.parameters['center']
        self.radius = self.parameters['radius']

        # Update region representation
        self.region = {
            'center': self.center,
            'radius': self.radius
        }

    def set_cage(self, region: Dict[str, Any]) -> None:
        """Set cage parameters from region dict."""
        updates = {}
        if 'center' in region:
            updates['center'] = np.array(region['center'])
            self.initial_parameters['center'] = np.array(region['center'])
        if 'radius' in region:
            updates['radius'] = region['radius']
            self.initial_parameters['radius'] = region['radius']

        if updates:
            self.update(**updates)

    def reset(self) -> None:
        """Reset cage to initial parameters and trajectory start."""
        if self.initial_parameters:
            # Reset to initial parameters
            for key, value in self.initial_parameters.items():
                if isinstance(value, np.ndarray):
                    self.parameters[key] = value.copy()
                else:
                    self.parameters[key] = value
            self._update_from_parameters()

        # Reset timestep tracker
        self.current_timestep = 0

    def initialize(self) -> None:
        """Initialize cage after parameters are set."""
        if 'center' not in self.parameters:
            raise ValueError("Cage center not set. Call set_cage() first or provide trajectory.")

        # Validate trajectory exists for time-varying cages
        if self.time_varying and (not self.trajectory or len(self.trajectory) == 0):
            raise ValueError("Time-varying cage requires non-empty trajectory")

        self.initialized = True

    def _generate_trajectory(self) -> List[np.ndarray]:
        """
        Generate cage trajectory from controller parameters.

        For Geometric3DCage, trajectory is provided externally.
        This method is called by the controller during initialization.
        """
        # Return the pre-computed trajectory
        return self.trajectory

    def apply_controller_updates(self, timestep: int) -> None:
        """
        Update cage position (and orientation if available) based on trajectory at given timestep.

        Args:
            timestep: Current timestep index
        """
        if not self.time_varying or not hasattr(self, 'controller'):
            return

        # Update current timestep
        self.current_timestep = timestep

        # Get pose from trajectory (clamp to valid range)
        traj_idx = min(timestep, len(self.trajectory) - 1)
        current_pose = self.trajectory[traj_idx]

        # Update cage center (always present)
        new_center = current_pose[:3].copy()
        updates = {'center': new_center}

        # Update orientation if trajectory has it
        if self.has_orientation:
            new_orientation = current_pose[3:6].copy()
            updates['orientation'] = new_orientation

        # Apply all updates
        self.update(**updates)

    def get_trajectory_direction(self, timestep: int) -> np.ndarray:
        """
        Get normalized direction vector from timestep t to t+1.

        Args:
            timestep: Current timestep index

        Returns:
            Normalized 3D direction vector [x, y, z] (only position component)
        """
        if timestep >= len(self.trajectory) - 1:
            # No next position, return zero vector
            return np.array([0.0, 0.0, 0.0])

        # Extract position components only (first 3 elements)
        current_pos = self.trajectory[timestep][:3]
        next_pos = self.trajectory[timestep + 1][:3]
        direction = next_pos - current_pos

        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            # Nearly zero movement
            return np.array([0.0, 0.0, 0.0])

        return direction / norm

    def distance_to_boundary(self, dris: DRIS) -> float:
        """
        Calculate signed distance to spherical cage boundary.

        Args:
            dris: DRIS containing state observation

        Returns:
            Positive if inside cage, negative if outside
        """
        # Extract 3D position from DRIS
        if isinstance(dris.observation, np.ndarray):
            if len(dris.observation.shape) == 1 and dris.observation.shape[0] >= 3:
                position = dris.observation[:3]
            elif len(dris.observation.shape) == 2 and dris.observation.shape[1] >= 3:
                # Multiple objects: use center of mass
                position = np.mean(dris.observation[:, :3], axis=0)
            else:
                raise ValueError("DRIS observation must have at least 3 dimensions for 3D cage")
        else:
            raise ValueError("DRIS observation must be numpy array")

        # Calculate distance to center
        distance = np.linalg.norm(np.array(position) - self.center)
        return self.radius - distance

    def validate_state(self, dris: DRIS) -> bool:
        """
        Check if state is within spherical cage boundary.

        Args:
            dris: DRIS containing state observation

        Returns:
            True if state is inside cage
        """
        # Extract 3D position from DRIS
        if isinstance(dris.observation, np.ndarray):
            if len(dris.observation.shape) == 1 and dris.observation.shape[0] >= 3:
                position = dris.observation[:3]
            elif len(dris.observation.shape) == 2 and dris.observation.shape[1] >= 3:
                # Multiple objects: use center of mass
                position = np.mean(dris.observation[:, :3], axis=0)
            else:
                raise ValueError("DRIS observation must have at least 3 dimensions for 3D cage")
        else:
            raise ValueError("DRIS observation must be numpy array")

        # Calculate distance to center
        distance = np.linalg.norm(np.array(position) - self.center)
        return distance <= self.radius

    def get_boundary(self) -> Dict[str, Any]:
        """Get cage boundary representation."""
        return {
            'type': 'spherical_3d',
            'center': self.center.tolist(),
            'radius': self.radius,
            'trajectory_length': len(self.trajectory)
        }

    def evaluate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[float]:
        """
        Evaluate 3D DRIS state(s) using configured cost function.

        Args:
            dris_input: Single DRIS or list of DRIS states

        Returns:
            List of costs (lower is better)
        """
        if self.cost_type == 'variance':
            return evaluate_position_variance(dris_input, self.center)
        return evaluate_distance_to_center(dris_input, self.center)

    def validate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[bool]:
        """
        Validate 3D DRIS state(s): check if within spherical cage.

        Args:
            dris_input: Single DRIS or list of DRIS states

        Returns:
            List of bools (True if valid - within cage radius)
        """
        return validate_within_radius(dris_input, self.center, self.radius)
