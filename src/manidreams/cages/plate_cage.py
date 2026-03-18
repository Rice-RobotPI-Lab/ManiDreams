"""
Plate Cage implementation for catching tasks.

Provides PlateCage constraint for evaluating ball catching performance,
measuring how well balls are positioned relative to the catching plate.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import numpy as np
import gymnasium as gym
from ..base.cage import Cage
from ..base.dris import DRIS

logger = logging.getLogger(__name__)


class PlateCage(Cage):
    """
    Plate cage constraint for catching tasks.

    Evaluates ball states relative to a catching plate:
    - Distance from ball to plate center (horizontal projection)
    - Ball velocity (stability metric)
    - Whether ball is above the plate surface

    DRIS observation format expected:
    [obj_pos(3), obj_vel(3), tcp_pos(3), tcp_quat(4)] = 13 dimensions
    """

    def __init__(self,
                 state_space: Any = None,
                 plate_radius: float = 0.12,
                 dist_threshold: float = 0.1,
                 vel_threshold: float = 0.2,
                 dist_weight: float = 0.7,
                 vel_weight: float = 0.3):
        """
        Initialize plate cage.

        Args:
            state_space: State space (optional, not used for plate cage)
            plate_radius: Radius of the catching plate
            dist_threshold: Distance threshold for validation
            vel_threshold: Velocity threshold for validation
            dist_weight: Weight for distance in cost function
            vel_weight: Weight for velocity in cost function
        """
        if state_space is None:
            state_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
            )

        super().__init__(state_space, time_varying=False)

        # Initialize parameters
        self.parameters = self._define_parameters()
        self.parameters['plate_radius'] = plate_radius
        self.parameters['dist_threshold'] = dist_threshold
        self.parameters['vel_threshold'] = vel_threshold
        self.parameters['dist_weight'] = dist_weight
        self.parameters['vel_weight'] = vel_weight

        # Direct access attributes
        self.plate_radius = plate_radius
        self.dist_threshold = dist_threshold
        self.vel_threshold = vel_threshold
        self.dist_weight = dist_weight
        self.vel_weight = vel_weight

        self.initialized = True

    def _define_parameters(self) -> Dict[str, Any]:
        """Define parameter schema for PlateCage."""
        return {
            'plate_radius': 0.12,
            'dist_threshold': 0.1,
            'vel_threshold': 0.2,
            'dist_weight': 0.7,
            'vel_weight': 0.3
        }

    def _update_from_parameters(self) -> None:
        """Update internal state from parameters."""
        self.plate_radius = self.parameters['plate_radius']
        self.dist_threshold = self.parameters['dist_threshold']
        self.vel_threshold = self.parameters['vel_threshold']
        self.dist_weight = self.parameters['dist_weight']
        self.vel_weight = self.parameters['vel_weight']

    def set_cage(self, region: Dict[str, Any]) -> None:
        """Set cage parameters from region dict."""
        updates = {}
        for key in ['plate_radius', 'dist_threshold', 'vel_threshold',
                    'dist_weight', 'vel_weight']:
            if key in region:
                updates[key] = region[key]
        if updates:
            self.update(**updates)

    def initialize(self) -> None:
        """Initialize cage (no-op for plate cage)."""
        self.initialized = True

    def _extract_state(self, dris: DRIS) -> Dict[str, np.ndarray]:
        """
        Extract ball and plate state from DRIS observation.

        Expected observation format:
        [obj_pos(3), obj_vel(3), tcp_pos(3), tcp_quat(4)] = 13 dimensions

        Or context-based format with separate fields.
        """
        obs = dris.observation

        # Check if observation has context with separate fields
        if hasattr(dris, 'context') and dris.context:
            ctx = dris.context
            if all(k in ctx for k in ['obj_pos', 'obj_vel', 'tcp_pos', 'tcp_quat']):
                return {
                    'obj_pos': np.array(ctx['obj_pos']),
                    'obj_vel': np.array(ctx['obj_vel']),
                    'tcp_pos': np.array(ctx['tcp_pos']),
                    'tcp_quat': np.array(ctx['tcp_quat'])
                }

        # Parse flat observation array
        if isinstance(obs, np.ndarray) and len(obs) >= 13:
            return {
                'obj_pos': obs[0:3],
                'obj_vel': obs[3:6],
                'tcp_pos': obs[6:9],
                'tcp_quat': obs[9:13]
            }

        # Fallback: return zeros
        return {
            'obj_pos': np.zeros(3),
            'obj_vel': np.zeros(3),
            'tcp_pos': np.zeros(3),
            'tcp_quat': np.array([1.0, 0.0, 0.0, 0.0])
        }

    def _quaternion_to_z_axis(self, quat: np.ndarray) -> np.ndarray:
        """
        Extract Z-axis (plate normal) from quaternion.

        Args:
            quat: Quaternion [w, x, y, z] or [x, y, z, w]

        Returns:
            Z-axis unit vector (plate normal pointing up)
        """
        # Assume quaternion format [w, x, y, z]
        if len(quat) != 4:
            return np.array([0.0, 0.0, 1.0])

        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Z-axis from rotation matrix (third column)
        z_axis = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ])

        # Normalize
        norm = np.linalg.norm(z_axis)
        if norm > 1e-6:
            z_axis = z_axis / norm
        else:
            z_axis = np.array([0.0, 0.0, 1.0])

        return z_axis

    def _compute_plate_metrics(self, state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute ball-to-plate metrics.

        Args:
            state: Dictionary with obj_pos, obj_vel, tcp_pos, tcp_quat

        Returns:
            Dictionary with:
            - u: horizontal distance to plate center
            - w: vertical distance (signed, positive = above plate)
            - vel_norm: ball velocity magnitude
        """
        obj_pos = state['obj_pos']
        obj_vel = state['obj_vel']
        tcp_pos = state['tcp_pos']
        tcp_quat = state['tcp_quat']

        # Get plate normal (Z-axis of TCP frame)
        tcp_normal = self._quaternion_to_z_axis(tcp_quat)

        # Vector from TCP to ball
        diff = obj_pos - tcp_pos

        # Vertical distance (projection onto normal)
        w = np.dot(diff, tcp_normal)

        # Horizontal distance (distance in plane perpendicular to normal)
        horizontal_component = diff - w * tcp_normal
        u = np.linalg.norm(horizontal_component)

        # Velocity magnitude
        vel_norm = np.linalg.norm(obj_vel)

        return {
            'u': u,  # horizontal distance
            'w': w,  # vertical distance (positive = above)
            'vel_norm': vel_norm
        }

    def evaluate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[float]:
        """
        Evaluate DRIS state(s) for catching quality.

        Cost = dist_weight * horizontal_distance + vel_weight * velocity

        Args:
            dris_input: Single DRIS or list of DRIS states

        Returns:
            List of costs (lower is better)
        """
        dris_list = dris_input if isinstance(dris_input, list) else [dris_input]

        costs = []
        for dris in dris_list:
            state = self._extract_state(dris)
            metrics = self._compute_plate_metrics(state)

            # Cost: weighted sum of horizontal distance and velocity
            cost = (self.dist_weight * metrics['u'] +
                    self.vel_weight * metrics['vel_norm'])
            costs.append(cost)

        # Debug output
        if len(dris_list) > 1 and logger.isEnabledFor(logging.DEBUG):
            costs_formatted = [f"{c:.3f}" for c in costs]
            logger.debug(f"Costs for {len(dris_list)} states: {costs_formatted}")

        return costs

    def validate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[bool]:
        """
        Validate DRIS state(s) against plate constraints.

        Valid if:
        - Ball is above the plate (w > 0)
        - Ball is within distance threshold of plate center (u < dist_threshold)

        Args:
            dris_input: Single DRIS or list of DRIS states

        Returns:
            List of bools (True if valid)
        """
        dris_list = dris_input if isinstance(dris_input, list) else [dris_input]

        results = []
        for dris in dris_list:
            state = self._extract_state(dris)
            metrics = self._compute_plate_metrics(state)

            # Valid if ball is above plate and within horizontal threshold
            above_plate = metrics['w'] > -0.05  # Small tolerance
            within_radius = metrics['u'] <= self.dist_threshold

            valid = above_plate and within_radius
            results.append(valid)

        return results

    def validate_state(self, dris: DRIS) -> bool:
        """Single state validation (convenience method)."""
        return self.validate(dris)[0]

    def get_boundary(self) -> Dict[str, Any]:
        """Get cage boundary representation."""
        return {
            'type': 'plate',
            'plate_radius': self.plate_radius,
            'dist_threshold': self.dist_threshold,
            'vel_threshold': self.vel_threshold
        }
