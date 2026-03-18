"""
Concrete geometric cage implementations.

Provides CircularCage and Geometric3DCage following the base class specification with
controller-based parameter management and time-varying behavior support.
"""

from typing import Optional, Dict, Any, List, Union
import logging
import numpy as np
import gymnasium as gym
from ..base.cage import Cage
from ..base.dris import DRIS

logger = logging.getLogger(__name__)


class CircularCage(Cage):
    """
    Circular cage constraint for 2D state spaces with controller-based trajectory support.
    
    Supports time-varying behavior with orbital motion through CageController.
    """
    
    def __init__(self,
                 state_space: Any,
                 center: Optional[List[float]] = None,
                 radius: float = 1.0,
                 time_varying: bool = False,
                 orbit_radius: float = 0.0,
                 orbit_speed: float = 0.1):
        """
        Initialize circular cage with orbital motion support.
        
        Args:
            state_space: State space this cage operates in
            center: Center coordinates [x, y]
            radius: Cage radius
            time_varying: Whether cage moves over time
            orbit_radius: Radius of orbital motion
            orbit_speed: Speed of orbital motion
        """
        # Initialize trajectory parameters for controller
        trajectory_params = {
            'orbit_radius': orbit_radius,
            'orbit_speed': orbit_speed
        } if time_varying else {}
        
        super().__init__(state_space, time_varying, trajectory_params)
        
        # Initialize parameters using new architecture
        initial_center = np.array(center) if center is not None else np.array([0.0, 0.0])
        self.parameters = self._define_parameters()
        self.parameters['center'] = initial_center
        self.parameters['radius'] = radius
        
        # Store initial parameters for trajectory calculations
        self.initial_parameters = self.parameters.copy()
        
        # Direct access attributes
        self.center = self.parameters['center']
        self.radius = self.parameters['radius']
        self.initial_center = self.initial_parameters['center'].copy()
        self.initial_radius = self.initial_parameters['radius']

        # Time-varying parameters
        self.orbit_radius = orbit_radius
        self.orbit_speed = orbit_speed
    
    def _define_parameters(self) -> Dict[str, Any]:
        """Define parameter schema for CircularCage."""
        return {
            'center': np.array([0.0, 0.0]),
            'radius': 1.0
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
            
        # Update initial state attributes
        self.initial_center = self.initial_parameters['center'].copy()
        self.initial_radius = self.initial_parameters['radius']
    
    def reset(self) -> None:
        """Reset cage to initial parameters."""
        if self.initial_parameters:
            # Reset to initial parameters
            for key, value in self.initial_parameters.items():
                if isinstance(value, np.ndarray):
                    self.parameters[key] = value.copy()
                else:
                    self.parameters[key] = value
            self._update_from_parameters()
        
        # Reset controller if time-varying and reset method exists
        if hasattr(self, 'controller') and self.controller and hasattr(self.controller, 'reset'):
            self.controller.reset()
    
    def initialize(self) -> None:
        """Initialize cage after parameters are set."""
        if 'center' not in self.parameters:
            raise ValueError("Cage center not set. Call set_cage() first.")
        
        # Generate trajectory if time_varying
        if self.time_varying:
            self.controller.generate_trajectory(horizon=1000, initial_center=self.initial_parameters['center'])
        
        self.initialized = True

    def validate_state(self, dris: DRIS) -> bool:
        """Check if state is within circular cage."""
        if not self.initialized:
            self.initialize()
        
        # Extract 2D position from DRIS
        if hasattr(dris.observation, '__len__') and len(dris.observation) >= 2:
            position = dris.observation[:2]
        else:
            raise ValueError("DRIS observation must have at least 2 dimensions for 2D circular cage")
        
        # Calculate distance to center
        distance = np.linalg.norm(np.array(position) - self.center)
        return distance <= self.radius
    
    def get_boundary(self) -> Dict[str, Any]:
        """Get cage boundary representation."""
        return {
            'type': 'circular',
            'center': self.center.tolist(),
            'radius': self.radius,
            'orbit_radius': self.orbit_radius,
            'orbit_speed': self.orbit_speed
        }

    def distance_to_boundary_pixels(self, dris: DRIS) -> float:
        """
        Calculate distance to cage boundary for pixel observations.

        Args:
            dris: DRIS with image observation

        Returns:
            Distance to boundary (positive if inside, negative if outside)
        """
        if not isinstance(dris.observation, np.ndarray) or len(dris.observation.shape) != 3:
            return self.distance_to_boundary(dris)  # Fall back to original method

        frame = dris.observation

        # Get PSS center from image
        red_mask = (frame[:, :, 0] > 0.7) & (frame[:, :, 1] < 0.4) & (frame[:, :, 2] < 0.4)
        if np.sum(red_mask) > 0:
            y_coords, x_coords = np.nonzero(red_mask)
            pss_center = np.array([np.mean(x_coords), np.mean(y_coords)])
        else:
            return 0.0  # No PSS detected

        # Get cage center
        if self.time_varying and hasattr(self.controller, 'current_center'):
            cage_center = self.controller.current_center
        else:
            cage_center = np.array([22, 64])

        # Calculate distance
        distance = np.linalg.norm(pss_center - cage_center)
        return self.radius - distance  # Positive if inside, negative if outside

    def evaluate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[float]:
        """
        Evaluate DRIS state(s) using combined metric: distance + convex hull area.

        Cost function from GeometricOptimizer:
        cost = distance_weight * distance + area_weight * area

        Args:
            dris_input: Single DRIS or list of DRIS states

        Returns:
            List of costs (lower is better)
        """
        # Always treat input as a list
        dris_list = dris_input if isinstance(dris_input, list) else [dris_input]

        # Cost function weights (from GeometricOptimizer config defaults)
        distance_weight = 0.2
        area_weight = 0.8

        costs = []
        avg_positions = []
        distances = []
        areas = []

        for dris in dris_list:
            obs = dris.observation
            # Extract x, y positions and compute average
            if obs is not None and isinstance(obs, np.ndarray) and len(obs) >= 14:  # At least 2 objects
                # Reshape to (num_objects, 7) and get x, y positions
                num_objects = len(obs) // 7
                obj_states = obs[:num_objects * 7].reshape(num_objects, 7)
                obj_positions = obj_states[:, :2]  # x, y columns
                avg_pos = np.mean(obj_positions, axis=0)

                # Calculate convex hull area
                area = self._convex_hull_area(obj_positions)
            else:
                avg_pos = np.array([0, 0])
                area = 0.0

            # Calculate distance from average position to cage center
            distance = np.linalg.norm(avg_pos - self.center)

            # Combined cost: distance_weight * distance + area_weight * area
            cost = distance_weight * distance + area_weight * area

            avg_positions.append(avg_pos.tolist())
            distances.append(distance)
            areas.append(area)
            costs.append(cost)

        # Debug output (only when DEBUG level is enabled)
        if logger.isEnabledFor(logging.DEBUG):
            cage_center_formatted = [f"{x:.2g}" for x in self.center.tolist()]
            avg_positions_formatted = [[f"{x:.2g}" for x in pos] for pos in avg_positions]
            distances_formatted = [f"{d:.2g}" for d in distances]
            areas_formatted = [f"{a:.2g}" for a in areas]
            costs_formatted = [f"{c:.2g}" for c in costs]

            logger.debug(f"Cage center: {cage_center_formatted}")
            logger.debug(f"Average positions for {len(dris_list)} DRIS: {avg_positions_formatted}")
            logger.debug(f"Distances to cage: {distances_formatted}")
            logger.debug(f"Convex hull areas: {areas_formatted}")
            logger.debug(f"Combined costs (0.2*dist + 0.8*area): {costs_formatted}")

        return costs

    def validate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[bool]:
        """
        Validate DRIS state(s) using simplified criteria: check if within cage radius.

        From GeometricOptimizer: objects are valid if their average position is within cage radius.

        Args:
            dris_input: Single DRIS or list of DRIS states

        Returns:
            List of bools (True if valid - within cage)
        """
        # Compute distances to cage center
        distances = self._compute_distances(dris_input)

        # Validate: objects are valid if their average position is within cage radius
        results = [distance <= self.radius for distance in distances]

        return results

    def _compute_distances(self, dris_input: Union[DRIS, List[DRIS]]) -> List[float]:
        """
        Helper method to compute distances from average object positions to cage center.

        Args:
            dris_input: Single DRIS or list of DRIS states

        Returns:
            List of distances
        """
        dris_list = dris_input if isinstance(dris_input, list) else [dris_input]

        distances = []
        for dris in dris_list:
            obs = dris.observation
            if obs is not None and isinstance(obs, np.ndarray) and len(obs) >= 14:
                num_objects = len(obs) // 7
                obj_states = obs[:num_objects * 7].reshape(num_objects, 7)
                obj_positions = obj_states[:, :2]
                avg_pos = np.mean(obj_positions, axis=0)
            else:
                avg_pos = np.array([0, 0])

            distance = np.linalg.norm(avg_pos - self.center)
            distances.append(distance)

        return distances

    def _convex_hull_area(self, points: np.ndarray) -> float:
        """
        Calculate convex hull area for 2D points.

        Args:
            points: Array of 2D points (N x 2)

        Returns:
            Convex hull area
        """
        if len(points) < 3:
            return 0.0
        try:
            from scipy.spatial import ConvexHull, QhullError
            hull = ConvexHull(points)
            return hull.volume  # In 2D, volume attribute is actually area
        except (QhullError, ValueError):
            return 0.0

