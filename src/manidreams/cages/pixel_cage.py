"""
Pixel-based Circular Cage for Image Observations

Replicates CageController from Diamond's play_cage16.py (lines 81-185).
"""

import logging
import numpy as np
from typing import Optional, List, Union
from ..base.cage import Cage
from ..base.dris import DRIS

logger = logging.getLogger(__name__)


class CircularPixelCage(Cage):
    """
    Circular cage operating on pixel coordinates.

    1:1 replication of Diamond's CageController class.
    """

    def __init__(self,
                 radius: float = 22,
                 orbit_radius: float = 42,
                 orbit_speed: float = 0.013,
                 time_varying: bool = True):
        """
        Initialize pixel-based circular cage (play_cage16.py:82-90).

        Args:
            radius: Cage radius in pixels (default: 22)
            orbit_radius: Orbital motion radius in pixels (default: 42)
            orbit_speed: Angular velocity for orbit (default: 0.013)
            time_varying: Enable orbital trajectory
        """
        super().__init__(state_space=None, time_varying=time_varying)

        self.radius = radius  # cage_radius
        self.orbit_radius = orbit_radius  # cage_orbit_radius
        self.orbit_speed = orbit_speed  # cage_orbit_speed
        self.cage_center = None
        self.initial_center = None
        self.time = 0
        self.action_duration = 4

        logger.debug(f"Initialized with: radius={radius}, orbit_radius={orbit_radius}, orbit_speed={orbit_speed}")

        # For backward compatibility
        self.center = self.cage_center
        self.parameters = self._define_parameters()

    def _define_parameters(self):
        """Define parameter schema for circular pixel cage"""
        return {
            'center': self.cage_center,
            'radius': self.radius
        }

    def _update_from_parameters(self):
        """Update internal state from parameters"""
        if 'center' in self.parameters:
            self.cage_center = self.parameters['center']
            self.center = self.cage_center
        if 'radius' in self.parameters:
            self.radius = self.parameters['radius']

    def initialize(self, dris: DRIS) -> None:
        """
        Initialize cage center on first call.

        Sets the initial cage position to a fixed default [22, 64] in pixel
        coordinates. Only runs once (subsequent calls are no-ops).
        """
        if self.initial_center is None:
            self.initial_center = np.array([22, 64])
            self.cage_center = self.initial_center.copy()
            self.center = self.cage_center
            logger.info(f"First PSS detected, initial cage position: ({self.initial_center[0]:.1f}, {self.initial_center[1]:.1f})")

    def step(self):
        """
        Update cage position following orbital trajectory (play_cage16.py:100-107).
        """
        if self.cage_center is not None:
            self.time += self.orbit_speed
            dx = -self.orbit_radius * np.cos(self.time) + self.orbit_radius
            dy = -self.orbit_radius * np.sin(self.time)
            self.cage_center = self.initial_center + np.array([dx, dy])
            self.center = self.cage_center  # Sync

    def update_cage_position(self):
        """Alias for step() to match Diamond naming"""
        self.step()

    def compute_position_from_time(self):
        """
        Compute cage position from current time without incrementing time.
        Used when resetting cage to a specific trajectory point.
        """
        if self.cage_center is not None and self.initial_center is not None:
            dx = -self.orbit_radius * np.cos(self.time) + self.orbit_radius
            dy = -self.orbit_radius * np.sin(self.time)
            self.cage_center = self.initial_center + np.array([dx, dy])
            self.center = self.cage_center  # Sync

    def get_pss_center(self, dris: DRIS, verbose: bool = False) -> Optional[np.ndarray]:
        """
        Detect PSS center from image (play_cage16.py:175-185).

        Uses strict threshold: R>0.7, G<0.4, B<0.4
        """
        frame = self._extract_frame(dris)
        if frame is None:
            return None

        red_mask = (frame[:, :, 0] > 0.7) & (frame[:, :, 1] < 0.4) & (frame[:, :, 2] < 0.4)
        red_pixels = np.sum(red_mask)

        if red_pixels > 0:
            y_coords, x_coords = np.nonzero(red_mask)
            center = np.array([np.mean(x_coords), np.mean(y_coords)])
            if verbose:
                logger.debug(f"PSS detected: {red_pixels} red pixels, center at ({center[0]:.1f}, {center[1]:.1f})")
            return center

        if verbose:
            logger.debug("No red pixels detected")
        return None

    def compute_direction(self, dris: DRIS) -> Optional[int]:
        """
        Compute optimal pushing direction (play_cage16.py:109-160).

        Uses looser threshold for direction: R>0.6, G<0.5, B<0.5

        Returns:
            direction_index: 0-15, or None if PSS fully inside cage
        """
        frame = self._extract_frame(dris)
        pss_center = self.get_pss_center(dris)

        if pss_center is None or self.cage_center is None:
            return None

        # Create PSS mask (looser threshold)
        red_mask = (frame[:, :, 0] > 0.6) & (frame[:, :, 1] < 0.5) & (frame[:, :, 2] < 0.5)

        # Create cage region mask
        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]
        cage_mask = (x - self.cage_center[0])**2 + (y - self.cage_center[1])**2 <= self.radius**2

        # Find PSS pixels outside the cage
        outside_pss_mask = red_mask & ~cage_mask
        outside_pixels = np.sum(outside_pss_mask)

        if outside_pixels == 0:
            logger.debug(f"All PSS pixels ({np.sum(red_mask)}) are inside cage region")
            return None

        # Get coordinates of PSS pixels outside cage
        y_coords, x_coords = np.nonzero(outside_pss_mask)

        # Compute distance from each pixel to cage center
        distance_ratio = 0.4
        distances = ((x_coords - self.cage_center[0])**2 + (y_coords - self.cage_center[1])**2)**distance_ratio

        # Use distance as weight (farther pixels have higher weight)
        weights = distances / np.sum(distances)

        # Compute weighted average position
        weighted_x = np.sum(x_coords * weights)
        weighted_y = np.sum(y_coords * weights)
        outside_center = np.array([weighted_x, weighted_y])

        # Compute direction relative to cage center
        direction = outside_center - self.cage_center
        distance = np.linalg.norm(direction)
        angle = np.arctan2(-direction[1], direction[0])
        angle_deg = np.degrees(angle)
        angle_deg = (angle_deg + 360) % 360

        # Map angle to one of 16 directions (22.5 degrees apart)
        direction_index = int(((angle_deg + 11.25 + 90) % 360) // 22.5)

        # Note: Removed verbose print output for performance
        # These prints can be re-enabled by passing verbose=True if needed
        return direction_index

    def validate_state(self, dris: DRIS) -> bool:
        """
        Validate if PSS is inside cage.

        Returns:
            True if all PSS pixels are inside cage region
        """
        frame = self._extract_frame(dris)
        if frame is None or self.cage_center is None:
            return False

        # Detect PSS (using looser threshold)
        red_mask = (frame[:, :, 0] > 0.6) & (frame[:, :, 1] < 0.5) & (frame[:, :, 2] < 0.5)

        # Create cage mask
        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]
        cage_mask = (x - self.cage_center[0])**2 + (y - self.cage_center[1])**2 <= self.radius**2

        # Check if all PSS inside cage
        outside_pss = red_mask & ~cage_mask
        return np.sum(outside_pss) == 0

    def _extract_frame(self, dris: DRIS) -> Optional[np.ndarray]:
        """
        Extract HWC image in [0,1] range from DRIS observation.

        Returns:
            Image array in HWC format [0, 1] range, or None if extraction fails
        """
        if not hasattr(dris, 'observation'):
            return None

        obs = dris.observation

        if isinstance(obs, np.ndarray):
            # Should already be HWC in [0,1] from DiffusionBackend
            if len(obs.shape) == 3 and obs.shape[2] in [1, 3]:
                return obs

        return None

    def set_cage(self, region):
        """Set cage region."""
        if isinstance(region, dict):
            if 'center' in region:
                self.cage_center = np.array(region['center'])
                self.center = self.cage_center
            if 'radius' in region:
                self.radius = region['radius']
            self.parameters = self._define_parameters()

    def evaluate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[float]:
        """
        Evaluate pixel-based DRIS using PSS center distance to cage center.

        Args:
            dris_input: Single DRIS or list of DRIS states

        Returns:
            List of distances (lower is better)
        """
        dris_list = dris_input if isinstance(dris_input, list) else [dris_input]

        costs = []
        for dris in dris_list:
            pss_center = self.get_pss_center(dris)
            if pss_center is None or self.cage_center is None:
                # No PSS detected or cage not initialized
                costs.append(0.0)
            else:
                distance = np.linalg.norm(pss_center - self.cage_center)
                costs.append(distance)

        return costs

    def validate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[bool]:
        """
        Validate pixel DRIS: check if PSS is within cage radius.

        Args:
            dris_input: Single DRIS or list of DRIS states

        Returns:
            List of bools (True if valid - PSS within cage)
        """
        dris_list = dris_input if isinstance(dris_input, list) else [dris_input]

        results = []
        for dris in dris_list:
            # Use existing validate_state method for each DRIS
            is_valid = self.validate_state(dris)
            results.append(is_valid)

        return results
