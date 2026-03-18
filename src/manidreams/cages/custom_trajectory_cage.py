"""
Custom Trajectory Pixel Cage

Extends CircularPixelCage to support user-defined trajectories
instead of fixed orbital motion.
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from .pixel_cage import CircularPixelCage
from ..base.dris import DRIS

logger = logging.getLogger(__name__)


class CustomTrajectoryPixelCage(CircularPixelCage):
    """
    Circular cage with custom user-defined trajectory.

    Trajectory is specified as a list of waypoints with frame indices and positions.
    Cage position is interpolated between waypoints during execution.
    """

    def __init__(self,
                 radius: float = 22,
                 waypoints: Optional[List[Dict]] = None,
                 time_varying: bool = True):
        """
        Initialize custom trajectory cage.

        Args:
            radius: Cage radius in pixels
            waypoints: List of trajectory waypoints
                      Format: [{'frame': int, 'pixel': [x, y]}, ...]
            time_varying: Always True for custom trajectories
        """
        # Initialize parent without orbit parameters (not used)
        super().__init__(
            radius=radius,
            orbit_radius=0,  # Not used
            orbit_speed=0,   # Not used
            time_varying=time_varying
        )

        self.waypoints = waypoints or []
        self.trajectory_cache = {}  # Cache interpolated positions
        self.current_frame = 0

        if self.waypoints:
            self._build_trajectory_cache()
            logger.debug(f"Initialized with {len(self.waypoints)} waypoints")

    def set_waypoints(self, waypoints: List[Dict]):
        """
        Set trajectory waypoints and rebuild cache.

        Args:
            waypoints: List of waypoints [{'frame': int, 'pixel': [x, y]}, ...]
        """
        self.waypoints = sorted(waypoints, key=lambda w: w['frame'])
        self._build_trajectory_cache()
        logger.debug(f"Trajectory updated: {len(self.waypoints)} waypoints")

    def _build_trajectory_cache(self):
        """Build interpolated trajectory cache from waypoints."""
        if len(self.waypoints) < 2:
            logger.warning("Need at least 2 waypoints for trajectory")
            return

        self.trajectory_cache = {}

        # Get frame range
        start_frame = self.waypoints[0]['frame']
        end_frame = self.waypoints[-1]['frame']

        # Interpolate position for each frame
        for frame in range(start_frame, end_frame + 1):
            position = self._interpolate_position(frame)
            self.trajectory_cache[frame] = position

        logger.debug(f"Trajectory cache built: {len(self.trajectory_cache)} frames")

    def _interpolate_position(self, frame: int) -> np.ndarray:
        """
        Interpolate cage position at given frame using linear interpolation.

        Args:
            frame: Frame index

        Returns:
            Interpolated position [x, y] in pixel coordinates
        """
        if not self.waypoints:
            return np.array([64, 64])  # Default center

        # Before first waypoint
        if frame <= self.waypoints[0]['frame']:
            return np.array(self.waypoints[0]['pixel'])

        # After last waypoint
        if frame >= self.waypoints[-1]['frame']:
            return np.array(self.waypoints[-1]['pixel'])

        # Find surrounding waypoints
        for i in range(len(self.waypoints) - 1):
            w1 = self.waypoints[i]
            w2 = self.waypoints[i + 1]

            if w1['frame'] <= frame <= w2['frame']:
                # Linear interpolation
                t = (frame - w1['frame']) / (w2['frame'] - w1['frame'])
                p1 = np.array(w1['pixel'])
                p2 = np.array(w2['pixel'])
                return p1 + t * (p2 - p1)

        return np.array(self.waypoints[-1]['pixel'])

    def initialize(self, dris: DRIS):
        """
        Initialize cage with first waypoint position.

        Args:
            dris: Initial DRIS state (not used for custom trajectory)
        """
        if self.waypoints and self.initial_center is None:
            self.initial_center = np.array(self.waypoints[0]['pixel'])
            self.cage_center = self.initial_center.copy()
            self.center = self.cage_center
            self.current_frame = self.waypoints[0]['frame']
            logger.info(f"Cage initialized at waypoint 0: ({self.cage_center[0]:.1f}, {self.cage_center[1]:.1f})")

    def step(self):
        """
        Update cage position from trajectory cache.

        Overrides orbital motion with user-defined trajectory.
        """
        if not self.trajectory_cache:
            return

        # Get position from cache
        if self.current_frame in self.trajectory_cache:
            self.cage_center = self.trajectory_cache[self.current_frame].copy()
            self.center = self.cage_center

        # Increment frame
        self.current_frame += 1

    def update_cage_position(self):
        """Alias for step() to match Diamond naming."""
        self.step()

    def compute_position_from_time(self):
        """
        Compute cage position from current frame without incrementing.

        Used when seeking to specific frame in trajectory.
        """
        if not self.trajectory_cache:
            return

        if self.current_frame in self.trajectory_cache:
            self.cage_center = self.trajectory_cache[self.current_frame].copy()
            self.center = self.cage_center

    def seek_to_frame(self, frame: int):
        """
        Seek to specific frame in trajectory.

        Args:
            frame: Target frame index
        """
        self.current_frame = frame
        self.compute_position_from_time()

    def get_trajectory_for_executor(self) -> List[Dict]:
        """
        Export trajectory in executor-compatible format.

        Returns:
            List of cage states: [{'center': [x, y], 'radius': r, 'timestep': t}, ...]
        """
        trajectory = []

        for frame, position in sorted(self.trajectory_cache.items()):
            # Convert pixel coordinates to physical if needed
            # For now, keep in pixel coordinates
            trajectory.append({
                'center': position.tolist(),
                'radius': self.radius,
                'timestep': frame
            })

        return trajectory

    def save_trajectory(self, filepath: str):
        """
        Save trajectory waypoints to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        import json

        data = {
            'radius': self.radius,
            'waypoints': self.waypoints
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Trajectory saved to {filepath}")

    @classmethod
    def load_trajectory(cls, filepath: str) -> 'CustomTrajectoryPixelCage':
        """
        Load trajectory from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            CustomTrajectoryPixelCage instance
        """
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        cage = cls(
            radius=data['radius'],
            waypoints=data['waypoints']
        )

        logger.info(f"Trajectory loaded from {filepath}")
        return cage
