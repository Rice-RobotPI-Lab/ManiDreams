"""Discrete action sampler for ManiDreams.

Enumerates all actions in a discrete action space.
Used by pushing tasks with GeometricOptimizer.
"""

from typing import List
import gymnasium as gym

from .base import SamplerBase


class DiscreteSampler(SamplerBase):
    """Enumerate all actions in a discrete action space.

    Returns each action wrapped as a single-step trajectory [action],
    compatible with MPCOptimizer's trajectory-based interface.
    """

    def __init__(self, action_space: gym.spaces.Discrete):
        """Initialize with a discrete action space.

        Args:
            action_space: Gymnasium Discrete action space.
        """
        self.action_space = action_space

    def sample(self, num_samples: int = None) -> List[List[int]]:
        """Enumerate all discrete actions as single-step trajectories.

        Args:
            num_samples: Ignored. All actions are always returned.

        Returns:
            List of single-step trajectories [[0], [1], ..., [n-1]].
        """
        return [[action] for action in range(self.action_space.n)]
