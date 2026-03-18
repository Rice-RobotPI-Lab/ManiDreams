"""Gaussian action sampler for ManiDreams.

Samples continuous action trajectories from a Gaussian distribution.
Used by picking tasks with MPPIOptimizer.
"""

from typing import List
import numpy as np

from .base import SamplerBase


class GaussianSampler(SamplerBase):
    """Sample continuous action trajectories from a Gaussian distribution.

    Each sample is a trajectory of length `horizon`, where each step
    is drawn from N(mean[t], std[t]^2).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, horizon: int = 1):
        """Initialize Gaussian sampler.

        Args:
            mean: Distribution mean, shape [horizon, action_dim].
            std: Distribution standard deviation, shape [horizon, action_dim].
            horizon: Trajectory length.
        """
        self.mean = mean
        self.std = std
        self.horizon = horizon

    def sample(self, num_samples: int = 16) -> List[List[np.ndarray]]:
        """Sample trajectories from the Gaussian distribution.

        Args:
            num_samples: Number of trajectories to sample.

        Returns:
            List of trajectories, each a list of action arrays.
        """
        trajectories = []
        for _ in range(num_samples):
            trajectory = []
            for t in range(self.horizon):
                offset = self.mean[t] + self.std[t] * np.random.randn(*self.mean[t].shape)
                trajectory.append(offset)
            trajectories.append(trajectory)
        return trajectories

    def update(self, mean: np.ndarray, std: np.ndarray):
        """Update distribution parameters (for CEM/MPPI iterations).

        Args:
            mean: New mean, shape [horizon, action_dim].
            std: New standard deviation, shape [horizon, action_dim].
        """
        self.mean = mean
        self.std = std
