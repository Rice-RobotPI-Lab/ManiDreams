"""
Geometric optimizer for ManiDreams.

This planner implements geometric cage-constrained optimization.
Evaluation and validation logic has been moved to CircularCage.
"""

from typing import Dict, Optional, List, Any
import gymnasium as gym
from .optimizer import MPCOptimizer
from ..samplers import DiscreteSampler


class GeometricOptimizer(MPCOptimizer):
    """
    Geometric cage-constrained optimizer.

    Uses cage's evaluate/validate methods for cost computation and validation.
    Responsible only for action generation and MPC control logic.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize geometric planner.

        Args:
            config: Planner configuration (horizon, num_trajectories, etc.)
        """
        super().__init__(config=config)
        self.name = "GeometricOptimizer"
    
    def action_generator(self, action_space: Optional[gym.spaces.Space] = None) -> List[List[Any]]:
        """
        Generate single-step action trajectories by enumerating all discrete actions.
        
        Args:
            action_space: The discrete action space to enumerate
            
        Returns:
            List of single-step action trajectories for all possible discrete actions
        """
        if action_space is None:
            raise ValueError("action_space must be provided to generate actions")

        if not hasattr(action_space, 'n'):
            raise ValueError("GeometricOptimizer only supports discrete action spaces")

        sampler = DiscreteSampler(action_space)
        return sampler.sample()
