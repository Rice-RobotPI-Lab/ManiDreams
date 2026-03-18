"""
Abstract base class for ManiDreams solvers.

This module defines the minimal interface that all solvers must implement.
Concrete implementations are in the solvers/ directory.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Any, Tuple, TYPE_CHECKING, Optional
import gymnasium as gym
from gymnasium import spaces

if TYPE_CHECKING:
    from .cage import Cage
    from .dris import DRIS
    from .tsip import TSIPBase


class SolverBase(ABC):
    """
    Abstract base class for all ManiDreams solvers.

    Solvers are responsible for action selection and optimization.
    Evaluation and validation are now handled by Cage classes.
    """

    def __init__(self, name: str = None):
        """
        Initialize solver.

        Args:
            name: Optional name for the solver
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def solve(self, action_space: gym.spaces.Space, cage: 'Cage',
              tsip: 'TSIPBase', current_dris: 'DRIS',
              verbose: bool = False) -> Tuple[Any, List[float], List[bool]]:
        """
        Execute complete action selection process.

        Args:
            action_space: Action space for trajectory generation
            cage: Current cage constraint (provides evaluate/validate methods)
            tsip: TSIP instance for physics prediction
            current_dris: Current DRIS state
            verbose: Whether to print detailed information

        Returns:
            Tuple containing:
            - best_action: The optimal action
            - costs: List of costs for all actions (or None if not computed)
            - validations: List of validation results for all actions (or None if not computed)
        """
        pass

    def reset(self):
        """Reset solver state if needed."""
        pass

    @abstractmethod
    def action_generator(self, action_space: Optional[gym.spaces.Space] = None) -> List[Any]:
        """
        Generate actions for parallel evaluation.

        Args:
            action_space: The action space to sample from (optional, can be stored internally)

        Returns:
            List of actions to evaluate in parallel

        Note:
            For discrete action spaces, typically returns all possible actions.
            For continuous action spaces, typically returns a batch of sampled actions.
        """
        pass


# Alias for compatibility
Solver = SolverBase