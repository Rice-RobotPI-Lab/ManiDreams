"""
Pixel-based Planner for ManiDreams Framework

Simplified planner that directly uses CircularPixelCage's compute_direction method.
Replicates Diamond's play_cage16.py action selection logic.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Any

from ...base.solver import SolverBase
from ...base.dris import DRIS
from ...base.cage import Cage

logger = logging.getLogger(__name__)


class PixelOptimizer(SolverBase):
    """
    Simplified planner that directly uses Cage's compute_direction.
    No prediction, no evaluation - completely trusts cage's computation.

    Replicates Diamond's AutoPlayGame action selection (play_cage16.py:272-288).
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.config = config or {}
        self.num_actions = 16

    def action_generator(self, action_space):
        """Generate 16 directional actions"""
        return list(range(self.num_actions))

    def solve(self,
              sampled_actions: Any,
              cage: Cage,
              tsip: Any,
              current_dris: DRIS,
              verbose: bool = False) -> Tuple[int, None, None]:
        """
        Solve for optimal action using cage's compute_direction.

        Replicates Diamond's action selection logic (play_cage16.py:272-288):
        - If direction_index is None (PSS inside cage), return action 0
        - Otherwise return the direction_index computed by cage

        PixelOptimizer does not compute costs or validations.

        Args:
            sampled_actions: List of action indices (0-15) - ignored
            cage: CircularPixelCage object with compute_direction method
            tsip: TSIP instance - ignored by this planner
            current_dris: Current DRIS with image observation
            verbose: Whether to log debug information

        Returns:
            Tuple of (best_action_idx, None, None)
        """

        # Direct call to cage's compute_direction (play_cage16.py:272)
        direction_index = cage.compute_direction(current_dris)

        if direction_index is None:
            # PSS is fully inside cage, no action needed
            # Diamond returns action 0 in this case (play_cage16.py:275-278)
            best_action_idx = 0
            if verbose:
                logger.info("PSS fully inside cage, action=0")
        else:
            # Use cage's computed direction directly (play_cage16.py:280-288)
            best_action_idx = direction_index
            if verbose:
                logger.info(f"Using cage direction: {direction_index}")

        return best_action_idx, None, None
