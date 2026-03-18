"""
DRIS Cage for ManiSkill tabletop tasks.

Evaluates DRIS states using distance-to-goal + variance reduction cost.
For distance-based tasks (PushCube, PickCube, PushT, etc.), distance to goal
is an effective proxy for -reward.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import gymnasium as gym
from ..base.cage import Cage
from ..base.dris import DRIS


class DRISCage(Cage):
    """
    DRIS-based cage for tabletop manipulation tasks.

    Cost = distance_to_goal + lambda_var * sum(dris_variance)

    DRIS context expected:
        mean_position: [3] mean position of m DRIS copies
        variance: [3] position variance across DRIS copies
    """

    def __init__(
        self,
        goal_pos: np.ndarray,
        lambda_var: float = 0.1,
        success_radius: float = 0.05,
        state_space: Any = None,
    ):
        if state_space is None:
            state_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )

        super().__init__(state_space, time_varying=False)

        self.goal_pos = np.asarray(goal_pos, dtype=np.float32)
        self.lambda_var = lambda_var
        self.success_radius = success_radius

        self.parameters = self._define_parameters()
        self.initialized = True

    def _define_parameters(self) -> Dict[str, Any]:
        return {
            'goal_pos': self.goal_pos,
            'lambda_var': self.lambda_var,
            'success_radius': self.success_radius,
        }

    def _update_from_parameters(self) -> None:
        self.goal_pos = np.asarray(self.parameters['goal_pos'], dtype=np.float32)
        self.lambda_var = self.parameters['lambda_var']
        self.success_radius = self.parameters['success_radius']

    def set_cage(self, region: Dict[str, Any]) -> None:
        updates = {}
        for key in ['goal_pos', 'lambda_var', 'success_radius']:
            if key in region:
                updates[key] = region[key]
        if updates:
            self.update(**updates)

    def initialize(self) -> None:
        self.initialized = True

    def _extract_from_dris(self, dris: DRIS):
        ctx = dris.context or {}
        mean_pos = ctx.get('mean_position', dris.observation)
        variance = ctx.get('variance', np.zeros(3))

        if hasattr(mean_pos, 'cpu'):
            mean_pos = mean_pos.cpu().numpy()
        if hasattr(variance, 'cpu'):
            variance = variance.cpu().numpy()

        return np.asarray(mean_pos).flatten()[:3], np.asarray(variance).flatten()[:3]

    def evaluate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[float]:
        dris_list = dris_input if isinstance(dris_input, list) else [dris_input]

        costs = []
        for dris in dris_list:
            mean_pos, variance = self._extract_from_dris(dris)
            dist = float(np.linalg.norm(mean_pos[:2] - self.goal_pos[:2]))
            cost = dist + self.lambda_var * float(np.sum(variance))
            costs.append(cost)
        return costs

    def validate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[bool]:
        dris_list = dris_input if isinstance(dris_input, list) else [dris_input]

        results = []
        for dris in dris_list:
            mean_pos, _ = self._extract_from_dris(dris)
            dist = float(np.linalg.norm(mean_pos[:2] - self.goal_pos[:2]))
            results.append(dist <= self.success_radius)
        return results

    def get_boundary(self) -> Dict[str, Any]:
        return {
            'type': 'dris',
            'goal_pos': self.goal_pos.tolist(),
            'lambda_var': self.lambda_var,
            'success_radius': self.success_radius,
        }
