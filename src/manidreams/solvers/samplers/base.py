"""Abstract base class for ManiDreams samplers.

Samplers generate candidate actions for evaluation by solvers.
"""

from abc import ABC, abstractmethod
from typing import List, Any


class SamplerBase(ABC):
    """Abstract base class for all ManiDreams samplers.

    Samplers are responsible for generating candidate actions.
    Solvers use samplers to obtain actions, then evaluate and select the best.
    """

    @abstractmethod
    def sample(self, num_samples: int = None) -> List[Any]:
        """Generate candidate actions.

        Args:
            num_samples: Number of actions to sample.
                For discrete samplers this may be ignored (returns all actions).
                For continuous samplers this controls the batch size.

        Returns:
            List of candidate actions.
        """
        pass

    def reset(self):
        """Reset sampler state."""
        pass
