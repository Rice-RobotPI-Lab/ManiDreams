"""Physics backend module for ManiDreams framework."""

from .simulation_tsip import SimulationBasedTSIP
from .learned_tsip import LearningBasedTSIP

__all__ = [
    "SimulationBasedTSIP",
    "LearningBasedTSIP"
]