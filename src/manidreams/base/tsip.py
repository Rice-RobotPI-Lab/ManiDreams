"""Abstract base class for Task-Specific Intuitive Physics (TSIP)."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, List
from .dris import DRIS


class TSIPBase(ABC):
    """Base class for Task-Specific Intuitive Physics."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize TSIP."""
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def next(self, dris: DRIS, action: Union[Any, List[Any]], 
             cage: Optional[Any] = None) -> Union[DRIS, List[DRIS]]:
        """
        Compute next DRIS state given current state and action.
        
        Args:
            dris: Current DRIS state
            action: Single action or list of actions for parallel evaluation
            cage: Optional cage constraint for physics simulation
            
        Returns:
            Next DRIS state (single or list for parallel evaluation)
        """
        pass
    
    def reset(self) -> None:
        """Reset TSIP state if needed."""
        pass


# Alias for compatibility
TSIP = TSIPBase