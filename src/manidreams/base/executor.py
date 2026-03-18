"""Abstract base class for action sequence execution."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union


class ExecutorBase(ABC):
    """Base class for executing action sequences with sensor feedback."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize executor."""
        self.name = name or self.__class__.__name__
        self.initialized = False
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with environment configuration."""
        pass
    
    @abstractmethod
    def execute(self, actions: Union[Any, List[Any]], 
                get_feedback: bool = True) -> Union[Tuple[Any, Dict], Tuple[List[Any], List[Dict]]]:
        """
        Execute single action or action sequence.
        
        Args:
            actions: Single action or list of actions
            get_feedback: Whether to collect feedback
            
        Returns:
            For single action: (observation, feedback_dict)
            For action sequence: (observations_list, feedback_dict_list)
        """
        pass
    
    @abstractmethod
    def reset(self) -> Any:
        """Reset environment to initial state."""
        pass
    
    @abstractmethod
    def get_obs(self) -> Any:
        """
        Get current observation of the environment.

        Returns observation containing object state information.
        For simulation: returns state-based observation (ground truth from simulator)
        For real hardware: returns sensor-based observation

        Returns:
            Observation dict containing object state (e.g., position, orientation)
        """
        pass
    
    def close(self) -> None:
        """Clean up resources."""
        pass