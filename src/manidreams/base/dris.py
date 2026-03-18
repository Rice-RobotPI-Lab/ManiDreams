"""
Domain-Randomized State Set (DRIS) core implementation.

This module provides the DRIS dataclass as the core data carrier and 
ContextSpace for domain randomization, following Python best practices
with automatic type inference and numpy compatibility.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Any, List
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym


class ContextSpace(ABC):
    """Abstract base class for domain randomization context spaces.
    
    Defines parameter spaces and sampling methods for domain randomization.
    Each task should implement its own ContextSpace subclass.
    """
    
    @abstractmethod
    def sample(self) -> Dict[str, Any]:
        """Sample randomized parameters from the context space.
        
        Returns:
            Dictionary of randomized parameters
        """
        pass
    
    @abstractmethod
    def get_default(self) -> Dict[str, Any]:
        """Get default context parameters without randomization.
        
        Returns:
            Dictionary of default parameters
        """
        pass
    
    def get_bounds(self) -> Dict[str, tuple]:
        """Get parameter bounds for visualization/validation.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        return {}


@dataclass
class DRIS:
    """
    Domain-Randomized State Set - Core data carrier for ManiDreams framework.
    
    Uses dataclass decorator with observation as the primary data carrier.
    Supports automatic type inference, domain randomization, and numpy compatibility.
    
    Attributes:
        observation: Core data (np.ndarray or complex structures)
        state_space: Associated state/observation space (optional)
        context: Domain randomization parameters
        representation_type: "state", "image", or "mixed" (auto-inferred)
        metadata: Additional information dictionary
    """
    observation: Union[np.ndarray, Any]  # Core data carrier
    state_space: Any = None  # Associated space (gym.Space or custom)
    context: Optional[Dict[str, Any]] = None  # Domain randomization context
    representation_type: str = "auto"  # "state", "image", "mixed", "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info
    

    
    def randomize(self, context_space: ContextSpace) -> 'DRIS':
        """Apply domain randomization using ContextSpace.
        
        Args:
            context_space: ContextSpace instance for parameter sampling
            
        Returns:
            New DRIS with randomized context
        """
        if context_space is not None:
            new_context = context_space.sample()
            return DRIS(
                observation=self.observation.copy() if isinstance(self.observation, np.ndarray) else self.observation,
                state_space=self.state_space,
                context=new_context,
                representation_type=self.representation_type,
                metadata=self.metadata.copy()
            )
        return self

    def update_dris(self, new_observation: np.ndarray):
        """Update DRIS with new observation
        
        Args:
            new_observation: New observation data
        """
        self.observation = new_observation
    
    def copy(self) -> 'DRIS':
        """Create a deep copy of this DRIS.
        
        Returns:
            New DRIS instance with copied data
        """
        return DRIS(
            observation=self.observation.copy() if isinstance(self.observation, np.ndarray) else self.observation,
            state_space=self.state_space,
            context=self.context.copy() if self.context else None,
            representation_type=self.representation_type,
            metadata=self.metadata.copy()
        )
    
    
    
    
    
    
    
    
    


