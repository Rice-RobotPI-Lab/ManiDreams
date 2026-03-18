"""
Abstract base class for Cage constraints and CageController.

Provides the foundation for implementing cage constraints and trajectory management
in the ManiDreams framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .dris import DRIS


class CageController:
    """
    Controller for managing cage trajectory and parameter updates over time.
    
    The CageController defines how cage parameters change over time and provides
    the interface for trajectory-based cage updates.
    """
    
    def __init__(self, time_varying: bool = False, trajectory_params: Optional[Dict[str, Any]] = None):
        """
        Initialize cage controller.
        
        Args:
            time_varying: Whether cage changes over time
            trajectory_params: Parameters defining trajectory behavior
        """
        self.time_varying = time_varying
        self.trajectory_params = trajectory_params or {}
        self.trajectory = {}  # Maps timestep -> parameter updates
        
    def set_trajectory(self, trajectory: Dict[int, Dict[str, Any]]) -> None:
        """Set complete trajectory for cage parameters.

        Args:
            trajectory: Dictionary mapping timestep to parameter updates,
                e.g. ``{0: {'center': [0,0], 'radius': 1.0}, 5: {'center': [1,0], 'radius': 1.2}}``.
        """
        self.trajectory = trajectory
        
    def generate_trajectory(self, horizon: int, **kwargs) -> None:
        """
        Generate trajectory automatically based on trajectory_params.
        
        Args:
            horizon: Number of timesteps to generate
            **kwargs: Additional parameters for trajectory generation
        """
        if not self.time_varying:
            return
            
        self.trajectory = {}
        for t in range(horizon):
            updates = self._compute_trajectory_updates(t, **kwargs)
            if updates:
                self.trajectory[t] = updates
    
    def get_updates_at_timestep(self, timestep: int) -> Optional[Dict[str, Any]]:
        """
        Get cage parameter updates for specific timestep.
        
        Args:
            timestep: Current timestep
            
        Returns:
            Dictionary with parameter updates or None if no updates needed
        """
        return self.trajectory.get(timestep, None)
    
    def _compute_trajectory_updates(self, timestep: int, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Compute parameter updates for given timestep (override in subclasses).
        
        Default implementation provides orbital motion for circular cages.
        
        Args:
            timestep: Current timestep
            **kwargs: Additional parameters (initial_center required for orbital motion)
            
        Returns:
            Parameter updates dictionary or None
        """
        if not self.time_varying:
            return None
            
        # Default orbital motion implementation
        if 'orbit_radius' in self.trajectory_params and 'orbit_speed' in self.trajectory_params:
            orbit_radius = self.trajectory_params['orbit_radius']
            orbit_speed = self.trajectory_params['orbit_speed']
            initial_center = kwargs.get('initial_center', np.array([0.0, 0.0]))
            
            if orbit_radius > 0:
                angle = orbit_speed * timestep
                orbit_offset = np.array([
                    orbit_radius * np.cos(angle),
                    orbit_radius * np.sin(angle)
                ])
                new_center = initial_center + orbit_offset
                return {'center': new_center}
        
        return None


class Cage(ABC):
    """
    Abstract base class for cage constraints.
    
    A cage defines safe/desired regions in state space and includes
    abstract parameter management for different cage types.
    """
    
    def __init__(self, 
                 state_space: Any, 
                 time_varying: bool = False, 
                 trajectory_params: Optional[Dict[str, Any]] = None):
        """
        Initialize cage with state space and controller.
        
        Args:
            state_space: State space this cage operates in
            time_varying: Whether cage parameters change over time
            trajectory_params: Parameters for trajectory generation
        """
        self.state_space = state_space
        self.initialized = False
        self.region = None  # Current region representation
        
        # Initialize cage controller
        self.controller = CageController(time_varying, trajectory_params)
        self.time_varying = time_varying
        
        # Abstract parameters - to be defined by subclasses
        self.parameters = {}  # Dict storing current parameter values
        self.initial_parameters = {}  # Dict storing initial parameter values
    
    @abstractmethod
    def _define_parameters(self) -> Dict[str, Any]:
        """
        Define the parameter schema for this cage type.
        
        Returns:
            Dictionary defining parameter names and their default values
            e.g., {'center': [0.0, 0.0], 'radius': 1.0} for CircularCage
        """
        pass
    
    @abstractmethod
    def _update_from_parameters(self) -> None:
        """
        Update internal cage representation from current parameters.
        Called after parameter updates to refresh cage state.
        """
        pass
    
    def update(self, **kwargs) -> None:
        """
        Update cage parameters dynamically.
        
        Args:
            **kwargs: Direct parameter updates (e.g., center=[1,0], radius=1.2)
                     If parameter not provided, keeps current value
        """
        for name, value in kwargs.items():
            if name in self.parameters:
                self.parameters[name] = value
        
        self._update_from_parameters()
    
    def apply_controller_updates(self, timestep: int) -> bool:
        """
        Apply trajectory updates from controller for given timestep.
        
        Args:
            timestep: Current timestep
            
        Returns:
            True if updates were applied, False otherwise
        """
        updates = self.controller.get_updates_at_timestep(timestep)
        if updates:
            self.update(**updates)
            return True
        return False
    
    @abstractmethod
    def set_cage(self, region: Any) -> None:
        """
        Set cage region parameters.
        
        Args:
            region: Region definition (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize cage after parameters are set.
        Must be called before use.
        """
        pass

    @abstractmethod
    def evaluate(self, dris_input: Union['DRIS', List['DRIS']]) -> List[float]:
        """
        Evaluate DRIS state(s) with respect to cage constraints.

        Computes cost/quality metric for given state(s). Lower values indicate
        better satisfaction of cage constraints.

        Args:
            dris_input: Single DRIS or list of DRIS states to evaluate

        Returns:
            List of costs (lower is better). Always returns a list even for single input.

        Note:
            Implementation should handle both single DRIS and list of DRIS uniformly.
            Cost computation is cage-specific (e.g., distance to boundary, area outside cage).
        """
        pass

    @abstractmethod
    def validate(self, dris_input: Union['DRIS', List['DRIS']]) -> List[bool]:
        """
        Validate DRIS state(s) against cage constraints.

        Checks whether given state(s) satisfy the cage constraints.

        Args:
            dris_input: Single DRIS or list of DRIS states to validate

        Returns:
            List of bools (True if valid - satisfies cage constraints).
            Always returns a list even for single input.

        Note:
            Implementation should handle both single DRIS and list of DRIS uniformly.
            Validation criteria are cage-specific (e.g., within radius, inside polygon).
        """
        pass

    def validate_state(self, dris: 'DRIS') -> bool:
        """
        Validate a single DRIS state against cage constraints.

        Convenience wrapper around validate() for single-element validation,
        used by ManiDreamsEnv.step().

        Args:
            dris: Single DRIS state to validate

        Returns:
            True if the state satisfies cage constraints
        """
        results = self.validate(dris)
        return results[0] if results else True
