"""
Simplified Learning-based TSIP

Thin wrapper around DiffusionBackend that replicates Diamond's PlayCageEnv behavior.
No complex state encoding - directly calls backend methods.
"""

from typing import Any, Dict, Optional, Union, List
import numpy as np
import warnings
from abc import ABC, abstractmethod

import gymnasium as gym
from ..base.tsip import TSIPBase
from ..base.dris import DRIS


class LearnedBackend(ABC):
    """
    Abstract base class for learned model backends.

    LearnedBackend works directly with DRIS (not state dictionaries).
    The observation format in DRIS is backend-specific (e.g., images for DiffusionBackend).
    """

    @abstractmethod
    def load_model(self, model_config: Dict[str, Any]) -> Any:
        """
        Load and return a learned model.

        Args:
            model_config: Model configuration including model_path, device, etc.

        Returns:
            Loaded model object
        """
        pass

    @abstractmethod
    def reset(self) -> DRIS:
        """
        Reset model and return initial DRIS.

        Returns:
            Initial DRIS (observation format is backend-specific)
        """
        pass

    @abstractmethod
    def predict_step(self, model: Any, current_state: Any, action: Any):
        """
        Predict next state given action.

        Args:
            model: Model object (can be ignored if backend maintains state)
            current_state: Current state (can be ignored if backend maintains state)
            action: Action index

        Returns:
            (next_observation, info) tuple
        """
        pass

    @abstractmethod
    def get_dris(self) -> DRIS:
        """
        Get current DRIS from backend.

        Backends maintain internal state and return current DRIS.
        For DiffusionBackend, this converts internal tensor to image DRIS.

        Returns:
            Current DRIS object
        """
        pass

    @abstractmethod
    def set_dris(self, dris: DRIS) -> None:
        """
        Set backend internal state from DRIS.

        Backends update their internal state from DRIS observation.
        For DiffusionBackend, this converts image DRIS to internal tensor.

        Args:
            dris: DRIS object to set (observation format is backend-specific)
        """
        pass

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """Get action space for the learned model"""
        pass

    @abstractmethod
    def get_observation_space(self) -> gym.Space:
        """Get observation space for the learned model"""
        pass


class LearningBasedTSIP(TSIPBase):
    """
    Simplified learned model-based TSIP.

    Thin wrapper around backend.predict_step() that replicates Diamond's
    PlayCageEnv behavior. Backend maintains internal state (self.obs).
    """

    def __init__(self,
                 backend: LearnedBackend,
                 model_config: Optional[Dict[str, Any]] = None,
                 context_info: Optional[Dict[str, Any]] = None):
        """
        Initialize learning-based TSIP with backend model.

        Args:
            backend: DiffusionBackend instance
            model_config: Model configuration (passed to backend.load_model)
            context_info: Additional context (unused in simplified version)
        """
        super().__init__()

        self.backend = backend
        self.model_config = model_config or {}
        self.context_info = context_info or {}

        # Load model through backend
        self.model = self.backend.load_model(self.model_config)
        self.current_dris: Optional[DRIS] = None

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> DRIS:
        """
        Reset TSIP model.

        Calls backend.reset() which replicates PlayCageEnv.reset()
        (play_cage_env.py:132-147).

        Args:
            seed: Random seed (ignored)
            options: Reset options (ignored)

        Returns:
            Initial DRIS with observation as HWC [0,1] numpy array
        """
        # Backend.reset() handles everything (wm_env.reset, tensor conversion)
        self.current_dris = self.backend.reset()
        return self.current_dris

    def get_dris(self, env_indices: Optional[List[int]] = None) -> List[DRIS]:
        """
        Get current DRIS state from backend.

        Delegates to backend.get_dris() for state retrieval.

        Args:
            env_indices: List of indices (ignored for single-env learned models)

        Returns:
            List containing current DRIS
        """
        # Delegate to backend to get current DRIS
        self.current_dris = self.backend.get_dris()
        return [self.current_dris] if self.current_dris else []

    def set_dris(self, dris: DRIS, env_indices: Optional[List[int]] = None) -> None:
        """Set DRIS state and synchronize to backend.

        Delegates to backend.set_dris() for state synchronization.
        Backend handles conversion from DRIS to internal representation.

        Args:
            dris: DRIS object to set
            env_indices: List of indices (ignored for single-env models)

        Example::

            # After executor feedback
            synthetic_image = generate_synthetic_observation(object_pos)
            new_dris = DRIS(
                observation=synthetic_image,
                context={},
                metadata={'timestep': 100}
            )
            tsip.set_dris(new_dris)  # Backend handles internal state update
        """
        # Update TSIP's reference
        self.current_dris = dris

        # Delegate to backend for internal state synchronization
        self.backend.set_dris(dris)

    def next(self, dris: Union[DRIS, List[DRIS]], action: Union[Any, List[Any]],
             cage: Optional[Any] = None) -> Union[DRIS, List[DRIS]]:
        """
        Predict next DRIS given current DRIS and action(s).

        Simplified version: Directly calls backend.predict_step() which
        replicates PlayCageEnv.step() (play_cage_env.py:190-239).

        NOTE: dris parameter is IGNORED because backend maintains internal
        state (self.obs tensor). This replicates Diamond's behavior where
        PlayCageEnv.step() uses self.obs instead of external state.

        Args:
            dris: Current DRIS (IGNORED - backend uses internal state)
            action: Action index (0-15) or list of actions
            cage: Optional cage object (ignored)

        Returns:
            Next DRIS or list of DRIS with observation as HWC [0,1] numpy
        """
        # Handle single vs list inputs
        single_input = not isinstance(action, list)
        action_list = [action] if single_input else action

        # Predict next states
        next_dris_list = []
        for current_action in action_list:
            # Direct call to backend.predict_step()
            # Backend maintains self.obs internally (like PlayCageEnv)
            # dris parameter is ignored - backend uses its own state
            next_frame, info = self.backend.predict_step(
                model=self.model,        # Can be ignored by backend
                current_state=None,      # Ignored - backend uses self.obs
                action=current_action
            )

            # Create next DRIS
            next_dris = DRIS(
                observation=next_frame,
                context=self.context_info,
                metadata={'timestep': info.get('timestep', 0)}
            )
            next_dris_list.append(next_dris)

            # Update current_dris for reference
            self.current_dris = next_dris

        # Return in same format as input
        return next_dris_list[0] if single_input else next_dris_list

    def get_observation_space(self) -> gym.Space:
        """Get observation space from backend."""
        return self.backend.get_observation_space()

    def get_action_space(self) -> gym.Space:
        """Get action space from backend."""
        return self.backend.get_action_space()

    def close(self) -> None:
        """Close learned model and clean up resources."""
        if hasattr(self.backend, 'close_model'):
            self.backend.close_model(self.model)

        # Clear stored data
        self.model = None
        self.current_dris = None
