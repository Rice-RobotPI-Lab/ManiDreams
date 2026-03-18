"""
Abstract Simulation-based TSIP

This module provides an abstract simulation-based TSIP that can work
with different simulation backends (ManiSkill, MuJoCo, Isaac, etc.).
It contains NO simulator-specific code.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import warnings
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

import gymnasium as gym
from ..base.tsip import TSIPBase
from ..base.dris import DRIS


class SimulationBackend(ABC):
    """
    Abstract base class for simulation backends.
    
    This defines the interface that all simulation backends must implement
    to work with the simulation-based TSIP.
    """
    
    @abstractmethod
    def create_environment(self, env_config: Dict[str, Any]) -> gym.Env:
        """
        Create and return a gym environment.
        
        Args:
            env_config: Environment configuration including context_info
            
        Returns:
            Gym environment (backend handles any parallelization internally)
        """
        pass
    
    @abstractmethod
    def reset_environment(self, env: gym.Env, seed: int = None, options: Dict = None) -> Tuple[Any, Dict]:
        """Reset environment"""
        pass
    
    def step_env(self, env: gym.Env, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Step environment with low-level action (default implementation)"""
        return env.step(action)
    
    
    @abstractmethod
    def get_state(self, env: gym.Env) -> Dict[str, np.ndarray]:
        """Get current state directly from simulation environment"""
        pass
    
    @abstractmethod
    def set_state(self, env: gym.Env, state: Dict[str, np.ndarray]) -> None:
        """Set environment state from state dictionary"""
        pass
    
    def dris2state(self, dris: DRIS) -> Dict[str, np.ndarray]:
        """Convert DRIS to state format for simulation"""
        if hasattr(dris, 'context') and dris.context and 'obj_poses' in dris.context:
            return {'obj_pose': dris.context['obj_poses']}
        return {}
    
    
    @abstractmethod
    def get_action_space(self, env: gym.Env) -> gym.Space:
        """Get action space from environment"""
        pass
    
    @abstractmethod
    def load_env(self, context: Dict[str, Any]) -> None:
        """Load simulation environment (table, walls, etc)"""
        pass
    
    @abstractmethod
    def load_object(self, context: Dict[str, Any]) -> None:
        """Load objects into the simulation based on context"""
        pass
    
    @abstractmethod
    def load_robot(self, context: Dict[str, Any]) -> None:
        """Load robot configuration based on context"""
        pass
    
    @abstractmethod
    def state2dris(self, observations: Any,
                   env_indices: Optional[List[int]] = None,
                   env_config: Optional[Dict[str, Any]] = None) -> List[DRIS]:
        """
        Convert state observation(s) to DRIS format.

        Args:
            observations: Single observation or list of observations
            env_indices: Optional list of environment indices to convert
            env_config: Optional environment configuration for context

        Returns:
            List of DRIS objects
        """
        pass


    @abstractmethod
    def step_act(self, actions: Union[Any, List[Any]], env: gym.Env = None,
                 cage=None, single_action: bool = False) -> Union[Any, List[Any]]:
        """
        Process high-level action(s) and execute environment step(s).

        This method should:
        1. Convert high-level action(s) to low-level control commands
        2. Call step_env() to execute the action(s) in parallel
        3. Return the observation(s)

        Args:
            actions: Single action or list of actions for parallel execution
            env: Environment to step
            cage: Optional cage object for context
            single_action: If True, a single action is being broadcast to all envs

        Returns:
            Single observation or list of observations after stepping
        """
        pass
    
    def close_environment(self, env: gym.Env) -> None:
        """Close environment (optional override)"""
        if hasattr(env, 'close'):
            env.close()
    


class SimulationBasedTSIP(TSIPBase):
    """Simulation-based TSIP implementation."""

    def __init__(self,
                 backend: SimulationBackend,
                 env_config: Optional[Dict[str, Any]] = None,
                 context_info: Optional[Dict[str, Any]] = None,
                 num_rollout_envs: int = 0):
        """
        Initialize simulation-based TSIP with backend environment.

        Args:
            backend: Simulation backend instance
            env_config: Main environment configuration
            context_info: Additional context information
            num_rollout_envs: Number of parallel rollout environments for MPPI.
                            If 0, no rollout environment is created (backward compatible).
                            If >0, creates a separate rollout environment for parallel optimization.
        """
        super().__init__()

        self.backend = backend
        self.env_config = env_config or {}
        self.num_rollout_envs = num_rollout_envs

        if context_info:
            self.env_config.update(context_info)

        # === Create main environment (unchanged) ===
        self.env = self.backend.create_environment(self.env_config)
        self.current_dris: Optional[DRIS] = None
        self._observation_space = None
        self._env_initialized = False
        self._last_step_obs = None

        # === Create rollout environment if requested (MPPI support) ===
        self.rollout_env = None
        self._rollout_env_initialized = False

        if num_rollout_envs > 0:
            logger.info(f"Creating rollout environment with {num_rollout_envs} envs...")
            rollout_config = self.env_config.copy()
            rollout_config['num_envs'] = num_rollout_envs
            rollout_config['render_mode'] = None  # No rendering for rollout
            rollout_config['enable_gui'] = False

            self.rollout_env = self.backend.create_environment(rollout_config)
            logger.info("Rollout environment created successfully")
        
    
    
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> DRIS:
        """
        Reset TSIP environment.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Initial DRIS
        """
        if options is None:
            options = {}
        
        # Reset environment through backend
        obs, _ = self.backend.reset_environment(
            self.env,
            seed=seed, 
            options=options
        )
        
        # Create DRIS from observation (state-based during reset)
        dris_list = self.backend.state2dris([obs])
        self.current_dris = dris_list[0] if dris_list else None
        
        # Mark environment as initialized
        self._env_initialized = True
        
        return self.current_dris
    
    def step(self, actions, cage=None):
        """
        Execute actions using vectorized operations.
        
        Args:
            actions: Single action or list of actions for parallel execution
            cage: Cage object (optional, passed to backend if needed)
        """
        if not isinstance(actions, list):
            actions = [actions]
        
        # Check if environment needs to be reset
        if not hasattr(self, '_env_initialized') or not self._env_initialized:
            # Reset the environment if it hasn't been initialized
            self.env.reset()
            self._env_initialized = True
        
        # Execute all actions using vectorized backend call
        # The backend (e.g., ManiSkill) handles parallel execution internally
        observations = self.backend.step_act(actions, self.env, cage=cage)
        
        # Store observations as dictionary for indexed retrieval
        if isinstance(observations, list):
            self.observations = {i: obs for i, obs in enumerate(observations)}
        else:
            self.observations = {0: observations}
    
    
    
    def get_dris(self, env_indices: Optional[List[int]] = None) -> List[DRIS]:
        """
        Get state-based DRIS from specified environments.
        
        Args:
            env_indices: List of environment indices to get DRIS from
            
        Returns:
            List of DRIS objects representing current states (one per environment)
        """
        # Get current state from backend
        all_states = self.backend.get_state(self.env)

        # Check for valid state dict
        if not isinstance(all_states, dict) or not all_states:
            return []

        # Pass environment context and indices to backend for vectorized processing
        return self.backend.state2dris(all_states, env_indices, self.env_config)

    def set_dris(self, dris: DRIS, env_indices: Optional[List[int]] = None) -> None:
        """
        Set DRIS state to specified environments.
        
        Args:
            dris: DRIS object to set as state  
            env_indices: List of environment indices to set state for.
                        If None, broadcasts to all environments.
        """
        # Convert DRIS to state format and use backend's set_state
        state = self.backend.dris2state(dris)
        if state:
            self.backend.set_state(self.env, state)
        else:
            logger.warning("Could not convert DRIS to state")

    
    def next(self, dris: Union[DRIS, List[DRIS]], action: Union[Any, List[Any]],
             cage: Optional[Any] = None) -> Union[DRIS, List[DRIS]]:
        """
        Compute next DRIS state using simulation backend.

        Handles two cases:
        1. Single DRIS + Single Action: Clone DRIS to all environments and execute same action
        2. Multiple DRIS + Multiple Actions: Execute each action on corresponding DRIS
        """

        # Get number of environments from current TSIP
        num_envs = self.env_config.get('num_envs')
        # Normalize inputs: convert single items to lists if needed
        dris_list = dris if isinstance(dris, list) else [dris]
        action_list = action if isinstance(action, list) else [action]


        # Handle DRIS: broadcast first DRIS as initial state for all environments.
        # This is correct for MPPI/MPC where all samples start from the same state.
        initial_dris = dris_list[0]
        state = self.backend.dris2state(initial_dris)
        if state:
            self.backend.set_state(self.env, state)
        single_action = False
        # Handle Actions: if single action provided, clone it for all environments
        if len(action_list) == 1:
            # Single action case - clone for all environments
            action_to_execute = action_list * num_envs
            single_action = True
        else:
            # Multiple actions case - use as provided
            action_to_execute = action_list

        # Execute actions in parallel (store raw obs for latent re-encoding)
        self._last_step_obs = self.backend.step_act(action_to_execute, self.env, cage=cage, single_action=single_action)

        # Get results
        result_dris_list = self.get_dris()

        # Return format based on original input
        if not isinstance(dris, list) and not isinstance(action, list):
            # Both were single - return single DRIS
            return result_dris_list[0] if result_dris_list else dris
        else:
            # At least one was a list - return list
            return result_dris_list
    
    

    
    
    
    
    def rollout_step(self, offset_batch: List[Any], cage: Optional[Any] = None) -> List[DRIS]:
        """
        Execute a batch of offsets in parallel using rollout environment (MPPI support).

        This method is used by MPPI planner for parallel trajectory rollout.

        Args:
            offset_batch: List of offsets [num_rollout_envs, action_dim]
            cage: Optional cage object for context

        Returns:
            List of DRIS states after executing offsets [num_rollout_envs]

        Raises:
            RuntimeError: If rollout environment was not initialized
        """
        if self.rollout_env is None:
            raise RuntimeError(
                "Rollout environment not initialized. "
                "Set num_rollout_envs > 0 in TSIP initialization."
            )

        # Initialize rollout env if needed
        if not self._rollout_env_initialized:
            self.rollout_env.reset()
            self._rollout_env_initialized = True

        # Execute offsets in parallel using rollout environment
        obs = self.backend.step_act(
            offset_batch,
            env=self.rollout_env,  # Use rollout environment
            cage=cage,
            single_action=False
        )

        # Convert observations to DRIS list
        # Pass rollout environment config to ensure proper batch size handling
        rollout_config = {'num_envs': self.num_rollout_envs}
        dris_list = self.backend.state2dris(obs, env_config=rollout_config)

        return dris_list

    def set_rollout_state(self, dris: DRIS) -> None:
        """
        Copy state from main environment to all rollout environments (MPPI support).

        This is called at the start of each MPPI solve to initialize rollout
        environments with the current state.

        Args:
            dris: DRIS from main environment to copy

        Raises:
            RuntimeError: If rollout environment was not initialized
        """
        if self.rollout_env is None:
            raise RuntimeError(
                "Rollout environment not initialized. "
                "Set num_rollout_envs > 0 in TSIP initialization."
            )

        # Convert DRIS to state format
        state = self.backend.dris2state(dris)

        if state:
            # Set state to rollout environment (broadcasts to all parallel envs)
            self.backend.set_state(self.rollout_env, state)
        else:
            logger.warning("Could not convert DRIS to state for rollout env")

    def close(self) -> None:
        """Close simulation environments"""
        self.backend.close_environment(self.env)

        # Close rollout environment if it exists
        if self.rollout_env is not None:
            self.backend.close_environment(self.rollout_env)
            self.rollout_env = None

        self.current_dris = None

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the simulation backend"""
        return {
            'backend_type': type(self.backend).__name__,
            'env_config': self.env_config,
            'num_rollout_envs': self.num_rollout_envs,
            'has_rollout_env': self.rollout_env is not None
        }