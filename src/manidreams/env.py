"""
ManiDreamsEnv - Main environment class for cage-constrained manipulation tasks.

Provides a unified Gym-compatible interface with built-in cage-constrained planning,
following Python best practices and OMPL's design philosophy.
"""

import logging
import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple, List, Union
import warnings

logger = logging.getLogger(__name__)

from .base.dris import DRIS, ContextSpace
from .base.tsip import TSIPBase
from .base.cage import Cage
from .base.solver import SolverBase


class ManiDreamsEnv(gym.Env):
    """
    Main ManiDreams environment with built-in cage-constrained planning.

    Provides a unified Gym-compatible interface for cage-constrained robotic manipulation
    tasks. Integrates TSIP physics prediction, cage constraints, and solver evaluation
    with the core ManiDreams planning algorithm built-in.

    Key features:
    - DRIS-based observation handling
    - Time-varying cage constraints
    - Parallel action evaluation
    - Built-in dream() planning loop
    - Clean OMPL-style abstract interfaces
    """
    
    metadata = {'render_modes': [], 'render_fps': 30}
    
    def __init__(self,
                 tsip: TSIPBase,
                 action_space: gym.Space,
                 solver: Optional[SolverBase] = None,
                 cage: Optional[Cage] = None,
                 max_timesteps: int = 1000,
                 observation_space: Optional[gym.Space] = None):
        """
        Initialize ManiDreams environment.
        
        Args:
            tsip: Task-Specific Intuitive Physics predictor (REQUIRED)
            action_space: Action space for the task (REQUIRED)
            solver: Optional solver for action evaluation
            cage: Optional cage constraint for validation
            max_timesteps: Maximum timesteps per episode
            observation_space: Custom observation space (inferred from TSIP if None)
            
        Raises:
            ValueError: If required components are not provided
        """
        if tsip is None:
            raise ValueError("TSIP instance is required")
        if action_space is None:
            raise ValueError("Action space is required")
        
        super().__init__()
        
        # Core components
        self.tsip = tsip
        self.action_space = action_space
        self.solver = solver
        self.cage = cage
        self.max_timesteps = max_timesteps
        
        # Set up observation space
        if observation_space is not None:
            self.observation_space = observation_space
        else:
            # Default observation space for DRIS (object state space)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
            )
        
        # Environment state
        self.current_timestep = 0
        self.current_dris: Optional[DRIS] = None
        self.context_space: Optional[ContextSpace] = None
        self.initial_dris: Optional[DRIS] = None
        self._is_initialized = False
    
    def initialize_dris(self, 
                       state_space: Any, 
                       initial_dris: DRIS,
                       context_space: Optional[ContextSpace] = None) -> None:
        """
        Initialize DRIS system with state space and context.
        
        Args:
            state_space: State space for DRIS validation
            initial_dris: Initial DRIS state
            context_space: Optional domain randomization context space
        """
        self.initial_dris = initial_dris
        self.context_space = context_space
        self.current_dris = initial_dris.copy()
        
        # Initialize cage if present
        if self.cage is not None and not self.cage.initialized:
            self.cage.initialize()
        
        self._is_initialized = True
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[DRIS, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            (initial_dris, info) tuple
        """
        super().reset(seed=seed)
        
        self.current_timestep = 0
        
        # Reset TSIP to initialize simulation environment
        if hasattr(self.tsip, 'reset'):
            initial_tsip_dris = self.tsip.reset(seed=seed, options=options)
        
        # Use TSIP's DRIS if available, then initial_dris, otherwise create default
        if hasattr(self.tsip, 'reset') and 'initial_tsip_dris' in locals() and initial_tsip_dris is not None:
            self.current_dris = initial_tsip_dris
        elif self.initial_dris is not None:
            self.current_dris = self.initial_dris.copy()
            
            # Apply randomization if context space is available
            if self.context_space is not None:
                self.current_dris = self.current_dris.randomize(self.context_space)
        else:
            # Create default DRIS with proper observation space handling
            if self.observation_space is not None and hasattr(self.observation_space, 'shape') and self.observation_space.shape is not None:
                # Handle different observation types (images, states, etc.)
                if len(self.observation_space.shape) == 3:  # RGB image: (H, W, C)
                    default_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
                else:  # State vector or other format
                    default_obs = np.zeros(self.observation_space.shape)
            else:
                # Fallback for unknown observation space
                default_obs = np.zeros((10,))
            
            self.current_dris = DRIS(
                observation=default_obs,
                state_space=self.observation_space,
                context=options.get('context', {}) if options else {}
            )
        
        # Cage doesn't need reset - initialize is sufficient
        
        info = {
            'timestep': self.current_timestep,
            'initialized': self._is_initialized
        }
        
        return self.current_dris, info
    
    def step(self, action: Any) -> Tuple[DRIS, float, bool, bool, Dict]:
        """
        Execute single action step.
        
        Args:
            action: Action to execute
            
        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        if self.current_dris is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.current_timestep += 1
        
        # Update time-varying cage using controller
        if self.cage is not None and self.cage.time_varying:
            self.cage.apply_controller_updates(self.current_timestep)
            logger.debug(f"Cage updated: {self.cage.center}")
        
        # Execute action through TSIP
        try:
            next_dris = self.tsip.next(self.current_dris, action, self.cage)
        except (RuntimeError, ValueError) as e:
            warnings.warn(f"TSIP step failed: {e}. Using current state.")
            next_dris = self.current_dris
        
        self.current_dris = next_dris
        
        # Validate with cage
        cage_valid = True
        if self.cage is not None:
            cage_valid = self.cage.validate_state(self.current_dris)
        
        # Basic termination conditions
        terminated = False
        truncated = self.current_timestep >= self.max_timesteps
        
        # Minimal reward (tasks should use solvers for evaluation)
        reward = 0.0
        
        info = {
            'timestep': self.current_timestep,
            'action': action,
            'cage_valid': cage_valid
        }
        
        return self.current_dris, reward, terminated, truncated, info
    
    def dream(self,
              horizon: int = 80,
              cage: Optional[Cage] = None,
              solver: Optional[SolverBase] = None,
              start_timestep: int = 0,
              verbose: bool = True) -> Tuple[List[DRIS], List[Any]]:
        """
        Dream possible futures and select optimal actions under cage constraints.

        Simulates multiple candidate actions in parallel via TSIP, evaluates each
        against the cage constraint, and selects the best action at every timestep.
        The cage acts as a virtual constraint on DRIS, bounding the state set from
        diverging throughout the planning horizon.

        Args:
            horizon: Planning horizon (number of timesteps)
            cage: Cage constraint (uses self.cage if None)
            solver: Solver for evaluation (uses self.solver if None)
            start_timestep: Starting timestep for cage trajectory (default: 0)
            verbose: Whether to print progress information

        Returns:
            Tuple of (trajectory, action_history, cage_history):
            - trajectory: List of DRIS representing the executed trajectory
            - action_history: List of actions that generated the trajectory
            - cage_history: List of cage states for each timestep

        Raises:
            ValueError: If required components are not available
            RuntimeError: If environment is not properly initialized
        """
        cage_to_use = cage or self.cage
        solver_to_use = solver or self.solver

        if not cage_to_use or not solver_to_use:
            raise ValueError("Cage and solver required for dream()")

        if not self.current_dris:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if verbose:
            logger.info("=== ManiDreams Planning ===")
            logger.info(f"Horizon: {horizon}")
            logger.info(f"Start timestep: {start_timestep}")
            logger.info(f"Solver: {solver_to_use.__class__.__name__}")

        trajectory = [self.current_dris.copy()]
        action_history = []
        cage_history = []  # Store cage info for each timestep

        for timestep in range(horizon):
            # Calculate global timestep for cage trajectory
            global_timestep = start_timestep + timestep

            if verbose:
                logger.info(f"--- Timestep {timestep + 1}/{horizon} (global: {global_timestep}) ---")

            # Update time-varying cage using global timestep
            if cage_to_use.time_varying:
                cage_to_use.apply_controller_updates(global_timestep)
                if verbose:
                    logger.info(f"Cage position: {cage_to_use.center}")

            # Save cage state for executor
            cage_info = {
                'center': cage_to_use.center.copy() if hasattr(cage_to_use.center, 'copy') else cage_to_use.center,
                'radius': cage_to_use.radius,
                'timestep': global_timestep
            }
            cage_history.append(cage_info)

            # Generate and evaluate actions
            optimal_action, costs, validations = solver_to_use.solve(
                self.action_space, cage_to_use, self.tsip, self.current_dris, verbose
            )
            logger.debug(f"Optimal action: {optimal_action}")
            # Execute best action
            best_dris = self.tsip.next(self.current_dris, optimal_action, cage_to_use)

            trajectory.append(best_dris.copy())
            action_history.append(optimal_action)
            self.current_dris = best_dris.copy()
        
        if verbose:
            logger.info("=== Execution Complete ===")
            logger.info(f"Trajectory length: {len(trajectory)}")
            logger.info(f"Action sequence: {action_history}")

        # Return trajectory, action history, and cage history for executor
        return trajectory, action_history, cage_history
    
    def sample_context(self) -> Optional[Dict[str, Any]]:
        """Sample context from context space."""
        if self.context_space is not None:
            return self.context_space.sample()
        return None
    
    
    def close(self) -> None:
        """Close environment and clean up resources."""
        if hasattr(self.tsip, 'close'):
            self.tsip.close()
        
        self.current_dris = None
        self.initial_dris = None
        self._is_initialized = False