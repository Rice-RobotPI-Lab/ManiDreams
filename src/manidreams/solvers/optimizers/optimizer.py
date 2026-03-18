"""MPC-based optimization solver with trajectory rollout."""

import logging
from typing import Dict, Optional, List, Any, Tuple, TYPE_CHECKING
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ...base.solver import SolverBase

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...base.dris import DRIS
    from ...base.cage import Cage
    from ...base.tsip import TSIPBase


class MPCOptimizer(SolverBase):
    """MPC-based optimization solver implementing the complete pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MPC optimization solver."""
        super().__init__(name="MPCOptimizer")

        self.config = config or {}
        self.horizon = self.config.get('horizon', 1)
        self.num_trajectories = self.config.get('num_trajectories', 16)
    
    def solve(self, action_space: gym.spaces.Space, cage: 'Cage', 
              tsip: 'TSIPBase', current_dris: 'DRIS',
              verbose: bool = False) -> Tuple[int, List[float], List[bool]]:
        """
        Execute MPC-based action selection with trajectory rollout.
        
        Args:
            action_space: Action space for trajectory generation
            cage: Current cage constraint
            tsip: TSIP instance for physics prediction
            current_dris: Current DRIS state
            verbose: Whether to print detailed information
            
        Returns:
            Tuple containing:
            - best_action: The optimal first action
            - costs: List of trajectory costs
            - validations: List of trajectory validation results
        """
        if verbose:
            logger.info(f"MPC Solver: Horizon={self.horizon}, Trajectories={self.num_trajectories}")
        
        # Generate action trajectories
        action_trajectories = self.action_generator(action_space)
        
        # Rollout trajectories
        trajectory_dris, cage_sequences = self._rollout(
            action_trajectories, current_dris, cage, tsip, verbose
        )
        
        # Evaluate trajectories
        costs, validations = self._evaluate(trajectory_dris, cage_sequences)
        
        # Select best trajectory
        best_traj_idx = self._select_best(costs, validations, verbose)
        
        # Return first action of best trajectory
        best_first_action = action_trajectories[best_traj_idx][0]
        
        if verbose:
            logger.info(f"Selected action {best_first_action} from trajectory {best_traj_idx}")
        
        return best_first_action, costs, validations
    
    
    def _rollout(self, action_trajectories: List[List[Any]],
                 initial_dris: 'DRIS', cage: 'Cage', tsip: 'TSIPBase',
                 verbose: bool) -> Tuple[List[List['DRIS']], List[List[Any]]]:
        """
        Rollout trajectories to get DRIS sequences.

        Automatically detects if TSIP supports parallel rollout and uses it if:
        1. TSIP has rollout_step and set_rollout_state methods
        2. num_trajectories matches num_rollout_envs
        3. horizon > 1 (parallel rollout is only beneficial for multi-step)

        Otherwise falls back to original serial or horizon=1 parallel execution.
        """
        # Check if parallel rollout is available and beneficial
        has_rollout_env = (
            hasattr(tsip, 'rollout_step') and
            hasattr(tsip, 'set_rollout_state') and
            hasattr(tsip, 'num_rollout_envs')
        )

        can_use_parallel = (
            has_rollout_env and
            tsip.num_rollout_envs > 0 and
            len(action_trajectories) == tsip.num_rollout_envs and
            self.horizon > 1
        )

        if can_use_parallel:
            # Use parallel rollout with rollout environment
            if verbose:
                logger.info(f"  Using parallel rollout: {len(action_trajectories)} trajs, horizon={self.horizon}")
            return self._rollout_parallel(action_trajectories, initial_dris, cage, tsip, verbose)

        elif self.horizon == 1:
            # Original horizon=1 parallel execution (unchanged)
            if verbose:
                logger.info(f"  Executing {len(action_trajectories)} actions in parallel")

            # Extract single actions
            actions = [traj[0] for traj in action_trajectories]

            # Execute all actions in parallel
            next_dris_list = tsip.next(initial_dris, actions, cage)

            # Ensure list format
            if not isinstance(next_dris_list, list):
                next_dris_list = [next_dris_list]

            # Package as single-step trajectories
            trajectory_dris = []
            cage_sequences = []
            for dris in next_dris_list:
                trajectory_dris.append([dris])
                cage_sequences.append([cage])

            return trajectory_dris, cage_sequences

        else:
            # Original sequential rollout for horizon>1 (unchanged)
            if verbose:
                logger.info(f"  Using sequential rollout: {len(action_trajectories)} trajs")

            trajectory_dris = []
            cage_sequences = []

            for traj_idx, action_trajectory in enumerate(action_trajectories):
                if verbose and traj_idx % 4 == 0:
                    logger.info(f"  Rolling out trajectory {traj_idx}/{len(action_trajectories)}...")

                dris_sequence = []
                cage_sequence = []
                current_dris = initial_dris

                for t, action in enumerate(action_trajectory):
                    # Update cage if time-varying
                    current_cage = cage
                    if hasattr(cage, 'time_varying') and cage.time_varying:
                        cage_copy = type(cage)(
                            state_space=cage.state_space,
                            center=cage.center.copy() if hasattr(cage.center, 'copy') else cage.center,
                            radius=cage.radius
                        )
                        cage_copy.apply_controller_updates(t)
                        current_cage = cage_copy

                    cage_sequence.append(current_cage)

                    # Execute single action
                    next_dris = tsip.next(current_dris, action, current_cage)
                    dris_sequence.append(next_dris)
                    current_dris = next_dris

                trajectory_dris.append(dris_sequence)
                cage_sequences.append(cage_sequence)

            return trajectory_dris, cage_sequences

    def _rollout_parallel(self, action_trajectories: List[List[Any]],
                         initial_dris: 'DRIS', cage: 'Cage', tsip: 'TSIPBase',
                         verbose: bool) -> Tuple[List[List['DRIS']], List[List[Any]]]:
        """
        Parallel rollout using TSIP's rollout environment.

        This method executes all trajectories in parallel by stepping through
        time horizons together, executing one action per trajectory per timestep.

        Args:
            action_trajectories: List of action trajectories [num_trajs, horizon, action_dim]
            initial_dris: Initial DRIS state
            cage: Cage constraint
            tsip: TSIP instance with rollout environment
            verbose: Whether to print progress

        Returns:
            trajectory_dris: List of DRIS sequences [num_trajs, horizon]
            cage_sequences: List of cage sequences [num_trajs, horizon]
        """
        num_trajs = len(action_trajectories)
        horizon = len(action_trajectories[0]) if action_trajectories else 0

        # Initialize rollout environments with current state
        tsip.set_rollout_state(initial_dris)

        # Initialize result containers
        dris_sequences = [[] for _ in range(num_trajs)]
        cage_sequences = [[] for _ in range(num_trajs)]

        # Roll out all trajectories in parallel by timestep
        for t in range(horizon):
            if verbose and t % 2 == 0:
                logger.info(f"    Parallel step {t}/{horizon}")

            # Collect actions for this timestep from all trajectories
            actions_at_t = [traj[t] for traj in action_trajectories]

            # Update cage if time-varying
            current_cage = cage
            if hasattr(cage, 'time_varying') and cage.time_varying:
                # TODO: Handle time-varying cage properly
                # For now, use the original cage
                pass

            # Execute all actions in parallel using rollout environment
            next_dris_list = tsip.rollout_step(actions_at_t, current_cage)

            # Collect results for each trajectory
            for i in range(num_trajs):
                dris_sequences[i].append(next_dris_list[i])
                cage_sequences[i].append(current_cage)

        return dris_sequences, cage_sequences
    
    def _evaluate(self, trajectory_dris: List[List['DRIS']],
                  cage_sequences: List[List[Any]]) -> Tuple[List[float], List[bool]]:
        """Evaluate all trajectories using CAGE's evaluate/validate methods."""

        discount = self.config.get('discount', 1.0)
        num_trajectories = len(trajectory_dris)
        horizon = len(trajectory_dris[0]) if trajectory_dris else 1

        # Initialize arrays for trajectory costs and validations
        trajectory_costs = []
        trajectory_validations = []

        # Process each timestep across all trajectories in batch
        for t in range(horizon):
            # Collect all DRIS at timestep t from all trajectories
            dris_at_t = [trajectory_dris[traj_idx][t] for traj_idx in range(num_trajectories)]

            # Use the cage at timestep t (same for all trajectories at this timestep)
            cage_t = cage_sequences[0][t] if cage_sequences and cage_sequences[0] else None

            # Batch evaluate all trajectories at this timestep using CAGE methods
            costs_at_t = cage_t.evaluate(dris_at_t)
            validations_at_t = cage_t.validate(dris_at_t)

            # Ensure they're lists
            if not isinstance(costs_at_t, list):
                costs_at_t = [costs_at_t]
            if not isinstance(validations_at_t, list):
                validations_at_t = [validations_at_t]

            # Store costs and validations for this timestep
            if t == 0:
                # Initialize trajectory accumulators
                trajectory_costs = [c * (discount ** t) for c in costs_at_t]
                trajectory_validations = validations_at_t
            else:
                # Accumulate discounted costs
                for traj_idx in range(num_trajectories):
                    trajectory_costs[traj_idx] += costs_at_t[traj_idx] * (discount ** t)
                    trajectory_validations[traj_idx] = trajectory_validations[traj_idx] and validations_at_t[traj_idx]

        return trajectory_costs, trajectory_validations
    
    def _select_best(self, costs: List[float], validations: List[bool], 
                     verbose: bool) -> int:
        """Select the best trajectory based on costs and validity."""
        valid_indices = [i for i, valid in enumerate(validations) if valid]
        
        if valid_indices:
            valid_costs = [costs[i] for i in valid_indices]
            best_local_idx = np.argmin(valid_costs)
            best_traj_idx = valid_indices[best_local_idx]
            if verbose:
                logger.info(f"Best valid trajectory: {best_traj_idx} with cost {costs[best_traj_idx]:.2g}")
        else:
            best_traj_idx = np.argmin(costs)
            if verbose:
                logger.info(f"No valid trajectories! Best trajectory: {best_traj_idx} with cost {costs[best_traj_idx]:.2g}")
        
        return best_traj_idx
    
    def reset(self):
        """Reset solver state if needed."""
        pass
    
    def action_generator(self, action_space: Optional[gym.spaces.Space] = None) -> List[List[Any]]:
        """
        Generate action trajectories for parallel evaluation.
        
        Args:
            action_space: The action space to sample from
            
        Returns:
            List of action trajectories, where each trajectory is a list of actions
            for the planning horizon
            
        Note:
            Must be implemented by subclasses. Should handle both single-step and 
            multi-step trajectory generation based on horizon.
        """
        raise NotImplementedError("action_generator() must be implemented by subclasses")