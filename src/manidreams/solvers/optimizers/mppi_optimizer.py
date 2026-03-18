"""
MPPI (Model Predictive Path Integral) Planner

Implements MPPI optimization for continuous action spaces using offset-based control.
Optimized for parallel rollout with 16 samples and 6D continuous offsets.
"""

import logging
from typing import Dict, Optional, List, Any, Tuple
import numpy as np
import gymnasium as gym
from .optimizer import MPCOptimizer
from ..samplers import GaussianSampler

logger = logging.getLogger(__name__)


class MPPIOptimizer(MPCOptimizer):
    """
    MPPI optimizer for continuous offset actions.

    Key features:
    - Optimizes 6D offsets [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
    - Uses Cross-Entropy Method (CEM) with MPPI weighting
    - Supports parallel rollout with TSIP rollout environments
    - Includes warm-start mechanism for temporal coherence

    Compatible with ManiDreams framework:
    - Inherits from MPCOptimizer (reuses _rollout, _evaluate)
    - Works with existing TSIP and Cage interfaces
    - Backward compatible (no changes to existing code)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MPPI planner.

        Args:
            config: Configuration dictionary with keys:
                - horizon: Planning horizon (default: 5)
                - num_samples: Number of trajectories to sample (default: 16)
                - num_elites: Number of elite samples for update (default: 8)
                - num_iterations: Number of CEM iterations (default: 6)
                - temperature: MPPI temperature parameter (default: 0.5)
                - discount: Discount factor for cost accumulation (default: 0.95)
                - init_std: Initial standard deviation [6] (default: [0.02, 0.02, 0.01, 0.1, 0.1, 0.1])
                - min_std: Minimum standard deviation [6] (default: [0.005, 0.005, 0.002, 0.02, 0.02, 0.02])
                - max_std: Maximum standard deviation [6] (default: [0.05, 0.05, 0.02, 0.26, 0.26, 0.26])
        """
        super().__init__(config=config)
        self.name = "MPPIOptimizer"

        # MPPI parameters
        self.num_samples = self.config.get('num_samples', 16)
        self.num_elites = self.config.get('num_elites', 8)
        self.num_iterations = self.config.get('num_iterations', 6)
        self.temperature = self.config.get('temperature', 0.5)

        # Std scaling factors (relative to action_space range)
        # These determine how std is computed from action_space bounds
        self.init_std_scale = self.config.get('init_std_scale', 0.25)    # init_std = range * 0.25
        self.min_std_scale = self.config.get('min_std_scale', 0.05)      # min_std = range * 0.05
        self.max_std_scale = self.config.get('max_std_scale', 0.5)       # max_std = range * 0.5

        # Fallback std values (used if action_space not provided or not Box)
        self._fallback_init_std = np.array(self.config.get('init_std',
            [0.02, 0.02, 0.01, 0.1, 0.1, 0.1]))
        self._fallback_min_std = np.array(self.config.get('min_std',
            [0.005, 0.005, 0.002, 0.02, 0.02, 0.02]))
        self._fallback_max_std = np.array(self.config.get('max_std',
            [0.05, 0.05, 0.02, 0.26, 0.26, 0.26]))

        # Actual std values (computed from action_space in solve())
        self.init_std = None
        self.min_std = None
        self.max_std = None

        # Warm-start state
        self._prev_offset_mean = None
        self._prev_offset_std = None
        self._is_first_step = True

        # Current distribution (used by action_generator)
        self._current_offset_mean = None
        self._current_offset_std = None

        logger.debug(f"Initialized with {self.num_samples} samples, "
                     f"{self.num_iterations} iterations, horizon={self.horizon}")
        logger.debug(f"  Std scales: init={self.init_std_scale}, min={self.min_std_scale}, max={self.max_std_scale}")

    def solve(self, action_space: gym.spaces.Space, cage, tsip, current_dris,
              verbose: bool = False) -> Tuple[Any, List[float], List[bool]]:
        """
        Execute MPPI optimization to find best offset.

        Args:
            action_space: Box space defining offset bounds
            cage: Cage constraint
            tsip: TSIP instance (should have rollout environment)
            current_dris: Current DRIS state
            verbose: Whether to print detailed information

        Returns:
            best_offset: Optimal 6D offset [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
            costs: List of trajectory costs
            validations: List of trajectory validations
        """
        if verbose:
            logger.info("[MPPI] === Starting Solve ===")
            logger.info(f"  Samples: {self.num_samples}, Iterations: {self.num_iterations}, Horizon: {self.horizon}")

        # Verify TSIP compatibility
        if hasattr(tsip, 'num_rollout_envs') and tsip.num_rollout_envs != self.num_samples:
            logger.warning(f"TSIP rollout envs ({tsip.num_rollout_envs}) != "
                           f"MPPI samples ({self.num_samples}). May use serial rollout.")

        # Step 1: Initialize offset distribution (use action_space center as initial mean)
        offset_mean, offset_std = self._initialize_offset_distribution(action_space)

        # Step 2: MPPI iterations
        for iteration in range(self.num_iterations):
            if verbose:
                logger.info(f"[MPPI] --- Iteration {iteration+1}/{self.num_iterations} ---")

            # 2.1 Sample offset trajectories
            offset_trajectories = self._sample_offset_trajectories(offset_mean, offset_std)

            # 2.2 Rollout and evaluate (uses MPCOptimizer._rollout)
            trajectory_dris, cage_sequences = self._rollout(
                offset_trajectories, current_dris, cage, tsip, verbose
            )
            costs, validations = self._evaluate(trajectory_dris, cage_sequences)

            if verbose:
                valid_count = np.sum(validations)
                if valid_count > 0:
                    avg_cost = np.mean([c for c, v in zip(costs, validations) if v])
                    logger.info(f"  Valid: {valid_count}/{len(costs)}, Avg cost: {avg_cost:.3f}")
                else:
                    logger.info(f"  Valid: 0/{len(costs)}, All invalid!")

            # 2.3 Compute MPPI weights
            weights = self._compute_mppi_weights(costs, validations)

            # 2.4 Update distribution
            offset_mean, offset_std = self._update_offset_distribution(
                offset_trajectories, weights, offset_mean, offset_std
            )

        # Step 3: Select best offset
        best_offset = self._select_best_offset(offset_trajectories, weights, exploration=False)

        # Step 4: Save for warm-start
        self._save_for_warm_start(offset_mean, offset_std)

        if verbose:
            logger.info("[MPPI] === Solve Complete ===")
            logger.info(f"  Best offset: {best_offset}")

        return best_offset, costs, validations

    def action_generator(self, action_space: gym.spaces.Space) -> List[List[np.ndarray]]:
        """
        Generate offset trajectories for evaluation.

        This method is called by solve() after distribution update.
        Returns trajectories sampled from current distribution.

        Args:
            action_space: Box space (not used, distribution comes from internal state)

        Returns:
            List of offset trajectories [num_samples, horizon, 6]
        """
        if self._current_offset_mean is None or self._current_offset_std is None:
            # Fallback: initialize with default
            mean = np.zeros((self.horizon, 6))
            std = np.tile(self.init_std, (self.horizon, 1))
        else:
            mean = self._current_offset_mean
            std = self._current_offset_std

        return self._sample_offset_trajectories(mean, std)

    def _sample_offset_trajectories(self, mean: np.ndarray,
                                     std: np.ndarray) -> List[List[np.ndarray]]:
        """
        Sample offset trajectories from Gaussian distribution.

        Args:
            mean: [horizon, 6] mean of distribution
            std: [horizon, 6] standard deviation

        Returns:
            List of trajectories, each is [horizon, 6] offsets
        """
        sampler = GaussianSampler(mean, std, self.horizon)
        return sampler.sample(self.num_samples)

    def _compute_mppi_weights(self, costs: List[float],
                              validations: List[bool]) -> np.ndarray:
        """
        Compute MPPI weights: w ∝ exp(-temperature * cost).

        Invalid samples are assigned zero weight.

        Args:
            costs: List of trajectory costs
            validations: List of trajectory validations

        Returns:
            Normalized weights [num_samples]
        """
        costs = np.array(costs)
        validations = np.array(validations)

        # Set invalid samples to infinite cost
        costs[~validations] = np.inf

        # Check if any valid samples exist
        if not np.any(validations):
            # All invalid: return uniform weights
            return np.ones(len(costs)) / len(costs)

        # Numerical stability: subtract minimum cost
        min_cost = np.min(costs[validations])

        # MPPI weight formula
        exp_costs = np.exp(-self.temperature * (costs - min_cost))
        exp_costs[~validations] = 0  # Zero weight for invalid

        # Normalize
        weights = exp_costs / (np.sum(exp_costs) + 1e-10)

        return weights

    def _update_offset_distribution(self, offset_trajectories: List[List[np.ndarray]],
                                     weights: np.ndarray,
                                     current_mean: np.ndarray,
                                     current_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update offset distribution using weighted mean and std.

        Args:
            offset_trajectories: List of sampled trajectories
            weights: MPPI weights for each trajectory
            current_mean: Current mean (unused, but kept for interface)
            current_std: Current std (unused, but kept for interface)

        Returns:
            new_mean: [horizon, 6]
            new_std: [horizon, 6]
        """
        # Convert to array [num_samples, horizon, 6]
        offsets_array = np.array(offset_trajectories)

        # Weighted mean
        new_mean = np.sum(
            weights[:, None, None] * offsets_array, axis=0
        )  # [horizon, 6]

        # Weighted standard deviation
        new_std = np.sqrt(
            np.sum(
                weights[:, None, None] * (offsets_array - new_mean)**2,
                axis=0
            ) + 1e-8
        )  # [horizon, 6]

        # Clamp std to valid range
        new_std = np.clip(new_std, self.min_std, self.max_std)

        return new_mean, new_std

    def _select_best_offset(self, offset_trajectories: List[List[np.ndarray]],
                            weights: np.ndarray, exploration: bool = False) -> np.ndarray:
        """
        Select best offset using weighted mean (standard MPPI).

        Weighted mean is smoother and more stable than stochastic sampling,
        avoiding jitter from randomly picking a sub-optimal trajectory.

        Args:
            offset_trajectories: List of sampled trajectories
            weights: MPPI weights
            exploration: Whether to add exploration noise

        Returns:
            Best first-step offset [6]
        """
        # Weighted mean over first-step offsets (standard MPPI formulation)
        first_steps = np.array([traj[0] for traj in offset_trajectories])  # [num_samples, 6]
        best_offset = np.sum(weights[:, None] * first_steps, axis=0)  # [6]

        # Add exploration noise if requested
        if exploration and self._current_offset_std is not None:
            noise = self._current_offset_std[0] * np.random.randn(6) * 0.1
            best_offset = best_offset + noise

        return best_offset

    def _initialize_offset_distribution(self, action_space: gym.spaces.Space = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize offset distribution with warm-start if available.

        Args:
            action_space: Box space defining offset bounds (used for computing center and std)

        Returns:
            offset_mean: [horizon, 6]
            offset_std: [horizon, 6]
        """
        # Compute action space center and std from bounds
        if action_space is not None and isinstance(action_space, gym.spaces.Box):
            action_center = (action_space.low + action_space.high) / 2.0
            action_range = action_space.high - action_space.low

            # Compute std values if not already computed
            if self.init_std is None:
                self.init_std = action_range * self.init_std_scale
                self.min_std = action_range * self.min_std_scale
                self.max_std = action_range * self.max_std_scale

                logger.debug(f"Computed std from action_space:")
                logger.debug(f"  Action range: {action_range}")
                logger.debug(f"  init_std: {self.init_std}")
                logger.debug(f"  min_std: {self.min_std}")
                logger.debug(f"  max_std: {self.max_std}")
        else:
            # Fallback to default values
            action_center = np.zeros(6)
            if self.init_std is None:
                self.init_std = self._fallback_init_std
                self.min_std = self._fallback_min_std
                self.max_std = self._fallback_max_std
                logger.debug("Using fallback std values")

        if self._is_first_step or self._prev_offset_mean is None:
            # Cold start: initialize with action space center
            offset_mean = np.tile(action_center, (self.horizon, 1))
            offset_std = np.tile(self.init_std, (self.horizon, 1))
        else:
            # Warm start: time-shift previous mean
            offset_mean = np.tile(action_center, (self.horizon, 1))
            offset_mean[:-1] = self._prev_offset_mean[1:]  # Shift forward
            # offset_mean[-1] uses action_center (new last step)

            offset_std = np.tile(self.init_std, (self.horizon, 1))
            offset_std[:-1] = self._prev_offset_std[1:]
            # offset_std[-1] uses init_std (new last step)

        # Save for action_generator
        self._current_offset_mean = offset_mean
        self._current_offset_std = offset_std

        return offset_mean, offset_std

    def _save_for_warm_start(self, offset_mean: np.ndarray,
                             offset_std: np.ndarray) -> None:
        """
        Save distribution parameters for next solve.

        Args:
            offset_mean: Current mean to save
            offset_std: Current std to save
        """
        self._prev_offset_mean = offset_mean.copy()
        self._prev_offset_std = offset_std.copy()
        self._is_first_step = False

    def reset(self) -> None:
        """
        Reset solver state (called at episode start).

        Clears warm-start cache.
        """
        self._prev_offset_mean = None
        self._prev_offset_std = None
        self._is_first_step = True
        self._current_offset_mean = None
        self._current_offset_std = None

        logger.debug("Reset: Cleared warm-start cache")
