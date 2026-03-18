"""
Policy-based action sampler for ManiDreams.

Combines RL policy execution with optional CAGE evaluation.
Automatically switches between baseline (direct policy) and CAGE
(sample N → evaluate → select best) modes.
"""

import logging
from typing import Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from manidreams.base.solver import SolverBase

logger = logging.getLogger(__name__)


class PolicySampler(SolverBase):
    """Policy-based action sampler with optional CAGE evaluation.

    Two modes determined at solve() time by whether cage/tsip are provided:
    - Baseline (cage=None): single deterministic action, returns (action, None, None)
    - CAGE (cage+tsip provided): sample N actions → TSIP evaluate → select best

    When horizon > 1, operates in action-chunk mode:
    - Samples N candidate trajectories (each of length horizon) from the policy
    - Evaluates each trajectory via multi-step TSIP rollout + CAGE
    - Returns the best trajectory as a list of actions

    Works with any policy model that has a get_action() method.
    """

    def __init__(self,
                 policy_model: nn.Module,
                 obs_encoder: nn.Module = None,
                 obs_mapper: Any = None,
                 action_mapper: Any = None,
                 num_samples: int = 1,
                 horizon: int = 1,
                 discount: float = 0.95,
                 deterministic: bool = True,
                 device: str = 'cpu',
                 name: str = "PolicySampler"):
        """
        Initialize PolicySampler.

        Args:
            policy_model: Trained policy with get_action() method
            obs_encoder: Optional observation encoder (e.g., PointCloudAE)
            obs_mapper: Mapper from raw obs to model input (e.g., ObsDRISMapper)
            action_mapper: Mapper from model output to control (e.g., ActionMapperStep)
            num_samples: Number of actions to sample (1=baseline, N>1=CAGE)
            horizon: Action chunk length. 1=single-step (default), >1=multi-step trajectory
            discount: Discount factor for multi-step cost accumulation (default: 0.95)
            deterministic: If True, use policy mean instead of sampling
            device: Torch device ('cpu' or 'cuda')
            name: Solver name
        """
        super().__init__(name=name)
        self.policy = policy_model
        self.encoder = obs_encoder
        self.obs_mapper = obs_mapper
        self.action_mapper = action_mapper
        self.num_samples = num_samples
        self.horizon = horizon
        self.discount = discount
        self.deterministic = deterministic
        self.device = device

        # Internal state for observation mapping
        self._dris_latent = None
        self._tcpn_m = None
        self._current_obs = None
        self._initialized = False

    def initialize(self, env, initial_obs):
        """
        Initialize solver with environment and initial observation.

        Registers reference observations for mappers and performs
        initial observation mapping and encoding.

        Args:
            env: ManiSkill environment
            initial_obs: Initial observation from env.reset()
        """
        if self.obs_mapper is not None:
            self.obs_mapper.register_ref_obs(initial_obs)
        if self.action_mapper is not None:
            self.action_mapper.register_ref_obs(initial_obs)

        self._update_latent_state(initial_obs)
        self._current_obs = initial_obs
        self._initialized = True

    def _update_latent_state(self, obs):
        """Update latent state from observation."""
        if self.obs_mapper is not None:
            obs_m, tcpn_m = self.obs_mapper.map(obs)
        else:
            obs_m = obs
            tcpn_m = None

        if self.encoder is not None:
            dris_latent = self.encoder.encode(obs_m)
        else:
            dris_latent = obs_m

        self._dris_latent = dris_latent
        self._tcpn_m = tcpn_m

    def _get_single_action(self):
        """Get a single action from policy (deterministic)."""
        with torch.no_grad():
            if hasattr(self.policy, 'get_action'):
                action = self.policy.get_action(
                    self._dris_latent,
                    self._tcpn_m,
                    deterministic=True
                ).detach()
            else:
                action = self.policy(self._dris_latent).detach()
        return action

    def _map_action_to_control(self, action):
        """Map a single action tensor to control space."""
        if self.action_mapper is not None:
            return self.action_mapper.map(self._current_obs, action)
        if isinstance(action, torch.Tensor):
            return action.cpu().numpy()
        return action

    def sample_actions(self, num_samples: int = None) -> torch.Tensor:
        """
        Sample multiple actions from policy distribution.

        Args:
            num_samples: Number of actions to sample (default: self.num_samples)

        Returns:
            Tensor of sampled actions (num_samples, action_dim)
        """
        if num_samples is None:
            num_samples = self.num_samples

        batch_latent = self._dris_latent.expand(num_samples, -1)
        batch_tcpn = self._tcpn_m.expand(num_samples, -1) if self._tcpn_m is not None else None

        with torch.no_grad():
            if hasattr(self.policy, 'get_action'):
                actions = self.policy.get_action(
                    batch_latent,
                    batch_tcpn,
                    deterministic=self.deterministic
                ).detach()
            else:
                actions = self.policy(batch_latent).detach()

        return actions

    def action_generator(self, action_space: Optional[gym.spaces.Space] = None) -> List[Any]:
        """
        Generate list of control-space actions for parallel evaluation.

        Args:
            action_space: Action space (not used, policy defines action space)

        Returns:
            List of control actions for parallel execution
        """
        if not self._initialized:
            raise RuntimeError("PolicySampler not initialized. Call initialize() first.")

        actions = self.sample_actions()
        ctrl_actions = []
        for i in range(actions.shape[0]):
            action = actions[i:i+1] if self.action_mapper is not None else actions[i]
            ctrl_actions.append(self._map_action_to_control(action))
        return ctrl_actions

    def solve(self, action_space: Optional[gym.spaces.Space],
              cage: Any,
              tsip: Any,
              current_dris: Any,
              verbose: bool = False,
              on_plan_step: Any = None) -> Tuple[Any, Optional[List[float]], Optional[List[bool]]]:
        """
        Select action(s) using policy, with optional CAGE evaluation.

        Automatically selects mode:
        - If cage/tsip are None: baseline (single deterministic action)
        - If cage/tsip provided, horizon=1: CAGE single-step
        - If cage/tsip provided, horizon>1: CAGE multi-step (action chunk)

        Args:
            action_space: Action space (not used)
            cage: Cage constraint (None for baseline mode)
            tsip: TSIP instance (None for baseline mode)
            current_dris: Current DRIS state
            verbose: Whether to print debug info
            on_plan_step: Optional callback called after each planning step
                in chunked mode. Signature: on_plan_step(step, best_idx, costs, validations)

        Returns:
            When horizon=1: (best_action, costs, validations)
            When horizon>1: (best_trajectory, costs, validations)
                where best_trajectory is a list of horizon actions
        """
        if not self._initialized:
            raise RuntimeError("PolicySampler not initialized. Call initialize() first.")

        if cage is None or tsip is None:
            return self._solve_baseline(verbose)
        if self.horizon > 1:
            return self._solve_cage_chunked(action_space, cage, tsip, current_dris, verbose, on_plan_step)
        return self._solve_cage(action_space, cage, tsip, current_dris, verbose)

    def _solve_baseline(self, verbose: bool) -> Tuple[Any, None, None]:
        """Baseline mode: single deterministic action."""
        action = self._get_single_action()
        ctrl_t = self._map_action_to_control(action)

        if verbose:
            logger.info(f"Baseline action: {ctrl_t}")

        return ctrl_t, None, None

    def _solve_cage(self, action_space, cage, tsip, current_dris,
                    verbose: bool) -> Tuple[Any, List[float], List[bool]]:
        """CAGE mode: sample N → evaluate → select best."""
        ctrl_actions = self.action_generator(action_space)

        if verbose:
            logger.info(f"Sampled {len(ctrl_actions)} actions")

        result_dris_list = tsip.next(current_dris, ctrl_actions, cage=cage)

        if verbose:
            logger.info(f"Got {len(result_dris_list)} result states")

        costs = cage.evaluate(result_dris_list)
        validations = cage.validate(result_dris_list)

        if verbose:
            costs_str = [f"{c:.3f}" for c in costs]
            logger.info(f"Costs: {costs_str}")
            logger.info(f"Validations: {validations}")

        best_idx = self._select_best(costs, validations, verbose)
        best_action = ctrl_actions[best_idx]

        if verbose:
            logger.info(f"Selected action {best_idx} with cost {costs[best_idx]:.3f}")

        return best_action, costs, validations

    def _select_best(self, costs: List[float], validations: List[bool],
                     verbose: bool = False) -> int:
        """
        Select best action: prefer valid actions, then lowest cost.

        Args:
            costs: List of costs from CAGE evaluation
            validations: List of validation results
            verbose: Whether to print debug info

        Returns:
            Index of best action
        """
        valid_indices = [i for i, v in enumerate(validations) if v]

        if valid_indices:
            valid_costs = [costs[i] for i in valid_indices]
            best_local_idx = np.argmin(valid_costs)
            best_idx = valid_indices[best_local_idx]
            if verbose:
                logger.info(f"{len(valid_indices)} valid actions, best={best_idx}")
        else:
            best_idx = np.argmin(costs)
            if verbose:
                logger.info(f"No valid actions, lowest cost={best_idx}")

        return best_idx

    def _solve_cage_chunked(self, action_space, cage, tsip, current_dris,
                            verbose: bool,
                            on_plan_step: Any = None) -> Tuple[List[Any], List[float], List[bool]]:
        """CAGE multi-step mode: sequential greedy planning in TSIP.

        For each step in the horizon:
        1. Sample N actions from policy, evaluate via single-step TSIP + CAGE
        2. Select best action, save to buffer
        3. Sync best env's DRIS to all TSIP envs as next starting state
        4. Re-encode latent from TSIP obs for next step's policy sampling

        Args:
            on_plan_step: Optional callback after each planning step.
                Signature: on_plan_step(step, best_idx, costs, validations)

        Returns the full action buffer for executor to execute.
        """
        H = self.horizon
        action_buffer = []
        cur_dris = current_dris
        last_costs, last_validations = None, None

        if verbose:
            logger.info(f"Chunked greedy: {self.num_samples} samples x {H} horizon")

        for t in range(H):
            # 1. Sample N actions from policy + map to control
            ctrl_actions = self.action_generator(action_space)

            # 2. TSIP single-step evaluate (sets all envs to cur_dris, then steps)
            result_dris_list = tsip.next(cur_dris, ctrl_actions, cage=cage)

            # 3. CAGE evaluate
            costs = cage.evaluate(result_dris_list)
            validations = cage.validate(result_dris_list)

            # 4. Select best action
            best_idx = self._select_best(costs, validations, verbose)
            action_buffer.append(ctrl_actions[best_idx])

            if verbose:
                logger.info(f"  Chunk step {t}/{H}: best={best_idx}, "
                            f"cost={costs[best_idx]:.3f}, valid={validations[best_idx]}")

            # Planning visualization callback
            if on_plan_step is not None:
                on_plan_step(t, best_idx, costs, validations)

            # 5. Sync: use best env's DRIS as next starting state
            cur_dris = result_dris_list[best_idx]

            # 6. Re-encode latent from TSIP obs for next step's policy sampling
            if t < H - 1:
                tsip_obs = tsip._last_step_obs             # (N, obs_dim) tensor
                best_obs = tsip_obs[best_idx:best_idx+1]   # (1, obs_dim)
                self._update_latent_state(best_obs)
                self._current_obs = best_obs

            last_costs, last_validations = costs, validations

        return action_buffer, last_costs, last_validations

    def update_after_step(self, new_obs):
        """
        Update internal state after environment step.

        Must be called after env.step() to keep the solver's
        observation encoding up to date.

        Args:
            new_obs: New observation from env.step()
        """
        self._current_obs = new_obs
        self._update_latent_state(new_obs)

    def reset(self):
        """Reset solver state."""
        self._dris_latent = None
        self._tcpn_m = None
        self._current_obs = None
        self._initialized = False
