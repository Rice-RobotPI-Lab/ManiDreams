# Solvers

**Solvers** are responsible for action selection in ManiDreams. A solver generates candidate actions, evaluates them using the TSIP and Cage, and returns the best action. Solvers are organized into two families: **Samplers** (policy-based) and **Optimizers** (planning-based).

## Abstract Interface

```python
class SolverBase(ABC):
    @abstractmethod
    def solve(self, action_space, cage, tsip, current_dris, verbose=False)
        -> Tuple[Any, List[float], List[bool]]:
        """Select the best action. Returns (best_action, costs, validations)."""

    @abstractmethod
    def action_generator(self, action_space=None) -> List[Any]:
        """Generate candidate actions for evaluation."""

    def reset(self) -> None: ...
```

The `solve()` method is the primary entry point. When operating in CAGE mode, it internally calls `action_generator()` to produce candidates, `tsip.next()` to predict outcomes, `cage.evaluate()`/`cage.validate()` to score them, and returns the best action. The selection strategy prefers valid actions first, then selects the one with lowest cost.

---

## Samplers

Samplers (`src/manidreams/solvers/samplers/`) provide policy-based action generation.

### SamplerBase

Abstract base class defining `sample(num_samples) -> List[Any]`.

### DiscreteSampler

Enumerates all actions in a discrete action space as single-step trajectories `[[0], [1], ..., [n-1]]`. Used with GeometricOptimizer for tasks with discrete action spaces.

### GaussianSampler

Samples continuous action trajectories from a Gaussian distribution `N(mean, std)` with shape `[horizon, action_dim]`. Supports online distribution updates for CEM-style optimization. Used with MPPIOptimizer.

### PolicySampler

The primary solver for RL-based tasks. Extends `SolverBase` (not `SamplerBase`) because it implements the full solve loop. Operates in two modes:

**Baseline mode** (`num_samples=1` or `cage=None`):
```
obs → policy.get_action(obs, deterministic=True) → single action → execute
```
No TSIP evaluation. No cage checking. The policy's output is directly executed.

**CAGE mode** (`num_samples > 1` with cage and TSIP):
```
obs → policy.sample_actions(N) → N candidate actions
    → tsip.next(dris, [a_0, ..., a_{N-1}]) → [DRIS_0, ..., DRIS_{N-1}]
    → cage.evaluate([DRIS_0, ..., DRIS_{N-1}]) → [cost_0, ..., cost_{N-1}]
    → cage.validate([DRIS_0, ..., DRIS_{N-1}]) → [valid_0, ..., valid_{N-1}]
    → select best: prefer valid, then lowest cost
    → execute best action
```

Constructor:
```python
PolicySampler(
    policy_model,          # RL policy with get_action() interface
    obs_encoder=None,      # Optional encoder (e.g., PointCloudAE)
    obs_mapper=None,       # Maps raw observations to model input
    action_mapper=None,    # Maps model output to control commands
    num_samples=1,         # Number of candidate actions (1 = baseline)
    deterministic=True,    # Use policy mean vs. stochastic sampling
    device='cpu',
)
```

Optional components:
- `obs_encoder`: Transforms raw observations before feeding to the policy (e.g., PointCloudAE encodes point clouds to a latent vector)
- `obs_mapper`: Maps raw environment observations to the format expected by the policy
- `action_mapper`: Maps policy output to the control command format expected by the backend

---

## Optimizers

Optimizers (`src/manidreams/solvers/optimizers/`) provide planning-based action selection via trajectory optimization.

### MPCOptimizer

Base MPC solver with configurable horizon, discount factor, and trajectory rollout. Implements `_rollout()` (serial or parallel), `_evaluate()` (discounted cumulative cost over trajectory), and `_select_best()`. Automatically detects whether the TSIP supports parallel rollout and falls back to serial execution otherwise.

For planning-based solvers, the pipeline extends to multi-step trajectory evaluation:

```
1. Generate K trajectories of horizon H:
   T_k = [a_k^0, a_k^1, ..., a_k^{H-1}]   for k = 0, ..., K-1

2. Rollout each trajectory through TSIP:
   DRIS_k^0 → (a_k^0) → DRIS_k^1 → (a_k^1) → ... → DRIS_k^{H-1}

3. Evaluate trajectory cost with discounting:
   cost_k = sum_{h=0}^{H-1} discount^h * cage.evaluate(DRIS_k^h)
   valid_k = all(cage.validate(DRIS_k^h) for h in 0..H-1)

4. Select best trajectory: k* = argmin_{k: valid_k} cost_k

5. Execute first action of best trajectory: a_{k*}^0
```

### GeometricOptimizer

Extends MPCOptimizer. Uses DiscreteSampler to enumerate all actions in a discrete action space and evaluates single-step trajectories. Used for the multi-object pushing task where the action space is 16 discrete directions.

### MPPIOptimizer

Implements Model Predictive Path Integral with Cross-Entropy Method (CEM) for continuous offset actions. Operates in 6D offset space `[dx, dy, dz, droll, dpitch, dyaw]`.

Key features:
- Multi-iteration CEM refinement (`num_iterations`, default 6)
- MPPI weighting: `w_i ~ exp(-temperature * cost_i)`, invalid samples receive zero weight
- Warm-start: caches offset distribution between timesteps for temporal coherence
- Configurable standard deviation bounds (`min_std`, `max_std`)
- Used for the picking task with multi-step horizon planning

### NaiveOptimizer

Geometric heuristic that selects actions based on the angle between the cage motion direction and the vector from cage center to DRIS center. No TSIP prediction — pure geometric reasoning.

### PixelOptimizer

Delegates action selection to the cage's `compute_direction()` method. No TSIP prediction. Used with CircularPixelCage where the cage directly computes the optimal 16-directional push action from the PSS-to-cage-center vector.

---

## Quick API Reference

```python
SolverBase
  .solve(action_space, cage, tsip, current_dris) -> (action, costs, validations)
  .action_generator(action_space) -> List[actions]
  .reset()

PolicySampler(policy_model, obs_encoder=None, obs_mapper=None, action_mapper=None,
              num_samples=1, deterministic=True, device='cpu')
  .initialize(env, initial_obs)
  .solve(action_space, cage, tsip, current_dris) -> (best_action, costs, validations)
  .sample_actions(num_samples) -> Tensor
  .update_after_step(new_obs)
  .reset()

MPPIOptimizer(config={'horizon': 5, 'num_samples': 16, 'num_elites': 8,
                       'num_iterations': 6, 'temperature': 0.5, 'discount': 0.95})
  .solve(action_space, cage, tsip, current_dris) -> (best_offset, costs, validations)
  .reset()
```
