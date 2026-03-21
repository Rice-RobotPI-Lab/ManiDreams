# TSIP

**TSIP (Task-Specific Intuitive Physics)** is the forward model abstraction in ManiDreams. Given the current state (DRIS) and an action, a TSIP predicts the next state. This abstraction allows the same solver and cage to work with either a physics simulator or a learned world model.

## Abstract Interface

```python
class TSIPBase(ABC):
    @abstractmethod
    def next(self, dris: DRIS, action: Union[Any, List[Any]], cage=None) -> Union[DRIS, List[DRIS]]:
        """Compute next DRIS state given current state and action(s).

        Single action -> single next DRIS
        List of N actions -> list of N next DRIS (parallel evaluation)
        """

    def reset(self) -> None: ...
```

The dual-mode `next()` interface is critical: when given a list of N actions, the TSIP evaluates all of them in parallel and returns N resulting DRIS states. This enables the cage-constrained algorithm to evaluate multiple candidate actions simultaneously.

## SimulationBasedTSIP

`SimulationBasedTSIP` (`src/manidreams/physics/simulation_tsip.py`) uses a physics simulator (ManiSkill/SAPIEN) for forward prediction. It manages parallel GPU-accelerated environments for simultaneous action evaluation.

```python
class SimulationBasedTSIP(TSIPBase):
    def __init__(self, backend: SimulationBackend, env_config=None,
                 context_info=None, num_rollout_envs=0)
```

Key features:
- Creates vectorized environments via `backend.create_environment()` for parallel action evaluation
- Optional separate rollout environment (`num_rollout_envs > 0`) for MPPI multi-step trajectory optimization
- State synchronization: `set_dris()` broadcasts the executor's current state to all parallel evaluation environments
- `next(dris, actions)` — steps all environments in parallel and returns resulting DRIS states via `backend.state2dris()`
- `rollout_step(offset_batch)` — batch trajectory rollout for MPPI optimizer

### SimulationBackend

`SimulationBackend` (abstract) defines the interface that task-specific simulator wrappers must implement:

```python
class SimulationBackend(ABC):
    def create_environment(env_config) -> gym.Env       # Create the environment
    def get_state(env) -> Dict[str, np.ndarray]          # Extract state for broadcasting
    def set_state(env, state) -> None                    # Set state on eval environments
    def state2dris(observations) -> List[DRIS]           # Convert observations to DRIS
    def dris2state(dris) -> Dict[str, np.ndarray]        # Convert DRIS back to state
    def step_act(actions, env, cage, single_action) -> Any  # Execute action(s)
    def load_env(context) -> None                        # Load environment configuration
    def load_object(context) -> None                     # Load object assets
    def load_robot(context) -> None                      # Load robot configuration
```

## LearningBasedTSIP

`LearningBasedTSIP` (`src/manidreams/physics/learned_tsip.py`) uses a learned world model (e.g., Diamond diffusion model) for forward prediction. Unlike SimulationBasedTSIP, the backend maintains internal state (the learned model's hidden state).

```python
class LearningBasedTSIP(TSIPBase):
    def __init__(self, backend: LearnedBackend, model_config=None, context_info=None)
```

### LearnedBackend

```python
class LearnedBackend(ABC):
    def load_model(model_config) -> Any                  # Load pretrained model
    def reset() -> DRIS                                  # Reset and return initial DRIS
    def predict_step(model, current_state, action) -> Tuple[Any, Dict]  # Predict next state
    def get_dris() -> DRIS                               # Get current DRIS
    def set_dris(dris) -> None                           # Set internal state from DRIS
```

## Simulation vs. Learned TSIP

The same Cage and Solver pipeline works with either TSIP backend. The key differences:

| Feature | SimulationBasedTSIP | LearningBasedTSIP |
|---------|-------------------|-----------------|
| Engine | ManiSkill/SAPIEN (GPU physics) | Trained diffusion model |
| Parallel evaluation | N environments natively | Sequential |
| Determinism | Deterministic | Stochastic (diffusion sampling) |
| State management | `set_state()` broadcasts to eval envs | Backend maintains internal model state |
| Observation space | Any (state, image, point cloud) | 64x64 RGB images |
| Action space | Any (continuous, discrete) | 16 discrete directions |

The learned TSIP enables cage-constrained planning with a world model instead of a physics simulator, for scenarios where an accurate simulator is unavailable. It trades off prediction fidelity for generality.

## Plan-Then-Execute Paradigm

Some tasks (e.g., object picking) use a two-phase workflow:

**Phase 1: Planning** — Run `dream()` on TSIP with the solver and cage. The TSIP uses N parallel environments (e.g., 16 envs on GPU) to evaluate candidates. No rendering, no executor. This produces a planned action sequence and cage parameter history.

**Phase 2: Execution** — Replay the planned action sequence on an independent executor environment (e.g., 1 env with rendering on a separate GPU). The executor creates its own simulation instance, ensuring complete isolation from the planning model.

```python
# Phase 1: Plan
env = ManiDreamsEnv(tsip=tsip, solver=mppi_optimizer, cage=cage, ...)
trajectory, action_history, cage_history = env.dream(horizon=80)

# Phase 2: Execute
executor = PickingTaskExecutor()
executor.initialize(env_config)
for action, cage_params in zip(action_history, cage_history):
    executor.set_cage(cage_params)
    obs, feedback = executor.execute(action)
executor.close_and_retract()  # Grasp + lift
```

This separation allows the planning model to differ from the execution environment (e.g., simplified physics for planning, full-fidelity for execution) and facilitates sim-to-real transfer.

## Quick API Reference

```python
TSIPBase
  .next(dris, action) -> DRIS | List[DRIS]    # Forward predict
  .reset()                                      # Reset state

SimulationBasedTSIP(backend, env_config=None, context_info=None, num_rollout_envs=0)
  .env                                          # The underlying gym.Env
  .reset() -> DRIS
  .next(dris, action) -> DRIS | List[DRIS]     # Step in parallel envs
  .get_dris(env_indices=None) -> List[DRIS]
  .set_dris(dris, env_indices=None)
  .rollout_step(offset_batch) -> List[DRIS]    # MPPI batch rollout
  .set_rollout_state(dris)
  .close()
```
