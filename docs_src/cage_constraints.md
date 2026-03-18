# Cage Constraints

The **Cage** is the core constraint mechanism in ManiDreams. It defines a spatial boundary that the system tries to maintain around manipulated objects. The cage provides two evaluation functions: a continuous cost (for ranking candidate actions) and a binary validation (for constraint satisfaction).

## Abstract Interface

```python
class Cage(ABC):
    def __init__(self, state_space, time_varying=False, trajectory_params=None):
        self.controller = CageController(time_varying, trajectory_params)
        self.parameters = {}       # Current parameter values
        self.initialized = False

    @abstractmethod
    def evaluate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[float]:
        """Compute cost for each state. Lower is better."""

    @abstractmethod
    def validate(self, dris_input: Union[DRIS, List[DRIS]]) -> List[bool]:
        """Check if each state satisfies the cage constraint."""

    @abstractmethod
    def set_cage(self, region) -> None: ...
    @abstractmethod
    def initialize(self) -> None: ...
    @abstractmethod
    def _define_parameters(self) -> Dict[str, Any]: ...
    @abstractmethod
    def _update_from_parameters(self) -> None: ...

    def update(self, **kwargs) -> None: ...
    def apply_controller_updates(self, timestep: int) -> bool: ...
    def get_boundary(self) -> Dict: ...
```

### CageController

`CageController` (`src/manidreams/base/cage.py`) manages time-varying cage behavior by mapping timesteps to parameter updates. It supports both explicit trajectory dictionaries (`set_trajectory()`) and automatic trajectory generation (`generate_trajectory()`). At each timestep, `cage.apply_controller_updates(t)` queries the controller and updates the cage's internal parameters (e.g., center position, radius).

## The Cage-Constrained Algorithm

At each timestep, the framework executes the following pipeline (implemented in `ManiDreamsEnv.dream()`):

**Algorithm 1: Cage-Constrained Action Selection**:

```
Input: TSIP, Cage, Solver, horizon T
Output: trajectory τ, action_history, cage_history

for t = 0, 1, ..., T-1:
    1. Update cage parameters: cage.apply_controller_updates(t)
    2. Sync executor state to TSIP eval environments
    3. Generate candidate actions: actions = solver.action_generator(action_space)
    4. Evaluate in parallel: next_dris_list = tsip.next(current_dris, actions)
    5. Score candidates:
       costs = cage.evaluate(next_dris_list)
       validations = cage.validate(next_dris_list)
    6. Select best: i* = argmin_{i: validations[i]=True} costs[i]
       (if no valid action, select i* = argmin_i costs[i])
    7. Execute: current_dris = tsip.next(current_dris, actions[i*])
    8. Record: τ.append(current_dris), action_history.append(actions[i*])
```

Steps 3-6 are encapsulated in `solver.solve()`. The separation between TSIP evaluation (steps 3-5) and execution (step 7) enables the plan-then-execute paradigm used in the picking task.

## Cage Implementations

All cage implementations reside in `src/manidreams/cages/` and extend the abstract `Cage` class.

| Cage Class | Input Modality | Cost Function | Validation | Time-Varying |
|------------|----------------|---------------|------------|--------------|
| `CircularCage` | 2D/3D positions | 0.2 * dist_to_center + 0.8 * convex_hull_area | Within circular radius | Yes (orbital) |
| `Geometric3DCage` | 3D trajectory poses | Distance to moving center or position variance | Within spherical radius | Yes (trajectory) |
| `PlateCage` | 13-dim [obj_pos, obj_vel, tcp_pos, tcp_quat] | 0.7 * horizontal_dist + 0.3 * velocity_norm | Ball above plate and within dist threshold | No |
| `DRISCage` | DRIS with variance context | dist_to_goal + lambda * sum(variance) | Distance within success radius | No |
| `CircularPixelCage` | RGB image [H, W, 3] | Pixel distance from PSS to cage center | PSS within cage circle | Yes (orbital) |
| `CustomTrajectoryCage` | 3D positions | Configurable trajectory-based cost | Within trajectory-defined boundary | Yes (trajectory) |

### CircularCage

Implements a 2D circular constraint with optional orbital motion. The cage center moves along a circular orbit at each timestep (controlled by `orbit_radius` and `orbit_speed`). The evaluate function combines distance-to-center (task progress) with convex-hull area of object positions (spread minimization). Used for the multi-object pushing task.

### Geometric3DCage

Supports pre-computed 3D (or 6D with orientation) trajectories. At each timestep, the cage center and optionally orientation update to the next waypoint in the trajectory. It supports two cost types: `'distance'` (L2 distance from state to cage center) and `'variance'` (position variance across multiple objects). Used for the picking task with a two-segment approach-then-push trajectory.

### PlateCage

Evaluates ball-catching configurations using a plate-relative coordinate system. It extracts ball and plate states from a 13-dimensional DRIS observation, computes horizontal distance (`u`) and vertical distance (`w`) between the ball and the plate surface normal, and validates that the ball is above the plate and within catching range. The cost function weights horizontal distance (0.7) against velocity magnitude (0.3).

### DRISCage

Designed for uncertainty-aware evaluation with DRIS copies. It reads `mean_position` and `variance` from the DRIS context and computes:

```
cost = ||mean_position - goal_position|| + lambda_var * sum(variance)
```

where the first term drives task progress and the second term penalizes high uncertainty. The `lambda_var` parameter controls the trade-off between progress and robustness.

### CircularPixelCage

Operates directly in image space, detecting the PSS (Physical State Surrogate) via color thresholding on the red channel (R > 0.7, G < 0.4, B < 0.4). It computes a 16-directional pushing action (22.5 degree increments) to move the PSS toward the cage center. The cage center follows an orbital trajectory in pixel coordinates.

## Quick API Reference

```python
Cage(state_space, time_varying=False, trajectory_params=None)
  .evaluate(dris_input) -> List[float]          # Cost (lower = better)
  .validate(dris_input) -> List[bool]           # Constraint check
  .update(**kwargs)                             # Update parameters
  .apply_controller_updates(timestep) -> bool   # Time-varying update
  .get_boundary() -> Dict                       # Constraint geometry
  .validate_state(dris) -> bool                 # Single-element convenience
```
