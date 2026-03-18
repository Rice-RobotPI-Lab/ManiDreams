# DRIS

DRIS (**Domain-Randomized Instance Set**) is the universal state representation in ManiDreams. It is a dataclass that carries observation data plus domain-randomization context through the entire pipeline, decoupling the observation format from the algorithms that consume it.

## Data Structure

```python
@dataclass
class DRIS:
    observation: Union[np.ndarray, Any]            # Core data: state vector, image, point cloud, etc.
    state_space: Any = None                        # Associated gym.Space
    context: Optional[Dict[str, Any]] = None       # Domain randomization metadata
    representation_type: str = "auto"              # "state", "image", "mixed", or "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)
```

DRIS is deliberately generic: `observation` can be a flat state vector, an RGB image, a point cloud, or any combination. The `context` dictionary carries task-specific metadata such as mean positions, variances across DRIS copies, or pixel-space coordinates. The `representation_type` field indicates the observation modality.

## Key Methods

- `randomize(context_space: ContextSpace) -> DRIS` — applies domain randomization by sampling from a `ContextSpace`
- `update_dris(new_observation)` — in-place observation update
- `copy() -> DRIS` — deep copy for parallel evaluation

## ContextSpace

`ContextSpace` (`src/manidreams/base/dris.py`) is an abstract base class for domain randomization parameter spaces. Concrete implementations define `sample() -> Dict` and `get_default() -> Dict` to produce randomization parameters.

## What DRIS Can Represent

| Use Case | `observation` | `context` |
|----------|--------------|-----------|
| Tabletop manipulation | Flat state vector `[qpos, qvel, obj_pose, ...]` | `{mean_position, variance, dris_poses}` |
| Object catching | `[obj_pos, obj_vel, tcp_pos, tcp_quat]` (13-dim) | `{ball_state, robot_state}` |
| Image-based pushing (Diamond) | RGB image `[H, W, 3]` | `{pss_center, cage_center}` |
| Point cloud manipulation | `[N, 3]` array | `{num_objects, segmentation}` |

## DRIS Copies and Uncertainty Quantification

For tasks that use the DRISBackend (tabletop manipulation), each evaluation environment contains not only the target object but also `m` DRIS copies — duplicates of the target object placed with random pose perturbations. This provides a physical mechanism for uncertainty quantification.

```
Environment layout with m=4 DRIS copies:

                  [copy_2]
          [copy_1]  [target]  [copy_3]
                  [copy_4]

Each copy has pose = target_pose + noise
  noise ~ Uniform(-pose_noise, +pose_noise)
  pose_noise = (dx, dy, dz, droll, dpitch, dyaw)
```

**Evaluation flow for N candidate actions with m copies:**

```
1. Executor has target object at position P

2. Sync to TSIP: broadcast P to all N evaluation environments.
   Each eval env also has m DRIS copies with randomized poses:

   Env 0: [target @ P] [copy_1 @ P+ε₁] [copy_2 @ P+ε₂] ... [copy_m @ P+εₘ]
   Env 1: [target @ P] [copy_1 @ P+ε₁] [copy_2 @ P+ε₂] ... [copy_m @ P+εₘ]
   ...
   Env N: [target @ P] [copy_1 @ P+ε₁] [copy_2 @ P+ε₂] ... [copy_m @ P+εₘ]

3. Step each env with its candidate action:
   Env i steps with action_i → all copies move → new positions

4. For each env i, compute:
   mean_position_i = mean of m copy positions
   variance_i      = var of m copy positions
   DRIS_i = DRIS(observation_i, context={mean_position_i, variance_i, dris_poses_i})

5. DRISCage evaluates each DRIS:
   cost_i = ||mean_position_i - goal||  +  lambda_var * sum(variance_i)
            ─────────────────────────     ─────────────────────────────
            task progress metric          uncertainty penalty
```

The variance term captures how sensitive the outcome is to initial pose uncertainty. Actions that produce consistent results across all copies (low variance) are preferred, even if their mean position is slightly further from the goal. The `lambda_var` hyperparameter (typically 0.1–0.2) controls this trade-off.

## Quick API Reference

```python
DRIS(observation, state_space=None, context=None, representation_type="auto", metadata={})
  .observation         # Core data (np.ndarray, image, etc.)
  .context             # Dict with domain randomization info
  .metadata            # Additional info
  .copy() -> DRIS      # Deep copy
  .update_dris(new_observation)
  .randomize(context_space) -> DRIS
```
