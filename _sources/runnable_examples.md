# Runnable Examples

Four manipulation tasks demonstrating different cage, solver, and TSIP configurations.

---

(object-pushing-simulation)=
## Object Pushing (Simulation)

Multi-object herding with orbital cage constraints. 64 polygon objects are pushed along a circular trajectory using 16 parallel ManiSkill environments for action evaluation.

| Component | Implementation | Details |
|---|---|---|
| TSIP Backend | `PushBackend` | ManiSkill, 16 parallel envs, `pd_joint_delta_pos` control |
| TSIP | `SimulationBasedTSIP` | Simulation-based forward prediction |
| Cage | `CircularCage` | Orbital motion (radius=0.28, orbit_radius=0.2, orbit_speed=0.1) |
| Solver | `GeometricOptimizer` | Enumerates 16 discrete push directions |
| Sampler | `DiscreteSampler` | All actions evaluated in parallel |
| Executor | `PushingTaskExecutor` | Independent single-object YCB environment |

```bash
python examples/tasks/object_pushing/main.py
```

(object-pushing-learned-world-model)=
## Object Pushing (Learned World Model)

Cage-constrained planning in pixel space using a Diamond diffusion world model as TSIP. The cage operates on rendered images rather than state vectors.

| Component | Implementation | Details |
|---|---|---|
| TSIP Backend | `DiffusionBackend` | Diamond world model (`push16.pt`) |
| TSIP | `LearningBasedTSIP` | Learned dynamics in pixel space |
| Cage | `CircularPixelCage` | Pixel-space circular constraint (radius=22px, orbit_radius=42px) |
| Solver | `PixelOptimizer` | Pixel-space distance optimization |
| Sampler | Built into solver | 16 candidate actions |
| Executor | `PushingTaskExecutor` | Independent single-object YCB environment |

```bash
python examples/tasks/object_pushing/main_pixel.py
```

With iterative feedback loop (re-encodes real observations back into the diffusion model):

```bash
python examples/tasks/object_pushing/main_pixel_feedback.py
```

(object-catching)=
## Object Catching

A Franka arm with a plate end-effector catches falling balls. The policy sampler generates candidate actions from a trained RL policy distribution, and CAGE selects the best one.

| Component | Implementation | Details |
|---|---|---|
| TSIP Backend | `CatchBackend` | ManiSkill `PlateCatch-v1`, 16 DRIS copies per env |
| TSIP | `SimulationBasedTSIP` | Parallel rollout for action evaluation |
| Cage | `PlateCage` | Distance + velocity cost (plate_radius=0.12) |
| Solver | `PolicySampler` | Samples from RL policy distribution (8 candidates) |
| Sampler | Integrated in `PolicySampler` | Stochastic sampling with action chunking (horizon=8) |
| Executor | `CatchingTaskExecutor` | Single environment with falling balls |

```bash
# Baseline (direct policy, no CAGE)
python examples/tasks/object_catching/main.py --num_samples 0

# CAGE mode
python examples/tasks/object_catching/main.py --num_samples 8 --num_objs_tsip 16
```

(object-picking)=
## Object Picking

MPPI-based trajectory optimization for pick-and-place. A 3D cage defines a time-varying trajectory of waypoints, and MPPI samples action perturbations to follow it.

| Component | Implementation | Details |
|---|---|---|
| TSIP Backend | `PickBackend` | ManiSkill, 10 parallel envs, 16 target objects |
| TSIP | `SimulationBasedTSIP` | Multi-step rollout (horizon=3) |
| Cage | `Geometric3DCage` | 50 waypoints, radius=0.15, variance-based cost |
| Solver | `MPPIOptimizer` | 10 samples, 3 elites, 2 iterations, temperature=1.0 |
| Sampler | Gaussian (built into MPPI) | Perturbation sampling around mean trajectory |
| Executor | `PickingTaskExecutor` | Plan-then-execute with full trajectory replay |

```bash
python examples/tasks/object_picking/main.py
```
