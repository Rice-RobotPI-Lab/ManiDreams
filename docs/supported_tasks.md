# Supported Tasks

ManiDreams includes five task categories demonstrating different aspects of the framework:

| Task | Environment | Cage | Solver | TSIP | Key Demonstration |
|------|-------------|------|--------|------|-------------------|
| Object Pushing (sim) | Custom 64-object ManiSkill | CircularCage | GeometricOptimizer | SimulationBasedTSIP | Multi-object herding with orbital cage |
| Object Pushing (learned) | Diamond world model | CircularPixelCage | PixelOptimizer | LearningBasedTSIP | Image-based caging with learned dynamics |
| Object Catching | PlateCatch-v1 ManiSkill | PlateCage | PolicySampler (CAGE) | SimulationBasedTSIP | RL + CAGE with 6D offsets |
| Object Picking | Custom PickEnv ManiSkill | Geometric3DCage | MPPIOptimizer | SimulationBasedTSIP | MPPI planning + plan-then-execute |
| ManiSkill Defaults | PushCube/PickCube/PushT | DRISCage | PolicySampler (CAGE) | SimulationBasedTSIP | DRIS uncertainty-aware evaluation |

Each task can be run in **baseline mode** (direct policy output, no CAGE) or **CAGE mode** (parallel evaluation with constraint selection) by changing the `--num_samples` argument.

---

## Running Example Tasks

### Tabletop Manipulation (PushCube, PickCube, PushT)

```bash
cd ManiDreams/

# Baseline mode — direct PPO policy, no CAGE
python examples/tasks/maniskill_defaults/main.py --task PushCube-v1

# CAGE mode — 8 candidate actions, 4 DRIS copies per eval env
python examples/tasks/maniskill_defaults/main.py \
    --task PushCube-v1 --num_samples 8 --n_dris_copies 4

# With custom parameters
python examples/tasks/maniskill_defaults/main.py \
    --task PickCube-v1 \
    --num_samples 16 \
    --n_dris_copies 8 \
    --lambda_var 0.2 \
    --pose_noise 0.05 0.05 0.0 0.0 0.0 0.1
```

Checkpoints are auto-detected from `examples/samplers/maniskill_defaults/ckpts/`. Use `--checkpoint /path/to/model.pt` to override.

### Object Catching

```bash
# Baseline mode
python examples/tasks/object_catching/main.py --num_samples 0

# CAGE mode with 8 samples, 16 DRIS copies per TSIP env
python examples/tasks/object_catching/main.py \
    --num_samples 8 --num_objs_tsip 16
```

### Object Pushing (simulation-based)

```bash
python examples/tasks/object_pushing/main.py
```

### Object Pushing (learned world model)

```bash
python examples/tasks/object_pushing/main_pixel.py \
    --model-dir /path/to/diamond_model --max-steps 500
```

