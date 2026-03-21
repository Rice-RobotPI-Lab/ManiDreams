# Supported Tasks

ManiDreams includes six task categories demonstrating different aspects of the framework:

| Task | Environment | Cage | Solver | TSIP | Key Demonstration |
|------|-------------|------|--------|------|-------------------|
| Object Pushing (sim) | Custom 64-object ManiSkill | CircularCage | GeometricOptimizer | SimulationBasedTSIP | Multi-object herding with orbital cage |
| Object Pushing (learned) | Diamond world model | CircularPixelCage | PixelOptimizer | LearningBasedTSIP | Image-based caging with learned dynamics |
| Object Catching | PlateCatch-v1 ManiSkill | PlateCage | PolicySampler (CAGE) | SimulationBasedTSIP | RL + CAGE with 6D offsets |
| Object Picking | Custom PickEnv ManiSkill | Geometric3DCage | MPPIOptimizer | SimulationBasedTSIP | MPPI planning + plan-then-execute |
| ManiSkill Defaults | PushCube/PickCube/PushT | DRISCage | PolicySampler (CAGE) | SimulationBasedTSIP | DRIS uncertainty-aware evaluation |
| Zero-shot Real2Sim | D415 + Newton XPBD | — (teleop) | — (teleop) | NewtonBackend | Real-time perception → domain-randomized sim |

Each task can be run in **baseline mode** (direct policy output, no CAGE) or **CAGE mode** (parallel evaluation with constraint selection) by changing the `--num_samples` argument.

---

## Running Example Tasks

(maniskill-default-tasks)=
### ManiSkill Default Tasks

PushCube, PickCube, PushT with PPO policies and optional CAGE enhancement. [Demo videos →](index.html#example-maniskill-defaults)

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

(object-catching)=
### Object Catching

Plate-based ball catching with optional CAGE enhancement. [Demo video →](index.html#example-catching)

```bash
# Baseline mode
python examples/tasks/object_catching/main.py --num_samples 0

# CAGE mode with 8 samples, 16 DRIS copies per TSIP env
python examples/tasks/object_catching/main.py \
    --num_samples 8 --num_objs_tsip 16
```

(object-pushing-simulation)=
### Object Pushing (Simulation)

Multi-object herding with orbital cage constraints in parallel ManiSkill environments. [Demo video →](index.html#example-pushing-sim)

```bash
python examples/tasks/object_pushing/main.py
```

(object-pushing-learned-world-model)=
### Object Pushing (Learned World Model)

Cage-constrained planning using a learned diffusion world model. [Demo video →](index.html#example-pushing-learned)

```bash
python examples/tasks/object_pushing/main_pixel.py \
    --model-dir /path/to/diamond_model --max-steps 500
```

(object-picking)=
### Object Picking

MPPI-based trajectory optimization with 3D cage constraints for pick-and-place. [Demo video →](index.html#example-picking)

```bash
python examples/tasks/object_picking/main.py
```

(zero-shot-real2sim)=
### Zero-shot Real-to-Sim Demo

Real-time object detection → Newton physics simulation with domain randomization. [Demo videos →](index.html#dris-real2sim)

This demo requires additional setup (RealSense D415, Fast-FoundationStereoPose, Newton). See the full [setup tutorial](zeroshot_real2sim.md).

```bash
python examples/tasks/zeroshot_real2sim_demo/main.py
```
